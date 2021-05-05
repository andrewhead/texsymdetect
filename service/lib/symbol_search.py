import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import scipy.spatial

from lib.instrument_tex import FontSize


@dataclass
class Point:
    x: float
    y: float


@dataclass
class PixelPosition:
    x: int
    y: int


@dataclass(frozen=True)
class Rectangle:
    """
    Rectangle within an image. Left and top refer to positions of pixels.
    """

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class Id:
    """
    To uniquely identify a symbol in the symbol search functionality (i.e., not confuse
    two symbols with each other), one needs both the MathML for the symbol, and the
    size it was rendered at.
    """

    mathml: str
    level: FontSize


@dataclass
class TokenInstance:
    id_: Id
    location: Rectangle


@dataclass
class SymbolInstance:
    id_: Id
    location: Rectangle


@dataclass
class TokenTemplate:
    symbol: Id
    images: List[np.array]


@dataclass
class Component:
    symbol_id: Id
    center: Point
    " Position of center of component, relative to center of anchor component. "


@dataclass
class SymbolTemplate:
    anchor: Id
    " Leftmost member of the composite template. "

    members: List[Component]
    " All members of the composite template except for the anchor. "


def create_bitstring_from_image(image: np.array, wildcard_padding: int = 0) -> str:
    """
    Convert image to flattened string representation appropriate for conducting efficient
    exact matching of pixesl. In the string, a '0' represents a blank pixel (e.g., white)
    and a '1' represents a non-blank pixel (e.g., gray or black).

    It is assumed that the input image is in grayscale format.

    If using this method to create a template to match against a larger image, set the
    'wildcard' to be the number of pixels difference between the width of the
    larger image and the template image. Wildcards (i.e., '.') will be inserted for these
    extra padded pixels, which will be able to match anything in the larger image.
    """

    thresholded = image.copy()
    blank_pixels = image == 255
    non_blank_pixels = image < 255

    thresholded[blank_pixels] = 0
    thresholded[non_blank_pixels] = 1

    s = ""
    for row_index, row in enumerate(thresholded):
        s += "".join([str(int(p)) for p in row.tolist()])
        if row_index < len(thresholded) - 1:
            s += "." * wildcard_padding

    return s


def find_in_bitstring(
    pattern_bitstring: str, image_bitstring: str, image_width: int
) -> Iterator[PixelPosition]:
    """
    Find appearances of pattern bit string (sequence of 0s, 1s, and wildcards ('.')) in
    an image bit string (sequence of 0s and 1s). Return a iterator over points (left, top)
    where the pattern can be found in the image. 'image_width' argument is used to convert
    character positions of the matches to 2D positions in the image.
    """

    search_start = 0
    pattern = re.compile(pattern_bitstring)
    while True:
        match = pattern.search(image_bitstring, pos=search_start)
        if match is None:
            break

        start_character = match.start()
        left = start_character % image_width
        top = math.floor(start_character / image_width)
        yield PixelPosition(left, top)

        search_start = start_character + 1


MathMl = str


def create_symbol_template(
    symbol_image: np.array,
    token_images: Dict[MathMl, Dict[FontSize, List[np.array]]],
    token_mathmls: Iterable[str],
) -> Optional[SymbolTemplate]:

    symbol_bitstring = create_bitstring_from_image(symbol_image)
    symbol_width = symbol_image.shape[1]

    # Find positions of child symbols in the composite symbol image.
    components: List[Component] = []
    for token_mathml in token_mathmls:
        for level, images in token_images[token_mathml].items():
            for child_image in images:
                child_height, child_width = child_image.shape
                child_bitstring = create_bitstring_from_image(
                    child_image, wildcard_padding=symbol_width - child_width
                )

                # Find the first appearance of the child symbol (from top-to-bottom)
                # in the image, and add it as a component to the template. Top-to-bottom search is
                # performed because of the order way find_in_bitstring operates (even though
                # it probably makes more sense to search left-to-right, given that the
                # child keys passed in as 'children' will likely be roughly in left-to-right order).
                for position in find_in_bitstring(
                    child_bitstring, symbol_bitstring, symbol_width
                ):
                    center = Point(
                        position.x + child_width / 2.0, position.y + child_height / 2.0
                    )
                    component = Component(Id(token_mathml, level), center)
                    if component not in components:
                        components.append(component)
                        break

    # Composite symbol needs at least one component.
    if not components:
        return None

    # Select 'anchor' for the template as the leftmost component.
    components.sort(key=lambda c: c.center.x)
    anchor = components.pop(0)

    # Normalize the positions of components relative to the anchor.
    for component in components:
        component.center.x -= anchor.center.x
        component.center.y -= anchor.center.y

    return SymbolTemplate(anchor.symbol_id, components)


class TokenIndex:
    " Index of appearances of all tokens on a page. "

    def __init__(self, tokens: Iterable[TokenInstance]) -> None:
        self._tokens: List[TokenInstance] = list(tokens)

        # Build a KD search tree over symbols to support faster spatial querying.
        token_centers = [
            (
                t.location.left + t.location.width / 2.0,
                t.location.top + t.location.height / 2.0,
            )
            for t in tokens
        ]
        self._tree = scipy.spatial.KDTree(token_centers)

    def get_instances(self, id_: Id = None) -> List[TokenInstance]:
        " Get all tokens with a specific key. "
        if not id_:
            return list(self._tokens)

        return [t for t in self._tokens if t.id_ == id_]

    def find(
        self, id_: Id, center: Point, tolerance: Optional[Point] = None,
    ) -> List[TokenInstance]:
        """
        Get all tokens near a specific point matching a specification for the token
        (its key and level). Matching tokens are returned if:

        * its center x falls within [center[0] - tolerance[0], center[0] + tolerance[0]]
        * its center y falls within [center[1] - tolerance[1], center[1] + tolerance[1]]
        """

        tolerance = tolerance or Point(1.0, 1.0)

        # Initial query for candidate symbols is made using the KDTree 'query_ball_point' method,
        # as it will in many cases filter symbols according to position in two-dimensional space
        # than an iteratively searching over a list of all symbols.
        radius = math.sqrt(tolerance.x * tolerance.x + tolerance.y * tolerance.y)
        nearby_points = self._tree.query_ball_point(x=[center.x, center.y], r=radius)

        matches = []
        for token_i in nearby_points:
            # Rule out symbols that are not the requested symbol.
            token = self._tokens[token_i]
            if token.id_ != id_:
                continue

            # Rule out symbols that are not within the tolerated distance of the query point.
            token_center_x = token.location.left + token.location.width / 2.0
            token_center_y = token.location.top + token.location.height / 2.0
            if (
                abs(token_center_x - center.x) > tolerance.x
                or abs(token_center_y - center.y) > tolerance.y
            ):
                continue

            matches.append(token)

        return matches


def find_symbols(template: SymbolTemplate, index: TokenIndex) -> Iterator[Rectangle]:
    """
    Search for appearances of a symbol given an index of tokens.
    """

    # Search for anchors---that is, leftmost glyphs in a symbol, relative
    # to which all other tokens in a composite symbol will be searched.
    anchor_candidates = index.get_instances(template.anchor)

    # For each anchor found, attempt to fill out the rest of the composite symbol template.
    for a in anchor_candidates:

        template_incomplete = False
        member_matches: List[TokenInstance] = []
        anchor_center_x = a.location.left + a.location.width / 2.0
        anchor_center_y = a.location.top + a.location.height / 2.0

        # For each expected member of the composite symbol (i.e., all simple symbols the composite
        # symbol should be made up of), search for appearances of the member at the expected
        # location relative to the anchor.
        for member in template.members:
            expected_center = Point(
                anchor_center_x + member.center.x, anchor_center_y + member.center.y
            )
            # Note that the tolerance for the position of a member symbol is higher the further away
            # that member is from the anchor, as it is assumed that TeX might insert or remove space
            # between members, which will accumulate the further away the member is from the anchor.
            tolerance = Point(
                math.ceil(abs(member.center.x) / 5.0) + 1,
                math.ceil(abs(member.center.y) / 5.0) + 1,
            )
            member_candidates = index.find(
                id_=member.symbol_id, center=expected_center, tolerance=tolerance,
            )
            # If multiple symbols could fill the member slot in the composite symbol, select the
            # leftmost symbol that has not yet been used to fill a slot.
            member_found = False
            member_candidates.sort(key=lambda c: c.location.left)
            for m in member_candidates:
                if m not in member_matches:
                    member_matches.append(m)
                    member_found = True
                    break

            # If any member slot of the template cannot be filled, a composite symbol cannot be
            # created. Advance to the next potential anchor.
            if not member_found:
                template_incomplete = True
                break

        # Create an instance of the composite symbol if the template has been completed.
        if not template_incomplete:
            tokens = [a] + member_matches
            left = min([t.location.left for t in tokens])
            top = min([t.location.top for t in tokens])
            right = max([t.location.left + t.location.width for t in tokens])
            bottom = max([t.location.top + t.location.height for t in tokens])
            yield Rectangle(left, top, right - left, bottom - top)
