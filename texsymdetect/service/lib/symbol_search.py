import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import scipy.spatial

from lib.image_processing import (
    Point,
    Rectangle,
    _contains_start_graphic,
    find_boxes_with_rgb,
    find_in_image,
)
from lib.instrument_tex import Detectable, FontSize
from lib.parse_formula_tex import TexSymbol, TexToken

logger = logging.getLogger("texsymdetect")

PageNumber = int
MathMl = str


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


def create_symbol_template(
    symbol_image: np.array,
    token_images: Dict[MathMl, Dict[FontSize, List[np.array]]],
    token_mathmls: Iterable[str],
    require_blank_border_around_tokens: bool = True,
) -> Optional[SymbolTemplate]:

    # Unpack token images into a 1-D list.
    token_image_list: List[np.array] = []
    mathmls: List[MathMl] = []
    font_sizes: List[FontSize] = []
    for mathml, sizes in token_images.items():
        if mathml not in token_mathmls:
            continue
        for font_size, images in sizes.items():
            for image in images:
                token_image_list.append(image)
                font_sizes.append(font_size)
                mathmls.append(mathml)

    # Search in image for tokens.
    rects = find_in_image(
        token_image_list,
        symbol_image,
        require_blank_border=require_blank_border_around_tokens,
    )

    # Unroll tokens into a 1-D list.
    rects_unrolled: List[Rectangle] = []
    mathmls_unrolled: List[MathMl] = []
    font_sizes_unrolled: List[FontSize] = []
    for mathml, font_size, rect_list in zip(mathmls, font_sizes, rects):
        for rect in rect_list:
            rects_unrolled.append(rect)
            mathmls_unrolled.append(mathml)
            font_sizes_unrolled.append(font_size)

    # Find positions of child symbols in the composite symbol image.
    components: List[Component] = []

    # Add tokens to the template left-to-right.
    for (mathml, font_size, rect) in sorted(
        zip(mathmls_unrolled, font_sizes_unrolled, rects_unrolled),
        key=lambda t: t[2].left,
    ):
        if mathml in token_mathmls:
            center = Point(rect.left + rect.width / 2.0, rect.top + rect.height / 2.0)
            component = Component(Id(mathml, font_size), center)
            if component not in components:
                components.append(component)

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

    # assert (
    #     False
    # ), "May want to filter out overlapping tokens... for instance, by blanking out the part of the image that matches."

    return SymbolTemplate(anchor.symbol_id, components)


def extract_templates(
    page_images: Dict[PageNumber, np.array], detectables: Sequence[Detectable],
) -> Tuple[Dict[Detectable, List[np.array]], Dict[Detectable, SymbolTemplate]]:
    """
    Given images of pages from a paper that has been modified to include appearances of many tokens
    and symbols (i.e., 'detectables'), extract templates for those tokens and symbols
    that can be used to identify them in other documents.

    Returns a collection of token templates (images), and symbol templates
    (a flexible template format).
    
    Note that both tokens and symbols must be passed in as detectables;
    symbols cannot be found without first finding their component tokens. All
    detectables should be provided in the order that they appear in the TeX,
    which should include all tokens first, followed by all symbols.
    """

    sorted_page_images = [page_images[pn] for pn in sorted(page_images.keys())]

    def dequeue_page() -> Optional[np.array]:
        " Remove image of the next page from the list of all pages in the document. "
        if not sorted_page_images:
            return None
        image = sorted_page_images.pop(0)
        return image

    page_image = dequeue_page()
    next_page_image = dequeue_page()

    # Scan all pages until the marker is found that suggests that the original LaTeX
    # document has ended, and the detectables (i.e., colorized tokens and symbols)
    # are about to appear.
    while True:
        if not _contains_start_graphic(page_image):
            page_image = next_page_image
            next_page_image = dequeue_page()
            continue

        # Once the marker has been found, skip forward one more page so that
        # symbols and tokens will be detected on the page after the marker.
        page_image = next_page_image
        next_page_image = dequeue_page()
        break

    # Templates are extracted for detecting both tokens and symbols. Templates
    # for tokens are images of single letters or marks. Templates for symbols
    # are groups of tokens and the expected (but somewhat flexible) spatial
    # relationships between them.
    token_images: Dict[Detectable, List[np.array]] = defaultdict(list)
    token_images_lookup: Dict[MathMl, Dict[FontSize, List[np.array]]] = defaultdict(
        dict
    )
    symbol_templates: Dict[Detectable, SymbolTemplate] = {}

    for d in detectables:

        # Find a bounding box around the token / symbol.
        red, green, blue = d.color
        rects = find_boxes_with_rgb(page_image, red, green, blue)

        if next_page_image is not None:
            if not rects:
                rects = find_boxes_with_rgb(next_page_image, red, green, blue)
                if not rects:
                    logger.warning("Could not find detectable %s.", d)
                    continue
                page_image = next_page_image
                next_page_image = dequeue_page()
            else:
                rects.extend(find_boxes_with_rgb(next_page_image, red, green, blue))
                if len(rects) > 1:
                    logger.warning(
                        "Unexpectedly more than one instance of detectable %s. "
                        + "There may have been a problem in the coloring code.",
                        d,
                    )

        if not rects:
            logger.warning("Could not find detectable %s.", d)

        box = rects[0]
        logger.debug(f"Found symbol at {box}.")

        # Extract a cropped, black-and-white image of the token or symbol.
        cropped_bw = page_image[
            box.top : box.top + box.height, box.left : box.left + box.width
        ]
        cropped_bw[
            np.where(
                (cropped_bw[:, :, 0] != 255)
                | (cropped_bw[:, :, 1] != 255)
                | (cropped_bw[:, :, 2] != 255)
            )
        ] = [0, 0, 0]
        cropped_bw = cv2.cvtColor(cropped_bw, cv2.COLOR_BGR2GRAY)

        # For simple symbols, extract images.
        if isinstance(d.entity, TexToken):

            # Only save a template if it has a different appearance from the other templates
            # saved for a symbol. This is important as a bunch of templates for the symbol
            # at the same size are created to try to make sure that templates are saved for
            # every way that extra space might have been introduced between characters in the
            # symbol when the PDF was rendered to an image.
            already_saved = False
            for img in token_images[d]:
                if np.array_equal(img, cropped_bw):
                    already_saved = True
                    break

            if not already_saved:
                token_images[d].append(cropped_bw)
                lookup_dict = token_images_lookup[d.entity.mathml]
                if d.font_size not in lookup_dict:
                    lookup_dict[d.font_size] = []
                lookup_dict[d.font_size].append(cropped_bw)

        # Note that, if the caller of this function did their job in ordering the list of
        # detectables, symbols will be processed only after all tokens have been processed.
        if isinstance(d.entity, TexSymbol):
            token_mathmls = [t.mathml for t in d.entity.tokens]
            template = create_symbol_template(
                cropped_bw, token_images_lookup, token_mathmls
            )
            if template:
                symbol_templates[d] = template

    return token_images, symbol_templates


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
        if not tokens:
            token_centers = np.empty(shape=(0, 2))
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


def detect_tokens(
    page_images: Dict[PageNumber, np.array],
    token_images: Dict[Detectable, List[np.array]],
    require_blank_border: bool = True,
) -> Dict[PageNumber, TokenIndex]:
    """
    Detect appearances of tokens in images of pages. If 'require_blank_border' is set,
    filter the detected tokens to just those that are surrounded with whitespace. This
    option is intended to help reduce the number of false positives. See the
    implementation comments below for more details.
    """

    tokens: Dict[PageNumber, TokenIndex] = {}

    # Unpack token images into a 1-D list.
    token_image_list = []
    token_list = []
    for (token, images) in token_images.items():
        for image in images:
            token_image_list.append(image)
            token_list.append(token)

    for page_no, page_image in sorted(page_images.items(), key=lambda t: t[0]):
        logger.debug("Detecting tokens on page %d.", page_no)
        page_image_gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        rects = find_in_image(
            token_image_list,
            page_image_gray,
            require_blank_border=require_blank_border,
        )
        token_instances: List[TokenInstance] = []
        for (token, rect_list) in zip(token_list, rects):
            for rect in rect_list:
                token_instances.append(
                    TokenInstance(
                        id_=Id(token.entity.mathml, token.font_size), location=rect
                    )
                )
        tokens[page_no] = TokenIndex(token_instances)

    return tokens


def detect_symbols(
    token_instances: Dict[PageNumber, TokenIndex],
    symbol_templates: Dict[Detectable, SymbolTemplate],
) -> Dict[PageNumber, List[SymbolInstance]]:

    symbol_instances: Dict[PageNumber, List[SymbolInstance]] = defaultdict(list)
    for page_no, token_index in token_instances.items():
        logger.debug("Scanning page %d for symbols.", page_no)
        for detectable, template in symbol_templates.items():
            for rect in find_symbols(template, token_index):
                instance = SymbolInstance(
                    Id(detectable.entity.mathml, detectable.font_size), rect
                )
                # Deduplicate symbols, in case two symbols are actually the same symbol (as
                # may happen if two symbols had different TeX, but the same MathML).
                if instance not in symbol_instances[page_no]:
                    symbol_instances[page_no].append(instance)

    return symbol_instances


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
