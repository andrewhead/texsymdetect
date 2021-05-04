import logging
import math
import os.path
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from lib.instrument_tex import Detectable, FontSize
from lib.parse_formula_tex import TexSymbol, TexToken
from lib.symbol_search import (Id, Rectangle, SymbolInstance, SymbolTemplate,
                               TokenIndex, TokenInstance,
                               create_bitstring_from_image,
                               create_symbol_template, find_symbols)

logger = logging.getLogger("symboldetector")

Path = str
MathMl = str
PageNumber = int


def extract_templates(
    page_images: Dict[PageNumber, np.array], detectables: Sequence[Detectable],
) -> Tuple[Dict[Detectable, np.array], Dict[Detectable, SymbolTemplate]]:
    """
    Given images of pages from a paper that has been modified to include appearances of many tokens
    and symbols (i.e., 'detectables'), extract templates for those tokens and symbols
    that can be used to identify them in other documents.

    Returns a collection of token templates (images), and symbol templates
    (a flexible template format).
    
    symbols. Note that both tokens and symbols must be passed in as detectables;
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
    symbol_templates: Dict[Detectable, SymbolTemplate] = defaultdict(list)

    token_detectables = filter(lambda d: isinstance(d.entity, TexToken), detectables)
    symbol_detectables = filter(lambda d: isinstance(d.entity, TexSymbol), detectables)

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


def detect_tokens(
    page_images: Dict[PageNumber, np.array], token_images: Dict[Detectable, np.array],
) -> Dict[PageNumber, TokenIndex]:
    tokens: Dict[PageNumber, TokenIndex] = {}

    # Load page images as black-and-white; symbol detection should not depend on color.
    # Load image into a bit-string (i.e., a sequence of 0's and 1's), which can
    # be faster to run template matching on using regular expressions in comparison to
    # OpenCV's image template matching module.
    page_bit_strings: Dict[int, str] = {}
    page_images_bw: Dict[PageNumber, np.array] = {}
    for page_no, page_image in page_images.items():

        BLACK = 0
        page_image_bw = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        non_blank_pixels = page_image_bw < 255
        page_image_bw[non_blank_pixels] = BLACK
        page_images_bw[page_no] = page_image_bw

        image_string = create_bitstring_from_image(page_image_bw)
        page_bit_strings[page_no] = image_string

    # Search for each simple symbol in the PDF using template matching on the rastered images,
    # using string matching on bit-strings (see note in comment above).
    for page_no, page_bit_string in page_bit_strings.items():

        logger.debug("Scanning page %d for tokens.", page_no)
        page_image_bw = page_images_bw[page_no]

        page_width = len(page_image[0])
        token_instances: List[TokenInstance] = []

        for detectable, images in token_images.items():
            for image in images:
                w: int
                h: int
                h, w = image.shape

                FOUR_PIXELS = 4
                if np.all(image == BLACK) and (w * h <= FOUR_PIXELS):
                    logger.warning(
                        "Skipping image for token (%s, size %s) that is very small and entirely "
                        + "black pixels, to avoid wasting time on detecting false positives.",
                        detectable.entity.mathml,
                        detectable.font_size,
                    )
                    continue

                token_bit_string = create_bitstring_from_image(
                    image,
                    # Pad the token bitstring with wildcards for the parts of the page bit
                    # string that it does not need to match.
                    page_width - w,
                )

                # Search for all string matches.
                search_start = 0
                pattern = re.compile(token_bit_string)
                while True:
                    match = pattern.search(page_bit_string, pos=search_start)
                    if match is None:
                        break
                    character_offset = match.start()
                    search_start = match.start() + 1

                    # Convert character position in bit string to pixel position.
                    x = character_offset % page_width
                    y = math.floor(character_offset / page_width)

                    # Skip over matches that are not surrounded with whitespace in
                    # the rectangle of pixels immediately surround it.
                    # This is important for ruling out false positives, like dots,
                    # which contain only a small, solid block of black pixels. Such
                    # tokens would be found everywhere that there are black pixels
                    # if there are not checks to make sure that the black pixels are
                    # surrounded in white pixels.
                    if np.any(page_image_bw[y - 1, x - 1 : x + w + 1] == 0):
                        continue
                    # Bottom boundary.
                    elif np.any(page_image_bw[y + h, x - 1 : x + w + 1] == 0):
                        continue
                    # Left boundary.
                    elif np.any(page_image_bw[y - 1 : y + h + 1, x - 1] == 0):
                        continue
                    # Right boundary.
                    elif np.any(page_image_bw[y - 1 : y + h + 1, x + w] == 0):
                        continue

                    # Save appearance of this token.
                    rect = Rectangle(left=x, top=y, width=w, height=h,)
                    token_instances.append(
                        TokenInstance(
                            Id(detectable.entity.mathml, detectable.font_size), rect
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
                symbol_instances[page_no].append(
                    SymbolInstance(
                        Id(detectable.entity.mathml, detectable.font_size), rect
                    )
                )

    return symbol_instances


def find_boxes_with_rgb(
    image: np.ndarray, red: int, green: int, blue: int
) -> List[Rectangle]:
    """
    Arguments:
    - 'red', 'green', 'blue': integer numbers between 0 and 255.
    - 'tolerance': is the amount of difference from 'hue' (from 0-to-1) still considered that hue.
    - 'masks': a set of masks to apply to the image, one at a time. Bounding boxes are extracted
      from within each of those boxes. Masks should be in pixel coordinates.
    """

    boxes = []
    matching_pixels = np.where(
        (image[:, :, 0] == blue) & (image[:, :, 1] == green) & (image[:, :, 2] == red)
    )

    if len(matching_pixels[0]) > 0:
        left = min(matching_pixels[1])
        right = max(matching_pixels[1])
        top = min(matching_pixels[0])
        bottom = max(matching_pixels[0])
        boxes.append(Rectangle(left, top, right - left + 1, bottom - top + 1))

    return boxes


def _contains_start_graphic(image: np.array) -> bool:
    num_pixels = image.shape[0] * image.shape[1]
    num_blue_pixels = len(
        np.where(
            (image[:, :, 0] == 80) & (image[:, :, 1] == 165) & (image[:, :, 2] == 250)
        )[0]
    )
    return num_blue_pixels / float(num_pixels) > 0.5


@dataclass
class LocatedEntity:
    left: int
    top: int
    width: int
    height: int
    page: int
    key: str


def save_debug_images(
    page_images: Dict[PageNumber, np.array],
    located_objects: Iterable[LocatedEntity],
    output_dir: Path,
) -> None:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    entity_colors: Dict[str, Tuple[int, int, int]] = {}
    for page_no, page_image in page_images.items():

        annotated_image = np.copy(page_image)

        for object in located_objects:
            if not object.page == page_no:
                continue

            key = object.key

            # Use a consistent (yet initially randomly-chosen) color for each entity detected.
            if key not in entity_colors:
                entity_colors[key] = (
                    random.randint(0, 256),
                    random.randint(0, 256),
                    random.randint(0, 256),
                )
            color = entity_colors[key]

            x = object.left
            y = object.top
            width = object.width
            height = object.height
            cv2.rectangle(
                annotated_image, (x, y), (x + width, y + height), color, 1,
            )

        cv2.imwrite(
            os.path.join(output_dir, f"page-{page_no:03d}.png"), annotated_image
        )

