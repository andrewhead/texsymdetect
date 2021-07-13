import logging
import math
import os.path
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("texsymdetect")

Path = str
MathMl = str
PageNumber = int


# Value of pixel in grayscale image that is black.
BLACK = 0


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


def find_in_image(
    targets: List[np.array],
    image: np.array,
    skip_small_filled_targets=True,
    require_blank_border: bool = True,
) -> List[List[Rectangle]]:
    """ This method assumes 'image' is a grayscale image. """

    # Load image as black-and-white; symbol detection should not depend on hue,
    # saturation or value.  Load image into a bit-string (i.e., a sequence of 0's and 1's),
    # which at the time of writing, provided a faster way to run template matching on using
    # regular expressions instead of OpenCV's image template matching module.
    image_bw = image.copy()
    non_blank_pixels = image_bw < 255
    image_bw[non_blank_pixels] = BLACK
    image_bit_string = create_bitstring_from_image(image_bw)
    image_height, image_width = image.shape

    detected_for_target: List[Rectangle]
    all_detected: List[List[Rectangle]] = []

    for target in targets:
        detected_for_target = []

        w: int
        h: int
        h, w = target.shape

        FOUR_PIXELS = 4
        if (
            skip_small_filled_targets
            and np.all(target == BLACK)
            and (w * h <= FOUR_PIXELS)
        ):
            all_detected.append([])
            continue

        token_bit_string = create_bitstring_from_image(
            target,
            # Pad the token bitstring with wildcards for the parts of the page bit
            # string that it does not need to match.
            image_width - w,
        )

        # Search for all string matches.
        search_start = 0
        pattern = re.compile(token_bit_string)
        while True:
            match = pattern.search(image_bit_string, pos=search_start)
            if match is None:
                break
            character_offset = match.start()
            search_start = match.start() + 1

            # Convert character position in bit string to pixel position.
            x = character_offset % image_width
            y = math.floor(character_offset / image_width)

            # Skip over matches that are not surrounded with whitespace in
            # the rectangle of pixels immediately surround it.
            # This is important for ruling out false positives, like dots,
            # which contain only a small, solid block of black pixels. Such
            # tokens would be found everywhere that there are black pixels
            # if there are not checks to make sure that the black pixels are
            # surrounded in white pixels.
            if require_blank_border:

                # Left, top, right, bottom of the match.
                l = x
                t = y
                r = x + w - 1
                b = t + h - 1

                # Left, top, right, bottom of box surrounding the match.
                # Adjust to be within the edges of the image.
                lb = max(x - 1, 0)
                tb = max(y - 1, 0)
                rb = min(x + w, image_width - 1)
                bb = min(y + h, image_height - 1)

                # Check top boundary (row of pixels just above the match)
                # for any non-blank pixels. Skip this check if the match is already
                # on the top border of the image.
                if t > 0 and np.any(image_bw[tb, lb : rb + 1] == 0):
                    continue
                # Bottom boundary.
                elif (b < image_height - 1) and np.any(image_bw[bb, lb : rb + 1] == 0):
                    continue
                # Left boundary.
                elif l > 0 and np.any(image_bw[tb : bb + 1, lb] == 0):
                    continue
                # Right boundary.
                elif r < image_width - 1 and np.any(image_bw[tb : bb + 1, rb] == 0):
                    continue

            # Save appearance of this token.
            # If the token has already been found at this location (e.g., if there
            # are multiple detectables with the same MathML) only save once.
            rect = Rectangle(left=x, top=y, width=w, height=h,)
            detected_for_target.append(rect)

        all_detected.append(detected_for_target)

    return all_detected


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
    START_MARKER_COLOR = (80, 165, 250)
    num_pixels = image.shape[0] * image.shape[1]
    new_colorized_pixels = len(
        np.where(
            (image[:, :, 0] == START_MARKER_COLOR[0])
            & (image[:, :, 1] == START_MARKER_COLOR[1])
            & (image[:, :, 2] == START_MARKER_COLOR[2])
        )[0]
    )
    return new_colorized_pixels / float(num_pixels) > 0.2


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

