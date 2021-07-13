from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest
from lib.instrument_tex import FontSize
from lib.symbol_search import Component
from lib.symbol_search import Id as SymbolId
from lib.symbol_search import (
    Point,
    Rectangle,
    SymbolInstance,
    SymbolTemplate,
    TokenIndex,
    create_symbol_template,
    find_symbols,
)

MathMl = str


def p(x: float, y: float) -> Point:
    return Point(x, y)


def test_construct_composite_template():
    # Initialize images of child symbols.
    images: Dict[MathMl, Dict[FontSize, List[np.array]]] = defaultdict(dict)
    x_img = np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]], dtype=np.int)
    images["x"]["normal"] = [x_img]
    i_img = np.array([[0], [255], [0]], dtype=np.int)
    images["i"]["script"] = [i_img]

    # Image is white, except for...
    composite_symbol_img = np.zeros((3, 5), dtype=np.int)
    composite_symbol_img[:] = 255
    # 'x' appearing at [left=0, top=0, width=3, height=3]
    composite_symbol_img[0:3, 0:3] = x_img
    # 'i' appearing at [left=4, top=0, width=1, height=3]
    composite_symbol_img[0:3, 4:5] = i_img

    # List of children expected to be found in the composite symbol image.
    children = ["x", "i"]

    # Create the composite symbol template.
    composite_template = create_symbol_template(composite_symbol_img, images, children)
    assert composite_template.anchor == SymbolId("x", "normal")
    members = composite_template.members
    assert len(members) == 1
    assert members[0].symbol_id == SymbolId("i", "script")
    assert members[0].center.x == pytest.approx(3.0)
    assert members[0].center.y == pytest.approx(0.0)


def test_expect_blank_border_around_tokens():
    x_img = np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]], dtype=np.int)
    images: Dict[MathMl, Dict[FontSize, List[np.array]]] = defaultdict(dict)
    x_img = np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]], dtype=np.int)
    images["x"]["normal"] = [x_img]
    i_img = np.array([[0], [0]], dtype=np.int)
    images["i"]["script"] = [i_img]

    # Composite image is white, except for...
    composite_symbol_img = np.zeros((3, 6), dtype=np.int)
    composite_symbol_img[:] = 255
    # 'x' appearing at [left=0, top=0, width=3, height=3]
    composite_symbol_img[0:3, 0:3] = x_img
    # 'i' appearing at [left=4, top=1, width=1, height=2]
    composite_symbol_img[1:3, 4:5] = i_img
    # And a junk black pixel bordering the 'i' on the right.
    composite_symbol_img[1, 5] = 0

    # List of children expected to be found in the composite symbol image.
    # (Though note that the 'i' should not be detected).
    children = ["x", "i"]

    # Create the composite symbol template.
    composite_template = create_symbol_template(
        composite_symbol_img, images, children, require_blank_border_around_tokens=True
    )
    assert composite_template.anchor == SymbolId("x", "normal")
    members = composite_template.members
    assert len(members) == 0


def test_construct_composite_template_without_repeating_symbols():
    # An earlier version of the template creation function would duplicate components if a single
    # subsymbol appeared multiple times in the composite symbol. Check that there is no duplication.
    images: Dict[MathMl, Dict[FontSize, List[np.array]]] = defaultdict(dict)
    i_img = np.array([[0], [255], [0]], dtype=np.int)
    images["i"]["normal"] = [i_img]

    # Composite image is white, except for...
    composite_symbol_img = np.zeros((3, 3), dtype=np.int)
    composite_symbol_img[:] = 255
    # 'i' appearing at [left=0, top=0, width=1, height=2]
    composite_symbol_img[0:3, 0:1] = i_img
    # 'i' appearing at [left=2, top=0, width=1, height=2]
    composite_symbol_img[0:3, 2:3] = i_img

    # List of children expected to be found in the composite symbol image.
    children = ["i", "i"]

    # Create the composite symbol template.
    composite_template = create_symbol_template(composite_symbol_img, images, children)
    assert composite_template.anchor == SymbolId("i", "normal")
    members = composite_template.members
    assert len(members) == 1


def test_find_symbols_in_index():
    index = TokenIndex(
        tokens=[
            SymbolInstance(
                id_=SymbolId(mathml="x", level="normal"),
                location=Rectangle(0, 0, 4, 2),  # Center: 2, 1
            ),
            SymbolInstance(
                id_=SymbolId(mathml="x", level="normal"),
                location=Rectangle(10, 10, 4, 2),  # Center: 12, 11
            ),
        ]
    )

    # Find instances of 'x'.
    assert len(index.find(SymbolId("x", "normal"), p(2, 1), p(1, 1))) == 1
    assert len(index.find(SymbolId("x", "normal"), p(12, 11), p(1, 1))) == 1
    assert len(index.find(SymbolId("x", "normal"), p(5, 5), p(10, 10))) == 2

    # Fail to find when query keys do not match, or tolerance is too small.
    assert len(index.find(SymbolId("y", "normal"), p(2, 1), p(1, 1))) == 0
    assert len(index.find(SymbolId("x", "script"), p(2, 1), p(1, 1))) == 0
    assert len(index.find(SymbolId("x", "normal"), p(0, 0), p(1, 1))) == 0


def instance(
    mathml: str, level: FontSize, left: int, top: int, width: int, height: int
) -> SymbolInstance:
    return SymbolInstance(
        id_=SymbolId(mathml, level), location=Rectangle(left, top, width, height)
    )


def test_exact_match_composite_template():
    template = SymbolTemplate(
        anchor=SymbolId(mathml="x", level="normal"),
        members=[Component(SymbolId("i", "script"), center=p(3.0, 0.5))],
    )
    index = TokenIndex(
        tokens=[
            instance("x", "normal", 100, 100, 3, 3),  # center @ 101.5, 101.5
            instance("i", "script", 104, 101, 1, 2),  # center @ 104.5, 102
        ]
    )
    locations = list(find_symbols(template, index))
    assert len(locations) == 1
    assert locations[0] == Rectangle(100, 100, 5, 3)


def test_do_not_match_composite_template_if_component_missing():
    template = SymbolTemplate(
        anchor=SymbolId(mathml="x", level="normal"),
        members=[Component(SymbolId("i", "script"), center=p(3.0, 0.5))],
    )
    index = TokenIndex(
        tokens=[
            instance("x", "normal", 100, 100, 3, 3),
            # Missing symbol "i"
        ]
    )
    locations = list(find_symbols(template, index))
    assert len(locations) == 0


def test_fuzzy_match_composite_template():
    template = SymbolTemplate(
        anchor=SymbolId(mathml="x", level="normal"),
        members=[Component(SymbolId("i", "script"), center=p(3.0, 0.5))],
    )
    index = TokenIndex(
        tokens=[
            instance("x", "normal", 100, 100, 3, 3),  # center @ 101.5, 101.5
            # 'i' is one pixel away from its expected position, which should be
            # just within the tolerated variance.
            instance("i", "script", 105, 101, 1, 2),  # center @ 105.5, 102
        ]
    )
    locations = list(find_symbols(template, index))
    assert len(locations) == 1
    assert locations[0] == Rectangle(100, 100, 6, 3)


def test_do_not_match_composite_template_if_component_too_far_from_expected_position():
    template = SymbolTemplate(
        anchor=SymbolId(mathml="x", level="normal"),
        members=[Component(SymbolId("i", "script"), center=p(3.0, 0.5))],
    )
    index = TokenIndex(
        tokens=[
            instance("x", "normal", 100, 100, 3, 3),  # center @ 101.5, 101.5
            # 'i' is too many pixels away to be considered a match for the 'i' component.
            instance("i", "script", 107, 101, 1, 2),  # center @ 107.5, 102
        ]
    )
    locations = list(find_symbols(template, index))
    assert len(locations) == 0
