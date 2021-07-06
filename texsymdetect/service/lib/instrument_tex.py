import os.path
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union

from typing_extensions import Literal

from lib.parse_formula_tex import TexSymbol, TexToken
from lib.parse_tex import (
    BeginDocumentExtractor,
    DocumentclassExtractor,
    EndDocumentExtractor,
)

# Load preambles for TeX files that will load the colorization commands.
with open(os.path.join("resources", "01-macros.tex")) as file_:
    COLOR_MACROS_BASE_MACROS = file_.read()
with open(os.path.join("resources", "02a-latex-import-color.tex")) as file_:
    COLOR_MACROS_LATEX_IMPORTS = file_.read()
with open(os.path.join("resources", "02b-tex-import-color.tex")) as file_:
    COLOR_MACROS_TEX_IMPORTS = file_.read()
with open(os.path.join("resources", "03-load-color-commands.tex")) as file_:
    COLOR_MACROS = file_.read()


TEX_COLOR_MACROS = "\n".join(
    [COLOR_MACROS_BASE_MACROS, COLOR_MACROS_TEX_IMPORTS, COLOR_MACROS]
)


FontSize = Literal[
    "normal",
    "normal_script",
    "normal_scriptscript",
    "small",
    "small_script",
    "small_scriptscript",
    "footnote",
    "footnote_script",
    "footnote_scriptscript",
]


@dataclass(frozen=True)
class Detectable:
    entity: Union[TexSymbol, TexToken]
    font_size: FontSize
    " The combination of entity and font size should be unique. "

    color: Tuple[int, int, int]
    " Color that was assigned to the entity, that can be used to find it using image processing."


class UnexpectedTexFormatException(Exception):
    pass


def add_colorized_symbols(
    tex: str, tokens: List[TexToken], symbols: List[TexSymbol]
) -> Tuple[str, List[Detectable]]:

    # Prepare a TeX file where each symbol is repeated, each time on a new line, in
    # a new size, and in a new color.
    FONT_SIZES: List[FontSize] = [
        "normal",
        "normal_script",
        "normal_scriptscript",
        "small",
        "small_script",
        "small_scriptscript",
        "footnote",
        "footnote_script",
        "footnote_scriptscript",
    ]

    detectables_tex = ""
    detectables: List[Detectable] = []

    # Colorize tokens (i.e., single letters) first, then symbols. This lets all tokens
    # be detected before processing any symbols, which requires a knowledge of the
    # appearance of the tokens that they are made of.
    color_generator = _get_color()
    for token in tokens:
        for size in FONT_SIZES:
            detectables_tex, detectable = _add_detectable(
                detectables_tex, token, size, color_generator
            )
            detectables.append(detectable)

    for symbol in symbols:
        for size in FONT_SIZES:
            detectables_tex, detectable = _add_detectable(
                detectables_tex, symbol, size, color_generator
            )
            detectables.append(detectable)

    tex_with_macros = add_helper_color_macros(tex)

    end_document_extractor = EndDocumentExtractor()
    end_document_command = end_document_extractor.parse(tex_with_macros)
    if not end_document_command:
        raise UnexpectedTexFormatException(
            "Could not find \\end{{document}} tag in TeX."
            + "Is this a LaTeX document (rather than a plain TeX document)?"
        )

    insertion_offset = end_document_command.start
    instrumented_tex = (
        tex_with_macros[:insertion_offset]
        # Add in special markers that will serve as fiducials for
        # the image processing pipeline (specifically, blank pages, and pages
        # that are filled entirely with certain expected colors.)
        + "\n\n"
        + (r"\scholaradvancepage{}" + "\n")
        + (r"\scholarfillandadvancepage{}" + "\n")
        + "\n"
        + detectables_tex
        + "\n\n"
        + (r"\scholaradvancepage{}" + "\n")
        + (r"\scholaradvancepage{}" + "\n")
        + "\n\n"
        + tex_with_macros[insertion_offset:]
    )

    return instrumented_tex, detectables


def _get_color(
    skip: Optional[List[Tuple[int, int, int]]] = None
) -> Iterator[Tuple[int, int, int]]:
    """
    This function is cyclical: it loops back to the initial colors when it is has reached the end
    of the unique colors. This will be after millions of unique colors have been produced.
    Skip black and white colors by default.
    """
    skip = skip or [(0, 0, 0), (255, 255, 255)]
    while True:
        for red in range(0, 255):
            for green in range(0, 255):
                for blue in range(0, 255):
                    if (red, green, blue) in skip:
                        continue
                    yield red, green, blue


def _add_detectable(
    tex: str,
    entity: Union[TexToken, TexSymbol],
    size: FontSize,
    color_generator: Iterator[Tuple[int, int, int]],
) -> Tuple[str, Detectable]:

    # Surround TeX for a token with the fonts that applied to it.
    entity_tex = entity.tex
    font_macros = getattr(entity, "font_macros", ())
    for macro in font_macros:
        entity_tex = f"\\{macro}{{{entity_tex}}}"

    if size == "normal":
        sized_tex = fr"{{\normalsize $${entity_tex}$$}}"
    elif size == "normal_script":
        sized_tex = fr"{{\normalsize $$_{{{entity_tex}}}$$}}"
    elif size == "normal_scriptscript":
        sized_tex = fr"{{\normalsize $$_{{_{{{entity_tex}}}}}$$}}"
    elif size == "small":
        sized_tex = fr"{{\small $${entity_tex}$$}}"
    elif size == "small_script":
        sized_tex = fr"{{\small $$_{{{entity_tex}}}$$}}"
    elif size == "small_scriptscript":
        sized_tex = fr"{{\small $$_{{_{{{entity_tex}}}}}$$}}"
    elif size == "footnote":
        sized_tex = fr"{{\footnotesize $${entity_tex}$$}}"
    elif size == "footnote_script":
        sized_tex = fr"{{\footnotesize $$_{{{entity_tex}}}$$}}"
    elif size == "footnote_scriptscript":
        sized_tex = fr"{{\footnotesize $$_{{_{{{entity_tex}}}}}$$}}"

    red, green, blue = next(color_generator)
    colorized = rf"\textcolor[RGB]{{{red},{green},{blue}}}{{{sized_tex}}}" + "\n\n"

    tex += colorized
    return tex, Detectable(entity, size, (red, green, blue))


PageNumber = int


def add_helper_color_macros(tex: str, after_macros: Optional[str] = None) -> str:
    documentclass_extractor = DocumentclassExtractor()
    documentclass = documentclass_extractor.parse(tex)
    if documentclass is not None:
        begin_document_extractor = BeginDocumentExtractor()
        begin_document = begin_document_extractor.parse(tex)
        if begin_document is not None:
            return (
                tex[: begin_document.start]
                + "\n"
                + COLOR_MACROS_BASE_MACROS
                + "\n"
                + COLOR_MACROS_LATEX_IMPORTS
                + "\n"
                + tex[begin_document.start : begin_document.end]
                + "\n"
                # These main color macros should be included *after* "\begin{document}". This is
                # because AutoTeX will sometimes add a "\RequirePackage{hyperref}" statement right
                # before "\begin{document}" in the moments before compiling the TeX. If we instead
                # put the color macros are put above "\begin{document}", what happens is that
                # hyperref reverts the hyperref macros that we had redefined to enable coloring.
                + COLOR_MACROS
                + ("\n" + after_macros if after_macros else "")
                + tex[begin_document.end :]
            )
    return (
        TEX_COLOR_MACROS + ("\n" + after_macros if after_macros else "") + "\n\n" + tex
    )
