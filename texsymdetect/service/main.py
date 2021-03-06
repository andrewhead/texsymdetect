import argparse
import dataclasses
import logging
import os
import os.path
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import aiofiles
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from texcompile.client import compile
from typing_extensions import Literal

from lib.expand_macros import (
    MacroDetectionException,
    apply_expansions,
    detect_expansions,
)
from lib.image_processing import LocatedEntity, save_debug_images
from lib.instrument_tex import add_colorized_symbols
from lib.parse_formula_tex import (
    Formula,
    TexSymbol,
    TexToken,
    convert_tex_to_mathml,
    create_symbol_from_node,
    filter_valid_formulas,
    parse_symbols_in_formulas,
)
from lib.parse_mathml import NodeType, parse_formula
from lib.parse_tex import FormulaExtractor
from lib.raster_document import raster_pages
from lib.symbol_search import (
    Rectangle,
    detect_symbols,
    detect_tokens,
    extract_templates,
)
from lib.unpack_tex import unpack_archive

app = FastAPI()

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("texsymdetect")
logger.setLevel(logging.DEBUG)


SymbolType = Literal["token", "symbol"]
Path = str
MathMl = str


@dataclass(frozen=True)
class FloatRectangle:
    left: float
    top: float
    width: float
    height: float


def extract_formulas(sources_dir: Path) -> Set[Formula]:
    " Extract list of unique formulas from the '.tex' files in the sources directory. "

    # Run equation extractor.
    extractor = FormulaExtractor()
    tex_files = []
    for (dirpath, _, filenames) in os.walk(sources_dir):
        for filename in filenames:
            __, ext = os.path.splitext(filename)
            if ext == ".tex":
                tex_files.append(os.path.join(dirpath, filename))

    formulas: Set[Formula] = set()
    for filename in tex_files:
        with open(filename, encoding="utf-8", errors="backslashescape") as file_:
            contents = file_.read()
            for formula in extractor.parse(filename, contents):
                formulas.add(formula.content_tex)

    return formulas


class TexCompilationException(Exception):
    pass


@dataclass(frozen=True)
class Location(Rectangle):
    page: int


@dataclass
class Symbol:
    id_: int
    type_: NodeType
    mathml: str
    tex: str
    location: Location
    parent: Optional[int]


def contains(outer: Rectangle, inner: Rectangle) -> bool:
    return (
        outer.left <= inner.left
        and ((outer.left + outer.width) >= (inner.left + inner.width))
        and outer.top <= inner.top
        and ((outer.top + outer.height) >= (inner.top + inner.height))
    )


def extract_symbols(
    sources: Path,
    texcompile_host: str,
    texcompile_port: int,
    try_expand_macros: Optional[bool] = None,
    require_blank_border: Optional[bool] = None,
    insert_function_elements: Optional[bool] = None,
    merge_adjacent_elements: Optional[bool] = None,
    debug_output_dir: Optional[Path] = None,
) -> List[Any]:
    """
    'sources' is a directory containing LaTeX sources (should not contain any files generated
    by processing or compiling the sources (e.g., '.aux', etc.)).

    If 'try_expand_macros' is set, this function will try to expand macros for symbols before
    detecting those symbols. For papers that make heavy use of macros, this should increase
    the number of symbols that are detected.

    If 'require_blank_border' is set, then the symbol detector will only detect tokens of
    symbols that are surrounded with whitespace on all sides. This parameter decreases the
    number of false positives (particularly for symbols like dots and dashes) and speeds
    up symbol extraction. However, it does result in some missed symbols, particularly when
    symbols are directly adjacent to each other, or when a symbol borders on a line (like
    when a symbol is a numerator on top of a fraction line).

    Options 'insert_function_elements' and 'merge_adjacent_elements' determine the extent
    to which the MathML for detected LaTeX symbols is modified to reflect inferred semantics
    of the MathML tree (for instance, whether consecutive elements in the MathML tree should be
    combined). See documentation for the 'parse_element' method for details.

    Defaults for options appear at the top of the function definition.

    'debug_output_dir', if set, is a directory where the images visualizing the results of
    token and symbol extraction, overlaid over images of the pages, will be output.
    """
    try_expand_macros = True if try_expand_macros is None else try_expand_macros
    require_blank_border = (
        True if require_blank_border is None else require_blank_border
    )
    insert_function_elements = (
        False if insert_function_elements is None else insert_function_elements
    )
    merge_adjacent_elements = (
        False if merge_adjacent_elements is None else merge_adjacent_elements
    )

    # Compile the TeX project to get the output PDF from which symbols will be extracted.
    logger.debug("Started processing paper.")
    with tempfile.TemporaryDirectory() as temp_dir:
        result = compile(
            sources,
            os.path.join(temp_dir, "outputs"),
            texcompile_host,
            texcompile_port,
        )
        if not result.success:
            raise TexCompilationException("Failed to compile TeX sources.", result.log)
        if len(result.output_files) > 1:
            raise TexCompilationException(
                "This service has not been yet programmed to handle projects that produce "
                + "more than one output file. %d output files were produced.",
                len(result.output_files),
            )

        main_tex_files = result.main_tex_files
        for output in result.output_files:
            logging.debug("Compiled file %s.", output.name)

        # Try to expand macros for formulas before parsing formulas. This makes it possible to
        # parse the formula into its tokens using an external parser like 'KaTeX,' which has no
        # knowledge of what macros for symbols are supposed to be expanded to.
        are_formulas_expanded = False
        if try_expand_macros:
            expanded_sources = os.path.join(temp_dir, "expanded-sources")
            shutil.copytree(sources, expanded_sources)

            # Expand formulas in each
            are_formulas_expanded = True
            for main_tex_file in main_tex_files:
                try:
                    expansions = detect_expansions(expanded_sources, main_tex_file)
                    apply_expansions(expansions)
                except MacroDetectionException:
                    are_formulas_expanded = False
                    logger.debug(  # pylint: disable=logging-not-lazy
                        "Failed to detect macro expansions when compiling file %s. "
                        + "Macros in formulas will not be expanded.",
                        main_tex_file,
                    )
                    break

        # Extract formulas from TeX sources.
        if are_formulas_expanded:
            formulas = extract_formulas(expanded_sources)
        else:
            formulas = extract_formulas(sources)

        # Convert formulas to MathML representation.
        formula_mathmls = convert_tex_to_mathml(formulas)

        # Extract all symbols and tokens for all formulas.
        all_symbols: Set[TexSymbol] = set()
        all_tokens: Set[TexToken] = set()
        mathml_parents: Dict[MathMl, Set[MathMl]] = defaultdict(set)
        mathml_texs: Dict[MathMl, Formula] = {}
        mathml_types: Dict[MathMl, NodeType] = {}

        for (formula, mathml) in formula_mathmls.items():
            if mathml is not None:
                nodes = parse_formula(
                    mathml,
                    merge_adjacent_elements=merge_adjacent_elements,
                    insert_function_elements=insert_function_elements,
                )
                for node in nodes:
                    instance = create_symbol_from_node(node, formula)
                    mathml_texs[str(node.element)] = instance.tex
                    mathml_types[str(node.element)] = instance.type_

                    # Save all unique symbols and tokens.
                    all_symbols.add(instance)
                    all_tokens.update(instance.tokens)

                    # Make a lookup table that answers whether one MathML element can have
                    # another MathML element as a parent. This table will be used later
                    # to determine parent-child relationships between detected symbols.
                    for child in node.children:
                        mathml_parents[str(child.element)].add(str(node.element))

        # Filter to only valid symbols and tokens.
        valid_formulas = filter_valid_formulas(
            {s.tex for s in all_symbols}.union({t.tex for t in all_tokens})
        )
        valid_symbols = list(filter(lambda s: s.tex in valid_formulas, all_symbols))
        valid_tokens = list(filter(lambda s: s.tex in valid_formulas, all_tokens))

        # Add colorized copies of tokens and symbols to the TeX.
        # Note that colorized tokens and symbols are added to the original TeX, and not the
        # copy of the TeX directory where macros are expanded. It is expected that this might
        # be more robust, because if any of the macros were expanded improperly (and there
        # might be thousands of macro usages), the paper might not compile.
        modified_sources_dir = os.path.join(temp_dir, "modified-sources")
        shutil.copytree(sources, modified_sources_dir)

        if len(main_tex_files) > 1:
            logger.warning(
                "More than one main TeX file was found by the TeX compilation service. "
                + "This service was created assuming there would be only one main TeX file. "
                + "Some symbols may not be recognized."
            )

        main_tex_filename = main_tex_files[-1]
        with open(
            os.path.join(sources, main_tex_filename), errors="backslashescape"
        ) as tex_file:
            tex = tex_file.read()

        tex_with_colorized_symbols, detectables = add_colorized_symbols(
            tex, valid_tokens, valid_symbols
        )
        with open(
            os.path.join(modified_sources_dir, main_tex_filename),
            "w",
            errors="backslashescape",
        ) as modified_tex_file:
            modified_tex_file.write(tex_with_colorized_symbols)

        # Compile the modified TeX.
        modified_outputs_dir = os.path.join(temp_dir, "modified-outputs")
        result = compile(
            modified_sources_dir,
            modified_outputs_dir,
            texcompile_host,
            texcompile_port,
        )
        if not result.success:
            raise TexCompilationException(
                "Failed to compile modified TeX sources.", result.log
            )
        if not len(result.output_files) == 1:
            raise TexCompilationException(
                "Expected 1 output file when compiling modified TeX sources. "
                + "Instead found %d.",
                len(result.output_files),
            )

        # Extract templates from compiled, modified PDF.
        modified_output = result.output_files[0]
        logger.debug("Started rastering modified PDF.")
        modified_page_images = raster_pages(
            os.path.join(temp_dir, "modified-outputs", modified_output.name),
            modified_output.type_,
        )
        logger.debug("Finished rastering modified PDF.")

        logger.debug("Started extracting templates.")
        token_images, symbol_templates = extract_templates(
            modified_page_images, detectables
        )
        logger.debug("Finished extracting templates.")

        # Detect tokens and symbols in the modified PDF using the extracted templates for
        # tokens and symbols.
        original_output = result.output_files[0]
        logger.debug("Started rastering original PDF.")
        original_page_images = raster_pages(
            os.path.join(temp_dir, "outputs", original_output.name),
            original_output.type_,
        )
        logger.debug("Finished rastering original PDF.")
        logger.debug("Started detecting token locations.")
        token_locations = detect_tokens(
            original_page_images, token_images, require_blank_border
        )
        logger.debug("Finished detecting token locations.")
        logger.debug("Started detecting symbol locations.")
        symbol_locations = detect_symbols(token_locations, symbol_templates)
        logger.debug("Finished detecting symbol locations.")

        # Clean up symbol data:
        # 1. Associate symbols with their parents.
        # 2. Remove incorrectly detected symbols when possible.
        symbols: List[Symbol] = []
        symbol_id = 0

        for page_no, page_symbol_instances in symbol_locations.items():

            page_symbols: List[Symbol] = []
            largest_to_smallest = sorted(
                page_symbol_instances, key=lambda si: si.location.width, reverse=True
            )
            for symbol_instance in largest_to_smallest:
                loc = symbol_instance.location
                contained_by = [s for s in page_symbols if contains(s.location, loc)]
                valid_parents = [
                    s
                    for s in contained_by
                    if s.mathml in mathml_parents[symbol_instance.id_.mathml]
                ]
                # Skip symbols that appear inside larger symbols, but which are not expected
                # to appear within those larger symbols based on known parent-child relationships.
                if contained_by and not valid_parents:
                    continue

                # The parent is the smallest candidate parent symbol that contains this symbol.
                parent: Optional[int] = None
                if valid_parents:
                    valid_parents.sort(key=lambda s: s.location.width)
                    parent = valid_parents[0].id_

                page_symbols.append(
                    Symbol(
                        id_=symbol_id,
                        type_=mathml_types[symbol_instance.id_.mathml],
                        mathml=symbol_instance.id_.mathml,
                        tex=mathml_texs[symbol_instance.id_.mathml],
                        location=Location(
                            loc.left, loc.top, loc.width, loc.height, page_no,
                        ),
                        parent=parent,
                    )
                )
                symbol_id += 1

            symbols.extend(page_symbols)

        # Save annotated images of paper for debugging.
        if debug_output_dir:
            token_debug_dir = os.path.join(debug_output_dir, "pages-with-tokens")
            detected_tokens = [
                LocatedEntity(
                    t.location.left,
                    t.location.top,
                    t.location.width,
                    t.location.height,
                    page_no,
                    t.id_.mathml,
                )
                for (page_no, token_index) in token_locations.items()
                for t in token_index.get_instances()
            ]
            save_debug_images(original_page_images, detected_tokens, token_debug_dir)

            symbol_debug_dir = os.path.join(debug_output_dir, "pages-with-symbols")
            detected_symbols = [
                LocatedEntity(
                    s.location.left,
                    s.location.top,
                    s.location.width,
                    s.location.height,
                    s.location.page,
                    s.mathml,
                )
                for s in symbols
            ]
            save_debug_images(original_page_images, detected_symbols, symbol_debug_dir)

    logger.debug("Finished processing paper.")
    symbols_json: List[Any] = []
    for symbol in symbols:

        # Dimensions (left, top, width, height) are expressed as a ratio of the page width (if left
        # or width) or page height (if top or height) with values between 0 and 1. This way of
        # expressing coordinates was chosen as a client of this library may wish to convert symbol
        # positions into pixels in a rendered image of the PDF, or in inches in a PDF document. This
        # representation allows a client to do either.
        page_shape = original_page_images[symbol.location.page].shape
        page_height = page_shape[0]
        page_width = page_shape[1]
        left_normalized = symbol.location.left / float(page_width)
        width_normalized = symbol.location.width / float(page_width)
        top_normalized = symbol.location.top / float(page_height)
        height_normalized = symbol.location.height / float(page_height)

        symbols_json.append(
            {
                "id": symbol.id_,
                "type": symbol.type_,
                "location": {
                    "left": left_normalized,
                    "top": top_normalized,
                    "width": width_normalized,
                    "height": height_normalized,
                    "page": symbol.location.page,
                },
                "tex": symbol.tex,
                "mathml": symbol.mathml,
                "parent": symbol.parent,
            }
        )

    return symbols_json


@app.post("/")
async def detect_upload_file(
    sources: UploadFile = File(...),
    try_expand_macros: Optional[bool] = None,
    require_blank_border: Optional[bool] = None,
    insert_function_elements: Optional[bool] = None,
    merge_adjacent_elements: Optional[bool] = None,
):
    " Documentation for options appears in the 'extract_symbols' method. "

    texcompile_host = os.getenv("TEXCOMPILE_HOST", "http://localhost")
    try:
        texcompile_port = int(os.getenv("TEXCOMPILE_PORT"))  # type: ignore
    except Exception:
        texcompile_port = 8000

    with tempfile.TemporaryDirectory() as tempdir:
        sources_filename = os.path.join(tempdir, "sources")
        async with aiofiles.open(sources_filename, "wb") as sources_file:
            content = await sources.read()  # async read
            await sources_file.write(content)  # async write

        unpacked_dir = os.path.join(tempdir, "unpacked_sources")
        unpack_archive(sources_filename, unpacked_dir)
        json_result = extract_symbols(
            unpacked_dir,
            texcompile_host,
            texcompile_port,
            try_expand_macros=try_expand_macros,
            require_blank_border=require_blank_border,
            insert_function_elements=insert_function_elements,
            merge_adjacent_elements=merge_adjacent_elements,
        )
        return json_result


class ParseFormulasRequest(BaseModel):
    formulas: List[str]


@app.post("/parse_formulas")
async def parse_formulas(request: ParseFormulasRequest):

    formulas = request.formulas
    result = []
    formula_symbols = parse_symbols_in_formulas(formulas)
    for formula in formulas:
        symbols = formula_symbols.get(formula, [])
        symbol_jsons = [dataclasses.asdict(s) for s in symbols]
        result.append(symbol_jsons)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run symbol detection service.")
    parser.add_argument(
        "--port", type=int, help="Port on which to run the service.", default=8001
    )
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
