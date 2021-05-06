import logging
import os
import os.path
import shutil
import tempfile
from collections import defaultdict
from configparser import ConfigParser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import aiofiles
import uvicorn
from fastapi import FastAPI, File, UploadFile
from texcompile.client import compile
from typing_extensions import Literal

from lib.image_processing import (
    LocatedEntity,
    detect_symbols,
    detect_tokens,
    extract_templates,
    save_debug_images,
)
from lib.instrument_tex import add_colorized_symbols
from lib.parse_formula_tex import (
    TexSymbol,
    TexToken,
    convert_tex_to_mathml,
    create_symbol_from_node,
    filter_valid_formulas,
)
from lib.parse_mathml import parse_formula
from lib.parse_tex import FormulaExtractor
from lib.raster_document import raster_pages
from lib.symbol_search import Rectangle
from lib.unpack_tex import unpack_archive

app = FastAPI()
logger = logging.getLogger("symboldetector")


SymbolType = Literal["token", "symbol"]
Path = str
MathMl = str


@dataclass(frozen=True)
class FloatRectangle:
    left: float
    top: float
    width: float
    height: float


Formula = str


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


@dataclass
class Location:
    left: int
    top: int
    width: int
    height: int
    page: int


@dataclass
class Symbol:
    id_: int
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
    debug_output_dir: Optional[Path] = None,
) -> List[Any]:
    """
    'sources' is a directory containing LaTeX sources (should not contain any files generated
    by processing or compiling the sources (e.g., '.aux', etc.)).

    'debug_output_dir', if set, is a directory where the images visualizing the results of
    token and symbol extraction, overlaid over images of the pages, will be output.
    """
    # TODO(andrewhead): Find out why symbols do not appear in algorithm listings.
    # TODO(andrewhead): Find out why some functions are not getting detected.
    # TODO(andrewhead): 'd' and 'm' probably need to be shifted by vertical pixels slightly
    # to be detected; for some reason they are not getting detected.

    # Compile the TeX project to get the output PDF from which symbols will be extracted.
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

        # Extract formulas from TeX sources.
        formulas = extract_formulas(sources)

        # Convert formulas to MathML representation.
        formula_mathmls = convert_tex_to_mathml(formulas)

        # Extract all symbols and tokens for all formulas.
        all_symbols: Set[TexSymbol] = set()
        all_tokens: Set[TexToken] = set()
        mathml_parents: Dict[MathMl, Set[MathMl]] = defaultdict(set)
        mathml_texs: Dict[MathMl, Formula] = {}

        for (formula, mathml) in formula_mathmls.items():
            if mathml is not None:
                nodes = parse_formula(mathml)
                for node in nodes:
                    instance = create_symbol_from_node(node, formula)
                    mathml_texs[str(node.element)] = instance.tex

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
        modified_page_images = raster_pages(
            os.path.join(temp_dir, "modified-outputs", modified_output.name),
            modified_output.type_,
        )
        token_images, symbol_templates = extract_templates(
            modified_page_images, detectables
        )

        # Detect tokens and symbols in the modified PDF using the extracted templates for
        # tokens and symbols.
        original_output = result.output_files[0]
        original_page_images = raster_pages(
            os.path.join(temp_dir, "outputs", original_output.name),
            original_output.type_,
        )
        token_locations = detect_tokens(original_page_images, token_images)
        symbol_locations = detect_symbols(token_locations, symbol_templates)

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
            for instance in largest_to_smallest:
                loc = instance.location
                contained_by = [s for s in page_symbols if contains(s.location, loc)]
                valid_parents = [
                    s
                    for s in contained_by
                    if s.mathml in mathml_parents[instance.id_.mathml]
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
                        mathml=instance.id_.mathml,
                        tex=mathml_texs[instance.id_.mathml],
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

    symbols_json: List[Any] = []
    for symbol in symbols:
        symbols_json.append(
            {
                "id": symbol.id_,
                "location": {
                    "left": symbol.location.left,
                    "top": symbol.location.top,
                    "width": symbol.location.width,
                    "height": symbol.location.height,
                    "page": symbol.location.page,
                },
                "tex": symbol.tex,
                "mathml": symbol.mathml,
                "parent": symbol.parent,
            }
        )

    return symbols_json


@app.post("/")
async def detect_upload_file(sources: UploadFile = File(...)):

    config = ConfigParser()
    config.read("config.ini")
    texcompile_host = config["texcompile"]["host"]
    texcompile_port = int(config["texcompile"]["port"])

    with tempfile.TemporaryDirectory() as tempdir:
        sources_filename = os.path.join(tempdir, "sources")
        async with aiofiles.open(sources_filename, "wb") as sources_file:
            content = await sources.read()  # async read
            await sources_file.write(content)  # async write

        unpacked_dir = os.path.join(tempdir, "unpacked_sources")
        unpack_archive(sources_filename, unpacked_dir)
        json_result = extract_symbols(unpacked_dir, texcompile_host, texcompile_port)
        return json_result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
