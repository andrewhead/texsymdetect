import logging
import os
import os.path
import shutil
import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Dict, Set

from texcompile.client import compile
from typing_extensions import Literal

from lib.image_processing import (
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


if __name__ == "__main__":
    parser = ArgumentParser(description=("Detect the positions of symbols."))
    parser.add_argument(
        "--sources",
        required=True,
        help=(
            "Directory containing LaTeX sources (should not contain auxiliary / generated "
            + "files)."
        ),
    )
    parser.add_argument(
        "--debug-output-dir",
        help=(
            "If set, rasters of the paper will be output to this directory, which lets "
            + "you preview the positions of detected tokens and symbols."
        ),
    )
    parser.add_argument(
        "--texcompile-host",
        default="http://127.0.0.1",
        help="Hostname of service for compiling LaTeX.",
    )
    parser.add_argument(
        "--texcompile-port",
        default=8000,
        type=int,
        help="Port of service for compiling LaTeX.",
    )

    args = parser.parse_args()

    # Compile the TeX project to get the output PDF from which symbols will be extracted.
    with tempfile.TemporaryDirectory() as temp_dir:
        result = compile(
            args.sources,
            os.path.join(temp_dir, "outputs"),
            args.texcompile_host,
            args.texcompile_port,
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
        formulas = extract_formulas(args.sources)

        # Convert formulas to MathML representation.
        formula_mathmls = convert_tex_to_mathml(formulas)

        # Extract all symbols and tokens for all formulas.
        all_symbols: Set[TexSymbol] = set()
        all_tokens: Set[TexToken] = set()
        for (formula, mathml) in formula_mathmls.items():
            if mathml is not None:
                nodes = parse_formula(mathml)
                for node in nodes:
                    symbol = create_symbol_from_node(node, formula)
                    all_symbols.add(symbol)
                    all_tokens.update(symbol.tokens)

        # Filter to only valid symbols and tokens.
        valid_formulas = filter_valid_formulas(
            {s.tex for s in all_symbols}.union({t.tex for t in all_tokens})
        )
        valid_symbols = list(filter(lambda s: s.tex in valid_formulas, all_symbols))
        valid_tokens = list(filter(lambda s: s.tex in valid_formulas, all_tokens))

        # Add colorized copies of tokens and symbols to the TeX.
        modified_sources_dir = os.path.join(temp_dir, "modified-sources")
        shutil.copytree(args.sources, modified_sources_dir)

        if len(main_tex_files) > 1:
            logger.warning(
                "More than one main TeX file was found by the TeX compilation service. "
                + "This service was created assuming there would be only one main TeX file. "
                + "Some symbols may not be recognized."
            )

        main_tex_filename = main_tex_files[-1]
        with open(
            os.path.join(args.sources, main_tex_filename), errors="backslashescape"
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
            args.texcompile_host,
            args.texcompile_port,
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

        if args.debug_output_dir:
            token_debug_dir = os.path.join(args.debug_output_dir, "pages-with-tokens")
            token_instances = {
                page_no: token_index.get_instances()
                for (page_no, token_index) in token_locations.items()
            }
            save_debug_images(original_page_images, token_instances, token_debug_dir)

            symbol_debug_dir = os.path.join(args.debug_output_dir, "pages-with-symbols")
            save_debug_images(original_page_images, symbol_locations, symbol_debug_dir)

    # TODO(andrewhead): Relate symbols to each other (find which ones are children of others)
    # TODO(andrewhead): Find out why symbols do not appear in algorithm listings.
    # TODO(andrewhead): Find out why some functions are not getting detected.
    # TODO(andrewhead): 'd' and 'm' probably need to be shifted by vertical pixels slightly
    # to be detected; for some reason they are not getting detected.
