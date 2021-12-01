import logging
import os
import os.path
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("symbol-extractor-client")


Path = str


class ServerConnectionException(Exception):
    pass


@dataclass
class Location:
    left: float
    top: float
    width: float
    height: float
    page: int


SymbolId = int


@dataclass
class Symbol:
    id_: SymbolId
    type_: str
    mathml: str
    tex: str
    location: Location
    parent: Optional["Symbol"]


@dataclass
class TexSymbol:
    """
    A symbol extracted from a TeX formula, with additional information about
    where that symbol came from within the formula.
    """

    id_: int
    type_: str
    mathml: str
    tex: str
    start: int
    end: int
    parent: Optional["TexSymbol"]


def detect_symbols(
    sources_dir: Path,
    host: str = "http://127.0.0.1",
    port: int = 8001,
    try_expand_macros: Optional[bool] = None,
    require_blank_border: Optional[bool] = None,
    insert_function_elements: Optional[bool] = None,
    merge_adjacent_elements: Optional[bool] = None,
) -> List[Symbol]:
    """
    Detect positions of symbols in LaTeX paper. Documentation and defaults for options (e.g.,
    'require_blank_border') appears in the server code for the 'extract_symbols' endpoint.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare a gzipped tarball file containing the sources.
        archive_filename = os.path.join(temp_dir, "archive.tgz")
        with tarfile.open(archive_filename, "w:gz") as archive:
            archive.add(sources_dir, arcname=os.path.sep)

        # Prepare query parameters.
        with open(archive_filename, "rb") as archive_file:
            files = {"sources": ("archive.tgz", archive_file, "multipart/form-data")}

            # Make request to service.
            endpoint = f"{host}:{port}/"
            try:
                response = requests.post(
                    endpoint,
                    files=files,
                    params={
                        "try_expand_macros": try_expand_macros,
                        "require_blank_border": require_blank_border,
                        "insert_function_elements": insert_function_elements,
                        "merge_adjacent_elements": merge_adjacent_elements,
                    },
                )
            except requests.exceptions.RequestException as e:
                raise ServerConnectionException(
                    f"Request to server {endpoint} failed.", e
                )

    # Get result
    data = response.json()

    # Create symbols from JSON.
    symbols: Dict[SymbolId, Symbol] = {}
    parents: Dict[SymbolId, SymbolId] = {}
    for item in data:
        symbol = Symbol(
            id_=item["id"],
            type_=item["type"],
            mathml=item["mathml"],
            tex=item["tex"],
            location=Location(
                item["location"]["left"],
                item["location"]["top"],
                item["location"]["width"],
                item["location"]["height"],
                item["location"]["page"],
            ),
            parent=None,
        )
        symbols[symbol.id_] = symbol
        parents[symbol.id_] = item["parent"]

    # Resolve parents of symbols.
    for id_, symbol in symbols.items():
        if parents[id_]:
            symbol.parent = symbols[parents[id_]]

    return [s for s in symbols.values()]


def parse_formulas(
    formulas: List[str], host: str = "http://127.0.0.1", port: int = 8001
) -> Dict[str, List[TexSymbol]]:
    """
    Parse a set of a LaTeX formulas to detect the symbols that the formulas
    are comprised of, along with normalized MathML representations of symbols.
    """

    # Make request to service.
    endpoint = f"{host}:{port}/parse_formulas"
    try:
        response = requests.post(endpoint, json={"formulas": formulas})
    except requests.exceptions.RequestException as e:
        raise ServerConnectionException(f"Request to server {endpoint} failed.", e)

    # Get result.
    data = response.json()

    formula_symbols: Dict[str, List[TexSymbol]] = {f: [] for f in formulas}
    for symbol_data, formula in zip(data, formulas):

        # Create symbols from JSON
        symbols: Dict[SymbolId, TexSymbol] = {}
        parents: Dict[SymbolId, SymbolId] = {}
        for item in symbol_data:
            symbol = TexSymbol(
                id_=item["id_"],
                type_=item["type_"],
                mathml=item["mathml"],
                tex=item["tex"],
                start=item["start"],
                end=item["end"],
                parent=None,
            )
            symbols[symbol.id_] = symbol
            parents[symbol.id_] = item["parent"]

        # Resonlve parents of symbols.
        for id_, symbol in symbols.items():
            if parents[id_]:
                symbol.parent = symbols[parents[id_]]

        formula_symbols[formula] = list(symbols.values())

    return formula_symbols
