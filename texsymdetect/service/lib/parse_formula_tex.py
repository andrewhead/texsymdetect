import csv
import json
import os.path
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from typing_extensions import Literal

from lib.parse_mathml import Node, NodeType, parse_formula

MathMl = str
Path = str
Formula = str


@dataclass(frozen=True)
class TexToken:
    tex: str
    type_: Literal["atom", "affix"]
    mathml: str
    font_macros: Tuple[str, ...]


@dataclass(frozen=True)
class TexSymbol:
    type_: NodeType
    tex: str
    mathml: str
    tokens: Tuple[TexToken, ...]


def convert_tex_to_mathml(
    formulas: Iterable[str], tolerate_parse_errors: bool = True
) -> Dict[str, Optional[MathMl]]:
    """
    Return map from formula TeX to its MathML.
    
    If 'tolerate_parse_errors' is False and a formula could not be successfully parsed by KaTeX
    (for instance, if it includes unknown macros), its MathML will be empty (i.e., 'None').
    
    By default, 'tolerate_parse_errors' is True, which means that this method will try to process
    all valid parts of the TeX while inserting special MathML nodes for the parts of the TeX that
    cannot be parsed.
    """

    formula_data = [{"id": i, "formula": e} for (i, e) in enumerate(formulas)]
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "formula.csv")
        with open(csv_path, "w", encoding="utf-8", errors="ignore") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=["id", "formula"], quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for row in formula_data:
                writer.writerow(row)

        command_args = [
            "npm",
            # Suppress 'npm' output we don't care about.
            "--silent",
            "start",
            "formulas-csv",
            os.path.abspath(csv_path),
        ]
        if not tolerate_parse_errors:
            command_args += ["--", "--throw-on-error"]

        result = subprocess.run(
            command_args,
            cwd="node",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=False,
        )

        mathmls: Dict[str, Optional[MathMl]] = {}
        for line in result.stdout.strip().splitlines():
            data = json.loads(line)
            formula = data["formula"]
            mathml = data["mathMl"]
            mathmls[formula] = mathml if data["success"] else None

        return mathmls


def filter_valid_formulas(formulas: Iterable[str]) -> Set[str]:
    """
    Return set of symbols that have been assessed as having valid TeX.
    """

    formula_mathmls = convert_tex_to_mathml(formulas, tolerate_parse_errors=False)
    valid_formulas = [f for (f, m) in formula_mathmls.items() if m is not None]
    return set(valid_formulas)


def create_symbol_from_node(node: Node, formula: str) -> TexSymbol:
    """
    Convert output node from MathML parser into symbol. The key part of this method
    is that it adds the TeX for tokens and symbols to the symbol data. Because of this,
    it requires the 'formula' that the symbol was found in.
    """
    tex_tokens: Tuple[TexToken, ...] = ()
    for token in node.tokens:

        tex = formula[token.start : token.end]
        # For affixes, add a empty group ('{}') to render the token without an argument.
        if token.type_ == "affix":
            tex += "{}"

        tex_tokens = tex_tokens + (
            TexToken(
                tex=tex,
                type_=token.type_,
                mathml=token.mathml,
                font_macros=token.font_macros,
            ),
        )

    symbol = TexSymbol(
        node.type_,
        formula[node.start : node.end],
        mathml=str(node.element),
        tokens=tex_tokens,
    )
    return symbol


@dataclass
class ExtendedTexSymbol:
    """
    A symbol extracted from a TeX formula, with additional information about
    where that symbol came from within the formula.
    """

    id_: int
    type_: NodeType
    mathml: str
    tex: str
    start: int
    end: int
    parent: Optional[int]


def parse_symbols_in_formulas(
    formulas: List[str],
) -> Dict[Formula, List[ExtendedTexSymbol]]:
    """ Parses formulas into their symbols. """

    # Convert formulas to MathML representation.
    formula_mathmls = convert_tex_to_mathml(formulas)

    # Extract all symbols and tokens for all formulas.
    formula_symbols: Dict[Formula, List[ExtendedTexSymbol]] = {f: [] for f in formulas}

    for (formula, mathml) in formula_mathmls.items():

        symbol_ids: Dict[int, int] = {}
        new_symbol_id = 0

        def _get_id(node: Node) -> int:
            " Maps from Python object ID for node to a simple, small numeric index. "
            object_id = id(node)
            nonlocal new_symbol_id
            if object_id not in symbol_ids:
                symbol_ids[object_id] = new_symbol_id
                new_symbol_id += 1
            return symbol_ids[object_id]

        new_symbol_id = 0
        parents: Dict[int, int] = {}

        if mathml is not None:
            nodes = parse_formula(
                mathml, merge_adjacent_elements=False, insert_function_elements=False,
            )
            # Build a map between parent and child symbols.
            for node in nodes:
                for child in node.child_symbols:
                    parents[_get_id(child)] = _get_id(node)

            # Create a symbol for each node in the formula's parse tree.
            for node in nodes:
                tex_symbol = create_symbol_from_node(node, formula)
                formula_symbols[formula].append(
                    ExtendedTexSymbol(
                        id_=_get_id(node),
                        type_=node.type_,
                        mathml=tex_symbol.mathml,
                        tex=tex_symbol.tex,
                        start=node.start,
                        end=node.end,
                        parent=parents.get(_get_id(node)),
                    )
                )

    return formula_symbols
