import csv
import json
import os.path
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from typing_extensions import Literal

from lib.parse_mathml import Node, NodeType, Token

MathMl = str
Path = str


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
