import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

from lib.scan_tex import (
    EndOfInput,
    Match,
    Pattern,
    TexScanner,
    has_balanced_braces,
    scan_tex,
)


@dataclass
class CharacterRange:
    start: int
    end: int


@dataclass
class Documentclass(CharacterRange):
    pass


class DocumentclassExtractor:
    """
    Detected calls to the documentclass command, wherever it appears in a TeX file. See past
    implementations of this class if you want a version of this extractor that enforced the
    requirement that the documentclass was the first command in the TeX.
    """

    def parse(self, tex: str) -> Optional[Documentclass]:
        patterns = [
            Pattern("documentclass", r"\\documentclass"),
            Pattern("optional_arg", r"\[[^\]]*?\]"),
            Pattern("required_arg", r"{[^}]*?}"),
        ]

        match_stage = "start"
        start: int = -1
        required_arg = None

        scanner = scan_tex(tex, patterns, include_unmatched=True)
        for match in scanner:

            if match_stage == "start":
                if match.pattern.name != "documentclass":
                    continue
                start = match.start
                match_stage = "awaiting-required-arg"

            # Once we hit a token that's not the document class or argument, return the document
            # class if the required argument has been found; otherwise, abort.
            elif match.pattern.name == "UNKNOWN":
                if match_stage == "awaiting-optional-arg":
                    return Documentclass(start, match.start)
                if not match.text.isspace():
                    break

            elif match_stage == "awaiting-required-arg":
                if match.pattern.name == "required_arg":
                    match_stage = "awaiting-optional-arg"
                    required_arg = match

            elif match_stage == "awaiting-optional-arg":
                if match.pattern.name == "optional_arg":
                    end = match.end
                    return Documentclass(start, end)

        if required_arg is not None:
            return Documentclass(start, required_arg.end)
        return None


@dataclass
class BeginDocument(CharacterRange):
    pass


class BeginDocumentExtractor:
    def parse(self, tex: str) -> Optional[BeginDocument]:
        pattern = Pattern("begin_document", r"\\begin{document}")
        scanner = scan_tex(tex, [pattern], include_unmatched=False)
        try:
            match = next(scanner)
            return BeginDocument(match.start, match.end)
        except StopIteration:
            return None


@dataclass
class EndDocument(CharacterRange):
    pass


class EndDocumentExtractor:
    def parse(self, tex: str) -> Optional[EndDocument]:
        pattern = Pattern("begin_document", r"\\end{document}")
        scanner = scan_tex(tex, [pattern], include_unmatched=False)
        try:
            match = next(scanner)
            return EndDocument(match.start, match.end)
        except StopIteration:
            return None


@dataclass
class Equation:

    tex_path: str
    """
    The path to the TeX file this entity was found in. Should be a short path, relative to the
    directory of sources this TeX file is in, not relative to the data directory, and not
    relative to the source code directory for this project.
    """

    id_: str
    """
    An ID which, when combined with the TeX path and arXiv ID for this entity, can uniquely
    identify the entity. In many cases, this will simply be a stringified index of which entity
    in the TeX file this is (e.g., 0, 1, 2, ...).
    """

    start: int
    "Character position where this range begins in the TeX."

    end: int
    "Character position where this range ends in the TeX."

    tex: str
    " The TeX for the detected entity. "

    i: int
    " Index of this equation in the TeX document. "

    content_start: int
    " Index of character where the contents (i.e. 'content_tex') of the equation starts. "

    content_end: int
    " Index of character where the contents of the equation ends. "

    content_tex: str
    " TeX for the equation contents, inside the environment (e.g., 'x + y'). "

    katex_compatible_tex: str
    " A santized version of the equation content meant for KaTeX parsing. "

    depth: int
    """
    Depth within a tree of equations. Most equations will not be nested in others, so will have a
    depth of 0 if not nested in another equation. As an example, if this equation is nested in
    another equation, which is nested in another equation, it will have a depth of 2.
    """


LEFT_BRACE = Pattern("left_brace", r"\{")
RIGHT_BRACE = Pattern("right_brace", r"\}")


@dataclass(frozen=True)
class NamedEnv:
    name: str
    star: bool
    arg_pattern: str = ""


@dataclass(frozen=True)
class DelimitedEnv:
    delimiter: str


@dataclass(frozen=True)
class StartEndEnv:
    start: str
    end: str


EnvSpec = Union[DelimitedEnv, NamedEnv, StartEndEnv]


"""
List of math environments from: https://latex.wikia.org/wiki/List_of_LaTeX_environments
TODO(andrewhead): Support 'alignat' and 'matrix'.
TODO(andrewhead): Determine if any other environments besides 'array' have arguments.
"""
MATH_ENVIRONMENT_SPECS: Dict[str, EnvSpec] = {
    # Inline math
    "dollar": DelimitedEnv(r"\$(?!\$)"),
    "parens": StartEndEnv(r"\\\(", r"\\\)"),
    "math": NamedEnv("math", star=False),
    # Display math
    "dollardollar": DelimitedEnv(r"\$\$"),
    "bracket": StartEndEnv(r"\\\[", r"\\\]"),
    "displaymath": NamedEnv("displaymath", star=True),
    "equation": NamedEnv("equation", star=True),
    "split": NamedEnv("split", star=True),
    "array": NamedEnv("array", arg_pattern=r"(?:\{[^}]*\})?", star=True),
    "eqnarray": NamedEnv("eqnarray", star=True),
    "multiline": NamedEnv("multiline", star=True),
    "gather": NamedEnv("gather", star=True),
    "align": NamedEnv("align", star=True),
    "flalign": NamedEnv("flalign", star=True),
}


def begin_environment_regex(name: str, arg_pattern: str = "") -> str:
    return r"\\begin{" + name + r"}" + r"(\s*" + arg_pattern + ")"


def end_environment_regex(name: str) -> str:
    return r"\\end{" + name + r"}"


def make_math_environment_patterns() -> List[Pattern]:

    begin = begin_environment_regex
    end = end_environment_regex

    patterns: List[Pattern] = []
    for name, spec in MATH_ENVIRONMENT_SPECS.items():
        if isinstance(spec, DelimitedEnv):
            patterns.append(Pattern(name + "_delimiter", spec.delimiter))
        elif isinstance(spec, StartEndEnv):
            patterns.append(Pattern(name + "_start", spec.start))
            patterns.append(Pattern(name + "_end", spec.end))
        elif isinstance(spec, NamedEnv):
            patterns.append(
                Pattern(name + "_start", begin(spec.name, spec.arg_pattern))
            )
            patterns.append(Pattern(name + "_end", end(spec.name)))
            if spec.star:
                patterns.append(
                    Pattern(
                        name + "s_start", begin(spec.name + r"\*", spec.arg_pattern)
                    )
                )
                patterns.append(Pattern(name + "s_end", end(spec.name + r"\*")))
    return patterns


class FormulaExtractor:
    """
    TODO(andrewhead): Cases that this doesn't yet handle:
    * Nested dollar signs: "$x + \\hbox{$y$}$"
    """

    def __init__(self) -> None:
        self.PATTERNS = make_math_environment_patterns()

    def parse(self, tex_path: str, tex: str) -> Iterator[Equation]:

        self._stack: List[Match] = []  # pylint: disable=attribute-defined-outside-init
        self._tex = tex  # pylint: disable=attribute-defined-outside-init
        self._tex_path = tex_path  # pylint: disable=attribute-defined-outside-init
        self._equation_index = 0  # pylint: disable=attribute-defined-outside-init

        scanner = scan_tex(tex, self.PATTERNS)
        for match in scanner:
            for equation in self._process_token(match):
                yield equation

    def _process_token(self, match: Match) -> Iterator[Equation]:
        pattern_name = match.pattern.name

        if pattern_name.endswith("_start"):
            self._stack.append(match)

        elif self._in_environment(pattern_name):
            start_pattern_name = self._get_start_pattern_name(pattern_name)
            while self._stack[-1].pattern.name != start_pattern_name:
                self._stack.pop()
            start_match = self._stack.pop()

            start = start_match.start
            end = match.end
            depth = len(self._stack)
            equation_tex = self._tex[start_match.start : match.end]
            content_tex = self._tex[start_match.end : match.start]

            yield Equation(
                tex_path=self._tex_path,
                id_=str(self._equation_index),
                tex=equation_tex,
                start=start_match.start,
                end=match.end,
                i=self._equation_index,
                content_start=start_match.end,
                content_end=match.start,
                content_tex=content_tex,
                katex_compatible_tex=sanitize_equation(content_tex),
                depth=depth,
            )
            self._equation_index += 1

        elif pattern_name.endswith("_delimiter"):
            self._stack.append(match)

    def _get_start_pattern_name(self, end_pattern_name: str) -> str:
        if end_pattern_name.endswith("_delimiter"):
            return end_pattern_name
        return re.sub("_end$", "_start", end_pattern_name)

    def _in_environment(self, end_pattern_name: str) -> bool:
        start_pattern_name = self._get_start_pattern_name(end_pattern_name)
        return any([m.pattern.name == start_pattern_name for m in self._stack])


"""
Valid units of measure in TeX and related engines.
"""
LENGTH_UNITS = [
    # From The TeXBook, "Chapter 10: Dimensions", p57.
    "pt",
    "pc",
    "in",
    "bp",
    "cm",
    "mm",
    "dd",
    "cc",
    "sp",
    # From the LaTeX book on Wikibooks: https://en.wikibooks.org/wiki/LaTeX/Lengths
    "ex",
    "em",
    "nd",
    "nc",
]

"""
Parameters for laying out arrays in LaTeX. From "The LaTeX2E Sources" by Braams et al.,
version 2019-10-01 Patch Level 1, page 341.
"""
ARRAY_PARAMETERS = [
    "arraycolsep",
    "tabcolsep",
    "arrayrulewidth",
    "doublerulesep",
    "arraystretch",
]


@dataclass
class LengthAssignment(CharacterRange):
    pass


class EquationLengthAssignmentExtractor:
    """
    Extracts length assignments of the form "\\[parameter]=[#][unit of measurement]",
    for example "\\arraycolsep=2pt"
    """

    def parse(self, tex: str) -> Iterator[LengthAssignment]:
        parameter_names_pattern = (
            r"(?:" + "|".join([r"\\" + p for p in ARRAY_PARAMETERS]) + ")"
        )
        unit_pattern = r"(?:" + "|".join(LENGTH_UNITS) + ")"
        assignment_pattern = (
            parameter_names_pattern + r"\s*=\s*[0-9\.]+\s*" + unit_pattern
        )
        pattern = Pattern("length_assignment", assignment_pattern)
        scanner = scan_tex(tex, [pattern])
        for match in scanner:
            yield LengthAssignment(match.start, match.end)


def sanitize_equation(tex: str) -> str:
    tex = _replace_unwanted_commands_with_spaces(tex)
    return tex


def _replace_unwanted_commands_with_spaces(tex: str) -> str:
    """
    KaTeX isn't programmed to support the entire vocabulary of LaTeX equation markup (though it
    does support a lot, see https://katex.org/docs/support_table.html).

    For those commands that we don't need to have parsed (e.g., 'label'), this function will
    strip those commands out, so that they cause KaTeX to crash or have unexpected behavior.
    'label', for example, if not removed, will have its argument parsed as an equation, and
    will be identified as consisting of many symbols.
    """
    UNWANTED_MACROS = [
        MacroDefinition("ref", "#1"),
        MacroDefinition("label", "#1"),
        MacroDefinition("nonumber", ""),
    ]
    macro_extractor = MacroExtractor()
    for macro_definition in UNWANTED_MACROS:
        for macro in macro_extractor.parse(tex, macro_definition):
            tex = _replace_substring_with_space(tex, macro.start, macro.end)

    length_assignment_extractor = EquationLengthAssignmentExtractor()
    length_assignments = length_assignment_extractor.parse(tex)
    for assignment in length_assignments:
        tex = _replace_substring_with_space(tex, assignment.start, assignment.end)

    UNWANTED_PATTERNS = [
        Pattern("ampersand", "&"),
        Pattern("split_start", begin_environment_regex("split")),
        Pattern("split_end", end_environment_regex("split")),
    ]
    unwanted_matches = scan_tex(tex, UNWANTED_PATTERNS)
    for match in unwanted_matches:
        tex = _replace_substring_with_space(tex, match.start, match.end)

    return tex


def _replace_substring_with_space(s: str, start: int, end: int) -> str:
    return s[:start] + (" " * (end - start)) + s[end:]


@dataclass
class MacroDefinition:
    name: str
    " Name of macro. For example, 'omega' if you want to detect \\omega{}. "

    parameter_string: str
    " Parameter string for macro, in TeX format, e.g., '[#1]#2'. As with TeX, can be empty string. "


@dataclass
class Macro(CharacterRange):
    tex: str


class MacroExtractor:
    """
    Extracts all instances of a macro defined by 'macro_definition'.
    This extractor follows the argument-parsing logic described on p203-4 of the TeXBook.
    """

    def parse(self, tex: str, macro_definition: MacroDefinition) -> Iterator[Macro]:
        parser = self._parse(tex, macro_definition)
        while True:
            try:
                macro = next(parser)
                yield macro
            # Run until the scanner has indicated that the end of input has been reached.
            except EndOfInput:
                return

    def _parse(self, tex: str, macro_definition: MacroDefinition) -> Iterator[Macro]:
        self.scanner = TexScanner(tex)  # pylint: disable=attribute-defined-outside-init
        name_pattern = Pattern("macro", r"\\" + macro_definition.name)

        # This loop will run until the scanner raises an 'EndOfInput' or indicates another error.
        while True:

            # Parse the macro name.
            step = self.scanner.next([name_pattern])
            macro_start = step.match.start
            token_end = step.match.end

            # Parse each of the expected tokens in the parameter string.
            tokens = re.split(r"(#\d+)", macro_definition.parameter_string)
            if tokens[0] == "":
                del tokens[0]
            if len(tokens) >= 1 and tokens[len(tokens) - 1] == "":
                del tokens[len(tokens) - 1]
            for i, token in enumerate(tokens):
                if re.match(r"#\d+", token):
                    if (i == len(tokens) - 1) or (re.match(r"#\d+", tokens[i + 1])):
                        token_end = self._scan_undelimited_parameter()
                    else:
                        token_end = self._scan_delimited_parameter(tokens[i + 1], tex)
                else:
                    token_end = self._scan_delimiter(token)

            # The macros text is the text of the name and all parameters.
            yield Macro(macro_start, token_end, tex[macro_start:token_end])

    def _scan_undelimited_parameter(self) -> int:
        patterns = [LEFT_BRACE, Pattern("nonspace_character", r"\S")]
        step = self.scanner.next(patterns)

        # If a non-space character, match just the first character.
        if step.match.pattern.name == "nonspace_character":
            return step.match.end

        # If the first match is a left-brace, parse until the braces are balanced.
        brace_depth = 1
        brace_patterns = [LEFT_BRACE, RIGHT_BRACE]
        while True:
            step = self.scanner.next(brace_patterns)
            if step.match.pattern.name == "left_brace":
                brace_depth += 1
            elif step.match.pattern.name == "right_brace":
                brace_depth -= 1
            if brace_depth == 0:
                return step.match.end

    def _scan_delimited_parameter(self, delimiter: str, tex: str) -> int:
        scan_start = self.scanner.i

        # Scan for the delimiter with a lookahead so that the scanner doesn't consume the tokens
        # for the delimiter while searching for it.
        delimiter_pattern = Pattern("delimiter", "(?=" + re.escape(delimiter) + ")")

        while True:
            step = self.scanner.next([delimiter_pattern])
            text_before_delimiter = tex[scan_start : step.match.start]
            if has_balanced_braces(text_before_delimiter):
                return step.match.start

    def _scan_delimiter(self, delimiter: str) -> int:
        pattern = Pattern("delimiter", re.escape(delimiter))
        step = self.scanner.next([pattern], include_unmatched=True)
        if step.skipped is not None and len(step.skipped) > 0:
            logging.warning(
                "Unexpectedly found unmatched text before macro argument delimiter."
            )
        return step.match.end
