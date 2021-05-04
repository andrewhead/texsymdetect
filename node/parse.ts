// @ts-ignore
import * as program from "commander";
import * as csv from "fast-csv";
import * as katex from "katex";
import { FormulaParseResult } from "./types";

program
  .command("formulas-csv <csv-file>")
  .option(
    "-t, --throw-on-error",
    "Make KaTeX throw a parse error when it fails to parse an formula."
  )
  .option(
    "-c, --error-color <color>",
    "Hex code for color of parse error nodes in MathML."
  )
  .description(
    "Parse formulas from CSV file. Should include columns 'id' and 'formula'"
  )
  .action((csvPath, cmdObj) => {
    csv.parseFile(csvPath, { headers: true }).on("data", (row) => {
      const { id, formula } = row;
      const baseResult = {
        id,
        formula,
      };
      let result: FormulaParseResult;
      if (!id || !formula) {
        result = {
          ...baseResult,
          success: false,
          mathMl: null,
          errorMessage:
            "FormatError: Unexpected format of CSV row. Check that parameters for writing the CSV and reading the CSV are consistent.",
        };
      } else {
        try {
          const mathMl = parse(formula, cmdObj.throwOnError, cmdObj.errorColor);
          result = { ...baseResult, success: true, mathMl, errorMessage: null };
        } catch (e) {
          result = {
            ...baseResult,
            success: false,
            mathMl: null,
            errorMessage: e.toString(),
          };
        }
      }
      console.log(JSON.stringify(result));
    });
  });

program
  .command("formula <formula>")
  .option(
    "-t, --throw-on-error",
    "Make KaTeX throw a parse error when it fails to parse an formula."
  )
  .option(
    "-c, --error-color <color>",
    "Hex code for color of parse error nodes in MathML."
  )
  .description(
    "Parse a TeX formula. Provide formula as a string, without surrounding delimiters (e.g., '$', '\\begin{equation}', etc."
  )
  .action((formula, cmdObj) => {
    console.log(parse(formula, cmdObj.throwOnError, cmdObj.errorColor));
  });

program.parse(process.argv);

/**
 * Parse a formula into a MathML tree with source location annotations.
 */
function parse(
  formula: string,
  throwOnError?: boolean,
  errorColor?: string
): string {
  return katex.renderToString(formula, {
    output: "mathml",
    throwOnError: throwOnError || false,
    errorColor,
  });
}
