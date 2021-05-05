export interface Token {
  index: number;
  start: number;
  end: number;
  text: string;
}

export interface FormulaParseResult {
  success: boolean;
  id: string;
  formula: string;
  mathMl: string | null;
  errorMessage: string | null;
}
