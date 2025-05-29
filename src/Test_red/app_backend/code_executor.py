# src/Test_red/app_backend/code_executor.py

import pandas as pd
import ast
import traceback

class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass

def execute_code_snippet(code: str, df: pd.DataFrame, return_all_vars: bool = False):
    """
    Executes the given Python code (as a string) in a restricted namespace where:
      - 'df' is the uploaded DataFrame (a copy)
      - 'pd' is pandas
    If return_all_vars=True, return a dict of all variables Gemini assigned (excluding 'pd' and 'df').
    Otherwise, return a single 'result' or last expression as before.
    Performs a syntax check (ast.parse) before execution to catch indentation/syntax errors early.
    Raises CodeExecutionError if anything goes wrong.
    """
    # 1) Basic syntax validation
    try:
        ast.parse(code)
    except SyntaxError as syn_err:
        raise CodeExecutionError(f"Syntax error in generated code:\n{syn_err}")

    # 2) Prepare the execution namespace
    local_vars = {"pd": pd, "df": df.copy()}
    try:
        lines = code.strip().splitlines()
        if not lines:
            return {} if return_all_vars else None

        body_lines = lines[:-1]
        last_line = lines[-1]

        # Execute all but the last line
        if body_lines:
            exec("\n".join(body_lines), {}, local_vars)

        # Try to eval the last line; if that fails, exec it instead
        try:
            last_val = eval(last_line, {}, local_vars)
        except Exception:
            exec(last_line, {}, local_vars)
            last_val = None

        # If caller wants all variables, return everything except 'pd' and 'df'
        if return_all_vars:
            return {k: v for k, v in local_vars.items() if k not in ("pd", "df")}

        # Otherwise, return 'result' if present, else last_val or common variable names
        if "result" in local_vars:
            return local_vars["result"]
        if isinstance(last_val, (pd.DataFrame, pd.Series, int, float, str, list, dict)):
            return last_val
        for var_name in ["df_result", "output", "res"]:
            if var_name in local_vars:
                return local_vars[var_name]
        return None

    except Exception as e:
        tb = traceback.format_exc()
        raise CodeExecutionError(f"{e}\nTraceback:\n{tb}")
