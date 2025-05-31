import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

class CodeExecutionError(Exception):
    """Raised when the dynamically generated code snippet fails to execute."""
    pass

def execute_code_snippet(code: str, df: pd.DataFrame, return_all_vars: bool = False) -> dict:
    """
    Executes a Gemini‐generated code snippet, but preloads the global namespace with:
      - pandas as pd
      - numpy as np
      - matplotlib.pyplot as plt
      - seaborn as sns
      - scipy.stats as stats
      - the DataFrame `df`
    Any exception is caught and rethrown as CodeExecutionError.

    Args:
        code:         A block of valid Python code (string) that references `df`, `pd`, `np`, etc.
        df:           The pandas DataFrame on which the code should operate.
        return_all_vars: If True, return a dict of all local variables created by the snippet.
                         Otherwise, return an empty dict.

    Returns:
        local_vars (dict): All variables defined in the snippet if return_all_vars=True, else {}.
    """
    # 1) Build a globals dict so that the snippet sees pd, np, plt, sns, stats, and df.
    preamble_globals = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "stats": stats,
        "df": df,
    }

    local_vars = {}
    try:
        # Execute the snippet. Any variable assigned in `code` ends up in local_vars.
        exec(code, preamble_globals, local_vars)
    except Exception as e:
        # Wrap any error as CodeExecutionError so calling code can catch it.
        raise CodeExecutionError(str(e))

    if return_all_vars:
        return local_vars
    else:
        return {}
