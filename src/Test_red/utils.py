import pandas as pd

def df_to_markdown_full(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Convert up to `max_rows` into a Markdown table.
    If the DataFrame is larger, it takes the head + tail samples.
    """
    if df.empty:
        return "*(empty dataset)*"
    sample = df if len(df) <= max_rows else pd.concat([df.head(max_rows//2), df.tail(max_rows//2)])
    try:
        return sample.to_markdown(index=False)
    except Exception:
        return sample.to_csv(index=False)

def get_column_summaries(df: pd.DataFrame) -> str:
    """
    Summarize each column: dtype, unique count, null count, (numeric) basic stats.
    """
    lines = []
    for col in df.columns:
        s = df[col]
        dtype = s.dtype
        nuniq = s.nunique(dropna=True)
        nnull = s.isna().sum()
        line = f"- **{col}** (`{dtype}`): {nuniq} unique, {nnull} missing"
        if pd.api.types.is_numeric_dtype(dtype):
            stats = s.agg(["mean", "std", "min", "max"]).to_dict()
            line += (f", mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                     f"min={stats['min']}, max={stats['max']}")
        lines.append(line)
    return "\n".join(lines)

def build_conversational_prompt(
    df_snippet: str,
    column_summaries: str,
    history: list[tuple[str, str]],
    user_question: str
) -> str:
    """
    Assemble the System + Context + History + Question into one Gemini‐ready prompt.
    We explicitly tell Gemini that the full DataFrame is available as `df` in memory,
    and that the snippet is only a small sample.
    """
    system_msg = (
        "You are a highly capable data‐analysis assistant. The user has just uploaded a "
        "dataset; in Python, that dataset is already loaded as a pandas DataFrame named `df`.  \n"
        "When you write code, you should always operate on the full DataFrame (`df`)—"
        "the snippet below is just a small sample.  \n"
        "Next, you will review the table schema (column names and types) and a small sample of rows. "
        "Then review the full conversation history, including the user’s last question. "
        "Finally, provide a precise, data‐driven answer that directly addresses their latest query.  \n"
        "Always:\n"
        "  1. Reference specific column names and values from the data.\n"
        "  2. Include any relevant calculations or summary statistics.\n"
        "  3. Highlight patterns, trends, or anomalies backed by the data.\n"
        "  4. Offer concise recommendations or next steps when appropriate.\n"
        "Avoid generic summaries—be concrete, analytical, and results‐oriented."
    )

    context = (
        "=== Dataset Context ===\n"
        "*(The full DataFrame is already loaded in memory as `df`. The table below is just a small sample.)*\n\n"
        "=== Column Summaries ===\n"
        f"{column_summaries}\n\n"
        "=== Data Sample (first/last rows) ===\n"
        f"{df_snippet}\n"
    )

    # Rebuild history into a single block
    history_block = ""
    for speaker, text in history:
        role = "User" if speaker.lower() == "you" else "Assistant"
        history_block += f"{role}: {text}\n"

    prompt = (
        f"{system_msg}\n\n"
        f"{context}\n\n"
        "=== Conversation History ===\n"
        f"{history_block}\n"
        f"User: {user_question}\nAssistant:"
    )
    return prompt

