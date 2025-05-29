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
    """
    system_msg = (
        "You are a highly capable data-analysis assistant. The user will upload a dataset, "
        "and you will first review the table schema (column names and types) and a small sample of rows. "
        "Next, you will review the full conversation history, including the user’s last question. "
        "Then, provide a precise, data-driven answer that directly addresses their latest query. "
        "Always:\n"
        "  1. Reference specific column names and values from the data.\n"
        "  2. Include any relevant calculations or summary statistics.\n"
        "  3. Highlight patterns, trends, or anomalies backed by the data.\n"
        "  4. Offer concise recommendations or next steps when appropriate.\n"
        "Avoid generic summaries—be concrete, analytical, and results-oriented."
    )

    context = (
        "=== Dataset Overview ===\n"
        f"{column_summaries}\n\n"
        "=== Data Sample ===\n"
        f"{df_snippet}\n"
    )

    history_block = ""
    for speaker, text in history:
        role = "User" if speaker.lower() == "you" else "Assistant"
        history_block += f"{role}: {text}\n"

    prompt = (
        f"{system_msg}\n\n"
        f"{context}\n"
        "=== Conversation History ===\n"
        f"{history_block}\n"
        f"User: {user_question}\nAssistant:"
    )
    return prompt
