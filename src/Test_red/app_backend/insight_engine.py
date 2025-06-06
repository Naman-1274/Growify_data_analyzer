# File: src/Test_red/app_backend/insight_engine.py

import os
import requests
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# 1) READ TOGETHER API KEY FROM ENV
# ────────────────────────────────────────────────────────────────────────────────
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if TOGETHER_API_KEY is None:
    raise EnvironmentError("⚠️ Please set TOGETHER_API_KEY in your environment or in .env.")

# ────────────────────────────────────────────────────────────────────────────────
# 2) SET UP TOGETHER ENDPOINT & HEADERS
#    (Using Meta Llama 3.3 70B Instruct Turbo Free via Together's API)
# ────────────────────────────────────────────────────────────────────────────────
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Meta Llama 3.3 70B Instruct Turbo Free
TOGETHER_ENDPOINT = "https://api.together.xyz/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

# ────────────────────────────────────────────────────────────────────────────────
# 3) STEP 1: QUICK DATA SUMMARY VIA TOGETHER
# ────────────────────────────────────────────────────────────────────────────────
def summarize_with_together(
    df: pd.DataFrame,
    csv_snippet: str,
    max_output_tokens: int = 512
) -> str:
    """
    Returns a one-paragraph summary of:
      1) Column names + inferred types
      2) Mean/median/min/max for numeric columns
      3) A concise "trends/anomalies" statement

    We supply a small CSV snippet (header + first rows) so the model sees sample values.
    """
    prompt = (
        "You are a data analyst. Below is a CSV snippet "
        "(header + first few rows). Please:\n"
        "1. List each column name and its inferred data type.\n"
        "2. For each numeric column, give mean, median, min, and max.\n"
        "3. Provide a concise, plain-English summary of any notable trends or anomalies.\n\n"
        "CSV_SNIPPET:\n"
        "```\n"
        f"{csv_snippet}"
        "\n```"
    )

    payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": ["<|eot_id|>"]
    }

    resp = requests.post(TOGETHER_ENDPOINT, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    # Together chat completions API returns: {"choices": [{"message": {"content": "..."}}]}
    generated = data["choices"][0]["message"]["content"]
    return generated.strip()

# ────────────────────────────────────────────────────────────────────────────────
# 4) STEP 2: DEEP "WHY / HOW / PREDICT" VIA TOGETHER
# ────────────────────────────────────────────────────────────────────────────────
def analyze_data_with_together(
    df: pd.DataFrame,
    user_question: str,
    summary_text: str,
    csv_snippet: str,
    max_output_tokens: int = 768
) -> str:
    """
    Given:
      • df: full DataFrame (for deeper reference if needed)
      • user_question: e.g. "Why did Q1 2025 sales drop vs Q4 2024? Predict next quarter."
      • summary_text: output from summarize_with_together(...)
      • csv_snippet: header + first rows again as string

    Asks Together to:
      1. Explain why the patterns/trends from the summary might be happening.
      2. Suggest possible root causes or drivers.
      3. Provide a short, data-driven prediction for the next quarter.
    Returns the multi-paragraph answer as plain text.
    """
    prompt = (
        "You are an expert data scientist.\n\n"
        f"A user asked: '{user_question}'\n\n"
        "Below is a concise summary of the data:\n"
        "```\n"
        f"{summary_text}\n"
        "```\n\n"
        "Below is the header + first rows of the CSV for context:\n"
        "```\n"
        f"{csv_snippet}\n"
        "```\n\n"
        "Please:\n"
        "1. Explain why the patterns/trends from the summary might be happening.\n"
        "2. Suggest possible root causes or drivers for those trends.\n"
        "3. Provide a short, data-driven prediction of what may happen next quarter.\n"
        "Give your answer as well-organized paragraphs."
    )

    payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": ["<|eot_id|>"]
    }

    resp = requests.post(TOGETHER_ENDPOINT, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    generated = data["choices"][0]["message"]["content"]
    return generated.strip()