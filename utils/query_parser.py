# utils/query_parser.py

import os
import json
import re
from datetime import datetime
from google.generativeai import GenerativeModel, configure

# Gemini model (lazy init)
model = None

def parse_timeframe_with_llm(query: str, today: str = None):
    """
    Uses Gemini to extract timeframe: start_date, end_date, and aggregation level.
    Returns those 3 strings.
    """
    global model

    if model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        print("✅ GEMINI_API_KEY loaded:", bool(api_key))
        if not api_key:
            raise EnvironmentError("❌ GEMINI_API_KEY not found in environment or .env file.")
        configure(api_key=api_key)
        model = GenerativeModel("gemini-1.5-flash")

    if not today:
        today = datetime.today().strftime("%Y-%m-%d")

    prompt = f"""
Today's date is {today}.
The user asked: "{query}"

Return only this JSON format:

{{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "aggregation_level": "month" | "day" | "none"
}}

Rules:
- Use "month" for ranges like quarters or multi-month spans.
- Use "day" for periods <30 days (e.g. last 2 weeks).
- Use "none" only for a single date.
- "Last year" = full previous calendar year.
- Do not add explanation. Return ONLY the raw JSON.
"""

    try:
        result = model.generate_content(prompt)
        if not hasattr(result, "text") or not result.text:
            raise ValueError("Gemini returned no response.")

        raw = result.text.strip()
        cleaned = re.sub(r"^```json|```$", "", raw).strip()
        parsed = json.loads(cleaned)

        return parsed["start_date"], parsed["end_date"], parsed["aggregation_level"]

    except Exception as e:
        raise RuntimeError(f"❌ Failed to parse timeframe: {e}")
