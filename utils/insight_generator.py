# utils/insight_generator.py

import time
from google.generativeai import GenerativeModel

model_name = "gemini-1.5-flash"  # Or "gemini-2.0-pro"
model = GenerativeModel(model_name)

def generate_insight(prompt: str, max_retries: int = 3, delay: float = 2.0) -> str:
    """
    Calls Gemini to generate insight using the provided prompt.
    Retries on transient errors (rate limit, network).
    """

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            else:
                raise ValueError("Gemini returned an empty response.")

        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    raise RuntimeError(f"❌ Gemini quota/rate limit hit. Retry later.\nDetails: {e}")
            else:
                raise RuntimeError(f"❌ Gemini error: {e}")

    raise RuntimeError("❌ Insight generation failed after retries.")
