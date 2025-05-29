import os
import google.generativeai as genai
from src.Test_red.exception import ModelAPIError
from src.Test_red.logger import logger
from dotenv import load_dotenv
import pandas as pd

# Load and validate API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY is missing.")
    raise ModelAPIError("Gemini API key not set in environment.")

# Configure Gemini model
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
except Exception as e:
    logger.error(f"Gemini configuration failed: {e}")
    raise ModelAPIError(f"Gemini setup error: {e}")

def generate_response(prompt: str) -> str:
    """
    Send the full text-prompt to Gemini and return a stripped response.
    """
    try:
        resp = model.generate_content(contents=[{"parts": [{"text": prompt}]}])
        return resp.text.strip()
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise ModelAPIError(f"Gemini API call failed: {e}")

def generate_response_df(df: pd.DataFrame, full_prompt: str) -> str:
    """
    Given a DataFrame + assembled prompt, forward to Gemini.
    """
    # NOTE: We assume the prompt already includes snippets, summaries, and history.
    return generate_response(full_prompt)
