# src/Test_red/app_backend/insight_engine.py
import os
import google.generativeai as genai
from src.Test_red.exception import ModelAPIError
from src.Test_red.logger import logger
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY is missing.")
    raise ModelAPIError("Gemini API key not set in environment.")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
except Exception as e:
    logger.error(f"Gemini configuration failed: {e}")
    raise ModelAPIError(f"Gemini setup error: {e}")

def generate_code_from_gemini(code_prompt: str) -> str:
    """
    Sends the code_prompt to Gemini and returns the raw code snippet (no explanation).
    """
    try:
        resp = model.generate_content(contents=[{"parts": [{"text": code_prompt}]}])
        code_text = resp.text.strip()
        # Strip triple backticks if present:
        if code_text.startswith("```"):
            lines = code_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            code_text = "\n".join(lines).strip()
        return code_text
    except Exception as e:
        logger.error(f"Gemini code generation failed: {e}")
        raise ModelAPIError(f"Gemini code generation error: {e}")

def generate_response(prompt: str) -> str:
    """
    (Your existing function for free-form Gemini replies, if you use the optional interpretation step.)
    """
    try:
        resp = model.generate_content(contents=[{"parts": [{"text": prompt}]}])
        return resp.text.strip()
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise ModelAPIError(f"Gemini API call failed: {e}")
