# File: src/Test_red/app_backend/api_client.py

import os
import requests
import json
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD ENV VARS AND STRIP WHITESPACE
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")


# ────────────────────────────────────────────────────────────────────────────────
# 2) VALIDATE PRESENCE OF BOTH KEYS
# ────────────────────────────────────────────────────────────────────────────────
if not TOGETHER_API_KEY:
    st.error("❌ Missing or empty TOGETHER_API_KEY in environment variables")
    st.stop()

if not GEMINI_API_KEY:
    st.error("❌ Missing or empty GEMINI_API_KEY in environment variables")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# 3) CONFIGURE GEMINI (do this once)
# ────────────────────────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

# ────────────────────────────────────────────────────────────────────────────────
# 4) DEFINE TOGETHER ENDPOINT & MODEL
# ────────────────────────────────────────────────────────────────────────────────
MODEL_NAME       = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"


def call_together_ai(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """
    Sends a POST to Together’s chat/completions endpoint with the given prompt.
    Returns the assistant’s reply, or an error message if something goes wrong.
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       MODEL_NAME,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # The “choices” array always exists on a successful call
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as http_err:
        # If Together returns a JSON error, show it in Streamlit
        try:
            error_json = response.json()
            st.error(f"Together AI API error {response.status_code}: {error_json}")
        except Exception:
            st.error(f"Together AI HTTP error: {http_err}")
        return "Error generating response"
    except Exception as e:
        st.error(f"Together AI unexpected error: {e}")
        return "Error generating response"


def call_gemini(prompt: str) -> str:
    """
    Sends the given prompt to Gemini (gemini-2.0-flash-exp) for polishing.
    Returns Gemini’s reply text, or an error message if it fails.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return "Error generating polished response"
