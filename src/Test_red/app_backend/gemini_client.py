# File: src/Test_red/app_backend/gemini_client.py

import os
import google.generativeai as genai

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD YOUR FREE GEMINI API KEY FROM ENV
# ────────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise EnvironmentError("⚠️ Please set GEMINI_API_KEY in your environment or .env.")

# ────────────────────────────────────────────────────────────────────────────────
# 2) CONFIGURE genai WITH api_key (Free Gemini endpoint)
# ────────────────────────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

# ────────────────────────────────────────────────────────────────────────────────
# 3) FUNCTION: CALL GEMINI FOR FINAL "WHY/HOW/PREDICT" NARRATIVE
# ────────────────────────────────────────────────────────────────────────────────
def generate_with_gemini(
    user_question: str,
    together_analysis: str,
    summary_text: str,
    csv_snippet: str
) -> str:
    """
    Builds a prompt containing:
      1. user_question
      2. together_analysis (the multi-paragraph text from Together)
      3. summary_text (the one-paragraph summary from Together)
      4. csv_snippet (header + first rows)

    Sends it to Gemini 2.0-Flash (free-tier) and returns the polished narrative.
    """
    prompt = (
        "You are a top-tier data consultant. A user asked:\n"
        f"{user_question}\n\n"
        "Below is a concise summary of the dataset:\n"
        "```\n"
        f"{summary_text}\n"
        "```\n\n"
        "Below is the detailed \"why/how/predict\" analysis from an open-source model:\n"
        "```\n"
        f"{together_analysis}\n"
        "```\n\n"
        "Below is the header + first few rows of the CSV for additional context:\n"
        "```\n"
        f"{csv_snippet}\n"
        "```\n\n"
        "Please rewrite the above analysis in a polished, professional consultant tone. "
        "Ensure clarity and coherence, break it into well-structured paragraphs, "
        "and keep it concise (no more than 600 words)."
    )

    try:
        # Create the Gemini model instance
        model = genai.GenerativeModel("gemini-2.0-flash-exp")  # Use the correct model name
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=800,  # Adjust for ~600 word limit
            temperature=0.7,
            top_p=0.8,
            top_k=40
        )
        
        # Generate the response
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Return the generated text
        return response.text.strip()
        
    except Exception as e:
        # Provide a more specific error message
        raise Exception(f"❌ Gemini call failed: {str(e)}")