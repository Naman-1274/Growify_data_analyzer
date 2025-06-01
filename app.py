import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import re
from dotenv import load_dotenv

from src.Test_red.app_backend.ingest import load_csv_file
from src.Test_red.app_backend.insight_engine import generate_code_from_gemini, generate_response
from src.Test_red.app_backend.code_executor import execute_code_snippet, CodeExecutionError
from src.Test_red.utils import df_to_markdown_full, get_column_summaries
from src.Test_red.exception import ModelAPIError, DataIngestionError

# -----------------------------
# 1) Load environment variables
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Gemini API key missing. Add it to your .env file as GEMINI_API_KEY.")
    st.stop()

# -----------------------------
# 2) Streamlit page config
# -----------------------------
st.set_page_config(page_title="📊 Chat with Your Dataset (Enhanced)", layout="wide")
st.title("📊 Chat with Your Dataset (Gemini + Advanced Summary)")

# -----------------------------
# 3) Sidebar file uploader
# -----------------------------
st.sidebar.header("Upload a file (CSV/Excel)")
uploaded = st.sidebar.file_uploader("Choose a dataset", type=["csv", "xlsx", "xls"])

# -----------------------------
# 4) Initialize session_state
# -----------------------------
for key, default in [
    ("messages", []),
    ("memory_log", []),
    ("last_question", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------
# 5) Define the enhanced validation
# ------------------------------------
# A small set of English stopwords. Expand as needed.
STOPWORDS = {
    "the", "is", "in", "at", "of", "for", "to", "a", "an",
    "and", "or", "but", "if", "what", "which", "on", "by",
    "with", "as", "that", "this", "these", "those", "from",
    "be", "been", "are", "was", "were", "how", "when", "where",
    "why", "who", "whom", "can", "could", "should", "would",
    "do", "does", "did", "my", "your", "our", "their", "its"
}

def is_question_valid(text: str, df: pd.DataFrame) -> bool:
    """
    Returns False if the question is too short, too generic, or doesn't mention any column name.
    1) Require at least two non-empty tokens.
    2) Require at least one alphabetic character.
    3) Strip punctuation, lowercase, remove stopwords, and check if any remaining token matches a column name.
    """
    # 1) Tokenize and require ≥ 2 tokens
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if len(tokens) < 2:
        return False

    # 2) Must have at least one letter (reject “1234”)
    if not any(c.isalpha() for c in text):
        return False

    # 3) Normalize tokens: lowercase & strip leading/trailing punctuation
    normalized = []
    for t in tokens:
        w = t.lower().strip(".,!?\"'():;")
        if w:
            normalized.append(w)

    if not normalized:
        return False

    # 4) Remove stopwords
    meaningful = [w for w in normalized if w not in STOPWORDS]
    if len(meaningful) < 1:
        return False

    # 5) Check if any meaningful token matches a column name (substring match)
    lower_columns = [col.lower() for col in df.columns]
    for w in meaningful:
        for col in lower_columns:
            if w in col or col in w:
                return True

    # If no match was found, reject
    return False

# ---------------------------------------
# 5.1) Helper: detect quarter patterns
# ---------------------------------------
def contains_quarter(text: str) -> bool:
    """
    Returns True if text contains a pattern like 'Q1 2025', 'Q3 2024', etc.
    """
    # Regex: a word boundary, Q[1-4], space, 4-digit year
    return bool(re.search(r"\bQ[1-4]\s+\d{4}\b", text, flags=re.IGNORECASE))

# ---------------------------------------
# 6) Main logic: only run if ‘uploaded’ exists
# ---------------------------------------
if uploaded is not None:
    try:
        # 6.1) Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # 6.2) Load DataFrame
        df = load_csv_file(tmp_path)
        st.success("✅ Data loaded into a DataFrame!")

        # 6.3) Show a preview of the DataFrame
        with st.expander("📄 Preview DataFrame"):
            st.dataframe(df.head(15), use_container_width=True)

        # 6.4) Ask the user for a question
        question = st.text_input("Ask a question about your dataset:")

        if question:
            # 6.4a) Early sanity check for illogical questions
            if not is_question_valid(question, df):
                st.warning(
                    "❗ Your question seems too short, too generic, or doesn’t reference any column.  \n"
                    "Please ask something more specific—e.g.:  \n"
                    "• “Total Sales (INR) by Region in Q1 2025”  \n"
                    "• “Average Units Sold by Product Category”  \n"
                    "• “Trend of Ads Spends (INR) over time”"
                )
                st.stop()

            # 5.2) If the user specifically mentions a quarter, show an info note
            if contains_quarter(question):
                st.info(
                    "ℹ️ For any question referencing quarters (e.g. “Q1 2025”), "
                    "we interpret them as follows:\n"
                    "- Q1 or q1 or 1st quarter = January to March  \n"
                    "- Q2 or q2 or 2nd quarter = April to June  \n"
                    "- Q3 or q3 or 3rd quarter = July to September  \n"
                    "- Q4 or q4 or 4th quarter = October to December"
                )

            # 6.4b) Initialize summary & recommendation so they never cause NameError
            summary = None
            recommendation = None

            # 6.5) Build context for Gemini: column summaries & sample snippet
            snippet = df_to_markdown_full(df)
            summaries = get_column_summaries(df)  # keep the existing logic

            history = [(item["input"], item["output"]) for item in st.session_state["memory_log"]]

            # 6.6) Define comparison instructions
            comparison_instr = (
                "First, examine the raw DataFrame `df` and perform **any** preprocessing, cleaning, "
                "filtering, or column transformations needed by the user's question.  \n"
                "Always store the final, cleaned DataFrame in a new variable called `df_processed`.  \n"
                "From that point onward, run every computation, grouping, and plot on `df_processed`, "
                "never on the original `df`.  \n\n"
                "Next, for time‐based questions, compute prev_metric and next_metric by comparing months in `df_processed`.  \n"
                "  For example:\n"
                "    # (no need to convert Period to float—compare via .dt.to_period('M'))\n"
                "    main_metric = df_processed[df_processed['Date'].dt.to_period('M') == pd.Period('2025-03', freq='M')]['Total Sales (INR)'].sum()\n"
                "    prev_metric = df_processed[df_processed['Date'].dt.to_period('M') == pd.Period('2025-02', freq='M')]['Total Sales (INR)'].sum()\n"
                "    next_metric = df_processed[df_processed['Date'].dt.to_period('M') == pd.Period('2025-04', freq='M')]['Total Sales (INR)'].sum()\n"
                "  # If you need a numeric x-axis for plotting, convert period to timestamp:\n"
                "  #    month_ts = pd.Period('2025-03', freq='M').to_timestamp()\n"
                "  #    plt.plot([month_ts], [main_metric])\n\n"
                "For ranking questions, compute runner-up values (e.g. second_metric and second_label) on `df_processed`.  \n"
            )

            # 6.7) Construct the Gemini code prompt
            #    — NOTE: we append a short note about quarters so Gemini knows how to interpret them
            quarter_note = (
                "\n"
                "Note: If the user's question references quarters (e.g., 'Q1 2025', 'Q2 2024'), "
                "interpret them as:\n"
                "- Q1 → January through March\n"
                "- Q2 → April through June\n"
                "- Q3 → July through September\n"
                "- Q4 → October through December\n"
            )

            code_prompt = (
                "You are a Python/Pandas expert. A pandas DataFrame named `df` is loaded in memory, "
                "and the following imports are available:\n"
                "```python\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "import scipy.stats as stats\n"
                "```\n\n"
                "Below are the column summaries (name and type):\n"
                f"{summaries}\n\n"
                "Below is a small data sample (Markdown table):\n"
                f"{snippet}\n\n"
                "When given a user question, generate **only valid Python code** that does the following:\n"
                "1) **Preprocessing**: Examine the raw `df` and perform any filtering, cleaning, type conversions, or column derives needed to answer the question.  \n"
                "   Store the result in a new DataFrame called `df_processed`.  \n"
                "2) **Analysis**: Use `df_processed` for all grouping, aggregation, and plotting steps.  \n"
                "   The key numeric result must be stored in `main_metric`.  \n"
                "   If a plot is required, put that figure into `main_plot_fig`.  \n"
                f"{comparison_instr}"
                f"{quarter_note}\n"  # ← here is our added quarter guidance
                "User's question:\n"
                f"\"\"\"{question}\"\"\"\n\n"
                "# Write Python code below (only code, no commentary):"
            )

            # 6.8) Call Gemini to generate the Pandas code
            with st.spinner("🧠 Generating Pandas code..."):
                try:
                    gemini_code = generate_code_from_gemini(code_prompt)
                except ModelAPIError as e:
                    st.error(f"❌ Error calling Gemini for code generation: {e}")
                    gemini_code = None
                except Exception as e:
                    st.error(f"⚠️ Unexpected error calling Gemini: {e}")
                    gemini_code = None

            if gemini_code:
                # 6.9) Display the generated code in an expander
                with st.expander("### 📝 Generated Python Code"):
                    st.code(gemini_code, language="python")

                # 6.10) Execute the code snippet
                with st.spinner("⚙️ Executing generated code..."):
                    try:
                        exec_locals = execute_code_snippet(gemini_code, df, return_all_vars=True)
                    except CodeExecutionError as ce:
                        st.error(f"❌ Code execution error:\n{ce}")
                        exec_locals = {}

                # 6.11) If exec_locals has content, display metrics & plot
                if exec_locals:
                    # 6.11a) Display computed metrics (excluding any figure)
                    with st.expander("### 📊 Computed Metrics"):
                        metrics_df = pd.DataFrame(
                            [(k, v) for k, v in exec_locals.items() if k != "main_plot_fig"],
                            columns=["variable", "value"]
                        )
                        st.dataframe(metrics_df, use_container_width=True)

                    # 6.11b) If a figure was generated, show it
                    if "main_plot_fig" in exec_locals:
                        st.markdown("### 📈 Generated Plot")
                        with st.expander("Plot"):
                            st.pyplot(exec_locals["main_plot_fig"])

                    # 6.12) Build a bullet list of metrics for prompting Gemini
                    metrics_text = "\n".join(f"• {k}: {v}" for k, v in exec_locals.items())

                    # 6.13) Generate a 2–3 sentence summary via Gemini
                    summary_prompt = (
                        "You are a concise marketing analyst. The user's question was:\n"
                        f"\"\"\"{question}\"\"\"\n\n"
                        "We computed:\n"
                        f"{metrics_text}\n\n"
                        "In 2–3 sentences, provide a brief, actionable summary that highlights the main finding "
                        "and how it compares to the other metrics."
                    )
                    with st.spinner("💬 Generating summary..."):
                        try:
                            summary = generate_response(summary_prompt)
                        except ModelAPIError as e:
                            st.error(f"❌ Error generating summary: {e}")
                            summary = None
                        except Exception as e:
                            st.error(f"⚠️ Unexpected error generating summary: {e}")
                            summary = None

                    if summary:
                        st.markdown("### ✍️ Summary")
                        st.write(summary)
                    else:
                        st.markdown("### ✍️ Summary")
                        st.info("Summary not available.")

                    # 6.14) Generate 2–4 bullet point strategic recommendation
                    rec_prompt = (
                        "You are a senior marketing strategist or a CEO. Based on the user's question, "
                        "the dataset structure, and the computed metrics, provide a clear recommendation for action. "
                        "This might include campaign adjustments, budget shifts, timing strategies, or deeper analysis directions.\n\n"
                        f"User's question:\n\"\"\"{question}\"\"\"\n\n"
                        f"Computed metrics:\n{metrics_text}\n\n"
                        "Deliver your advice in 2–4 concise bullet points with a focus on ROI, customer acquisition, and growth strategy. "
                        "Be direct and actionable like a marketing leader in a business meeting."
                    )
                    with st.spinner("📈 Generating strategic recommendation..."):
                        try:
                            recommendation = generate_response(rec_prompt)
                        except ModelAPIError as e:
                            st.error(f"❌ Error generating recommendation: {e}")
                            recommendation = None
                        except Exception as e:
                            st.error(f"⚠️ Unexpected error during recommendation: {e}")
                            recommendation = None

                    if recommendation:
                        st.markdown("### 📈 Strategic Recommendation")
                        st.write(recommendation)
                    else:
                        st.markdown("### 📈 Strategic Recommendation")
                        st.info("Recommendation not available.")

                else:
                    # Code ran but returned no variables (likely snippet failed)
                    st.error("❌ Code executed but returned no variables. Unable to compute metrics.")

                # 6.15) Guarantee summary_text & recommendation_text exist
                summary_text = summary if summary is not None else "Summary not available."
                recommendation_text = (
                    recommendation if recommendation is not None else "Recommendation not available."
                )

                # 6.16) Record conversation in session_state
                st.session_state["last_question"] = question
                st.session_state["memory_log"].append({"input": question, "output": summary_text})
                st.session_state["messages"].append({
                    "question": question,
                    "summary": summary_text,
                    "recommendation": recommendation_text
                })

        # 7) Display chat history if any
        if st.session_state["messages"]:
            st.markdown("### 💬 Chat History (Summary + Recommendations)")
            for entry in reversed(st.session_state["messages"]):
                st.markdown(f"**🧠 You asked:** {entry['question']}")
                st.markdown(f"**✍️ Summary:** {entry['summary']}")
                st.markdown(f"**📈 Recommendation:** {entry['recommendation']}")
                st.markdown("---")

    except DataIngestionError as die:
        st.error(f"❌ Data ingestion error: {die}")
    except Exception as e:
        st.error(f"⚠️ Failed to process dataset: {e}")
else:
    st.info("👈 Upload a CSV or Excel file to begin.")

# -----------------------------
# 8) Optional: CSS styling for buttons
# -----------------------------
st.markdown(
    """
    <style>
    .stButton>button {
        margin-top: 1rem;
        width: 100%;
        background-color: #444;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
