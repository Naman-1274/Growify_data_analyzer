import streamlit as st
import pandas as pd
import tempfile
import os
from dotenv import load_dotenv

from src.Test_red.app_backend.ingest import load_csv_file
from src.Test_red.app_backend.insight_engine import generate_code_from_gemini, generate_response
from src.Test_red.app_backend.code_executor import execute_code_snippet, CodeExecutionError
from src.Test_red.utils import df_to_markdown_full, get_column_summaries
from src.Test_red.exception import ModelAPIError, DataIngestionError

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Gemini API key missing. Add it to your .env file as GEMINI_API_KEY.")
    st.stop()

# Page config
st.set_page_config(page_title="📊 Chat with Your Dataset (Enhanced)", layout="wide")
st.title("📊 Chat with Your Dataset (Gemini + Advanced Summary)")

# Sidebar: file upload
st.sidebar.header("Upload a file (CSV/Excel)")
uploaded = st.sidebar.file_uploader("Choose a dataset", type=["csv", "xlsx", "xls"])

# Initialize session state
for key, default in [
    ("messages", []),
    ("memory_log", []),
    ("last_question", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

if uploaded is not None:
    try:
        # Save upload to temp and load into DataFrame
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        df = load_csv_file(tmp_path)
        st.success("✅ Data loaded into a DataFrame!")

        # Preview dataset
        with st.expander("📄 Preview DataFrame"):
            st.dataframe(df, use_container_width=True)

        # User question input
        question = st.text_input("Ask a question about your dataset:")
        regen = st.button("🔄 Regenerate Last Response")

        if regen and st.session_state["last_question"] is None:
            st.warning("⚠️ No previous question to regenerate.")
            regen = False

        if question or regen:
            current_q = question if question else st.session_state["last_question"]

            # Build context: snippet, stats, history
            snippet = df_to_markdown_full(df)
            summaries = get_column_summaries(df)
            history = [(item["input"], item["output"]) for item in st.session_state["memory_log"]]

            # Comparison instructions: let Gemini decide how to compute prev/next or runner-up
            comparison_instr = (
                "Also compute a relevant comparison metric: "
                "if this is a time-based question, compute the same metric for the previous and next period "
                "(store in prev_metric and next_metric). "
                "If this is a ranking question, compute the runner-up value (store in second_metric and second_label)."
            )

            # Assemble the code-generation prompt
            code_prompt = (
                "You are a Python/Pandas expert. A pandas DataFrame named `df` is loaded in memory, "
                "and `import pandas as pd` (and `import matplotlib.pyplot as plt`) have been run.\n\n"
                "Below are the column names, types, and a small data sample from `df`:\n\n"
                "=== Column Summaries ===\n"
                f"{summaries}\n\n"
                "=== Data Sample (Markdown) ===\n"
                f"{snippet}\n\n"
                "When given a user question, you must generate **only valid Python code** that runs against `df` to compute the answer. "
                "Ensure the final result of the main query is stored in a variable named `main_metric`.\n"
                f"{comparison_instr}\n\n"
                "User's question:\n"
                f"\"\"\"{current_q}\"\"\"\n\n"
                "# Write Python code below (only code, no commentary):"
            )

            # Call Gemini to get Pandas code
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
                # Display generated code
                st.markdown("### 📝 Generated Python Code")
                st.code(gemini_code, language="python")

                # Execute the code and capture all variables
                with st.spinner("⚙️ Executing generated code..."):
                    try:
                        # return_all_vars=True returns a dict of all variables Gemini assigned
                        exec_locals = execute_code_snippet(gemini_code, df, return_all_vars=True)
                    except CodeExecutionError as ce:
                        st.error(f"❌ Code execution error:\n{ce}")
                        exec_locals = {}

                # Display computed metrics as a DataFrame
                if exec_locals:
                    st.markdown("### 📊 Computed Metrics")
                    metrics_df = pd.DataFrame(
                        list(exec_locals.items()), columns=["variable", "value"]
                    )
                    st.dataframe(metrics_df, use_container_width=True)

                    # Prepare metrics text for summary prompt
                    metrics_text = "\n".join(f"• {k}: {v}" for k, v in exec_locals.items())
                    summary_prompt = (
                        "You are a concise marketing analyst. The user's question was:\n"
                        f"\"\"\"{current_q}\"\"\"\n\n"
                        "We computed:\n"
                        f"{metrics_text}\n\n"
                        "In 2–3 sentences, provide a brief, actionable summary that highlights the main finding and how it compares to the other metrics."
                    )

                    # Get summary from Gemini
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

                # Record conversation
                if question:
                    st.session_state["last_question"] = question
                st.session_state["memory_log"].append({"input": current_q, "output": gemini_code})
                st.session_state["messages"].append((current_q, gemini_code))

        # Display chat history of questions and code
        if st.session_state["messages"]:
            st.markdown("### 💬 Chat History (Question → Generated Code)")
            for q, a_code in reversed(st.session_state["messages"]):
                st.markdown(f"**You:** {q}")
                st.markdown("```python\n" + a_code + "\n```")

    except DataIngestionError as die:
        st.error(f"❌ Data ingestion error: {die}")
    except Exception as e:
        st.error(f"⚠️ Failed to process dataset: {e}")
else:
    st.info("👈 Upload a CSV or Excel file to begin.")

# Styling for buttons
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
