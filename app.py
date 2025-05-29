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
            st.dataframe(df.head(15), use_container_width=True)

        # User question input
        question = st.text_input("Ask a question about your dataset:")
    
        if question:
            current_q = question if question else st.session_state["last_question"]

            # Build context: snippet, stats, history
            snippet = df_to_markdown_full(df)
            summaries = get_column_summaries(df)
            history = [(item["input"], item["output"]) for item in st.session_state["memory_log"]]

            # Comparison instructions: let Gemini decide how to compute prev/next or runner-up
            comparison_instr = (
                "Also compute a relevant comparison metric:\n"
                "• If this is a time‐based question, compute the same metric for the previous and next period "
                "(store in prev_metric and next_metric).\n"
                "  For example, if they ask “sales in March 2025 vs February 2025,” then:\n"
                "    main_metric = df[df['Date'].dt.to_period('M') == pd.Period('2025-03', freq='M')]['Total Sales (INR)'].sum()\n"
                "    prev_metric = df[df['Date'].dt.to_period('M') == pd.Period('2025-02', freq='M')]['Total Sales (INR)'].sum()\n"
                "    next_metric = df[df['Date'].dt.to_period('M') == pd.Period('2025-04', freq='M')]['Total Sales (INR)'].sum()\n"
                "• If this is a ranking question, compute the runner‐up value (store in second_metric and second_label).\n"
            )

            # Assemble the code-generation prompt
            code_prompt = (
                "You are a Python/Pandas expert. A pandas DataFrame named `df` is loaded in memory, "
                "and 'import pandas as pd' and 'import numpy as np' (and 'import matplotlib.pyplot as plt')  You also have access to 'scipy as stats' and 'seaborn as sns'.\n\n"
                "and can create visualizations if needed or the user requests it.\n\n"
                "Below are the column names, types, and a small data sample from `df`:\n\n"
                "=== Column Summaries ===\n"
                f"{summaries}\n\n"
                "=== Data Sample (Markdown) ===\n"
                f"{snippet}\n\n"
                "When given a user question, you must generate **only valid Python code** that runs against `df` to compute the answer. "
                "if the question is about trends, comparisons, or time-series data, also generate a plot. "
                "Use 'plt.subplots()' and store the plot in 'main_plot_fig'. "
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
                with st.expander("### 📝 Generated Python Code"):
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
                    with st.expander("### 📊 Computed Metrics"):
                        metrics_df = pd.DataFrame(
                            [(k, v) for k, v in exec_locals.items() if k != "main_plot_fig"], 
                                columns=["variable", "value"]
                        )
                        st.dataframe(metrics_df, use_container_width=True)

                if 'main_plot_fig' in exec_locals:
                    st.markdown("### 📈 Generated Plot")
                    with st.expander("Plot"):
                        st.pyplot(exec_locals['main_plot_fig'])

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
                        
                        
                        # === Strategic Recommendation AI ===
                        rec_prompt = (
                            "You are a senior marketing strategist or a CEO. Based on the user's question, "
                            "the dataset structure, and the computed metrics, provide a clear recommendation for action. "
                            "This might include campaign adjustments, budget shifts, timing strategies, or deeper analysis directions.\n\n"
                            f"User's question:\n\"\"\"{current_q}\"\"\"\n\n"
                            f"Computed metrics:\n{metrics_text}\n\n"
                            "Deliver your advice in 2–4 concise bullet points with a focus on ROI, customer acquisition, and growth strategy. "
                            "Be direct and actionable like a marketing leader in a business meeting."
                        )

                        with st.spinner("📈 Generating strategic recommendation..."):
                            try:
                                recommendation = generate_response(rec_prompt)
                                st.markdown("### 📈 Strategic Recommendation")
                                st.write(recommendation)
                            except ModelAPIError as e:
                                st.error(f"❌ Error generating recommendation: {e}")
                            except Exception as e:
                                st.error(f"⚠️ Unexpected error during recommendation: {e}")

                # Record conversation
                if question:
                    st.session_state["last_question"] = question

                # Fallback if summary or recommendation is None
                summary_text = summary or "Summary not available."
                recommendation_text = recommendation or "Recommendation not available."

                st.session_state["memory_log"].append({"input": current_q, "output": summary_text})
                st.session_state["messages"].append({
                    "question": current_q,
                    "summary": summary_text,
                    "recommendation": recommendation_text
                })

        # Display chat history of questions and code
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