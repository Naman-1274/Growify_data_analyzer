
# app.py
import streamlit as st
import pandas as pd
import tempfile
import os
from dotenv import load_dotenv
from src.Test_red.app_backend.ingest import load_csv_file
from src.Test_red.app_backend.insight_engine import generate_response_df
from src.Test_red.utils import df_to_markdown_full, get_column_summaries, build_conversational_prompt
from src.Test_red.exception import ModelAPIError, DataIngestionError

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Add it to your .env file as GEMINI_API_KEY.")
    st.stop()

# Page config
st.set_page_config(page_title="📊 Chat with Your Dataset", layout="wide")
st.title("📊 Chat with Your Dataset (Gemini Agent)")

# Sidebar: file upload
st.sidebar.header("Upload a file (CSV/Excel)")
uploaded = st.sidebar.file_uploader("Choose a dataset", type=["csv", "xlsx", "xls"] )

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
        # Save upload to temp and load into DuckDB
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        df = load_csv_file(tmp_path)
        st.success("✅ Data loaded and registered in DuckDB!")

        # Preview dataset
        with st.expander("📄 Preview DataFrame"):
            st.dataframe(df, use_container_width=True)

        # User question input
        question = st.text_input("Ask a question about your dataset:")
        regen = st.button("🔄 Regenerate Last Response")

        if regen and st.session_state.last_question is None:
            st.warning("⚠️ No previous question to regenerate.")
            regen = False

        if question or regen:
            current_q = question if question else st.session_state.last_question

            # Build context: snippet, stats, history
            snippet = df_to_markdown_full(df)
            summaries = get_column_summaries(df)
            history = [(item['input'], item['output']) for item in st.session_state.memory_log]

            # Assemble prompt via utils
            full_prompt = build_conversational_prompt(
                df_snippet=snippet,
                column_summaries=summaries,
                history=history,
                user_question=current_q
            )

            # Call Gemini-based agent
            with st.spinner("Analyzing..."):
                try:
                    answer = generate_response_df(df, full_prompt)
                except (ModelAPIError, DataIngestionError) as e:
                    st.error(f"❌ Error: {e}")
                    answer = None
                except Exception as e:
                    st.error(f"⚠️ Unexpected error: {e}")
                    answer = None

            if answer:
                # Record conversation
                if question:
                    st.session_state.last_question = question
                st.session_state.memory_log.append({"input": current_q, "output": answer})
                st.session_state.messages.append((current_q, answer))

        # Display chat history
        if st.session_state.messages:
            st.markdown("### 💬 Chat History")
            for q, a in reversed(st.session_state.messages):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Gemini:** {a}")

    except DataIngestionError as die:
        st.error(f"❌ Data ingestion error: {die}")
    except Exception as e:
        st.error(f"⚠️ Failed to process dataset: {e}")
else:
    st.info("👈 Upload a CSV or Excel file to begin.")

# Styling for buttons
st.markdown("""
<style>
.stButton>button {
    margin-top: 1rem;
    width: 100%;
    background-color: #444;
    color: white;
}
</style>
""", unsafe_allow_html=True)

