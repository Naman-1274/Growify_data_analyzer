# app.py

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# Utils
from utils.query_parser import parse_timeframe_with_llm
from utils.data_filter import filter_dataframe_by_dates
from utils.aggregator import summarize_dataframe
from utils.prompt_builder import build_insight_prompt
from utils.insight_generator import generate_insight
from utils.data_cleaner import clean_and_standardize_data
from utils.metrics_calculator import add_derived_metrics

# Load env vars
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file.")
    st.stop()

from google.generativeai import configure
configure(api_key=api_key)

# Page setup
st.set_page_config(page_title="📊 Marketing Analyst Chatbot", layout="wide")
st.title("📊 Ask Anything About Your Marketing Data")

# File upload
uploaded_file = st.file_uploader("Upload your marketing dataset (.csv)", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.subheader("📄 Raw Dataset Preview")
        st.dataframe(df.head())
        st.session_state["raw_df"] = df
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

# User query
st.divider()
user_query = st.text_input("Ask a question (e.g., 'Compare ROAS in Q1 vs Q2 2024')")

# Aggregated processing
if st.button("📊 Preview Aggregated Data") and uploaded_file and user_query:
    try:
        with st.spinner("🔄 Cleaning, enriching, and summarizing your data..."):
            cleaned_df = clean_and_standardize_data(st.session_state["raw_df"])
            enriched_df = add_derived_metrics(cleaned_df)
            start_date, end_date, aggregation = parse_timeframe_with_llm(user_query)
            st.session_state["parsed_dates"] = (start_date, end_date, aggregation)
            filtered_df = filter_dataframe_by_dates(enriched_df, start_date, end_date)
            summary_str, preview_df = summarize_dataframe(filtered_df, start_date, end_date, aggregation)

            st.session_state["summary_csv"] = summary_str
            st.session_state["preview_df"] = preview_df
            st.session_state["preview_ready"] = True

        st.subheader("📆 Gemini Parsed Timeframe")
        st.markdown(f"""
        - **Start Date:** `{start_date}`  
        - **End Date:** `{end_date}`  
        - **Aggregation:** `{aggregation}`
        """)

        st.success("✅ Aggregated Dataset Preview")
        st.dataframe(preview_df)

    except Exception as e:
        st.error(f"❌ Failed to process aggregation: {e}")
        st.session_state["preview_ready"] = False

# Generate Insights
if st.session_state.get("preview_ready", False):
    if st.button("🧠 Generate Gemini Insights"):
        try:
            with st.spinner("🧠 Generating insights from Gemini..."):
                prompt = build_insight_prompt(
                    user_query,
                    st.session_state["summary_csv"],
                    brand_name="Drzya"
                )
                insight = generate_insight(prompt)

            st.subheader("📊 Aggregated Data Sent to Gemini")
            st.dataframe(st.session_state["preview_df"])

            st.subheader("🧠 Gemini Insight")
            st.markdown(insight)

        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
