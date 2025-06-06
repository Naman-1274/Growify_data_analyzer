# File: app.py

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD ENV VARS FROM .env (TOGETHER_API_KEY & GEMINI_API_KEY)
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ────────────────────────────────────────────────────────────────────────────────
# 2) ENSURE `src/` IS ON PYTHON PATH FOR IMPORTS
# ────────────────────────────────────────────────────────────────────────────────
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))     # …/Test_red
SRC_FOLDER = os.path.join(THIS_DIR, "src")
if SRC_FOLDER not in sys.path:
    sys.path.insert(0, SRC_FOLDER)

# ────────────────────────────────────────────────────────────────────────────────
# 3) IMPORT OUR BACKEND MODULES
# ────────────────────────────────────────────────────────────────────────────────
from Test_red.app_backend.ingest import read_dataframe_from_upload, IngestError
from Test_red.app_backend.insight_engine import (
    summarize_with_together,
    analyze_data_with_together,
)
from Test_red.app_backend.gemini_client import generate_with_gemini

# ────────────────────────────────────────────────────────────────────────────────
# 4) STREAMLIT PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chat with Your Data ▶ Together + Gemini", layout="wide")
st.title("🗣️ Chat with Your Data ▶ Together (Analysis) + Gemini (Polish)")

st.markdown(
    """
    **Workflow**:
    1. Upload a CSV/Excel file.  
    2. Together API runs an open-source model (Meta Llama 3.3 70B Instruct) to get:
       - Column names & types  
       - Basic stats (mean/median/min/max)  
       - A one-paragraph "trends/anomalies" summary  
       - A multi-paragraph "why/how/predict" explanation  
    3. Free Gemini 2.0-Flash (via your `GEMINI_API_KEY`) polishes the multi-paragraph analysis into a consultant-style narrative.
    """
)

# ────────────────────────────────────────────────────────────────────────────────
# 5) STEP 1: FILE UPLOAD
# ────────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label="1. Upload your CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False
)

if uploaded_file:
    try:
        df, csv_snippet = read_dataframe_from_upload(uploaded_file, snippet_rows=50)
        st.success(f"✅ Loaded `{uploaded_file.name}`: {df.shape[0]} rows × {df.shape[1]} columns")
    except IngestError as e:
        st.error(str(e))
        st.stop()

    # ────────────────────────────────────────────────────────────────────────────
    # SIDEBAR: DATA SUMMARY
    # ────────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Data Overview")
        
        with st.expander("📋 Dataset Summary", expanded=True):
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
            
            # Column types summary
            st.write("**Column Types:**")
            for dtype, count in df.dtypes.value_counts().items():
                st.write(f"- {dtype}: {count} columns")
        
        with st.expander("🔢 Column Details"):
            st.write("**Column Names and Data Types:**")
            for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
                # Infer more readable data type
                if dtype == 'object':
                    readable_type = "String"
                elif dtype in ['int64', 'int32']:
                    readable_type = "Integer"
                elif dtype in ['float64', 'float32']:
                    readable_type = "Float"
                elif dtype == 'datetime64[ns]':
                    readable_type = "Date"
                else:
                    readable_type = str(dtype)
                
                st.write(f"{i}. **{col}** - {readable_type}")
        
        with st.expander("📈 Numeric Statistics"):
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 0:
                for i, col in enumerate(numeric_cols, 1):
                    st.write(f"**{i}. {col}**")
                    stats = df[col].describe()
                    st.write(f"   * Mean: {stats['mean']:,.2f}")
                    st.write(f"   * Median: {stats['50%']:,.2f}")
                    st.write(f"   * Min: {stats['min']:,.0f}")
                    st.write(f"   * Max: {stats['max']:,.0f}")
                    st.write("")
            else:
                st.write("No numeric columns found.")
        
        with st.expander("⚠️ Data Quality"):
            st.write("**Missing Values:**")
            missing = df.isnull().sum()
            has_missing = False
            for col, count in missing.items():
                if count > 0:
                    percentage = (count / len(df)) * 100
                    st.write(f"- **{col}**: {count} ({percentage:.1f}%)")
                    has_missing = True
            
            if not has_missing:
                st.write("✅ No missing values detected")
            
            # Duplicates check
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                st.write(f"**Duplicate Rows**: {duplicates}")
            else:
                st.write("✅ No duplicate rows")

    # Preview data in main area
    with st.expander("👀 Preview Raw Data (First 100 Rows)"):
        st.dataframe(df.head(100), use_container_width=True)

    st.markdown("---")

    # ───────────────────────────────────────────────────────────────────────────
    # 6) STEP 2: USER QUESTION INPUT
    # ───────────────────────────────────────────────────────────────────────────
    user_question = st.text_input(
        label="2. Enter your question about this data",
        placeholder="e.g. 'Why did Q1 2025 revenue drop vs Q4 2024? Predict next quarter.'",
        help="Ask specific questions about trends, patterns, or predictions based on your data."
    )

    if user_question:
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # ───────────────────────────────────────────────────────────────────────
            # STEP 3A: QUICK SUMMARY WITH TOGETHER
            # ───────────────────────────────────────────────────────────────────────
            with st.spinner("🔄 Running quick data summary via Together..."):
                try:
                    summary_text = summarize_with_together(
                        df=df,
                        csv_snippet=csv_snippet,
                        max_output_tokens=512
                    )
                except Exception as e:
                    st.error(f"❌ Together summary failed: {e}")
                    st.stop()

            st.subheader("📝 Data Summary (Together AI)")
            with st.expander("View Summary", expanded=True):
                st.markdown(summary_text)

            # ───────────────────────────────────────────────────────────────────────
            # STEP 3B: DETAILED "WHY/HOW/PREDICT" WITH TOGETHER
            # ───────────────────────────────────────────────────────────────────────
            with st.spinner("🔄 Running detailed analysis via Together..."):
                try:
                    together_analysis = analyze_data_with_together(
                        df=df,
                        user_question=user_question,
                        summary_text=summary_text,
                        csv_snippet=csv_snippet,
                        max_output_tokens=768
                    )
                except Exception as e:
                    st.error(f"❌ Together detailed analysis failed: {e}")
                    st.stop()

            st.subheader("🔍 Detailed Analysis (Together AI)")
            with st.expander("View Analysis", expanded=True):
                st.markdown(together_analysis)

            # ───────────────────────────────────────────────────────────────────────
            # STEP 4: FINAL POLISH WITH GEMINI 2.0-FLASH
            # ───────────────────────────────────────────────────────────────────────
            with st.spinner("🔄 Polishing final narrative with Gemini..."):
                try:
                    final_response = generate_with_gemini(
                        user_question=user_question,
                        together_analysis=together_analysis,
                        summary_text=summary_text,
                        csv_snippet=csv_snippet
                    )
                except Exception as e:
                    st.error(f"❌ Gemini call failed: {e}")
                    st.stop()

            st.subheader("✨ Final Polished Report (Gemini)")
            st.markdown(final_response)
        
        with col2:
            # Additional insights or charts could go here
            st.info("💡 **Tip**: The sidebar contains detailed data statistics and quality information.")

    else:
        st.info("🔍 Enter a question to get started with your data analysis.")
else:
    st.info("🔼 Please upload a CSV or Excel file to proceed.")
    
    # Show example in sidebar when no file is uploaded
    with st.sidebar:
        st.header("📚 Example")
        st.markdown("""
        **Sample Questions:**
        - Why did sales drop in March?
        - What's driving the ROI variations?
        - Predict next quarter's performance
        - Which advertising channel is most effective?
        """)