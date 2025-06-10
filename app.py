import os
import sys
import streamlit as st
import pandas as pd
import duckdb
from datetime import datetime
# Assuming all your backend scripts are in the specified paths
from src.Test_red.app_backend.data_utils import detect_column_types
from src.Test_red.app_backend.analysis_utils import analyze_marketing_question, polish_with_gemini, _clean_and_prepare_data
from src.Test_red.app_backend.viz_utils import create_dynamic_visualizations
from src.Test_red.app_backend.sql_utils import generate_sql_query

def setup_dynamic_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=':memory:')

def main():
    st.set_page_config(page_title="Dynamic Marketing Data Analyzer", page_icon="📊", layout="wide")
    st.title("🚀 Dynamic Marketing Data Analyzer")
    st.markdown("### AI-Powered Analysis with Together AI + Gemini Integration")
    
    conn = setup_dynamic_db()
    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Upload your marketing dataset", type=['csv', 'xlsx', 'xls'])

    if 'user_question' not in st.session_state:
        st.session_state.user_question = ''

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            df_raw.columns = df_raw.columns.str.strip().str.replace(' ', '_')
            
            # **CORRECTED**: Use the dedicated cleaning function from analysis_utils
            df = _clean_and_prepare_data(df_raw)
            
            conn.register('marketing_data', df)
            st.success(f"✅ Successfully loaded and cleaned: {len(df):,} rows × {len(df.columns)} columns")
            
            with st.spinner("🔍 Analyzing data structure..."):
                # This can be used for UI elements but not for core analysis functions
                column_analysis = detect_column_types(df)
                # Create a simple text summary for the AI context
                summary_text = f"The dataset has {len(df)} rows and columns like {', '.join(df.columns[:5])}."

            with st.sidebar:
                st.header("📊 Data Analysis")
                with st.expander("📋 Dataset Overview", expanded=True):
                    st.metric("Rows", f"{len(df):,}")
                    st.metric("Columns", len(df.columns))
                    total_nulls = df.isnull().sum().sum()
                    if len(df) > 0 and len(df.columns) > 0:
                        quality_score = max(0, 100 - (total_nulls / (df.size) * 100))
                        st.metric("Data Quality", f"{quality_score:.1f}%")
                
                with st.expander("🔍 Column Details"):
                    st.json(column_analysis, expanded=False)

            # Main layout
            col1, col2 = st.columns([2, 1])
            with col1:
                with st.expander("👀 Data Preview", expanded=False):
                    st.dataframe(df.head(20), use_container_width=True)
                
                with st.expander("📊 Automatic Visualizations", expanded=True):
                    with st.spinner("📈 Creating visualizations..."):
                        figures = create_dynamic_visualizations(df, column_analysis)
                    if figures:
                        for i, fig in enumerate(figures):
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
                    else:
                        st.info("No suitable visualizations could be generated automatically.")
            
            with col2:
                st.subheader("💡 Smart Insights")
                numeric_cols = df.select_dtypes(include='number').columns
                if not numeric_cols.empty:
                    for col in numeric_cols[:4]:
                        try:
                            value = df[col].sum() if 'spend' in col.lower() or 'cost' in col.lower() or 'sales' in col.lower() else df[col].mean()
                            st.metric(col, f"{value:,.2f}")
                        except Exception:
                            pass # Failsafe for display
                st.markdown("**Data Quality:**")
                missing_cols = [col for col, info in column_analysis.items() if info['null_percentage'] > 5]
                if missing_cols:
                    st.warning(f"⚠️ {len(missing_cols)} columns with >5% missing data: {', '.join(missing_cols)}")
                else:
                    st.success("✅ Good data completeness")

            st.markdown("---")
            st.subheader("🤖 Ask Questions About Your Data")
            
            user_question = st.text_input(
                "Enter your question:",
                value=st.session_state.user_question,
                placeholder="e.g., 'Why did performance drop in March?'"
            )

            if st.button("Analyze Question"):
                if user_question:
                    st.session_state.user_question = user_question
                    st.markdown("---")
                    
                    with st.spinner("🧠 Analyzing with Together AI..."):
                        # **CORRECTED**: Call analyze_marketing_question with the correct arguments
                        together_analysis = analyze_marketing_question(df, user_question, summary_text)
                    
                    st.subheader("🔍 Detailed Analysis (Together AI)")
                    st.markdown(together_analysis)

                    with st.spinner("✨ Polishing with Gemini..."):
                        # **CORRECTED**: Call polish_with_gemini with the correct arguments
                        final_report = polish_with_gemini(user_question, together_analysis, summary_text)
                    
                    st.subheader("📊 Executive Report (Gemini)")
                    st.markdown(final_report)
                    
                    with st.expander("🔧 SQL Query (Advanced)"):
                        try:
                            sql_candidate = generate_sql_query(user_question, df, column_analysis)
                            st.code(sql_candidate, language='sql')
                            if st.button("Execute SQL"):
                                result_df = conn.execute(sql_candidate).df()
                                st.dataframe(result_df, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate SQL: {e}")
                else:
                    st.warning("Please enter a question to analyze.")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.error("Please ensure the file is a valid CSV or Excel file and try again.")
    else:
        st.info("📤 Upload your marketing dataset to begin AI-powered analysis")

if __name__ == "__main__":
    main()
