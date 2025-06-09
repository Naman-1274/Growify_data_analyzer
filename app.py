import os
import sys
import streamlit as st
import pandas as pd
import duckdb
from datetime import datetime
from src.Test_red.app_backend.data_utils import clean_numeric_data, detect_column_types
from src.Test_red.app_backend.analysis_utils import generate_data_summary, analyze_marketing_question, polish_with_gemini
from src.Test_red.app_backend.viz_utils import create_dynamic_visualizations
from src.Test_red.app_backend.sql_utils import generate_sql_query


def setup_dynamic_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=':memory:')


def main():
    st.set_page_config(page_title="Dynamic Marketing Data Analyzer", page_icon="📊", layout="wide")
    st.title("🚀 Dynamic Marketing Data Analyzer")
    st.markdown("### AI-Powered Analysis with Together AI + Gemini Integration")
    st.markdown("""
    **Workflow:**
    1. 📤 Upload any marketing dataset (CSV/Excel)
    2. 🔍 AI automatically analyzes columns and data structure
    3. 📊 Get comprehensive data summary and insights
    4. ❓ Ask specific questions about your data
    5. 🧠 Together AI performs deep analysis
    6. ✨ Gemini polishes results into executive reports
    """)

    conn = setup_dynamic_db()
    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Upload your marketing dataset", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            df = clean_numeric_data(df)
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except:
                        pass
            conn.register('marketing_data', df)
            st.success(f"✅ Successfully loaded: {len(df):,} rows × {len(df.columns)} columns")
            
            with st.spinner("🔍 Analyzing data structure..."):
                column_analysis = detect_column_types(df)

            with st.sidebar:
                st.header("📊 Data Analysis")
                with st.expander("📋 Dataset Overview", expanded=True):
                    st.metric("Rows", f"{len(df):,}")
                    st.metric("Columns", len(df.columns))
                    total_nulls = df.isnull().sum().sum()
                    quality_score = max(0, 100 - (total_nulls / (len(df) * len(df.columns)) * 100))
                    st.metric("Data Quality", f"{quality_score:.1f}%")
                with st.expander("🔍 Column Analysis"):
                    purpose_counts = {}
                    for info in column_analysis.values():
                        purpose = info['likely_purpose']
                        purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
                    st.write("**Column Types Detected:**")
                    for purpose, count in purpose_counts.items():
                        emoji = {
                            'temporal': '📅',
                            'financial': '💰',
                            'performance_metric': '📈',
                            'volume_metric': '📊',
                            'categorical': '🏷️',
                            'numeric': '🔢',
                            'unknown': '❓'
                        }.get(purpose, '❓')
                        st.write(f"{emoji} {purpose.replace('_', ' ').title()}: {count}")
                with st.expander("📝 Column Details"):
                    for col, info in column_analysis.items():
                        st.write(f"**{col}**")
                        st.write(f"- Type: {info['likely_purpose']}")
                        st.write(f"- Data Type: {info['dtype']}")
                        st.write(f"- Missing: {info['null_count']} ({info['null_percentage']:.1f}%)")
                        st.write(f"- Unique: {info['unique_count']}")
                        if info.get('mean') is not None:
                            st.write(f"- Mean: {info['mean']:.2f}")
                        st.write("---")

            col1, col2 = st.columns([2, 1])
            with col1:
                with st.expander("👀 Data Preview", expanded=False):
                    st.dataframe(df.head(20), use_container_width=True)
                with st.expander("📋 AI Data Summary", expanded=True):
                    with st.spinner("📝 Generating AI data summary..."):
                        summary = generate_data_summary(df, column_analysis)
                    st.markdown(summary)
                with st.expander("📊 Automatic Visualizations", expanded=False):
                    with st.spinner("📈 Creating visualizations..."):
                        figures = create_dynamic_visualizations(df, column_analysis)
                    if figures:
                        for i, fig in enumerate(figures):
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
                    else:
                        st.info("No suitable visualizations could be generated automatically.")

            with col2:
                st.subheader("💡 Smart Insights")
                numeric_cols = [col for col, info in column_analysis.items() 
                                if info['likely_purpose'] in ['financial', 'performance_metric', 'volume_metric']]
                if numeric_cols:
                    for col in numeric_cols[:4]:
                        try:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                value = df[col].sum() if 'spend' in col.lower() or 'cost' in col.lower() or 'sales' in col.lower() else df[col].mean()
                                st.metric(col, f"{value:,.2f}")
                        except:
                            pass
                st.markdown("**Data Quality:**")
                missing_cols = [col for col, info in column_analysis.items() if info['null_percentage'] > 5]
                if missing_cols:
                    st.warning(f"⚠️ {len(missing_cols)} columns with >5% missing data")
                else:
                    st.success("✅ Good data completeness")

            st.markdown("---")
            st.subheader("🤖 Ask Questions About Your Data")
            suggested_questions = []
            if any(info['likely_purpose'] == 'financial' for info in column_analysis.values()):
                suggested_questions.extend([
                    "What are the spending trends over time?",
                    "Which periods had the highest ROI?",
                    "How do different channels compare in performance?"
                ])
            if any(info['likely_purpose'] == 'performance_metric' for info in column_analysis.values()):
                suggested_questions.extend([
                    "What factors drive performance variations?",
                    "Are there seasonal patterns in the metrics?",
                    "Which metrics are most correlated?"
                ])
            if suggested_questions:
                st.markdown("**💡 Suggested Questions:**")
                cols = st.columns(len(suggested_questions))
                for i, question in enumerate(suggested_questions):
                    if cols[i].button(question, key=f"suggest_{i}"):
                        st.session_state.user_question = question
            user_question = st.text_input(
                "Enter your question:",
                value=getattr(st.session_state, 'user_question', ''),
                placeholder="e.g., 'Why did performance drop in March?' or 'What's driving the ROI variations?'"
            )
            if user_question:
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                with col1:
                    with st.spinner("🧠 Analyzing with Together AI..."):
                        together_analysis = analyze_marketing_question(df, user_question, column_analysis, summary)
                    st.subheader("🔍 Detailed Analysis (Together AI)")
                    with st.expander("View Technical Analysis", expanded=True):
                        st.markdown(together_analysis)
                    with st.spinner("✨ Polishing with Gemini..."):
                        final_report = polish_with_gemini(user_question, together_analysis, summary)
                    st.subheader("📊 Executive Report (Gemini)")
                    st.markdown(final_report)
                    with st.expander("🔧 SQL Query (Advanced)"):
                        try:
                            # 1) Ask Together to build the SQL query
                            sql_candidate = generate_sql_query(user_question, df, column_analysis)

                            # 2) If Together failed, generate_sql_query returns a string beginning with "Error"
                            #    (or otherwise is clearly not valid SQL). We check for that:
                            if not sql_candidate or sql_candidate.strip().lower().startswith("error"):
                                # Show a friendly error message instead of raw SQL
                                st.error("❌ Could not generate a valid SQL query from Together AI.")
                            else:
                                # It looks like a real SQL statement—display it and offer to execute it
                                st.code(sql_candidate, language='sql')

                                if st.button("Execute SQL"):
                                    try:
                                        # Run the SQL against DuckDB
                                        result_df = conn.execute(sql_candidate).df()
                                        st.dataframe(result_df, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"SQL execution error: {e}")

                        except Exception as e:
                            # Any unexpected exception in generate_sql_query()
                            st.warning(f"Could not generate SQL: {e}")

                with col2:
                    st.info("💡 **Analysis Pipeline:**\n\n1. ✅ Data Structure Detection\n2. ✅ Together AI Analysis\n3. ✅ Gemini Report Polish\n4. 📊 Visual Insights")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("📤 Upload your marketing dataset to begin AI-powered analysis")
        st.markdown("### 📋 Supported Data Formats")
        st.markdown("""
        The AI automatically detects and analyzes:
        - **📅 Date/Time columns** - for trend analysis
        - **💰 Financial metrics** - spend, revenue, cost, sales
        - **📈 Performance metrics** - ROI, ROAS, CTR, conversion rates
        - **📊 Volume metrics** - clicks, impressions, leads
        - **🏷️ Categorical data** - campaigns, channels, sources
        
        Just upload your CSV or Excel file - no formatting required!
        """)

if __name__ == "__main__":
    main()