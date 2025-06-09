import json
import pandas as pd
from .api_clients import call_together_ai, call_gemini


def generate_data_summary(df: pd.DataFrame, column_analysis: dict) -> str:
    """Generate comprehensive data summary using Together AI"""
    date_cols = df.select_dtypes(include=['datetime'])
    min_date = date_cols.min().min() if not date_cols.empty else 'No date columns detected'
    max_date = date_cols.max().max() if not date_cols.empty else ''

    data_desc = f"""
    Dataset Overview:
    - Total rows: {len(df):,}
    - Total columns: {len(df.columns)}
    - Date range: {min_date} to {max_date}
    
    Column Analysis:
    """
    for col, info in column_analysis.items():
        data_desc += f"""
    - {col} ({info['likely_purpose']}): {info['dtype']}, {info['null_count']} nulls ({info['null_percentage']:.1f}%), {info['unique_count']} unique values
      Sample values: {info['sample_values']}"""
        if info.get('mean') is not None:
            data_desc += f", Stats: min={info['min']:.2f}, max={info['max']:.2f}, mean={info['mean']:.2f}"  

    prompt = f"""
    As a marketing data analyst, provide a comprehensive summary of this dataset:
    {data_desc}
    Sample data (first 3 rows):
    {df.head(3).to_string()}
    Please provide:
    1. Overall assessment of data quality and completeness
    2. Key marketing metrics identified
    3. Potential analysis opportunities
    4. Any data quality concerns or recommendations
    Keep the summary concise but informative (3-4 paragraphs).
    """
    return call_together_ai(prompt, max_tokens=600)


def analyze_marketing_question(
    df: pd.DataFrame,
    question: str,
    column_analysis: dict,
    summary: str
) -> str:
    """Analyze specific marketing question using Together AI"""
    question_lower = question.lower()
    relevant_cols = []
    for col, info in column_analysis.items():
        col_lower = col.lower()
        if any(word in col_lower for word in question_lower.split()) or info['likely_purpose'] in ['financial', 'performance_metric']:
            relevant_cols.append(col)

    if relevant_cols:
        relevant_data = df[relevant_cols].head(10)
    else:
        relevant_data = df.head(10)

    prompt = f"""
    As a marketing analytics expert, analyze this question: \"{question}\"
    Dataset Summary:
    {summary}
    Relevant Data Sample:
    {relevant_data.to_string()}
    Column Information:
    {json.dumps({col: column_analysis[col] for col in relevant_cols[:10]}, indent=2, default=str)}
    Provide a detailed analysis including:
    1. Data patterns and trends related to the question
    2. Key insights and findings
    3. Potential causes or explanations
    4. Actionable recommendations
    5. Areas for further investigation
    Be specific and use data points where possible.
    """
    return call_together_ai(prompt, max_tokens=800)


def polish_with_gemini(question: str, together_analysis: str, summary: str) -> str:
    """Polish the analysis using Gemini for consultant-style presentation"""
    prompt = f"""
    You are a senior marketing consultant preparing a client presentation. 
    Transform this technical analysis into a polished, executive-ready report:
    Client Question: \"{question}\"
    Technical Analysis:
    {together_analysis}
    Data Summary Context:
    {summary}
    Please create a professional, consultant-style report with:
    ## Executive Summary
    Brief overview of key findings (2-3 sentences)
    ## Key Insights
    Main discoveries from the data analysis
    ## Root Cause Analysis
    Explanations for observed patterns
    ## Strategic Recommendations
    Actionable next steps and optimizations
    ## Risk Assessment
    Potential challenges or considerations
    Use professional language, include specific data points, and make it suitable for stakeholder presentation.
    """
    return call_gemini(prompt)