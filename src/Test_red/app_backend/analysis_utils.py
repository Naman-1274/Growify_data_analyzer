# analysis_utils.py

import json
import pandas as pd
import numpy as np
from .api_clients import call_together_ai, call_gemini # Assuming these are defined elsewhere and accessible

def _clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically clean and prepare a DataFrame for analysis.
    - Converts object columns with comma-separated numbers to numeric.
    - Converts object columns with date-like strings to datetime.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Attempt to convert to numeric after removing commas
            try:
                series_no_commas = df_clean[col].str.replace(',', '', regex=False)
                numeric_series = pd.to_numeric(series_no_commas, errors='coerce')
                
                if numeric_series.notna().mean() > 0.8: 
                    df_clean[col] = numeric_series
                    continue 
            except AttributeError:
                pass

            try:
                datetime_series = pd.to_datetime(df_clean[col], errors='coerce')
                if datetime_series.notna().mean() > 0.8: 
                    df_clean[col] = datetime_series
            except (ValueError, TypeError):
                pass
                
    return df_clean

def summarize_full_dataframe(df: pd.DataFrame, max_categories: int = 5) -> dict:
    """
    Compute essential summary statistics for all columns in a cleaned DataFrame.
    This is a significantly lighter version for prompt efficiency.
    """
    summary = {}
    n = len(df)
    for col in df.columns:
        col_data = df[col]
        null_count = int(col_data.isnull().sum())
        null_pct = (null_count / n * 100) if n > 0 else 0.0
        col_summary = {
            "dtype": str(col_data.dtype),
            "null_pct": round(null_pct, 2),
        }
        
        # Numeric columns (only essential stats)
        if pd.api.types.is_numeric_dtype(col_data):
            ser = col_data.dropna()
            if not ser.empty:
                desc = ser.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
                col_summary.update({
                    "count": int(desc.get("count", 0)),
                    "min": float(desc.get("min", 0)),
                    "25%": float(desc.get("25%", 0)),
                    "50%": float(desc.get("50%", 0)),
                    "75%": float(desc.get("75%", 0)),
                    "max": float(desc.get("max", 0)),
                    "mean": float(desc.get("mean", 0)),
                    "std": float(desc.get("std", 0)),
                })
            else:
                col_summary["note"] = "no non-null numeric values"

        # Datetime columns (only essential stats)
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            ser = col_data.dropna()
            if not ser.empty:
                years = ser.dt.year.value_counts().sort_index().to_dict()
                col_summary.update({
                    "min_date": str(ser.min()),
                    "max_date": str(ser.max()),
                    "year_counts_top": years, # Keep all years to avoid issues with top_n for small sets
                })
            else:
                col_summary["note"] = "no valid datetime values"
                
        # Categorical/text columns (only essential stats)
        else:
            ser = col_data.astype("string").dropna()
            unique_count = ser.nunique()
            col_summary["unique_count"] = int(unique_count)
            top = ser.value_counts().head(max_categories).to_dict()
            col_summary["top_categories"] = {str(k): int(v) for k, v in top.items()}
            if unique_count > max_categories:
                col_summary["note"] = f"{unique_count - max_categories} more categories"
                
        summary[col] = col_summary
    return summary

def build_structured_analysis_prompt_full(
    question: str,
    full_summary: dict,
    summary_text: str,
) -> str:
    """
    Builds a very light prompt using essential summary statistics.
    Designed for maximum prompt efficiency while maintaining a structured analysis.
    """
    # Use compressed JSON
    columns_section = json.dumps(full_summary, separators=(',', ':'))
    summary_one_line = summary_text.strip().replace("\n", " ")
    
    prompt = f"""
You are a marketing analytics expert. The user question is: "{question}"

You must work exclusively with the data described in the `full_column_summary` below. Do not invent columns or assume external data. Refer to columns by their EXACT names.

High-level dataset summary (for quick context): {summary_one_line}

Full-column summary with key statistics (use this to understand data types, ranges, and basic distributions. Perform all calculations and reasoning based on these statistics):
{columns_section}

Please analyze the question step by step following these phases:
1. UNDERSTAND & DECOMPOSE:
    - Restate intent, identify key metrics, filters, timeframes, segments.
    - Break question into sub-tasks.
2. MAP TO DATA:
    - Specify which column(s) apply by name.
    - Use EXACT COLUMN NAMES as they appear in the `full_column_summary`.
    - Assume the pandas DataFrame `df` is already loaded and cleaned in the execution environment.
3. CLEAN & PREPARE:
    - Describe parsing dates, normalizing numerics, handling missing/bad values.
    - Provide concise pandas code snippets for cleaning steps.
4. FILTER & AGGREGATE:
    - Explain filtering logic (e.g., year, quarter, segment).
    - Provide pandas code to compute aggregates (sum, mean, groupby, correlation).
    - Explicitly state which data points are included/excluded and why.
5. VALIDATE & SANITY-CHECK:
    - List basic checks for data integrity.
    - Provide pandas/logic code for these checks.
6. INTERPRET FINDINGS:
    - State the main insight in one sentence.
    - Provide 2–3 bullets with exact figures or percentages.
7. CONTEXTUALIZE & EXPLAIN “WHY”:
    - Provide domain reasoning (seasonality, campaign cycles) and state assumptions.
8. RECOMMEND NEXT STEPS:
    - Actionable advice grounded in data, suggest further analyses or tests.
9. FLAG DATA-QUALITY:
    - Note any data limitations or anomalies from the `full_summary` (e.g., high null_pct).
10. FORMAT FOR READABILITY:
    - Use headings “Step 1: …” through “Step 10: …”.
    - Include code blocks for Python/pandas snippets.
    - Conclude with an emoji-summary block:
      🎯 KEY INSIGHT, 📊 DATA, 💡 WHY, ⚡ ACTION, ⚠️ QUALITY.

Return the full detailed reasoning, including code snippets, ending with the emoji-summary block.
"""
    return prompt

def analyze_marketing_question(
    df: pd.DataFrame,
    question: str,
    summary_text: str
) -> str:
    """
    Main analysis function: Cleans the data, computes a full summary, 
    builds a prompt, and calls the analysis AI.
    """
    cleaned_df = _clean_and_prepare_data(df)
    full_summary = summarize_full_dataframe(cleaned_df)
    prompt = build_structured_analysis_prompt_full(question, full_summary, summary_text)
    return call_together_ai(prompt, max_tokens=1500) # Uncommented this line
    # return prompt # Commented out this line

def polish_with_gemini(question: str, together_analysis: str, summary: str) -> str:
    """
    Takes the technical analysis and polishes it into an executive report using Gemini.
    """
    prompt = f"""
You are a senior marketing consultant preparing a polished, executive-ready report. The user question: "{question}"

Technical Analysis (detailed step-by-step with code) from Together AI:
{together_analysis}

Dataset summary context (for reference; do not repeat verbatim):
{summary.strip().replace(chr(10), ' ')}

Please transform into a clean report with sections labeled as below, using emojis:

## 🎯 EXECUTIVE SUMMARY
- 2–3 sentences capturing the core finding.

## 📊 KEY INSIGHTS
- Bullet points with main discoveries, each including exact figures or percentages.
- Optionally include a simple markdown table.

## 💡 ROOT CAUSE ANALYSIS
- Explain why observed patterns likely occurred (domain reasoning, seasonality, etc.).
- Highlight assumptions.

## ⚡ STRATEGIC RECOMMENDATIONS
- Actionable steps grounded in data.
- Suggest further analyses or monitoring (no code here).

## ⚠️ RISK & DATA QUALITY CONCERNS
- Note any data limitations or anomalies encountered.
- Warn about potential pitfalls if proceeding without validation.

## 📈 NEXT STEPS
- Outline follow-up analyses, dashboards, or experiments.
- Suggest how to track progress over time.

Ensure a professional, concise tone. Return only the markdown content of the report.
"""
    return call_gemini(prompt) # Uncommented this line
    # return prompt # Commented out this line
