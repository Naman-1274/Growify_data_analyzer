# utils/aggregator.py

import pandas as pd

def summarize_dataframe(df: pd.DataFrame, start_date: str, end_date: str, aggregation_level: str):
    """
    Filters and summarizes the DataFrame based on the given date range and aggregation level.
    Returns a CSV-formatted string and the DataFrame.
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Assign grouping key
    if aggregation_level == "month":
        df["Time Period"] = df["Date"].dt.to_period("M").astype(str)
    elif aggregation_level == "day":
        df["Time Period"] = df["Date"].dt.strftime("%Y-%m-%d")
    else:
        df["Time Period"] = "All Data"

    # Define custom aggregation logic
    agg_map = {}

    for col in df.columns:
        if col in ["Date", "Time Period"]:
            continue
        if "CTR" in col or "CPM" in col or "CPC" in col or "ROAS" in col or "ROI" in col:
            agg_map[col] = "mean"
        elif any(kw in col.lower() for kw in ["cost", "spent", "revenue", "sales", "clicks", "impressions"]):
            agg_map[col] = "sum"
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = "mean"

    summary = df.groupby("Time Period").agg(agg_map).reset_index()
    summary = summary.round(2)
    summary_str = summary.to_csv(index=False)

    return summary_str, summary
