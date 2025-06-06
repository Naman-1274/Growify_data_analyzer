# utils/metrics_calculator.py

import pandas as pd

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calculated metrics like CTR, CPM, CPC, ROAS, and Total ROAS.
    """

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

    # Google metrics
    if "Google Revenue" in df.columns and "Google Cost" in df.columns:
        df["Google ROAS"] = df.apply(lambda row: safe_divide(row["Google Revenue"], row["Google Cost"]), axis=1)

    if "Google Clicks" in df.columns and "Google Impressions" in df.columns:
        df["Google CTR (%)"] = df.apply(lambda row: safe_divide(row["Google Clicks"], row["Google Impressions"]) * 100, axis=1)

    if "Google Cost" in df.columns and "Google Clicks" in df.columns:
        df["Google CPC"] = df.apply(lambda row: safe_divide(row["Google Cost"], row["Google Clicks"]), axis=1)

    if "Google Cost" in df.columns and "Google Impressions" in df.columns:
        df["Google CPM"] = df.apply(lambda row: safe_divide(row["Google Cost"], row["Google Impressions"]) * 1000, axis=1)

    # Meta/Facebook metrics
    if "Meta revenue" in df.columns and "Meta Cost" in df.columns:
        df["Meta ROAS"] = df.apply(lambda row: safe_divide(row["Meta revenue"], row["Meta Cost"]), axis=1)

    if "Meta Clicks" in df.columns and "Meta Impressions" in df.columns:
        df["Meta CTR (%)"] = df.apply(lambda row: safe_divide(row["Meta Clicks"], row["Meta Impressions"]) * 100, axis=1)

    if "Meta Cost" in df.columns and "Meta Clicks" in df.columns:
        df["Meta CPC"] = df.apply(lambda row: safe_divide(row["Meta Cost"], row["Meta Clicks"]), axis=1)

    if "Meta Cost" in df.columns and "Meta Impressions" in df.columns:
        df["Meta CPM"] = df.apply(lambda row: safe_divide(row["Meta Cost"], row["Meta Impressions"]) * 1000, axis=1)

    # Total ROAS
    if "Total Sales" in df.columns and "Total Cost" in df.columns:
        df["Total ROAS"] = df.apply(lambda row: safe_divide(row["Total Sales"], row["Total Ads Cost"]), axis=1)

    # Round all metrics to 2 decimal places
    df = df.round(2)

    return df
