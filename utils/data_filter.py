# utils/data_filter.py

import pandas as pd
from datetime import datetime

def filter_dataframe_by_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filters the DataFrame based on the provided date range.

    Parameters:
    - df: The full CSV-loaded DataFrame
    - start_date: 'YYYY-MM-DD' (from LLM)
    - end_date: 'YYYY-MM-DD' (from LLM)

    Returns:
    - Filtered DataFrame containing only rows in the given date range
    """
    try:
        # Convert date column
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        # Drop rows with invalid dates
        df = df.dropna(subset=["Date"])

        # Convert start/end dates to datetime objects
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Filter rows between start and end date
        filtered_df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]

        return filtered_df

    except Exception as e:
        raise ValueError(f"❌ Error filtering DataFrame: {e}")
