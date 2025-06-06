# utils/data_cleaner.py

import pandas as pd

def clean_and_standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the uploaded dataset to make it compatible.
    """


    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
    else:
        raise ValueError("❌ The dataset must contain a 'Date' column.")

    df.dropna(how="all", inplace=True)

    return df
