import pandas as pd
import re


def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert numeric columns, handling commas and other formatting"""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                sample_values = df_clean[col].dropna().astype(str).head(10)
                if any(bool(re.search(r'[\d,.$%]', str(val))) for val in sample_values):
                    cleaned_series = (
                        df_clean[col].astype(str)
                            .str.replace(',', '', regex=False)
                            .str.replace('$', '', regex=False)
                            .str.replace('%', '', regex=False)
                            .str.replace(' ', '', regex=False)
                            .str.strip()
                    )
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    if numeric_series.notna().sum() > len(df_clean) * 0.7:
                        df_clean[col] = numeric_series
            except Exception:
                continue
    return df_clean


def detect_column_types(df: pd.DataFrame) -> dict:
    """Analyze columns and detect their purpose and characteristics"""
    column_analysis = {}
    for col in df.columns:
        info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'sample_values': df[col].dropna().head(5).tolist(),
            'likely_purpose': 'unknown'
        }
        col_lower = col.lower()
        if any(k in col_lower for k in ['date', 'time', 'day', 'month', 'year']):
            info['likely_purpose'] = 'temporal'
        elif any(k in col_lower for k in ['spend', 'cost', 'budget', 'price', 'revenue', 'sales', 'amount']):
            info['likely_purpose'] = 'financial'
        elif any(k in col_lower for k in ['roi', 'roas', 'ctr', 'cpc', 'cpm', 'conversion', 'rate', '%']):
            info['likely_purpose'] = 'performance_metric'
        elif any(k in col_lower for k in ['click', 'impression', 'view', 'lead', 'acquisition']):
            info['likely_purpose'] = 'volume_metric'
        elif any(k in col_lower for k in ['campaign', 'channel', 'source', 'medium', 'platform', 'category']):
            info['likely_purpose'] = 'categorical'
        elif df[col].dtype == 'object' and info['unique_count'] < len(df) * 0.5:
            info['likely_purpose'] = 'categorical'
        elif pd.api.types.is_numeric_dtype(df[col]):
            info['likely_purpose'] = 'numeric'
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                info.update({
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std())
                })
            except Exception:
                info.update({ 'min': None, 'max': None, 'mean': None, 'median': None, 'std': None })
        column_analysis[col] = info
    return column_analysis