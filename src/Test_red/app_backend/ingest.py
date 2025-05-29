import pandas as pd
import duckdb
import os
from src.Test_red.exception import DataIngestionError
from src.Test_red.logger import logger

# Single in-memory DuckDB connection
_duck_conn = duckdb.connect(database=":memory:")

def load_csv_file(filepath: str) -> pd.DataFrame:
    """
    Read a CSV into pandas, sanitize, register in DuckDB, and return.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read CSV {filepath}: {e}")
        raise DataIngestionError(f"CSV read failed: {e}")

    table_name = _table_name_from_path(filepath)
    try:
        _duck_conn.register(table_name, df)
        logger.info(f"Registered CSV as DuckDB table `{table_name}`")
    except Exception as e:
        logger.error(f"Failed to register CSV in DuckDB: {e}")
        raise DataIngestionError(f"DuckDB registration failed: {e}")

    return df

def load_excel_file(filepath: str) -> pd.DataFrame:
    """
    Read an Excel file (first sheet) into pandas, register in DuckDB, and return.
    """
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        logger.error(f"Failed to read Excel {filepath}: {e}")
        raise DataIngestionError(f"Excel read failed: {e}")

    table_name = _table_name_from_path(filepath)
    try:
        _duck_conn.register(table_name, df)
        logger.info(f"Registered Excel as DuckDB table `{table_name}`")
    except Exception as e:
        logger.error(f"Failed to register Excel in DuckDB: {e}")
        raise DataIngestionError(f"DuckDB registration failed: {e}")

    return df

def list_upload_files(directory: str, extensions: list[str]) -> list[str]:
    """
    Return full paths of all files in `directory` matching any of `extensions`.
    """
    try:
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if any(f.lower().endswith(ext) for ext in extensions)
        ]
    except FileNotFoundError:
        return []

def _table_name_from_path(filepath: str) -> str:
    """
    Turn a filename into a sanitized DuckDB table name.
    """
    fname = os.path.basename(filepath)
    name, _ = os.path.splitext(fname)
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "t_" + sanitized
    return sanitized.lower()

def run_sql(query: str) -> pd.DataFrame:
    """
    Execute a SQL query on the in-memory DuckDB and return a DataFrame.
    """
    try:
        return _duck_conn.execute(query).df()
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        raise DataIngestionError(f"SQL execution failed: {e}")
