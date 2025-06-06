# File: src/Test_red/app_backend/ingest.py

import pandas as pd
from io import BytesIO, StringIO

class IngestError(Exception):
    """Raised if the uploaded file cannot be read or parsed."""
    pass

def read_dataframe_from_upload(uploaded_file, snippet_rows: int = 50):
    """
    Given a Streamlit uploader (with .read()), returns:
      • df: pandas.DataFrame for the entire file
      • csv_snippet: header + first `snippet_rows` rows, CSV-formatted string
    """
    try:
        file_name = uploaded_file.name.lower()
        raw_bytes = uploaded_file.read()

        if file_name.endswith((".xls", ".xlsx")):
            buffer = BytesIO(raw_bytes)
            df = pd.read_excel(buffer)
        elif file_name.endswith(".csv"):
            text_io = StringIO(raw_bytes.decode("utf-8"))
            df = pd.read_csv(text_io)
        else:
            raise IngestError(f"Unsupported file type: {file_name}")

        snippet_df = df.head(snippet_rows)
        csv_snippet = snippet_df.to_csv(index=False)
        return df, csv_snippet

    except Exception as e:
        raise IngestError(f"Failed to ingest file: {e}")
