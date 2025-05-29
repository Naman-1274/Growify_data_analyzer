# exception.py

class DataIngestionError(Exception):
    """Raised when CSV/Excel ingestion fails."""
    pass

class ModelAPIError(Exception):
    """Raised when the LLM/API call fails."""
    pass
