import pandas as pd
from typing import Optional, Any
from pathlib import Path
from .base import BaseConnector

class CsvConnector(BaseConnector):
    """Connector for CSV files."""

    def __init__(self, filepath: str, **kwargs: Any):
        """
        Initialize CSV connector.

        Args:
            filepath: Path to the CSV file.
            **kwargs: Additional arguments passed to pandas.read_csv.
        """
        self.filepath = filepath
        self.kwargs = kwargs

    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        
        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not Path(self.filepath).exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
            
        return pd.read_csv(self.filepath, **self.kwargs)
