from abc import ABC, abstractmethod

import pandas as pd


class BaseConnector(ABC):
    """Abstract base class for data connectors."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data from the source into a pandas DataFrame."""
        pass
