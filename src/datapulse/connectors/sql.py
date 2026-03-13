import pandas as pd
import logging
from typing import Optional
from sqlalchemy import create_url, create_engine

logger = logging.getLogger(__name__)


class SQLConnector:
    def __init__(self, connection_url: str):
        self.url = connection_url
        self.engine = create_engine(self.url)

    def read_table(self, table_name: str, query: Optional[str] = None) -> pd.DataFrame:
        """Read data from a SQL table or query."""
        if query:
            return pd.read_sql_query(query, self.engine)
        return pd.read_sql_table(table_name, self.engine)

    def write_table(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        """Write DataFrame to a SQL table."""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
