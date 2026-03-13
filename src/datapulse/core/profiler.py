import json
import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class Profile:
    """Represents a data quality profile report."""

    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics

    def summary(self) -> str:
        """Returns a text summary of the profile."""
        lines = ["DataPulse Quality Report", "=" * 24]
        lines.append(f"Total Rows: {self.metrics['rows']}")
        lines.append(f"Total Columns: {self.metrics['columns']}")
        lines.append(
            f"Missing Cells: {self.metrics['missing_cells']} ({self.metrics['missing_cells_pct']:.2f}%)"
        )
        lines.append("\nColumn Profiles:")
        lines.append("-" * 16)

        for col, stats in self.metrics["columns_profile"].items():
            lines.append(f"• {col} ({stats['dtype']})")
            lines.append(
                f"  - Missing: {stats['missing']} ({stats['missing_pct']:.2f}%)"
            )
            lines.append(f"  - Unique: {stats['unique']}")
            if stats["dtype"] in ["int64", "float64", "Int64", "Float64"]:
                lines.append(f"  - Mean: {stats.get('mean', 'N/A')}")
                lines.append(
                    f"  - Min/Max: {stats.get('min', 'N/A')} / {stats.get('max', 'N/A')}"
                )

        return "\n".join(lines)

    def to_json(self, path: str = None) -> str:
        """Export profile to JSON."""
        js = json.dumps(self.metrics, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(js)
        return js

    def to_html(self, path: str):
        """Export profile to HTML dashboard."""
        from datapulse.reports.html import generate_html_report

        generate_html_report(self, path)


class DataPulse:
    """Core profiling engine."""

    def __init__(self):
        pass

    def profile(self, df: pd.DataFrame) -> Profile:
        """Generates a quality profile for a given DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        rows, cols = df.shape
        missing_cells = int(df.isna().sum().sum())
        total_cells = rows * cols

        metrics = {
            "rows": rows,
            "columns": cols,
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_cells_pct": (
                (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            ),
            "columns_profile": {},
        }

        for col in df.columns:
            s = df[col]
            missing = int(s.isna().sum())
            col_stats = {
                "dtype": str(s.dtype),
                "missing": missing,
                "missing_pct": (missing / rows) * 100 if rows > 0 else 0,
                "unique": int(s.nunique()),
            }

            if pd.api.types.is_numeric_dtype(s):
                col_stats["mean"] = float(s.mean()) if not pd.isna(s.mean()) else None
                col_stats["min"] = float(s.min()) if not pd.isna(s.min()) else None
                col_stats["max"] = float(s.max()) if not pd.isna(s.max()) else None
                col_stats["std"] = float(s.std()) if not pd.isna(s.std()) else None

            metrics["columns_profile"][str(col)] = col_stats

        return Profile(metrics)
