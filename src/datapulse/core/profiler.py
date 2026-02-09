"""Core profiling engine for DataPulse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union
from pathlib import Path
import json

import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""
    
    name: str
    dtype: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Numeric stats (None for non-numeric)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Categorical stats
    top_values: Optional[list[tuple[Any, int]]] = None
    
    # Quality indicators
    has_outliers: bool = False
    outlier_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "null_count": self.null_count,
            "null_percentage": round(self.null_percentage, 2),
            "unique_count": self.unique_count,
            "unique_percentage": round(self.unique_percentage, 2),
            "mean": round(self.mean, 4) if self.mean is not None else None,
            "std": round(self.std, 4) if self.std is not None else None,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "skewness": round(self.skewness, 4) if self.skewness is not None else None,
            "kurtosis": round(self.kurtosis, 4) if self.kurtosis is not None else None,
            "top_values": self.top_values,
            "has_outliers": self.has_outliers,
            "outlier_count": self.outlier_count,
        }


@dataclass
class Profile:
    """Complete profile of a dataset."""
    
    row_count: int
    column_count: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    duplicate_row_count: int = 0
    memory_usage_mb: float = 0.0
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "DATAPULSE QUALITY REPORT",
            "=" * 60,
            f"Rows: {self.row_count:,}",
            f"Columns: {self.column_count}",
            f"Duplicates: {self.duplicate_row_count:,} ({self.duplicate_row_count/self.row_count*100:.1f}%)" if self.row_count > 0 else "Duplicates: 0",
            f"Memory: {self.memory_usage_mb:.2f} MB",
            "",
            "-" * 60,
            "COLUMN SUMMARY",
            "-" * 60,
        ]
        
        for col_name, col_profile in self.columns.items():
            null_indicator = "⚠️" if col_profile.null_percentage > 5 else "✓"
            outlier_indicator = "⚠️" if col_profile.has_outliers else ""
            
            lines.append(
                f"{null_indicator} {col_name}: {col_profile.dtype} | "
                f"nulls: {col_profile.null_percentage:.1f}% | "
                f"unique: {col_profile.unique_count} {outlier_indicator}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_row_count": self.duplicate_row_count,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }
    
    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export profile to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(json_str)
        return json_str
    
    def to_html(self, path: Union[str, Path]) -> None:
        """Generate HTML report."""
        from datapulse.reports.html import generate_html_report
        generate_html_report(self, path)


class DataPulse:
    """Main profiling engine."""
    
    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        top_n_values: int = 10,
    ):
        """
        Initialize DataPulse profiler.
        
        Args:
            outlier_method: Method for outlier detection ('iqr' or 'zscore')
            outlier_threshold: Threshold for outlier detection
            top_n_values: Number of top values to capture for categorical columns
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.top_n_values = top_n_values
    
    def profile(self, df: pd.DataFrame) -> Profile:
        """
        Generate a complete profile of the dataset.
        
        Args:
            df: pandas DataFrame to profile
            
        Returns:
            Profile object containing all statistics
        """
        profile = Profile(
            row_count=len(df),
            column_count=len(df.columns),
            duplicate_row_count=df.duplicated().sum(),
            memory_usage_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
        )
        
        for col_name in df.columns:
            profile.columns[col_name] = self._profile_column(df[col_name])
        
        return profile
    
    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Profile a single column."""
        count = len(series)
        null_count = series.isna().sum()
        unique_count = series.nunique()
        
        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            count=count,
            null_count=null_count,
            null_percentage=(null_count / count * 100) if count > 0 else 0,
            unique_count=unique_count,
            unique_percentage=(unique_count / count * 100) if count > 0 else 0,
        )
        
        # Numeric column statistics
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                profile.mean = float(clean_series.mean())
                profile.std = float(clean_series.std())
                profile.min = float(clean_series.min())
                profile.max = float(clean_series.max())
                profile.median = float(clean_series.median())
                profile.q1 = float(clean_series.quantile(0.25))
                profile.q3 = float(clean_series.quantile(0.75))
                
                if len(clean_series) > 2:
                    profile.skewness = float(stats.skew(clean_series))
                    profile.kurtosis = float(stats.kurtosis(clean_series))
                
                # Outlier detection
                outliers = self._detect_outliers(clean_series)
                profile.has_outliers = len(outliers) > 0
                profile.outlier_count = len(outliers)
        
        # Top values for categorical-like columns
        if unique_count <= 100 or not pd.api.types.is_numeric_dtype(series):
            value_counts = series.value_counts().head(self.top_n_values)
            profile.top_values = list(zip(value_counts.index.tolist(), value_counts.values.tolist()))
        
        return profile
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers in a numeric series."""
        if self.outlier_method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr
            return series[(series < lower_bound) | (series > upper_bound)]
        
        elif self.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(series))
            return series[z_scores > self.outlier_threshold]
        
        return pd.Series(dtype=series.dtype)
