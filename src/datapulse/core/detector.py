"""Drift detection for monitoring data distribution changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats


class DriftMethod(Enum):
    """Statistical methods for drift detection."""
    KS_TEST = "ks_test"  # Kolmogorov-Smirnov test
    CHI_SQUARE = "chi_square"  # Chi-square test for categorical
    PSI = "psi"  # Population Stability Index
    JS_DIVERGENCE = "js_divergence"  # Jensen-Shannon divergence


@dataclass
class ColumnDrift:
    """Drift analysis for a single column."""
    
    column: str
    has_drift: bool
    method: DriftMethod
    statistic: float
    p_value: Optional[float] = None
    threshold: float = 0.05
    baseline_stats: dict[str, Any] = field(default_factory=dict)
    current_stats: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "has_drift": self.has_drift,
            "method": self.method.value,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6) if self.p_value else None,
            "threshold": self.threshold,
        }


@dataclass
class DriftReport:
    """Complete drift analysis report."""
    
    has_drift: bool
    columns_analyzed: int
    columns_with_drift: int
    drift_results: dict[str, ColumnDrift] = field(default_factory=dict)
    
    @property
    def drifted_columns(self) -> list[str]:
        """Get list of columns with detected drift."""
        return [name for name, result in self.drift_results.items() if result.has_drift]
    
    @property
    def stable_columns(self) -> list[str]:
        """Get list of columns without drift."""
        return [name for name, result in self.drift_results.items() if not result.has_drift]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "⚠️ DRIFT DETECTED" if self.has_drift else "✓ NO DRIFT"
        
        lines = [
            "=" * 50,
            f"DRIFT REPORT: {status}",
            "=" * 50,
            f"Columns analyzed: {self.columns_analyzed}",
            f"Columns with drift: {self.columns_with_drift}",
            "",
        ]
        
        if self.drifted_columns:
            lines.append("Drifted columns:")
            for col in self.drifted_columns:
                result = self.drift_results[col]
                lines.append(f"  ⚠️ {col}: {result.method.value} (stat={result.statistic:.4f})")
        
        if self.stable_columns:
            lines.append("\nStable columns:")
            for col in self.stable_columns[:5]:
                lines.append(f"  ✓ {col}")
            if len(self.stable_columns) > 5:
                lines.append(f"  ... and {len(self.stable_columns) - 5} more")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_drift": self.has_drift,
            "columns_analyzed": self.columns_analyzed,
            "columns_with_drift": self.columns_with_drift,
            "drifted_columns": self.drifted_columns,
            "results": {name: result.to_dict() for name, result in self.drift_results.items()},
        }


class DriftDetector:
    """Detect distribution drift between datasets."""
    
    def __init__(
        self,
        method: Literal["auto", "ks_test", "chi_square", "psi"] = "auto",
        threshold: float = 0.05,
        psi_threshold: float = 0.2,
    ):
        """
        Initialize drift detector.
        
        Args:
            method: Detection method ('auto' selects based on column type)
            threshold: P-value threshold for statistical tests
            psi_threshold: PSI threshold for drift detection
        """
        self.method = method
        self.threshold = threshold
        self.psi_threshold = psi_threshold
    
    def compare(
        self,
        baseline: pd.DataFrame,
        current: pd.DataFrame,
        columns: Optional[list[str]] = None,
    ) -> DriftReport:
        """
        Compare current data against baseline for drift.
        
        Args:
            baseline: Reference dataset (historical/training data)
            current: Current dataset to check for drift
            columns: Specific columns to analyze (None = all common columns)
            
        Returns:
            DriftReport with analysis results
        """
        # Determine columns to analyze
        if columns is None:
            columns = list(set(baseline.columns) & set(current.columns))
        else:
            columns = [c for c in columns if c in baseline.columns and c in current.columns]
        
        drift_results: dict[str, ColumnDrift] = {}
        
        for col in columns:
            baseline_col = baseline[col].dropna()
            current_col = current[col].dropna()
            
            if len(baseline_col) == 0 or len(current_col) == 0:
                continue
            
            # Select appropriate method
            if self.method == "auto":
                if pd.api.types.is_numeric_dtype(baseline_col):
                    result = self._ks_test(col, baseline_col, current_col)
                else:
                    result = self._chi_square_test(col, baseline_col, current_col)
            elif self.method == "ks_test":
                result = self._ks_test(col, baseline_col, current_col)
            elif self.method == "chi_square":
                result = self._chi_square_test(col, baseline_col, current_col)
            elif self.method == "psi":
                result = self._psi_test(col, baseline_col, current_col)
            else:
                continue
            
            drift_results[col] = result
        
        has_drift = any(r.has_drift for r in drift_results.values())
        
        return DriftReport(
            has_drift=has_drift,
            columns_analyzed=len(drift_results),
            columns_with_drift=sum(1 for r in drift_results.values() if r.has_drift),
            drift_results=drift_results,
        )
    
    def _ks_test(self, column: str, baseline: pd.Series, current: pd.Series) -> ColumnDrift:
        """Kolmogorov-Smirnov test for numeric columns."""
        statistic, p_value = stats.ks_2samp(baseline, current)
        
        return ColumnDrift(
            column=column,
            has_drift=p_value < self.threshold,
            method=DriftMethod.KS_TEST,
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=self.threshold,
            baseline_stats={"mean": float(baseline.mean()), "std": float(baseline.std())},
            current_stats={"mean": float(current.mean()), "std": float(current.std())},
        )
    
    def _chi_square_test(self, column: str, baseline: pd.Series, current: pd.Series) -> ColumnDrift:
        """Chi-square test for categorical columns."""
        # Get all unique categories
        all_categories = set(baseline.unique()) | set(current.unique())
        
        # Calculate frequencies
        baseline_counts = baseline.value_counts()
        current_counts = current.value_counts()
        
        # Align to same categories
        baseline_freq = np.array([baseline_counts.get(cat, 0) for cat in all_categories])
        current_freq = np.array([current_counts.get(cat, 0) for cat in all_categories])
        
        # Normalize to expected frequencies
        baseline_freq = baseline_freq / baseline_freq.sum() * current_freq.sum()
        
        # Avoid division by zero
        baseline_freq = np.maximum(baseline_freq, 0.001)
        
        statistic, p_value = stats.chisquare(current_freq, baseline_freq)
        
        return ColumnDrift(
            column=column,
            has_drift=p_value < self.threshold,
            method=DriftMethod.CHI_SQUARE,
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=self.threshold,
        )
    
    def _psi_test(self, column: str, baseline: pd.Series, current: pd.Series) -> ColumnDrift:
        """Population Stability Index for drift detection."""
        # Create bins from baseline
        if pd.api.types.is_numeric_dtype(baseline):
            bins = pd.qcut(baseline, q=10, duplicates="drop").categories
            baseline_binned = pd.cut(baseline, bins=bins)
            current_binned = pd.cut(current, bins=bins)
        else:
            baseline_binned = baseline
            current_binned = current
        
        # Calculate proportions
        baseline_props = baseline_binned.value_counts(normalize=True)
        current_props = current_binned.value_counts(normalize=True)
        
        # Align indices
        all_bins = set(baseline_props.index) | set(current_props.index)
        
        psi = 0.0
        for bin_val in all_bins:
            baseline_pct = baseline_props.get(bin_val, 0.0001)
            current_pct = current_props.get(bin_val, 0.0001)
            
            # PSI formula
            psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
        
        return ColumnDrift(
            column=column,
            has_drift=psi > self.psi_threshold,
            method=DriftMethod.PSI,
            statistic=float(psi),
            threshold=self.psi_threshold,
        )
