"""Data validation and expectation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
from enum import Enum
import re

import pandas as pd
import numpy as np


class CheckType(Enum):
    """Types of validation checks."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BETWEEN = "between"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    REGEX = "regex"
    IN_SET = "in_set"
    RECENT = "recent"
    CUSTOM = "custom"


@dataclass
class ExpectationResult:
    """Result of a single expectation check."""
    
    column: str
    check_type: CheckType
    passed: bool
    message: str
    failed_count: int = 0
    failed_percentage: float = 0.0
    failed_examples: list[Any] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Complete validation result."""
    
    passed: bool
    expectations_run: int
    expectations_passed: int
    expectations_failed: int
    results: list[ExpectationResult] = field(default_factory=list)
    
    @property
    def failures(self) -> list[ExpectationResult]:
        """Get failed expectations."""
        return [r for r in self.results if not r.passed]
    
    def summary(self) -> str:
        """Generate validation summary."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [
            f"Validation {status}",
            f"Expectations: {self.expectations_passed}/{self.expectations_run} passed",
            "",
        ]
        
        for result in self.results:
            icon = "✓" if result.passed else "✗"
            lines.append(f"  {icon} {result.column}: {result.message}")
            
            if not result.passed and result.failed_examples:
                examples = result.failed_examples[:3]
                lines.append(f"      Examples: {examples}")
        
        return "\n".join(lines)


class Expectation:
    """Fluent interface for building column expectations."""
    
    def __init__(self, column: str, monitor: Monitor):
        self.column = column
        self.monitor = monitor
        self._checks: list[tuple[CheckType, dict[str, Any]]] = []
    
    def to_be_positive(self) -> Expectation:
        """Values must be greater than 0."""
        self._checks.append((CheckType.POSITIVE, {}))
        return self
    
    def to_be_negative(self) -> Expectation:
        """Values must be less than 0."""
        self._checks.append((CheckType.NEGATIVE, {}))
        return self
    
    def to_be_between(self, min_val: float, max_val: float) -> Expectation:
        """Values must be within range [min, max]."""
        self._checks.append((CheckType.BETWEEN, {"min": min_val, "max": max_val}))
        return self
    
    def to_be_unique(self) -> Expectation:
        """Values must be unique (no duplicates)."""
        self._checks.append((CheckType.UNIQUE, {}))
        return self
    
    def to_not_be_null(self) -> Expectation:
        """Values must not be null/NaN."""
        self._checks.append((CheckType.NOT_NULL, {}))
        return self
    
    def to_match_regex(self, pattern: str) -> Expectation:
        """String values must match regex pattern."""
        self._checks.append((CheckType.REGEX, {"pattern": pattern}))
        return self
    
    def to_be_in(self, values: list[Any]) -> Expectation:
        """Values must be in the specified set."""
        self._checks.append((CheckType.IN_SET, {"values": values}))
        return self
    
    def to_be_recent(self, days: int) -> Expectation:
        """Date values must be within N days of now."""
        self._checks.append((CheckType.RECENT, {"days": days}))
        return self
    
    def to_satisfy(self, func: Callable[[pd.Series], pd.Series], description: str = "custom check") -> Expectation:
        """Values must satisfy custom function."""
        self._checks.append((CheckType.CUSTOM, {"func": func, "description": description}))
        return self


class Monitor:
    """Data quality monitor with expectation-based validation."""
    
    def __init__(
        self,
        name: str = "default",
        alerts: Optional[list] = None,
        fail_on_error: bool = False,
    ):
        """
        Initialize monitor.
        
        Args:
            name: Monitor name for identification
            alerts: List of alert handlers
            fail_on_error: Whether to raise exception on validation failure
        """
        self.name = name
        self.alerts = alerts or []
        self.fail_on_error = fail_on_error
        self._expectations: list[Expectation] = []
    
    def expect(self, column: str) -> Expectation:
        """Create expectation for a column."""
        expectation = Expectation(column, self)
        self._expectations.append(expectation)
        return expectation
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all expectations against the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with all check results
        """
        results: list[ExpectationResult] = []
        
        for expectation in self._expectations:
            if expectation.column not in df.columns:
                results.append(ExpectationResult(
                    column=expectation.column,
                    check_type=CheckType.CUSTOM,
                    passed=False,
                    message=f"Column '{expectation.column}' not found in DataFrame",
                ))
                continue
            
            series = df[expectation.column]
            
            for check_type, params in expectation._checks:
                result = self._run_check(series, check_type, params)
                results.append(result)
        
        passed = all(r.passed for r in results)
        
        validation_result = ValidationResult(
            passed=passed,
            expectations_run=len(results),
            expectations_passed=sum(1 for r in results if r.passed),
            expectations_failed=sum(1 for r in results if not r.passed),
            results=results,
        )
        
        # Send alerts if validation failed
        if not passed:
            self._send_alerts(validation_result)
        
        if not passed and self.fail_on_error:
            raise ValueError(f"Data validation failed: {validation_result.failures}")
        
        return validation_result
    
    def _run_check(
        self,
        series: pd.Series,
        check_type: CheckType,
        params: dict[str, Any],
    ) -> ExpectationResult:
        """Run a single check on a series."""
        column = str(series.name)
        
        if check_type == CheckType.POSITIVE:
            mask = series.dropna() <= 0
            failed = series[mask]
            return self._make_result(column, check_type, mask, failed, "be positive")
        
        elif check_type == CheckType.NEGATIVE:
            mask = series.dropna() >= 0
            failed = series[mask]
            return self._make_result(column, check_type, mask, failed, "be negative")
        
        elif check_type == CheckType.BETWEEN:
            min_val, max_val = params["min"], params["max"]
            mask = ~series.between(min_val, max_val)
            failed = series[mask].dropna()
            return self._make_result(column, check_type, mask, failed, f"be between {min_val} and {max_val}")
        
        elif check_type == CheckType.UNIQUE:
            duplicates = series[series.duplicated(keep=False)]
            passed = len(duplicates) == 0
            return ExpectationResult(
                column=column,
                check_type=check_type,
                passed=passed,
                message="be unique" if passed else f"be unique (found {len(duplicates)} duplicates)",
                failed_count=len(duplicates),
                failed_percentage=len(duplicates) / len(series) * 100 if len(series) > 0 else 0,
                failed_examples=duplicates.head(5).tolist(),
            )
        
        elif check_type == CheckType.NOT_NULL:
            mask = series.isna()
            return self._make_result(column, check_type, mask, series[mask], "not be null")
        
        elif check_type == CheckType.REGEX:
            pattern = params["pattern"]
            mask = ~series.astype(str).str.match(pattern, na=False)
            failed = series[mask].dropna()
            return self._make_result(column, check_type, mask, failed, f"match pattern '{pattern}'")
        
        elif check_type == CheckType.IN_SET:
            valid_values = set(params["values"])
            mask = ~series.isin(valid_values)
            failed = series[mask].dropna()
            return self._make_result(column, check_type, mask, failed, f"be in {list(valid_values)[:5]}...")
        
        elif check_type == CheckType.RECENT:
            days = params["days"]
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            mask = pd.to_datetime(series, errors="coerce") < cutoff
            failed = series[mask].dropna()
            return self._make_result(column, check_type, mask, failed, f"be within last {days} days")
        
        elif check_type == CheckType.CUSTOM:
            func = params["func"]
            description = params["description"]
            try:
                mask = ~func(series)
                failed = series[mask]
                return self._make_result(column, check_type, mask, failed, description)
            except Exception as e:
                return ExpectationResult(
                    column=column,
                    check_type=check_type,
                    passed=False,
                    message=f"Custom check failed with error: {e}",
                )
        
        return ExpectationResult(
            column=column,
            check_type=check_type,
            passed=False,
            message=f"Unknown check type: {check_type}",
        )
    
    def _make_result(
        self,
        column: str,
        check_type: CheckType,
        mask: pd.Series,
        failed: pd.Series,
        description: str,
    ) -> ExpectationResult:
        """Create an ExpectationResult from check outputs."""
        failed_count = mask.sum()
        total = len(mask)
        passed = failed_count == 0
        
        return ExpectationResult(
            column=column,
            check_type=check_type,
            passed=passed,
            message=description if passed else f"{description} ({failed_count} failures)",
            failed_count=failed_count,
            failed_percentage=(failed_count / total * 100) if total > 0 else 0,
            failed_examples=failed.head(5).tolist() if len(failed) > 0 else [],
        )
    
    def _send_alerts(self, result: ValidationResult) -> None:
        """Send alerts for validation failures."""
        for alert in self.alerts:
            try:
                alert.send(self.name, result)
            except Exception as e:
                print(f"Failed to send alert: {e}")
