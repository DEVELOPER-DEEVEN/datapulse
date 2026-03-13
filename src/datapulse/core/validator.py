from typing import Any, Callable, Dict, List

import pandas as pd


class ValidationResult:
    def __init__(
        self,
        passed: bool,
        total_checked: int,
        total_failed: int,
        failures: Dict[str, Any],
    ):
        self.passed = passed
        self.total_checked = total_checked
        self.total_failed = total_failed
        self.failures = failures

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation Result: {status}"]
        lines.append(f"Checks Run: {self.total_checked}")
        lines.append(f"Checks Failed: {self.total_failed}")
        if not self.passed:
            lines.append("Failures:")
            for k, v in self.failures.items():
                lines.append(f" - {k}: {v}")
        return "\n".join(lines)


class Expectation:
    def __init__(self, column: str):
        self.column = column
        self.checks: List[Callable[[pd.Series], tuple[bool, str]]] = []

    def to_be_positive(self):
        def check(s: pd.Series):
            failed = (s <= 0).sum()
            return failed == 0, f"{failed} non-positive values found"

        self.checks.append(check)
        return self

    def to_be_negative(self):
        def check(s: pd.Series):
            failed = (s >= 0).sum()
            return failed == 0, f"{failed} non-negative values found"

        self.checks.append(check)
        return self

    def to_be_unique(self):
        def check(s: pd.Series):
            dupes = s.duplicated().sum()
            return dupes == 0, f"{dupes} duplicate values found"

        self.checks.append(check)
        return self

    def to_not_be_null(self):
        def check(s: pd.Series):
            nulls = s.isna().sum()
            return nulls == 0, f"{nulls} null values found"

        self.checks.append(check)
        return self

    def to_be_in(self, values: List[Any]):
        def check(s: pd.Series):
            failed = (~s.isin(values) & s.notna()).sum()
            return failed == 0, f"{failed} values not in {values}"

        self.checks.append(check)
        return self

    def to_be_between(self, min_val, max_val):
        def check(s: pd.Series):
            failed = ((s < min_val) | (s > max_val)).sum()
            return failed == 0, f"{failed} values out of bounds [{min_val}, {max_val}]"

        self.checks.append(check)
        return self

    def to_match_regex(self, pattern: str):
        def check(s: pd.Series):
            if not pd.api.types.is_string_dtype(s):
                return False, "Column is not string dtype"
            failed = (~s.astype(str).str.match(pattern) & s.notna()).sum()
            return failed == 0, f"{failed} values do not match regex '{pattern}'"

        self.checks.append(check)
        return self

    def validate(self, s: pd.Series) -> List[tuple[bool, str]]:
        return [check(s) for check in self.checks]


class Monitor:
    def __init__(self, name: str, alerts=None, fail_on_error: bool = False):
        self.name = name
        self.alerts = alerts or []
        self.expectations: Dict[str, Expectation] = {}
        self.fail_on_error = fail_on_error

    def expect(self, column: str) -> Expectation:
        if column not in self.expectations:
            self.expectations[column] = Expectation(column)
        return self.expectations[column]

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        total_checked = 0
        total_failed = 0
        failures = {}

        for col, exp in self.expectations.items():
            if col not in df.columns:
                total_checked += 1
                total_failed += 1
                failures[f"{col}_exists"] = "Column not found in dataset"
                continue

            results = exp.validate(df[col])
            for idx, (passed, msg) in enumerate(results):
                total_checked += 1
                if not passed:
                    total_failed += 1
                    failures[f"{col}_check_{idx}"] = msg

        return ValidationResult(
            passed=(total_failed == 0),
            total_checked=total_checked,
            total_failed=total_failed,
            failures=failures,
        )
