from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class DriftReport:
    def __init__(self, drift_results: Dict[str, Any]):
        self.results = drift_results
        self.drifted_columns = [
            col for col, res in self.results.items() if res["drift_detected"]
        ]
        self.has_drift = len(self.drifted_columns) > 0

    def summary(self) -> str:
        lines = ["DataPulse Drift Report", "=" * 22]
        lines.append(f"Columns Checked: {len(self.results)}")
        lines.append(f"Columns Drifted: {len(self.drifted_columns)}")

        if self.has_drift:
            lines.append("\nDrift Detected In:")
            for col in self.drifted_columns:
                res = self.results[col]
                lines.append(
                    f"• {col} (Method: {res['method']}, p-value: {res.get('p_value', 'N/A'):.4f})"
                )

        return "\n".join(lines)


class DriftDetector:
    def __init__(self, method: str = "auto", p_value_threshold: float = 0.05):
        self.method = method
        self.threshold = p_value_threshold

    def compare(
        self,
        baseline: pd.DataFrame,
        current: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> DriftReport:
        if columns is None:
            columns = list(set(baseline.columns).intersection(set(current.columns)))

        results = {}
        for col in columns:
            if col not in baseline.columns or col not in current.columns:
                continue

            base_s = baseline[col].dropna()
            curr_s = current[col].dropna()

            if len(base_s) == 0 or len(curr_s) == 0:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(
                base_s
            ) and pd.api.types.is_numeric_dtype(curr_s)
            method_to_use = self.method
            if method_to_use == "auto":
                method_to_use = "ks_test" if is_numeric else "chi_square"

            if method_to_use == "ks_test" and is_numeric:
                stat, p_val = stats.ks_2samp(base_s, curr_s)
                results[col] = {
                    "method": "ks_test",
                    "statistic": stat,
                    "p_value": p_val,
                    "drift_detected": p_val < self.threshold,
                }
            elif method_to_use in ["chi_square", "auto"]:
                # Categorical drift using Chi-Square
                base_counts = base_s.value_counts(normalize=True)
                curr_counts = curr_s.value_counts(normalize=True)

                all_cats = list(set(base_counts.index).union(set(curr_counts.index)))

                b_freq = [base_counts.get(c, 0.0) * len(base_s) for c in all_cats]
                c_freq = [curr_counts.get(c, 0.0) * len(curr_s) for c in all_cats]

                # Add small epsilon to avoid divide by zero
                b_freq = np.array(b_freq) + 1e-5
                c_freq = np.array(c_freq) + 1e-5

                stat, p_val = stats.chisquare(f_obs=c_freq, f_exp=b_freq)
                results[col] = {
                    "method": "chi_square",
                    "statistic": stat,
                    "p_value": p_val,
                    "drift_detected": p_val < self.threshold,
                }

        return DriftReport(results)
