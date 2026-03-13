import numpy as np
import pandas as pd

from datapulse.core.detector import DriftDetector
from datapulse.core.profiler import DataPulse
from datapulse.core.validator import Monitor


def test_profiler():
    df = pd.DataFrame({"a": [1, 2, 3, np.nan], "b": ["x", "y", "x", "z"]})

    pulse = DataPulse()
    report = pulse.profile(df)
    metrics = report.metrics

    assert metrics["rows"] == 4
    assert metrics["columns"] == 2
    assert metrics["missing_cells"] == 1
    assert metrics["columns_profile"]["a"]["missing"] == 1
    assert metrics["columns_profile"]["b"]["unique"] == 3
    assert metrics["columns_profile"]["a"]["max"] == 3.0


def test_validator():
    df = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
            "customer_id": [1, 2, 3],
            "category": ["electronics", "clothing", "food"],
            "age": [25, 30, -5],
        }
    )

    monitor = Monitor(name="test")
    monitor.expect("revenue").to_be_positive()
    monitor.expect("customer_id").to_be_unique().to_not_be_null()
    monitor.expect("category").to_be_in(["electronics", "clothing", "food", "toys"])
    monitor.expect("age").to_be_between(0, 120)

    result = monitor.validate(df)
    # The 'age' expectation will fail because of -5
    assert not result.passed
    assert result.total_failed == 1
    assert "age_check_0" in result.failures

    # Fix the data
    df["age"] = [25, 30, 45]
    result2 = monitor.validate(df)
    assert result2.passed


def test_drift_detector():
    # Create baseline
    np.random.seed(42)
    baseline_df = pd.DataFrame(
        {
            "metric": np.random.normal(0, 1, 1000),
            "cat": np.random.choice(["A", "B", "C"], 1000),
        }
    )

    # Create current (drifted)
    current_df = pd.DataFrame(
        {
            "metric": np.random.normal(5, 1, 1000),  # Drifted mean
            "cat": np.random.choice(
                ["A", "B", "C"], 1000, p=[0.8, 0.1, 0.1]
            ),  # Drifted distribution
        }
    )

    detector = DriftDetector()
    report = detector.compare(baseline_df, current_df)

    assert report.has_drift
    assert "metric" in report.drifted_columns
    assert "cat" in report.drifted_columns
    assert report.results["metric"]["method"] == "ks_test"
    assert report.results["cat"]["method"] == "chi_square"
