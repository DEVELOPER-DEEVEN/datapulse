import pytest
import pandas as pd
import numpy as np
from datapulse.core.detector import DriftDetector, DriftReport, DriftMethod

@pytest.fixture
def baseline_data():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(30, 5, 1000),
        "income": np.random.normal(50000, 10000, 1000),
        "category": np.random.choice(["A", "B", "C"], 1000)
    })

@pytest.fixture
def drifted_data():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(40, 5, 1000),  # Drifted mean
        "income": np.random.normal(50000, 10000, 1000),  # No drift
        "category": np.random.choice(["A", "B"], 1000)  # Drifted category (missing C)
    })

def test_detector_initialization():
    detector = DriftDetector(method="auto", threshold=0.05)
    assert detector.method == "auto"
    assert detector.threshold == 0.05

def test_drift_detection_numeric(baseline_data, drifted_data):
    detector = DriftDetector(method="ks_test")
    report = detector.compare(baseline_data, drifted_data, columns=["age", "income"])
    
    assert isinstance(report, DriftReport)
    assert "age" in report.drifted_columns
    assert "income" in report.stable_columns
    assert report.drift_results["age"].method == DriftMethod.KS_TEST

def test_drift_detection_categorical(baseline_data, drifted_data):
    detector = DriftDetector(method="chi_square")
    report = detector.compare(baseline_data, drifted_data, columns=["category"])
    
    assert report.has_drift
    assert "category" in report.drifted_columns
    assert report.drift_results["category"].method == DriftMethod.CHI_SQUARE

def test_auto_method_selection(baseline_data, drifted_data):
    detector = DriftDetector(method="auto")
    report = detector.compare(baseline_data, drifted_data)
    
    assert report.drift_results["age"].method == DriftMethod.KS_TEST
    assert report.drift_results["category"].method == DriftMethod.CHI_SQUARE
