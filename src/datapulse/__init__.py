"""
DataPulse - Automated Data Quality Monitoring

A lightweight, extensible data quality monitoring framework that automatically
profiles datasets, detects anomalies, and alerts before bad data propagates.
"""

__version__ = "0.1.0"
__author__ = "Deeven Seru"
__email__ = "deevenseru11@gmail.com"

from datapulse.core.profiler import DataPulse, Profile
from datapulse.core.validator import Monitor, Expectation, ValidationResult
from datapulse.core.detector import DriftDetector, DriftReport

__all__ = [
    "DataPulse",
    "Profile",
    "Monitor",
    "Expectation",
    "ValidationResult",
    "DriftDetector",
    "DriftReport",
]
