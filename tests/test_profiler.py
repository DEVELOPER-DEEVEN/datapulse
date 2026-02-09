"""Tests for DataPulse profiler."""

import pytest
import pandas as pd
import numpy as np

from datapulse import DataPulse, Profile


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(1000),
        "name": [f"user_{i}" for i in range(1000)],
        "age": np.random.randint(18, 80, 1000),
        "salary": np.random.normal(50000, 15000, 1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
        "has_nulls": [None if i % 10 == 0 else i for i in range(1000)],
    })


class TestDataPulse:
    """Test suite for DataPulse profiler."""
    
    def test_profile_basic(self, sample_df):
        """Test basic profiling functionality."""
        pulse = DataPulse()
        profile = pulse.profile(sample_df)
        
        assert isinstance(profile, Profile)
        assert profile.row_count == 1000
        assert profile.column_count == 6
    
    def test_profile_columns(self, sample_df):
        """Test column-level profiling."""
        pulse = DataPulse()
        profile = pulse.profile(sample_df)
        
        assert "id" in profile.columns
        assert "salary" in profile.columns
        
        # Check numeric stats
        salary_profile = profile.columns["salary"]
        assert salary_profile.mean is not None
        assert salary_profile.std is not None
        assert salary_profile.min is not None
        assert salary_profile.max is not None
    
    def test_null_detection(self, sample_df):
        """Test null value detection."""
        pulse = DataPulse()
        profile = pulse.profile(sample_df)
        
        null_profile = profile.columns["has_nulls"]
        assert null_profile.null_count == 100  # Every 10th value is null
        assert null_profile.null_percentage == 10.0
    
    def test_unique_count(self, sample_df):
        """Test unique value counting."""
        pulse = DataPulse()
        profile = pulse.profile(sample_df)
        
        id_profile = profile.columns["id"]
        assert id_profile.unique_count == 1000
        assert id_profile.unique_percentage == 100.0
        
        cat_profile = profile.columns["category"]
        assert cat_profile.unique_count == 3
    
    def test_outlier_detection(self, sample_df):
        """Test outlier detection with IQR method."""
        # Add obvious outliers
        df = sample_df.copy()
        df.loc[0, "salary"] = 1000000  # Extreme outlier
        
        pulse = DataPulse(outlier_method="iqr", outlier_threshold=1.5)
        profile = pulse.profile(df)
        
        salary_profile = profile.columns["salary"]
        assert salary_profile.has_outliers
        assert salary_profile.outlier_count > 0
    
    def test_summary_output(self, sample_df):
        """Test summary string generation."""
        pulse = DataPulse()
        profile = pulse.profile(sample_df)
        summary = profile.summary()
        
        assert "DATAPULSE QUALITY REPORT" in summary
        assert "1,000" in summary  # Row count
        assert "6" in summary  # Column count
    
    def test_json_export(self, sample_df, tmp_path):
        """Test JSON export functionality."""
        pulse = DataPulse()
        profile = pulse.profile(sample_df)
        
        # Export to file
        output_path = tmp_path / "report.json"
        json_str = profile.to_json(output_path)
        
        assert output_path.exists()
        assert "row_count" in json_str
        assert "columns" in json_str
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        pulse = DataPulse()
        profile = pulse.profile(pd.DataFrame())
        
        assert profile.row_count == 0
        assert profile.column_count == 0
