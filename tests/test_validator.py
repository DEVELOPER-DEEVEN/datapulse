"""Tests for DataPulse validator."""

import pytest
import pandas as pd
import numpy as np

from datapulse import Monitor, ValidationResult


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, -5, 40, 35],  # -5 is invalid
        "email": ["a@test.com", "b@test.com", "invalid", "d@test.com", "e@test.com"],
        "category": ["A", "B", "A", "C", "D"],  # D is invalid if expecting A, B, C
        "score": [100, 200, 150, 175, 200],  # 200 is duplicate
    })


class TestMonitor:
    """Test suite for Monitor validation."""
    
    def test_positive_check(self, sample_df):
        """Test positive value validation."""
        monitor = Monitor()
        monitor.expect("age").to_be_positive()
        
        result = monitor.validate(sample_df)
        
        assert not result.passed  # -5 should fail
        assert result.expectations_failed == 1
    
    def test_between_check(self, sample_df):
        """Test range validation."""
        monitor = Monitor()
        monitor.expect("age").to_be_between(0, 50)
        
        result = monitor.validate(sample_df)
        
        assert not result.passed  # -5 is outside range
    
    def test_not_null_check(self):
        """Test null validation."""
        df = pd.DataFrame({
            "values": [1, 2, None, 4, 5]
        })
        
        monitor = Monitor()
        monitor.expect("values").to_not_be_null()
        
        result = monitor.validate(df)
        
        assert not result.passed
        assert result.failures[0].failed_count == 1
    
    def test_unique_check(self, sample_df):
        """Test uniqueness validation."""
        monitor = Monitor()
        monitor.expect("score").to_be_unique()
        
        result = monitor.validate(sample_df)
        
        assert not result.passed  # 200 appears twice
    
    def test_in_set_check(self, sample_df):
        """Test categorical validation."""
        monitor = Monitor()
        monitor.expect("category").to_be_in(["A", "B", "C"])
        
        result = monitor.validate(sample_df)
        
        assert not result.passed  # "D" is not in set
    
    def test_regex_check(self, sample_df):
        """Test regex pattern validation."""
        monitor = Monitor()
        monitor.expect("email").to_match_regex(r"^[\w]+@[\w]+\.[\w]+$")
        
        result = monitor.validate(sample_df)
        
        assert not result.passed  # "invalid" doesn't match
    
    def test_multiple_expectations(self, sample_df):
        """Test multiple expectations on same column."""
        monitor = Monitor()
        monitor.expect("age").to_be_positive().to_be_between(0, 100)
        
        result = monitor.validate(sample_df)
        
        # Both checks should fail for -5
        assert not result.passed
        assert result.expectations_failed >= 1
    
    def test_passing_validation(self):
        """Test validation that passes."""
        df = pd.DataFrame({
            "positive_values": [1, 2, 3, 4, 5],
            "unique_ids": [1, 2, 3, 4, 5],
        })
        
        monitor = Monitor()
        monitor.expect("positive_values").to_be_positive()
        monitor.expect("unique_ids").to_be_unique()
        
        result = monitor.validate(df)
        
        assert result.passed
        assert result.expectations_passed == 2
        assert result.expectations_failed == 0
    
    def test_summary_output(self, sample_df):
        """Test summary string generation."""
        monitor = Monitor()
        monitor.expect("age").to_be_positive()
        
        result = monitor.validate(sample_df)
        summary = result.summary()
        
        assert "FAILED" in summary or "PASSED" in summary
        assert "age" in summary
    
    def test_missing_column(self, sample_df):
        """Test handling of missing column."""
        monitor = Monitor()
        monitor.expect("nonexistent").to_be_positive()
        
        result = monitor.validate(sample_df)
        
        assert not result.passed
        assert "not found" in result.failures[0].message
    
    def test_custom_check(self, sample_df):
        """Test custom validation function."""
        monitor = Monitor()
        monitor.expect("age").to_satisfy(
            lambda s: s > 20,
            description="be greater than 20"
        )
        
        result = monitor.validate(sample_df)
        
        # -5 fails the check
        assert not result.passed
