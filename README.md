<p align="center">
  <img src="assets/logo.svg" width="200" alt="DataPulse Logo">
</p>

# DataPulse

**Automated Data Quality Monitoring for Modern Data Teams**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-success.svg?style=flat-square)](tests/test_core.py)

DataPulse is a lightweight, extensible data quality monitoring framework that automatically profiles datasets, detects anomalies, and alerts stakeholders before bad data propagates through the pipeline.

## Overview

Data pipelines often fail silently. Schema changes, null values, or statistical distribution shifts can degrade downstream models and dashboards without triggering standard system alerts. DataPulse provides the empirical visibility required to maintain data integrity at scale.

### Key Features

*   **Data Profiling**: Comprehensive analysis of distributions, null percentages, and descriptive statistics.
*   **Drift Detection**: Automated statistical tests (Kolmogorov-Smirnov and Chi-Square) to identify distribution shifts.
*   **Expectations API**: Fluent, vectorized validation for data quality constraints.
*   **Proactive Alerting**: Native integration with Slack and Webhooks for real-time failure notification.
*   **Interactive Dashboards**: Automated generation of responsive HTML reports for quality assessment.
*   **Flexible Connectivity**: Native support for Pandas DataFrames and SQL-based data sources.

## Installation

Install via pip:

```bash
pip install datapulse
```

Or install from source for development:

```bash
git clone https://github.com/Deeven-Seru/datapulse.git
cd datapulse
pip install -e ".[all]"
```

## Quick Start

### Statistical Profiling

```python
import pandas as pd
from datapulse import DataPulse

# Load dataset
df = pd.read_csv("sales_data.csv")

# Initialize and profile
pulse = DataPulse()
report = pulse.profile(df)

# Output summary to console
print(report.summary())

# Generate interactive dashboard
report.to_html("quality_dashboard.html")
```

### Automated Monitoring with Alerting

```python
from datapulse import Monitor, SlackAlert

# Configure monitoring and alerting
monitor = Monitor(
    name="production_pipeline",
    alerts=[SlackAlert(webhook_url="https://hooks.slack.com/services/...")]
)

# Define precise expectations
monitor.expect("revenue").to_be_positive()
monitor.expect("customer_id").to_be_unique().to_not_be_null()
monitor.expect("region").to_be_in(["US", "EU", "APAC"])
monitor.expect("age").to_be_between(0, 120)

# Validate dataset (triggers alerts on failure)
results = monitor.validate(df)
```

### Distribution Drift Detection

```python
from datapulse import DriftDetector

detector = DriftDetector()

# Detect shifts between baseline and current data
drift_report = detector.compare(
    baseline=historical_df,
    current=new_df
)

if drift_report.has_drift:
    print(drift_report.summary())
```

## SQL Connectivity

DataPulse supports direct integration with SQL databases via SQLAlchemy.

```python
from datapulse.connectors.sql import SQLConnector
from datapulse import DataPulse

connector = SQLConnector("postgresql://user:pass@host:5432/dbname")
df = connector.read_table("transactions")

pulse = DataPulse()
report = pulse.profile(df)
```

## Core Architecture

The project is structured for modularity and high performance:

*   **core/**: Primary engines for profiling, validation, and statistical detection.
*   **connectors/**: Abstraction layer for different data sources (Pandas, SQL).
*   **alerts/**: Dispatchers for stakeholder notifications.
*   **reports/**: Visualization and reporting generators.

## Expectations API Reference

| Method | Description |
| :--- | :--- |
| `to_be_positive()` | Values must be strictly greater than 0 |
| `to_be_negative()` | Values must be strictly less than 0 |
| `to_be_unique()` | Validates that all values are distinct |
| `to_not_be_null()` | Ensures no missing or NaN values exist |
| `to_be_in(values)` | Checks for membership in a specific categorical set |
| `to_be_between(min, max)` | Validates numeric range constraints |
| `to_match_regex(pattern)` | Ensures strings match a specific regular expression |

## Testing

DataPulse maintains a 100% pass rate on core modules. Run the test suite using pytest:

```bash
pytest tests/test_core.py
```

## Contributing

Professional contributions are welcome. Please ensure all code is formatted using `black` and adheres to the stylistic guidelines before submitting a pull request.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built by [Deeven Seru](https://github.com/Deeven-Seru)
