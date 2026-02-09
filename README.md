# DataPulse

**Automated Data Quality Monitoring for Modern Data Teams**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DataPulse is a lightweight, extensible data quality monitoring framework that automatically profiles your datasets, detects anomalies, and alerts you before bad data propagates through your pipelines.

## Why DataPulse?

Data pipelines fail silently. A null value sneaks in, a distribution shifts, a schema changes — and nobody notices until dashboards break or models degrade. DataPulse catches these issues at the source.

**Key Features:**
- 🔍 **Auto-Profiling**: Automatically analyze distributions, nulls, outliers, and data types
- 📊 **Drift Detection**: Statistical tests to catch distribution shifts over time
- 🚨 **Smart Alerts**: Slack, Email, Telegram, or webhook notifications
- 📈 **Quality Reports**: Beautiful HTML reports with visualizations
- 🔌 **Flexible Connectors**: Works with pandas, SQL databases, Spark, and cloud storage
- ⚡ **Lightweight**: Minimal dependencies, easy to integrate

## Installation

```bash
pip install datapulse
```

Or install from source:

```bash
git clone https://github.com/DEVELOPER-DEEVEN/datapulse.git
cd datapulse
pip install -e .
```

## Quick Start

### Basic Profiling

```python
import pandas as pd
from datapulse import DataPulse

# Load your data
df = pd.read_csv("sales_data.csv")

# Create a DataPulse instance
pulse = DataPulse()

# Generate a quality report
report = pulse.profile(df)
print(report.summary())

# Save HTML report
report.to_html("quality_report.html")
```

### Automated Monitoring

```python
from datapulse import Monitor, SlackAlert

# Configure monitoring
monitor = Monitor(
    name="sales_pipeline",
    alerts=[SlackAlert(webhook_url="https://hooks.slack.com/...")]
)

# Define quality expectations
monitor.expect("revenue").to_be_positive()
monitor.expect("customer_id").to_be_unique()
monitor.expect("order_date").to_be_recent(days=30)
monitor.expect("category").to_be_in(["electronics", "clothing", "food"])

# Run validation
results = monitor.validate(df)

if not results.passed:
    print(f"Quality issues found: {results.failures}")
```

### Drift Detection

```python
from datapulse import DriftDetector

detector = DriftDetector()

# Compare current data against baseline
drift_report = detector.compare(
    baseline=historical_df,
    current=new_df,
    columns=["price", "quantity", "category"]
)

if drift_report.has_drift:
    print(f"Drift detected in: {drift_report.drifted_columns}")
```

## Architecture

```
datapulse/
├── core/
│   ├── profiler.py      # Data profiling engine
│   ├── validator.py     # Expectation validation
│   └── detector.py      # Drift detection
├── connectors/
│   ├── pandas.py        # DataFrame connector
│   ├── sql.py           # Database connector
│   └── spark.py         # Spark connector
├── alerts/
│   ├── slack.py         # Slack notifications
│   ├── email.py         # Email alerts
│   └── telegram.py      # Telegram alerts
├── reports/
│   ├── html.py          # HTML report generator
│   └── json.py          # JSON export
└── cli.py               # Command-line interface
```

## Expectations API

DataPulse provides a fluent API for defining data quality expectations:

| Method | Description |
|--------|-------------|
| `to_be_positive()` | Values must be > 0 |
| `to_be_negative()` | Values must be < 0 |
| `to_be_between(min, max)` | Values within range |
| `to_be_unique()` | No duplicate values |
| `to_not_be_null()` | No null/NaN values |
| `to_match_regex(pattern)` | String pattern matching |
| `to_be_in(values)` | Categorical validation |
| `to_be_recent(days)` | Date recency check |
| `to_have_distribution(type)` | Distribution shape |

## Configuration

Create a `datapulse.yaml` for persistent configuration:

```yaml
project: my_data_pipeline
version: 1.0

sources:
  sales:
    type: postgres
    connection: postgresql://user:pass@host/db
    table: sales_data
    
  events:
    type: csv
    path: /data/events/*.csv

monitors:
  sales_quality:
    source: sales
    schedule: "0 * * * *"  # Hourly
    expectations:
      - column: revenue
        check: positive
      - column: customer_id
        check: unique
    alerts:
      - type: slack
        channel: "#data-alerts"
```

## CLI Usage

```bash
# Profile a CSV file
datapulse profile data.csv

# Run validation
datapulse validate --config datapulse.yaml

# Generate report
datapulse report data.csv --output report.html

# Watch mode (continuous monitoring)
datapulse watch --config datapulse.yaml
```

## Integrations

- **Airflow**: `DataPulseOperator` for DAG integration
- **Prefect**: Native task decorators
- **dbt**: Post-hook validation
- **GitHub Actions**: CI/CD quality gates

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with ❤️ by [Deeven Seru](https://github.com/DEVELOPER-DEEVEN)
