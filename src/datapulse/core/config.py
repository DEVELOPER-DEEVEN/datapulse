import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datapulse.core.validator import Monitor
from datapulse.alerts.base import SlackAlert, WebhookAlert


class ExpectationConfig(BaseModel):
    column: str
    check: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MonitorConfig(BaseModel):
    name: str
    source: str
    expectations: list[ExpectationConfig]
    alerts: Optional[list[Dict[str, str]]] = Field(default_factory=list)


class DataPulseConfig(BaseModel):
    project: str
    version: str
    monitors: Dict[str, MonitorConfig]


def load_config(path: str | Path) -> DataPulseConfig:
    """Load and validate datapulse.yaml."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return DataPulseConfig(**data)


def build_monitor_from_config(name: str, config: DataPulseConfig) -> Monitor:
    """Construct a Monitor instance from the YAML configuration."""
    if name not in config.monitors:
        raise ValueError(f"Monitor '{name}' not found in configuration.")

    m_cfg = config.monitors[name]
    alerts = []

    for a in m_cfg.alerts:
        if a["type"] == "slack":
            alerts.append(SlackAlert(webhook_url=a["url"]))
        elif a["type"] == "webhook":
            alerts.append(WebhookAlert(url=a["url"]))

    monitor = Monitor(name=m_cfg.name, alerts=alerts)

    for exp in m_cfg.expectations:
        expectation = monitor.expect(exp.column)
        check_name = exp.check
        params = exp.params or {}

        # Dynamic mapping of strings to expectation methods
        if check_name == "positive":
            expectation.to_be_positive()
        elif check_name == "unique":
            expectation.to_be_unique()
        elif check_name == "not_null":
            expectation.to_not_be_null()
        elif check_name == "between":
            expectation.to_be_between(params.get("min"), params.get("max"))
        elif check_name == "in":
            expectation.to_be_in(params.get("values", []))

    return monitor
