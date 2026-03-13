"""Alerting system for DataPulse."""

import requests
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class BaseAlert:
    def send(self, message: str, context: Dict[str, Any] = None):
        raise NotImplementedError


class SlackAlert(BaseAlert):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str, context: Dict[str, Any] = None):
        payload = {
            "text": f"🚨 *DataPulse Alert: {message}*",
            "attachments": [
                {
                    "color": "#ef4444",
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in (context or {}).items()
                    ],
                }
            ],
        }
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


class WebhookAlert(BaseAlert):
    def __init__(self, url: str):
        self.url = url

    def send(self, message: str, context: Dict[str, Any] = None):
        payload = {
            "event": "datapulse_alert",
            "message": message,
            "context": context or {},
        }
        try:
            requests.post(self.url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send Webhook alert: {e}")
