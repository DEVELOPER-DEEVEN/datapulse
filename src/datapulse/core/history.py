import sqlite_utils
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class HistoryEngine:
    """Persistent storage for DataPulse quality metrics."""
    def __init__(self, db_path: str | Path = "datapulse_history.db"):
        self.db = sqlite_utils.Database(db_path)
        
    def log_profile(self, monitor_name: str, profile_metrics: Dict[str, Any]):
        """Store profiling results in the history table."""
        row = {
            "timestamp": datetime.now().isoformat(),
            "monitor_name": monitor_name,
            "rows": profile_metrics["rows"],
            "columns": profile_metrics["columns"],
            "missing_cells_pct": profile_metrics["missing_cells_pct"]
        }
        self.db["profiles"].insert(row)
        
    def log_validation(self, monitor_name: str, result_summary: Dict[str, Any]):
        """Store validation results in the history table."""
        row = {
            "timestamp": datetime.now().isoformat(),
            "monitor_name": monitor_name,
            "passed": result_summary["passed"],
            "total_checked": result_summary["total_checked"],
            "total_failed": result_summary["total_failed"]
        }
        self.db["validations"].insert(row)
        
    def get_history(self, monitor_name: str):
        """Retrieve historical quality trends."""
        return list(self.db["validations"].rows_where("monitor_name = ?", [monitor_name]))
