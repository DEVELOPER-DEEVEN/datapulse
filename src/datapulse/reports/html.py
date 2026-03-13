"""HTML report generation for DataPulse."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datapulse.core.profiler import Profile

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataPulse Quality Report</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #3b82f6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: sans-serif; background: var(--bg-primary); color: var(--text-primary); padding: 2rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 2rem; border-bottom: 1px solid var(--bg-card); padding-bottom: 2rem; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .stat-card {{ background: var(--bg-secondary); padding: 1.5rem; border-radius: 12px; text-align: center; }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: var(--accent); }}
        .section {{ background: var(--bg-secondary); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--bg-card); }}
        th {{ color: var(--text-secondary); text-transform: uppercase; font-size: 0.8rem; }}
        .badge {{ padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }}
        .badge-success {{ background: var(--success); color: white; }}
        .badge-danger {{ background: var(--danger); color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 DataPulse Quality Report</h1>
        </header>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{rows:,}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{columns}</div>
                <div class="stat-label">Columns</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{missing_pct:.2f}%</div>
                <div class="stat-label">Missing Cells</div>
            </div>
        </div>
        <div class="section">
            <h2>Column Profiles</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Missing</th>
                        <th>Unique</th>
                        <th>Mean</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {column_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""


def generate_html_report(profile: Profile, path: str | Path) -> None:
    metrics = profile.metrics
    column_rows = []

    for col_name, col_stats in metrics["columns_profile"].items():
        missing_pct = col_stats["missing_pct"]
        status = '<span class="badge badge-success">OK</span>'
        if missing_pct > 10:
            status = '<span class="badge badge-danger">High Missing</span>'

        row = f"""
        <tr>
            <td><strong>{col_name}</strong></td>
            <td>{col_stats['dtype']}</td>
            <td>{missing_pct:.2f}%</td>
            <td>{col_stats['unique']:,}</td>
            <td>{col_stats.get('mean', 'N/A')}</td>
            <td>{status}</td>
        </tr>
        """
        column_rows.append(row)

    html = HTML_TEMPLATE.format(
        rows=metrics["rows"],
        columns=metrics["columns"],
        missing_pct=metrics["missing_cells_pct"],
        column_rows="\n".join(column_rows),
    )

    Path(path).write_text(html)
