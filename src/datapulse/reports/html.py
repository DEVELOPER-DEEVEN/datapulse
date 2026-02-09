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
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #3b82f6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--bg-card);
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--text-secondary);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
        }
        
        .stat-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .section {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .section h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--bg-card);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--bg-card);
        }
        
        th {
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
        }
        
        tr:hover {
            background: var(--bg-card);
        }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .badge-success { background: var(--success); color: white; }
        .badge-warning { background: var(--warning); color: black; }
        .badge-danger { background: var(--danger); color: white; }
        
        .progress-bar {
            height: 8px;
            background: var(--bg-card);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.3s ease;
        }
        
        footer {
            text-align: center;
            color: var(--text-secondary);
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid var(--bg-card);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 DataPulse Quality Report</h1>
            <p class="subtitle">Automated data quality analysis</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{row_count:,}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{column_count}</div>
                <div class="stat-label">Columns</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{duplicate_count:,}</div>
                <div class="stat-label">Duplicates</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{memory_mb:.1f} MB</div>
                <div class="stat-label">Memory</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Column Profiles</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Nulls</th>
                        <th>Unique</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {column_rows}
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>Generated by <strong>DataPulse</strong> | Built by Deeven Seru</p>
        </footer>
    </div>
</body>
</html>
"""


def generate_html_report(profile: Profile, path: str | Path) -> None:
    """Generate an HTML report from a profile."""
    column_rows = []
    
    for col_name, col_profile in profile.columns.items():
        null_pct = col_profile.null_percentage
        
        if null_pct > 20:
            status = '<span class="badge badge-danger">High Nulls</span>'
        elif null_pct > 5:
            status = '<span class="badge badge-warning">Some Nulls</span>'
        elif col_profile.has_outliers:
            status = '<span class="badge badge-warning">Outliers</span>'
        else:
            status = '<span class="badge badge-success">OK</span>'
        
        row = f"""
        <tr>
            <td><strong>{col_name}</strong></td>
            <td>{col_profile.dtype}</td>
            <td>{null_pct:.1f}%</td>
            <td>{col_profile.unique_count:,}</td>
            <td>{status}</td>
        </tr>
        """
        column_rows.append(row)
    
    html = HTML_TEMPLATE.format(
        row_count=profile.row_count,
        column_count=profile.column_count,
        duplicate_count=profile.duplicate_row_count,
        memory_mb=profile.memory_usage_mb,
        column_rows="\n".join(column_rows),
    )
    
    Path(path).write_text(html)
