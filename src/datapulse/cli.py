"""CLI interface for DataPulse."""

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

app = typer.Typer(
    name="datapulse",
    help="Automated data quality monitoring for modern data teams",
    add_completion=False,
)
console = Console()


@app.command()
def profile(
    path: Path = typer.Argument(..., help="Path to CSV or Parquet file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for report"
    ),
    format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, json, html, pdf"
    ),
    ):
    """Profile a dataset and generate quality report."""
    from datapulse import DataPulse

    # Load data
    console.print(f"[blue]Loading data from {path}...[/blue]")

    try:
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Loaded {len(df):,} rows, {len(df.columns)} columns[/green]")

    # Profile
    pulse = DataPulse()
    report = pulse.profile(df)

    # Output
    if format == "text":
        console.print(report.summary())
    elif format == "json":
        json_output = report.to_json(output)
        if not output:
            console.print(json_output)
    elif format == "html":
        if output:
            report.to_html(output)
            console.print(f"[green]HTML report saved to {output}[/green]")
        else:
            console.print("[red]HTML format requires --output path[/red]")
            raise typer.Exit(1)
    elif format == "pdf":
        if output:
            report.to_pdf(output)
            console.print(f"[green]PDF report saved to {output}[/green]")
        else:
            console.print("[red]PDF format requires --output path[/red]")
            raise typer.Exit(1)


    if output and format != "html":
        console.print(f"[green]Report saved to {output}[/green]")


@app.command()
def validate(
    path: Path = typer.Argument(..., help="Path to CSV or Parquet file"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to datapulse.yaml config"
    ),
    monitor_name: Optional[str] = typer.Option(
        None, "--monitor", "-m", help="Specific monitor name from config"
    ),
    fail_on_error: bool = typer.Option(
        False, "--fail", help="Exit with error code on validation failure"
    ),
):
    """Validate a dataset against expectations."""
    from datapulse import Monitor
    from datapulse.core.config import load_config, build_monitor_from_config

    # Load data
    console.print(f"[blue]Loading data from {path}...[/blue]")
    try:
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)

    if config:
        # Load from YAML
        console.print(f"[blue]Loading expectations from {config}...[/blue]")
        full_config = load_config(config)
        target_monitor = monitor_name or list(full_config.monitors.keys())[0]
        monitor = build_monitor_from_config(target_monitor, full_config)
    else:
        # Intelligent Auto-Expectations
        console.print(
            "[yellow]No config provided. Running intelligent auto-validation...[/yellow]"
        )
        monitor = Monitor(name="auto_validation", fail_on_error=fail_on_error)
        for col in df.columns:
            # 1. Non-null expectation for relatively full columns
            null_pct = df[col].isna().sum() / len(df)
            if null_pct < 0.2:
                monitor.expect(col).to_not_be_null()

            # 2. Uniqueness for high-cardinality ID-like columns
            if "id" in col.lower() and df[col].nunique() / len(df) > 0.9:
                monitor.expect(col).to_be_unique()

            # 3. Positivity for financial/metric columns
            if any(k in col.lower() for k in ["price", "revenue", "cost", "score"]):
                if pd.api.types.is_numeric_dtype(df[col]):
                    monitor.expect(col).to_be_positive()

    # Run validation
    result = monitor.validate(df)
    console.print(result.summary())

    if not result.passed and fail_on_error:
        raise typer.Exit(1)


@app.command()
def drift(
    baseline: Path = typer.Argument(..., help="Path to baseline dataset"),
    current: Path = typer.Argument(..., help="Path to current dataset"),
    columns: Optional[str] = typer.Option(
        None, "--columns", "-c", help="Comma-separated column names"
    ),
    method: str = typer.Option(
        "auto",
        "--method",
        "-m",
        help="Detection method: auto, ks_test, chi_square, psi",
    ),
):
    """Detect distribution drift between datasets."""
    from datapulse import DriftDetector

    # Load data
    console.print(f"[blue]Loading baseline from {baseline}...[/blue]")
    baseline_df = (
        pd.read_csv(baseline)
        if baseline.suffix == ".csv"
        else pd.read_parquet(baseline)
    )

    console.print(f"[blue]Loading current from {current}...[/blue]")
    current_df = (
        pd.read_csv(current) if current.suffix == ".csv" else pd.read_parquet(current)
    )

    # Parse columns
    column_list = columns.split(",") if columns else None

    # Detect drift
    detector = DriftDetector(method=method)
    report = detector.compare(baseline_df, current_df, columns=column_list)

    console.print(report.summary())

    if report.has_drift:
        raise typer.Exit(1)


@app.command()
def version():
    """Show DataPulse version."""
    from datapulse import __version__

    console.print(f"DataPulse v{__version__}")


if __name__ == "__main__":
    app()
