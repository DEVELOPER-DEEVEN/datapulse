"""CLI interface for DataPulse."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

import pandas as pd

app = typer.Typer(
    name="datapulse",
    help="Automated data quality monitoring for modern data teams",
    add_completion=False,
)
console = Console()


@app.command()
def profile(
    path: Path = typer.Argument(..., help="Path to CSV or Parquet file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for report"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, html"),
):
    """Profile a dataset and generate quality report."""
    from datapulse import DataPulse
    
    # Load data
    console.print(f"[blue]Loading data from {path}...[/blue]")
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        console.print(f"[red]Unsupported file format: {path.suffix}[/red]")
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
    
    if output and format != "html":
        console.print(f"[green]Report saved to {output}[/green]")


@app.command()
def validate(
    path: Path = typer.Argument(..., help="Path to CSV or Parquet file"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to datapulse.yaml config"),
    fail_on_error: bool = typer.Option(False, "--fail", help="Exit with error code on validation failure"),
):
    """Validate a dataset against expectations."""
    from datapulse import Monitor
    
    # Load data
    console.print(f"[blue]Loading data from {path}...[/blue]")
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        console.print(f"[red]Unsupported file format: {path.suffix}[/red]")
        raise typer.Exit(1)
    
    # Create monitor with basic expectations
    monitor = Monitor(name="cli_validation", fail_on_error=fail_on_error)
    
    # Add basic expectations for all columns
    for col in df.columns:
        if df[col].isna().sum() / len(df) > 0.5:
            continue  # Skip mostly-null columns
        monitor.expect(col).to_not_be_null()
    
    # Run validation
    result = monitor.validate(df)
    
    console.print(result.summary())
    
    if not result.passed and fail_on_error:
        raise typer.Exit(1)


@app.command()
def drift(
    baseline: Path = typer.Argument(..., help="Path to baseline dataset"),
    current: Path = typer.Argument(..., help="Path to current dataset"),
    columns: Optional[str] = typer.Option(None, "--columns", "-c", help="Comma-separated column names"),
    method: str = typer.Option("auto", "--method", "-m", help="Detection method: auto, ks_test, chi_square, psi"),
):
    """Detect distribution drift between datasets."""
    from datapulse import DriftDetector
    
    # Load data
    console.print(f"[blue]Loading baseline from {baseline}...[/blue]")
    baseline_df = pd.read_csv(baseline) if baseline.suffix == ".csv" else pd.read_parquet(baseline)
    
    console.print(f"[blue]Loading current from {current}...[/blue]")
    current_df = pd.read_csv(current) if current.suffix == ".csv" else pd.read_parquet(current)
    
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
