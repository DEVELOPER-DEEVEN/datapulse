"""PDF report generation for DataPulse."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datapulse.core.profiler import Profile

def generate_pdf_report(profile: Profile, path: str | Path) -> None:
    """Generate a beautiful PDF report from a profile."""
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError(
            "WeasyPrint is required for PDF reporting. "
            "Install it with: pip install 'datapulse[reports]'"
        )

    # We reuse the HTML template logic but can add PDF-specific styling here if needed
    from datapulse.reports.html import generate_html_report
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
        temp_html_path = tf.name
    
    try:
        generate_html_report(profile, temp_html_path)
        HTML(filename=temp_html_path).write_pdf(path)
    finally:
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)
