from pathlib import Path

from ingestion.html_parser import parse_html
from ingestion.pdf_parser import parse_pdf


def load_pages(path: str | Path) -> tuple[list[dict], str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path), "pdf"
    if suffix in {".html", ".htm"}:
        return parse_html(path), "html"
    raise ValueError(f"Unsupported document type: {path.suffix}")
