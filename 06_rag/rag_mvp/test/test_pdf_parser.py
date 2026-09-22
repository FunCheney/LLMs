from pathlib import Path

from ingestion.pdf_parser import parse_pdf


def test_parse_pdf():
    pdf_path = Path("data/raw/Claude Code设计指南.pdf")

    if not pdf_path.exists():
        return

    pages = parse_pdf(pdf_path)

    assert len(pages) > 0
    assert pages[0]["page_number"] == 1
    assert "text" in pages[0]