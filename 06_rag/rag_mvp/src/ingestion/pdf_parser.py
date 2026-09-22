from pathlib import Path
import pymupdf

def parse_pdf(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file {pdf_path} does not exist")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF file {pdf_path} is not a PDF file")

    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "page_number": page_index + 1,
                "text": text.strip(),
            })


    return pages


def summarize_pages(pages: list[dict]) -> dict:
    non_empty_pages = [
        page for page in pages
        if page["text"]
    ]

    total_chars = sum(
        len(page["text"])
        for page in pages
    )

    return {
        "total_pages": len(pages),
        "non_empty_pages": len(non_empty_pages),
        "total_chars": total_chars,
    }

