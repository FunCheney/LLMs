from ingestion.pdf_parser import parse_pdf, summarize_pages


def main():
    pdf_path = "../data/raw/Claude Code设计指南.pdf"

    pages = parse_pdf(pdf_path)

    print(f"Total pages: {len(pages)}")

    summary = summarize_pages(pages)

    print(f"Total summary: {summary}")

if __name__ == "__main__":
    main()

