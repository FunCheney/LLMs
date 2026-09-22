from ingestion.page_aware_splitter import (
    build_page_aware_chunks,
    combine_pages,
    split_paragraph_units,
)


def test_combine_pages():
    pages = [
        {
            "page_number": 1,
            "text": "Page one text.",
        },
        {
            "page_number": 2,
            "text": "Page two text.",
        },
    ]

    combined_text, page_ranges = combine_pages(pages)

    assert combined_text == (
        "Page one text.\n"
        "Page two text."
    )

    assert len(page_ranges) == 2


def test_paragraph_can_cross_pages():
    pages = [
        {
            "page_number": 1,
            "text": (
                "Kafka uses partitions to provide"
            ),
        },
        {
            "page_number": 2,
            "text": (
                " parallelism and scalability."
            ),
        },
    ]

    units = split_paragraph_units(pages)

    assert len(units) == 1
    assert units[0].page_start == 1
    assert units[0].page_end == 2


def test_chunk_contains_multiple_pages():
    pages = [
        {
            "page_number": 1,
            "text": "Paragraph A.",
        },
        {
            "page_number": 2,
            "text": "Paragraph B.",
        },
    ]

    chunks = build_page_aware_chunks(
        pages=pages,
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 2
    assert "Paragraph A." in chunks[0]["content"]
    assert "Paragraph B." in chunks[0]["content"]


def test_chunk_does_not_exceed_size():
    pages = [
        {
            "page_number": 1,
            "text": "A" * 40,
        },
        {
            "page_number": 2,
            "text": "B" * 40,
        },
        {
            "page_number": 3,
            "text": "C" * 40,
        },
    ]

    chunks = build_page_aware_chunks(
        pages=pages,
        chunk_size=85,
        overlap=10,
    )

    assert len(chunks) == 2

    for chunk in chunks:
        assert len(chunk["content"]) <= 85


