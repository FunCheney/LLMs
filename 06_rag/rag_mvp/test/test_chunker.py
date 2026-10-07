from ingestion.chunker import build_chunks


def test_build_chunks_keeps_section_and_cross_page_range():
    pages = [
        {
            "page_number": 7,
            "text": "1.1 Background\nConsumer Groups are logical groups of consumers.",
        },
        {
            "page_number": 8,
            "text": "Consumers read records from partitions.",
        },
    ]

    chunks = build_chunks(
        pages=pages,
        document_id="claude-code-0001",
        document_title="Claude Code Documentation",
        chunk_size=200,
        overlap=10,
        source_uri="data/raw/example.pdf",
        source_type="pdf",
    )

    assert len(chunks) == 1
    assert chunks[0].section_title == "1.1 Background"
    assert chunks[0].section_path == ["1.1 Background"]
    assert chunks[0].page_start == 7
    assert chunks[0].page_end == 8
    assert chunks[0].source_uri == "data/raw/example.pdf"
    assert "Consumer Groups" in chunks[0].content
    assert "Consumers read records" in chunks[0].content


def test_does_not_merge_across_sections():
    pages = [
        {
            "page_number": 1,
            "text": "1.1 A\nShort paragraph A.\n1.2 B\nShort paragraph B.",
        }
    ]

    chunks = build_chunks(
        pages=pages,
        document_id="doc",
        document_title="Doc",
        chunk_size=500,
        overlap=10,
    )

    assert len(chunks) == 2
    assert chunks[0].section_title == "1.1 A"
    assert chunks[1].section_title == "1.2 B"


def test_long_paragraph_maps_pages_by_offset():
    pages = [
        {
            "page_number": 3,
            "text": "1.1 A\n" + ("A" * 40),
        },
        {
            "page_number": 4,
            "text": "B" * 40,
        },
    ]

    chunks = build_chunks(
        pages=pages,
        document_id="doc",
        document_title="Doc",
        chunk_size=50,
        overlap=2,
    )

    assert len(chunks) >= 2
    assert chunks[0].page_start == 3
    assert chunks[0].page_end <= 4
    assert all(chunk.page_end - chunk.page_start <= 1 for chunk in chunks)
    assert all(chunk.section_title == "1.1 A" for chunk in chunks)
