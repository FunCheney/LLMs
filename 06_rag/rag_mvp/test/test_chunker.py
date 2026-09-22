from ingestion.chunker import build_chunks


def test_build_chunks():
    pages = [
        {
            "page_number": 7,
            "text": "Consumer Groups are logical groups of consumers.",
        },
        {
            "page_number": 8,
            "text": "Consumers read records from partitions.",
        },
    ]

    chunks = build_chunks(
        pages=pages,
        document_id="claude-Code-0001",
        document_title="claude-Code Documentation",
        chunk_size=50,
        overlap=2
    )

    assert len(chunks) == 2
    print(chunks[0].chunk_id)
    print(chunks[1].chunk_id)
    assert chunks[0].chunk_id == (
        "claude-Code-0001-p007-p008-c000"
    )

    assert chunks[0].page_start == 7
    assert chunks[0].page_end == 8

    assert chunks[1].chunk_id == (
        "claude-Code-0001-p007-p008-c001"
    )



if __name__ == "__main__":
    test_build_chunks()