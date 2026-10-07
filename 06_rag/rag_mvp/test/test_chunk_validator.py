from schemas import DocumentChunk
from validation.chunk_validator import validate_chunks


def _chunk(**kwargs) -> DocumentChunk:
    defaults = dict(
        chunk_id="doc-p001-p001-c000",
        document_id="doc",
        document_title="Doc",
        section_title="1.1 A",
        section_path=["1.1 A"],
        page_start=1,
        page_end=1,
        content="hello world",
        source_uri="data/raw/doc.pdf",
        source_type="pdf",
    )
    defaults.update(kwargs)
    return DocumentChunk(**defaults)


def test_validate_chunks_accepts_healthy_corpus():
    report = validate_chunks(
        [_chunk(), _chunk(chunk_id="doc-p002-p002-c001", page_start=2, page_end=2)]
    )
    assert report.ok
    assert report.stats["chunk_count"] == 2


def test_validate_chunks_detects_duplicate_and_empty():
    report = validate_chunks(
        [
            _chunk(),
            _chunk(content="   "),
        ]
    )
    codes = {issue.code for issue in report.errors}
    assert "duplicate_chunk_id" in codes
    assert "empty_content" in codes


def test_validate_chunks_warns_wide_page_span():
    report = validate_chunks(
        [_chunk(page_end=20)],
        max_page_span=8,
    )
    assert report.ok
    assert any(issue.code == "wide_page_span" for issue in report.warnings)
