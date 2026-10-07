from ingestion.structured_parser import parse_structured_paragraphs
from ingestion.text_splitter import split_text_windows
from schemas import DocumentChunk, StructuredParagraph


def _pages_for_range(
    start_offset: int,
    end_offset: int,
    page_spans: list[tuple[int, int, int]],
    fallback_start: int,
    fallback_end: int,
) -> tuple[int, int]:
    matched = [
        page
        for span_start, span_end, page in page_spans
        if start_offset < span_end and end_offset > span_start
    ]
    if not matched:
        return fallback_start, fallback_end
    return min(matched), max(matched)


def _flush_group(
    group: list[StructuredParagraph],
) -> tuple[str, int, int, str, list[str]]:
    content = "\n\n".join(paragraph.content for paragraph in group)
    return (
        content,
        min(paragraph.page_start for paragraph in group),
        max(paragraph.page_end for paragraph in group),
        group[0].section_title,
        list(group[0].section_path),
    )


def _make_chunk(
    *,
    document_id: str,
    document_title: str,
    chunk_index: int,
    page_start: int,
    page_end: int,
    content: str,
    section_title: str,
    section_path: list[str],
    source_uri: str,
    source_type: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=(
            f"{document_id}-"
            f"p{page_start:03d}-"
            f"p{page_end:03d}-"
            f"c{chunk_index:03d}"
        ),
        document_id=document_id,
        document_title=document_title,
        section_title=section_title,
        section_path=section_path,
        page_start=page_start,
        page_end=page_end,
        content=content,
        source_uri=source_uri,
        source_type=source_type,
    )


def build_chunks(
    pages: list[dict],
    document_id: str,
    document_title: str,
    chunk_size: int = 500,
    overlap: int = 50,
    source_uri: str = "",
    source_type: str = "pdf",
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    paragraphs = parse_structured_paragraphs(pages)
    chunks: list[DocumentChunk] = []
    current_group: list[StructuredParagraph] = []
    current_length = 0

    def emit(content: str, page_start: int, page_end: int, section_title: str, section_path: list[str]) -> None:
        chunks.append(
            _make_chunk(
                document_id=document_id,
                document_title=document_title,
                chunk_index=len(chunks),
                page_start=page_start,
                page_end=page_end,
                content=content,
                section_title=section_title,
                section_path=section_path,
                source_uri=source_uri,
                source_type=source_type,
            )
        )

    def flush_group() -> None:
        nonlocal current_group, current_length
        if not current_group:
            return
        content, page_start, page_end, section_title, section_path = _flush_group(current_group)
        emit(content, page_start, page_end, section_title, section_path)
        current_group = []
        current_length = 0

    for paragraph in paragraphs:
        content = paragraph.content
        if len(content) > chunk_size:
            flush_group()
            for piece, start, end in split_text_windows(content, chunk_size, overlap):
                page_start, page_end = _pages_for_range(
                    start,
                    end,
                    paragraph.page_spans,
                    paragraph.page_start,
                    paragraph.page_end,
                )
                emit(piece, page_start, page_end, paragraph.section_title, paragraph.section_path)
            continue

        same_section = (
            current_group
            and current_group[0].section_path == paragraph.section_path
        )
        additional_length = len(content) + (2 if current_group else 0)

        if same_section and current_length + additional_length <= chunk_size:
            current_group.append(paragraph)
            current_length += additional_length
            continue

        flush_group()
        current_group.append(paragraph)
        current_length = len(content)

    flush_group()
    return chunks
