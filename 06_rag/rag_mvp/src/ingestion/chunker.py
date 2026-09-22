from ingestion.page_aware_splitter import build_page_aware_chunks
from schemas import DocumentChunk


def build_chunks(
        pages: list[dict],
        document_id: str,
        document_title: str,
        chunk_size: int = 500,
        overlap: int = 50,
) -> list[DocumentChunk]:
    chunks = []
    raw_chunks = build_page_aware_chunks(pages, chunk_size, overlap)
    for chunk_index, raw_chunk in enumerate(raw_chunks):
        page_start = raw_chunk["page_start"]
        page_end = raw_chunk["page_end"]
        chunk = DocumentChunk(
            chunk_id=(
                f"{document_id}-"
                f"p{page_start:03d}-"  # 03d 表示将整数格式化为三位数
                f"p{page_end:03d}-"
                f"c{chunk_index:03d}"
                      ),
            document_id=document_id,
            document_title=document_title,
            section_title="UNKNOW",
            section_path=[],
            page_start=page_start,
            page_end=page_end,
            content=raw_chunk["content"], )
        chunks.append(chunk)

    return chunks
