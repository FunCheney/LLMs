"""
检索结果映射：把底层存储对象还原成上层可直接使用的结构。

- metadata_to_document_chunk：metadata(dict) + 正文 → DocumentChunk
- vector_record_to_hit：VectorRecord + 分数 → RetrievalResult

这是本地向量库 / BM25 的统一出口，保证不同后端产出的结果结构一致。
"""

from typing import Any

from retrieval.types import RetrievalResult
from schemas import DocumentChunk
from vector_store.vector_store_types import VectorRecord


def metadata_to_document_chunk(metadata: dict[str, Any], content: str) -> DocumentChunk:
    """由 metadata 字段还原出完整的 DocumentChunk（缺失字段用安全默认值兜底）。"""
    return DocumentChunk(
        chunk_id=metadata.get("chunk_id", ""),
        document_id=metadata.get("document_id", ""),
        document_title=metadata.get("document_title", ""),
        section_title=metadata.get("section_title", ""),
        section_path=list(metadata.get("section_path") or []),
        page_start=int(metadata.get("page_start") or 0),
        page_end=int(metadata.get("page_end") or 0),
        content=content,
        source_uri=metadata.get("source_uri", ""),
        source_type=metadata.get("source_type", ""),
    )


def vector_record_to_hit(
    record: VectorRecord,
    score: float,
    source: str,
) -> RetrievalResult:
    """把一条存储记录和它的分数包装成 RetrievalResult。"""
    metadata = dict(record.metadata)
    # chunk_id 优先取 metadata，缺失时退回 record.id
    chunk_id = metadata.get("chunk_id") or record.id
    chunk = metadata_to_document_chunk(metadata, record.text)

    return RetrievalResult(
        chunk_id=chunk_id,
        text=record.text,
        metadata=metadata,
        score=score,
        source=source,
        chunk=chunk,
    )
