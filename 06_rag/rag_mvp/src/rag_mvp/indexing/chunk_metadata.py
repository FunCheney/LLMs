"""
Chunk → metadata 的字段投影。

向量库 / BM25 索引只需要「溯源用」的元数据，不需要把整段正文重复存进去
（正文单独放在 VectorRecord.text / payload 里），因此这里统一挑出关键字段：
chunk_id、文档信息、章节路径、页码范围、来源。

注意：索引构建与检索结果映射都依赖这份字段约定，修改需上下游同步。
"""

from typing import Any


def chunk_to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """从 chunks.jsonl 的一行记录中提取可检索/可展示的元数据字段。"""
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "document_title": chunk["document_title"],
        "section_title": chunk.get("section_title", ""),
        "section_path": chunk.get("section_path") or [],
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "source_uri": chunk.get("source_uri", ""),
        "source_type": chunk.get("source_type", ""),
    }
