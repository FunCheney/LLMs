from typing import Any


def chunk_to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
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
