from dataclasses import dataclass
from typing import Any

from schemas import DocumentChunk


@dataclass
class RetrievalResult:
    """
        单条检索结果，供 CLI / 问答层使用。
        这里不返回 embedding，因为用户不需要。
        VectorRecord --> RetrievalResult: 相当于对底层的数据做了一个抽象
    """
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    source: str
    chunk: DocumentChunk
    rank: int | None = None
