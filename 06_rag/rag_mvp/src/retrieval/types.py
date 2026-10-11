"""
检索层的数据结构。

RetrievalResult 是「检索器的统一输出」。相比底层的 VectorRecord，
它少了一个 embedding 字段（用户/上层不需要向量），
并补充了 score、source、rank、以及还原好的 DocumentChunk：

    VectorRecord --> RetrievalResult: 相当于对底层的数据做了一个抽象
"""

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
    source: str                 # 来源检索器：dense / sparse / hybrid
    chunk: DocumentChunk        # 还原出的完整 Chunk（含章节、页码、来源）
    rank: int | None = None     # 在结果列表中的名次（从 1 开始）
