"""
向量存储的通用数据结构。

VectorRecord 是「一条可被检索的记录」的统一表示：
- id：通常等于 chunk_id；
- text：原始文本，供展示 / BM25 / 生成上下文使用；
- metadata：章节、页码、来源等溯源信息；
- embedding：稠密向量（BM25 场景下可为空列表）。

本地向量库、Milvus、Qdrant 三种实现共用这一结构，方便互相替换。
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class VectorRecord:
    id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]