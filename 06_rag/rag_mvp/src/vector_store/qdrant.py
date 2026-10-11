"""
Qdrant 向量库实现的占位骨架（尚未完成）。

现状：与 milvus.py 类似，仅保留接口形状，search 尚未接入 Qdrant 客户端，
当前实际不可用。

后续待办：
1. 用 QdrantClient 连接（本地内存或远端服务）；
2. recreate_collection 定义向量维度（BGE-M3 为 1024）与距离度量（Cosine）；
3. add / search 映射到 upsert / search，并把 payload 还原成 metadata；
4. save / load 可视 Qdrant 持久化策略决定是否保留。

在此之前，本项目统一使用 vector_store.local.LocalVectorIndex。
"""

from pathlib import Path
from typing import List

from vector_store import VectorRecord
from vector_store.base import VectorOpt


class QdrantVectorOpt(VectorOpt):

    def __init__(self):
        self.records: List[VectorRecord] = []

    def add(self, vector: VectorRecord):
        self.records.append(vector)

    def count(self) -> int:
        return len(self.records)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[VectorRecord, float]]:
        # TODO: 接入 QdrantClient.search，此处仅为占位，尚未实现
        scored_vectors = []

        scored_vectors.sort(key=lambda x: x[1], reverse=True)

        return scored_vectors[:top_k]

    def save(self, path: str| Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)