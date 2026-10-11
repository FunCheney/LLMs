"""
Milvus 向量库实现的占位骨架（尚未完成）。

现状：目前只实现了与本地方案一致的 add / count 与 save 目录创建，
search 尚未接入 Milvus 客户端（scored_vectors 为空，实际不可用）。

后续待办：
1. 用 MilvusClient 连接（见 example/test_milvus.py），建 collection / schema；
2. add 时写入向量 + metadata，search 时调用 client.search 并映射回 VectorRecord；
3. save 可改为「导出 collection 快照」或直接去掉（DB 自身持久化）。

在此之前，本项目统一使用 vector_store.local.LocalVectorIndex。
"""

from pathlib import Path
from typing import List

from vector_store import VectorRecord
from vector_store.base import VectorOpt


class MilvusVectorOpt(VectorOpt):

    def __init__(self):
        self.records: List[VectorRecord] = []

    def add(self, vector: VectorRecord):
        self.records.append(vector)

    def count(self) -> int:
        return len(self.records)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[VectorRecord, float]]:
        # TODO: 接入 MilvusClient.search，此处仅为占位，尚未实现
        scored_vectors = []

        scored_vectors.sort(key=lambda x: x[1], reverse=True)

        return scored_vectors[:top_k]

    def save(self, path: str| Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)