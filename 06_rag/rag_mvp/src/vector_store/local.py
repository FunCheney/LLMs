"""
本地向量索引：暴力扫描 + 点积，零外部依赖的最小实现。

原理：由于 BGE-M3 输出已做 L2 归一化，两个向量点积即等价于余弦相似度，
因此直接对所有记录算点积、排序、取 Top-K 即可，无需建 ANN 索引。

适用场景：小规模语料（本项目约 444 条 chunk）的原型与测试；
数据量大时应替换为 Milvus / Qdrant 等支持近似最近邻的实现。

持久化：整份 records 用 pickle 落盘到 data/index/local.pkl。
"""

from typing import List
import pickle
from pathlib import Path

from vector_store.base import VectorOpt
from vector_store.vector_store_types import VectorRecord

def dot_product(a: List[float], b: List[float]) -> float:
    """两个等长向量的点积（归一化后即余弦相似度）。"""
    return sum(x * y for x, y in zip(a, b))


class LocalVectorIndex(VectorOpt):
    """内存中的向量索引，支持存取与持久化。"""

    def __init__(self):
        self.records: List[VectorRecord] = []

    def add(self, vector: VectorRecord):
        """追加一条记录。"""
        self.records.append(vector)

    def count(self) -> int:
        """返回记录数。"""
        return len(self.records)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[VectorRecord, float]]:
        """对全部记录打点积分并降序排序，返回 Top-K。"""
        scored_vectors = []
        for record in self.records:
            score = dot_product(query_embedding, record.embedding)

            scored_vectors.append(
                (record, score)
            )

        scored_vectors.sort(key=lambda x: x[1], reverse=True)

        return scored_vectors[:top_k]

    def save(self, path: str| Path) -> None:
        """将 records 序列化为 pickle 文件（自动创建父目录）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('wb') as f:
            pickle.dump(self.records, f)


    @classmethod
    def load(cls, path: str| Path) -> "LocalVectorIndex":
        """从 pickle 文件读取 records 并构造索引实例。"""
        path = Path(path)
        with path.open('rb') as f:
            records = pickle.load(f)

        index = cls()
        index.records = records
        return index
