"""
本地向量索引：暴力扫描 + 点积，零外部依赖的最小实现。

原理：由于 BGE-M3 输出已做 L2 归一化，两个向量点积即等价于余弦相似度，
因此直接对所有记录算点积、排序、取 Top-K 即可，无需建 ANN 索引。

适用场景：小规模语料（本项目约 425 条 chunk）的原型与测试；
数据量大时应替换为 Milvus / Qdrant 等支持近似最近邻的实现。

持久化：整份 records 用 pickle 落盘到 data/index/local.pkl。

语义说明（对齐 VectorStore 契约）：
- local 后端没有「服务端」概念，records 内存即全部数据，
  save(path) 之前的数据仅存在于内存，进程退出即丢失；
- count()/search() 面向当前内存数据（load 回来的即「已持久化」数据）；
- upsert() 按 id 替换旧记录并立即落盘（需已 save 过或显式给 path）。
"""

from typing import List
import pickle
from pathlib import Path

from vector_store.base import VectorStore
from vector_store.vector_store_types import VectorRecord


def dot_product(a: List[float], b: List[float]) -> float:
    """两个等长向量的点积（归一化后即余弦相似度）。"""
    return sum(x * y for x, y in zip(a, b))


class LocalVectorIndex(VectorStore):
    """内存中的向量索引，支持存取与持久化。"""

    def __init__(self):
        self.records: List[VectorRecord] = []
        # 记录最近一次 save 的路径，供 upsert 后立即持久化使用
        self._persist_path: Path | None = None

    def add(self, vector: VectorRecord):
        """追加一条记录到内存缓冲。"""
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

    def save(self, path: str | Path = "") -> None:
        """将 records 序列化为 pickle 文件（自动创建父目录）。"""
        if not path:
            if self._persist_path is None:
                raise ValueError("save() requires a path on first call")
            path = self._persist_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('wb') as f:
            pickle.dump(self.records, f)
        self._persist_path = path

    def upsert(self, vector: VectorRecord) -> None:
        """按 id 幂等写入：存在则替换，不存在则追加；已落盘过则立即持久化。"""
        for i, record in enumerate(self.records):
            if record.id == vector.id:
                self.records[i] = vector
                break
        else:
            self.records.append(vector)
        if self._persist_path is not None:
            self.save(self._persist_path)

    def drop(self) -> None:
        """清空内存数据；若已落盘则删除索引文件。"""
        self.records.clear()
        if self._persist_path is not None and self._persist_path.exists():
            self._persist_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "LocalVectorIndex":
        """从 pickle 文件读取 records 并构造索引实例。"""
        path = Path(path)
        with path.open('rb') as f:
            records = pickle.load(f)

        index = cls()
        index.records = records
        index._persist_path = path
        return index
