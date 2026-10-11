"""
向量库抽象接口：屏蔽底层存储差异（本地 / Milvus / Qdrant）。

上层检索器只依赖 VectorOpt 的 add / count / search / save 四个方法，
切换后端时无需改检索逻辑。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from vector_store.vector_store_types import VectorRecord


class VectorOpt(ABC):
    """向量操作接口"""

    @abstractmethod
    def add(self, vector: VectorRecord):
        """
        添加
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        统计
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[VectorRecord, float]]:
        """
        查询
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        保存
        """
        raise NotImplementedError
