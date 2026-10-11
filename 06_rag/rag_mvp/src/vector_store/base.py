from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from vector_store import VectorRecord


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
