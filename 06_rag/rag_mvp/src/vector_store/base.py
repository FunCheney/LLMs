"""
向量库抽象接口：读写分离（接口隔离原则），屏蔽底层存储差异（本地 / Milvus / 其他）。

设计要点：
- 检索层只依赖 VectorReader（search / count），构建层只依赖 VectorWriter
  （add / save / upsert / drop），上层切换后端时无需改动业务逻辑。
- 统一语义约定（所有后端必须遵守，否则抽象会漏实现）：
    * add()      只进内存缓冲，不保证立即可检索；
    * save()     把缓冲数据持久化（local 落盘 pickle / milvus upsert+flush）；
    * count()    统计「已持久化」的数据，而不是内存缓冲；
    * search()   面向「已持久化」的数据检索；
    * upsert()   按主键（chunk_id）幂等写入，重复执行不产生重复数据；
    * drop()     清空后端数据（local 清内存并删除索引文件，milvus 重建 collection）。
- VectorStore = Reader + Writer 组合接口；VectorOpt 为旧名别名，保留向后兼容。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from vector_store.vector_store_types import VectorRecord


class VectorReader(ABC):
    """读侧接口：检索层（DenseRetriever）只依赖它。"""

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[VectorRecord, float]]:
        """向量检索 Top-K，返回 (记录, 相似度分数) 列表，按分数降序。"""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """返回已持久化的记录数。"""
        raise NotImplementedError


class VectorWriter(ABC):
    """写侧接口：索引构建层只依赖它。"""

    @abstractmethod
    def add(self, vector: VectorRecord):
        """追加一条记录到写缓冲（不保证立即可检索，需 save 持久化）。"""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path = "") -> None:
        """持久化缓冲数据；local 后端落盘到 path，DB 后端忽略 path（自身持久化）。"""
        raise NotImplementedError

    @abstractmethod
    def upsert(self, vector: VectorRecord) -> None:
        """按主键幂等写入一条记录并立即持久化（重复写入不产生重复数据）。"""
        raise NotImplementedError

    @abstractmethod
    def drop(self) -> None:
        """清空后端全部数据（用于重建索引）。"""
        raise NotImplementedError


class VectorStore(VectorReader, VectorWriter):
    """读写合一的完整向量库接口（大多数后端同时实现两侧）。"""


# 旧名别名：保留向后兼容（历史代码/文档中的 VectorOpt 仍可用）。
# 新代码请按需依赖 VectorReader 或 VectorWriter。
VectorOpt = VectorStore
