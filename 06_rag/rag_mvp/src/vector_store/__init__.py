"""
vector_store 包：向量存储抽象与多种后端实现。

- local.LocalVectorIndex：本地暴力扫描（当前默认，见 README/road_map）
- milvus.MilvusVectorOpt / qdrant.QdrantVectorOpt：外部向量库占位骨架
"""

from vector_store.local import LocalVectorIndex
from vector_store.vector_store_types import VectorRecord
from vector_store.base import VectorOpt


__all__ = [
    "VectorRecord",
    "LocalVectorIndex"
]