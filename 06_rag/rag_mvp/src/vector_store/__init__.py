"""
vector_store 包：向量存储抽象与多种后端实现。

- base：VectorReader / VectorWriter / VectorStore 接口（读写分离，VectorOpt 为旧名别名）
- config：VectorStoreConfig 双后端配置（store_type=local|milvus，env 兜底）
- factory：open_reader / create_writer（Composition Root，pymilvus 惰性导入）
- local.LocalVectorIndex：本地暴力扫描（默认后端）
- milvus.MilvusVectorOpt：Milvus / Milvus Lite 后端（惰性导出，不加载 pymilvus）
"""

from vector_store.base import VectorOpt, VectorReader, VectorStore, VectorWriter
from vector_store.config import VectorStoreConfig
from vector_store.local import LocalVectorIndex
from vector_store.vector_store_types import VectorRecord

_LAZY_EXPORTS = {
    "MilvusVectorOpt": ("vector_store.milvus", "MilvusVectorOpt"),
    "open_reader": ("vector_store.factory", "open_reader"),
    "create_writer": ("vector_store.factory", "create_writer"),
}


def __getattr__(name: str):
    """PEP 562 惰性导出：milvus/factory 相关符号按需加载（避免 eager 引入 pymilvus）。"""
    if name in _LAZY_EXPORTS:
        import importlib

        module_path, attr = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value  # 缓存，后续访问走模块字典
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VectorRecord",
    "VectorReader",
    "VectorWriter",
    "VectorStore",
    "VectorOpt",
    "VectorStoreConfig",
    "LocalVectorIndex",
    "MilvusVectorOpt",
    "open_reader",
    "create_writer",
]
