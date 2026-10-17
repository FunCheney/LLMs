"""
retrieval 包：统一检索接口与三种实现（稠密 / 稀疏 / 混合）。

对外暴露检索器、结果类型、配置与工厂方法。

注意：这里用 ``__getattr__`` 做惰性导出——``retrieval.factory`` 会连带加载
sentence-transformers/torch（很重），如果像普通包那样在 __init__ 顶层 ``import``，
那么连 ``from retrieval.types import RetrievalResult`` 这类纯类型导入都要等几十秒。
改成惰性后，只有真正访问到工厂相关符号（即真正要跑检索）时才加载 torch。
"""

__all__ = [
    "Retriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "RetrieverConfig",
    "create_retriever",
    "reciprocal_rank_fusion",
]

# name -> (module, attribute)：仅在首次访问时按需 import
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Retriever": ("retrieval.base", "Retriever"),
    "SparseRetriever": ("retrieval.spare_retriever", "SparseRetriever"),
    "RetrieverConfig": ("retrieval.factory", "RetrieverConfig"),
    "create_retriever": ("retrieval.factory", "create_retriever"),
    "HybridRetriever": ("retrieval.hybrid", "HybridRetriever"),
    "DenseRetriever": ("retrieval.retriever", "DenseRetriever"),
    "reciprocal_rank_fusion": ("retrieval.rrf", "reciprocal_rank_fusion"),
    "RetrievalResult": ("retrieval.types", "RetrievalResult"),
}


def __getattr__(name: str):
    """模块级惰性导入：首次访问 __all__ 里的符号时才真正 import 对应子模块。"""
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr)
        # 缓存到模块全局，避免重复 import
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
