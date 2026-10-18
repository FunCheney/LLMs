"""
reranker 包：检索后重排序（精排）。

- base.Reranker：重排序器统一接口；
- bge.BGEReranker：BAAI/bge-reranker-v2-m3 CrossEncoder 实现；
- pipeline.RerankedRetriever：「召回放大 + 重排」包装成普通 Retriever。

使用 PEP 562 惰性导出：import reranker 不会加载 torch，
只有真正访问 BGEReranker 才会触发重依赖（见 bge.py 内部的惰性导入）。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reranker.base import Reranker
    from reranker.bge import BGEReranker
    from reranker.pipeline import RerankedRetriever

__all__ = ["BGEReranker", "Reranker", "RerankedRetriever"]


def __getattr__(name: str):
    """惰性导出：首次访问时才加载对应子模块。"""
    if name == "Reranker":
        from reranker.base import Reranker

        return Reranker
    if name == "BGEReranker":
        from reranker.bge import BGEReranker

        return BGEReranker
    if name == "RerankedRetriever":
        from reranker.pipeline import RerankedRetriever

        return RerankedRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
