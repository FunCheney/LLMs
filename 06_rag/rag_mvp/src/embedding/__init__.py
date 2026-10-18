"""
embedding 包：向量化抽象、BGE-M3 实现、BM25 稀疏索引与分词工具。

对外暴露常用入口，调用方可直接 `from embedding import BGEM3EmbeddingProvider`。

惰性导出（PEP 562）：BGEM3EmbeddingProvider 首次被访问时才加载
sentence-transformers / torch（数百 MB、数十秒），避免 `import embedding`
或仅使用 BM25 时也被拖慢。详见 .workbuddy/memory/MEMORY.md 的项目约定。
"""

from embedding.base import EmbeddingProvider
from embedding.embedding_types import EmbeddingResult
from embedding.bm25 import BM25Index
from embedding.bm25 import build_bm25_index

_LAZY_EXPORTS = {
    "BGEM3EmbeddingProvider": ("embedding.bge_m3", "BGEM3EmbeddingProvider"),
}


def __getattr__(name: str):
    """PEP 562 惰性导出：重模型相关的符号按需加载。"""
    if name in _LAZY_EXPORTS:
        import importlib

        module_path, attr = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value  # 缓存，后续访问走模块字典
        return value
    if name == "VisualBGEEmbeddingProvider":
        # VisualBGE 依赖 third_party/FlagEmbedding/research/visual_bge（多模态扩展，可选）。
        # 该目录当前不存在，顶层 import 会让整个 embedding 包不可用，
        # 因此这里做保护：缺失时置为 None，不影响主链路。
        try:
            from embedding.visual_bge import VisualBGEEmbeddingProvider as _V
        except ModuleNotFoundError:
            _V = None
        globals()["VisualBGEEmbeddingProvider"] = _V
        return _V
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EmbeddingProvider",
    "BGEM3EmbeddingProvider",
    "EmbeddingResult",
    "BM25Index",
    "build_bm25_index",
    "VisualBGEEmbeddingProvider"
]
