"""检索层（W5）：召回 → 融合 → 精排 → 组装上下文。

- `base.py`          Retriever / Reranker / QueryRewriter 抽象
- `fusion.py`        RRF 融合（名次加权，不看原始分数量纲）
- `pipeline.py`      串成一条可评测的链路 + 超时降级 + 预算递减
- `query_rewrite.py` multi-query / HyDE（走 gateway 的 cheap 档，默认关）

后端实现（BM25 / 向量库 / cross-encoder）不在本包内：框架只定义协议，
实现由 apps 层注入，否则框架包会被 torch 绑死。

约定见 `base.py` 顶部：空结果 ≠ 失败。
"""

from .base import QueryRewriter, Reranker, Retriever
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion, renumber
from .pipeline import (
    DEFAULT_BUDGET_MS,
    DEGRADE_BUDGET,
    DEGRADE_RERANK,
    DEGRADE_REWRITE,
    RetrievalDiagnostics,
    RetrievalOutcome,
    RetrievalPipeline,
)
from .query_rewrite import (
    DEFAULT_N,
    HydeRewriter,
    MultiQueryRewriter,
    RewriteUsage,
    parse_queries,
)

__all__ = [
    "DEFAULT_BUDGET_MS",
    "DEFAULT_N",
    "DEFAULT_RRF_K",
    "DEGRADE_BUDGET",
    "DEGRADE_RERANK",
    "DEGRADE_REWRITE",
    "HydeRewriter",
    "MultiQueryRewriter",
    "QueryRewriter",
    "Reranker",
    "RetrievalDiagnostics",
    "RetrievalOutcome",
    "RetrievalPipeline",
    "Retriever",
    "RewriteUsage",
    "parse_queries",
    "reciprocal_rank_fusion",
    "renumber",
]
