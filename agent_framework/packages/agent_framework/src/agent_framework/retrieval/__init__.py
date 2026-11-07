"""检索层（W5–W6）：召回 → 融合 → 精排 → ACL → 治理 → 组装上下文 → 引用。

- `base.py`              Retriever / Reranker / QueryRewriter 抽象
- `fusion.py`            RRF 融合（名次加权，不看原始分数量纲）
- `pipeline.py`          串成一条可评测的链路 + 超时降级 + 预算递减
- `query_rewrite.py`     multi-query / HyDE（走 gateway 的 cheap 档，默认关）
- `acl.py`               ACL 过滤（W6 D3，默认拒绝未标注，主体按请求传入）
- `context_assemble.py`  去冗余 / 预算裁剪 / 首尾放置（W6 D4，编号之前完成）

后端实现（BM25 / 向量库 / cross-encoder）不在本包内：框架只定义协议，
实现由 apps 层注入，否则框架包会被 torch 绑死。

约定见 `base.py` 顶部：空结果 ≠ 失败。
"""

from .acl import (
    ACLDecision,
    ACLPolicy,
    Principal,
    filter_visible,
    make_acl_check,
)
from .base import QueryRewriter, Reranker, Retriever
from .context_assemble import (
    DEFAULT_DEDUPE_THRESHOLD,
    DEFAULT_TOKEN_BUDGET,
    AssembledContext,
    assemble,
    deduplicate,
    estimate_tokens,
    order_for_attention,
    similarity,
    trim_to_budget,
)
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
    "ACLDecision",
    "ACLPolicy",
    "DEFAULT_BUDGET_MS",
    "DEFAULT_DEDUPE_THRESHOLD",
    "DEFAULT_N",
    "DEFAULT_RRF_K",
    "DEFAULT_TOKEN_BUDGET",
    "DEGRADE_BUDGET",
    "DEGRADE_RERANK",
    "DEGRADE_REWRITE",
    "AssembledContext",
    "HydeRewriter",
    "MultiQueryRewriter",
    "Principal",
    "QueryRewriter",
    "Reranker",
    "RetrievalDiagnostics",
    "RetrievalOutcome",
    "RetrievalPipeline",
    "Retriever",
    "RewriteUsage",
    "assemble",
    "deduplicate",
    "estimate_tokens",
    "filter_visible",
    "make_acl_check",
    "order_for_attention",
    "parse_queries",
    "reciprocal_rank_fusion",
    "renumber",
    "similarity",
    "trim_to_budget",
]
