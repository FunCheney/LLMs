"""产品侧的检索装配——**唯一的装配点**。

依赖方向是单向的：

    growth_agent  →  rag_mvp  →  agent_framework.knowledge / retrieval（只是类型）

`agent_framework` 本身**不依赖** rag_mvp（它只定义 Chunk / ScoredChunk /
Retriever 协议，保持零 ML 依赖）。所以"换个向量库""换 embedding 供应商"
"关掉精排"这些事，全部只改本文件，不动框架、不动检索实现。

装配放在启动阶段调用一次：`create_base_retriever` 会加载 bge-m3（本机实测约
18–20 秒）。放进请求路径意味着第一个请求吃掉整个加载时间，p95 被一次冷启动
污染，之后所有延迟结论都不可信。

**精排与召回分开装**：`AsyncReranker` 单独持有，交给 `RetrievalPipeline`
按剩余预算施加超时。合成一个 `RerankedRetriever` 的话，精排超时后拿不到
RRF 顺序，只能整个失败——那是"demo RAG"的做法。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from agent_framework.config import Settings
from agent_framework.retrieval import RetrievalPipeline
from rag_mvp.contrib.agent_framework import (
    DEFAULT_MAX_WORKERS,
    AsyncReranker,
    AsyncRetriever,
    build_async_retriever,
    configure_local_threads,
)
from rag_mvp.retrieval.factory import (
    RetrieverConfig,
    create_base_retriever,
    create_reranker,
)

logger = logging.getLogger(__name__)

# 仓库布局：<repo>/agent_framework/apps/growth_agent/src/growth_agent/rag.py
# parents[5] = <repo>。装成 wheel 后这个推导不成立，此时必须用 RAG_ROOT 指定。
_REPO_ROOT_DEPTH = 5


def _default_rag_root() -> Path:
    """定位 rag_mvp 的数据目录（语料与索引由它自己维护）。"""
    env = os.getenv("RAG_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    parts = Path(__file__).resolve().parents
    if len(parts) > _REPO_ROOT_DEPTH:
        candidate = parts[_REPO_ROOT_DEPTH] / "06_rag" / "rag_mvp"
        if (candidate / "data" / "index").is_dir():
            return candidate

    raise FileNotFoundError(
        "找不到 rag_mvp 的索引目录。设置环境变量 RAG_ROOT 指向 rag_mvp 仓库根"
        f"（当前推导路径：{parts[_REPO_ROOT_DEPTH] if len(parts) > _REPO_ROOT_DEPTH else 'N/A'}）"
    )


def retriever_config(settings: Settings | None = None) -> RetrieverConfig:
    """按 Settings 组装检索器配置。

    **没有任何硬编码路径**：索引目录从 RAG_ROOT 或仓库布局推导，
    连接串从 env 读（`PGVECTOR_DSN`），密钥不进代码。
    """
    resolved = settings or Settings()
    root = _default_rag_root()

    dsn = resolved.pgvector_dsn.get_secret_value() if resolved.pgvector_dsn else None

    return RetrieverConfig(
        mode=resolved.retrieval_mode,
        dense_index_path=root / "data" / "index" / "local.pkl",
        bm25_index_path=root / "data" / "index" / "bm25.pkl",
        store_type=resolved.vector_store,
        pgvector_dsn=dsn,
        # 精排模型只在这里决定"用不用"；是否真的精排由 pipeline 的预算决定。
        reranker="bge" if resolved.retrieval_rerank else None,
    )


@dataclass
class RagRuntime:
    """装配结果。持有需要显式释放的资源（线程池）。

    `close()` 必须在 lifespan 里调用：`ThreadPoolExecutor` 不关会让进程
    在退出时挂住（等待线程池空闲），容器里表现为"优雅停止超时"。
    """

    pipeline: RetrievalPipeline
    retriever: AsyncRetriever
    reranker: AsyncReranker | None = None

    def close(self) -> None:
        if self.reranker is not None:
            self.reranker.close()
        self.retriever.close()


def build_runtime(settings: Settings | None = None) -> RagRuntime:
    """构造检索链路。同步、慢（加载模型），**只在启动阶段调一次**。"""
    resolved = settings or Settings()
    config = retriever_config(resolved)

    # 不设的话每个工作线程内部还会开 OMP 线程，2 线程 × 8 核 = 16 个计算线程
    # 互抢缓存，表现为"并发越高越慢"，且很难归因。
    configure_local_threads(1)

    try:
        inner = create_base_retriever(config)
    except ImportError as exc:
        raise ImportError(
            "当前模式需要本地模型（sentence-transformers / torch）。"
            "执行 `uv sync --extra local` 后重试；"
            "或设 RETRIEVAL_MODE=sparse 只用 BM25（不需要模型）。"
        ) from exc

    retriever = AsyncRetriever(
        inner,
        max_workers=DEFAULT_MAX_WORKERS,
        # 这里的超时是**单次召回**的上限，整链路预算在 pipeline 里。
        # 两个都要有：召回卡住时不该继续等精排的预算。
        timeout_s=max(resolved.retrieval_budget_ms / 1000.0, 1.0),
    )

    reranker: AsyncReranker | None = None
    if resolved.retrieval_rerank:
        sync_reranker = create_reranker(config)
        if sync_reranker is not None:
            reranker = AsyncReranker(sync_reranker)

    pipeline = RetrievalPipeline(
        retriever,
        reranker=reranker,
        budget_ms=resolved.retrieval_budget_ms,
        rerank_timeout_ms=resolved.retrieval_rerank_timeout_ms,
        rewrite_timeout_ms=resolved.retrieval_rewrite_timeout_ms,
        rerank_candidates=resolved.retrieval_rerank_candidates,
        recall_multiplier=resolved.retrieval_recall_multiplier,
    )

    logger.info(
        "retrieval ready: mode=%s store=%s rerank=%s budget=%.0fms",
        resolved.retrieval_mode,
        resolved.vector_store,
        resolved.retrieval_rerank,
        resolved.retrieval_budget_ms,
    )
    return RagRuntime(pipeline=pipeline, retriever=retriever, reranker=reranker)


def build_retriever(
    config: RetrieverConfig | None = None,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout_s: float | None = None,
) -> AsyncRetriever:
    """只构造召回层（给脚本与冒烟用，不装配 pipeline）。"""
    configure_local_threads(1)
    settings = Settings()
    return build_async_retriever(
        config or retriever_config(settings),
        max_workers=max_workers,
        timeout_s=timeout_s or max(settings.retrieval_budget_ms / 1000.0, 1.0),
    )
