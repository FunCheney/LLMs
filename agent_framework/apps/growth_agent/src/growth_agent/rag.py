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
import sys
from dataclasses import dataclass
from pathlib import Path

from agent_framework.bootstrap import build_gateway
from agent_framework.config import Settings
from agent_framework.knowledge import GovernancePolicy
from agent_framework.llm.client import LLMGateway
from agent_framework.retrieval import MultiQueryRewriter, QueryRewriter, RetrievalPipeline
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
        # None = 让 sentence-transformers 自动选设备；Mac 上开精排必须显式 cpu，
        # 否则 MPS 在两个线程各持一个 encoder 时会断言失败（见 Settings 注释）。
        device=resolved.retrieval_device,
        # 精排模型只在这里决定"用不用"；是否真的精排由 pipeline 的预算决定。
        reranker="bge" if resolved.retrieval_rerank else None,
    )


def build_pipeline(
    retriever: AsyncRetriever,
    *,
    reranker: AsyncReranker | None = None,
    rewriter: QueryRewriter | None = None,
    governance: GovernancePolicy | None = None,
    settings: Settings | None = None,
) -> RetrievalPipeline:
    """把 `Settings` 的检索段翻译成 `RetrievalPipeline` 参数。

    **唯一一处**做这个翻译。评测脚本（`evals/scripts/retrieval_metrics.py`）要跑
    "同样口径、不同配置"的 A/B，如果它自己再写一遍 `budget_ms=...` 这串参数，
    那是两份可能漂移的真相：改了这里忘了那里，报告上的数字就不再是生产数字。
    所以评测只允许改 `Settings`，装配走同一个函数。

    治理（W6）也在这里挂：它乘的是**分数**，必须与评测口径同源——
    否则 A/B 报告里的排序差异会分不清是检索变强了还是治理降权变了。
    """

    resolved = settings or Settings()
    policy = governance
    if policy is None and resolved.retrieval_governance:
        policy = GovernancePolicy()

    return RetrievalPipeline(
        retriever,
        reranker=reranker,
        rewriter=rewriter,
        governance=policy,
        budget_ms=resolved.retrieval_budget_ms,
        rerank_timeout_ms=resolved.retrieval_rerank_timeout_ms,
        rewrite_timeout_ms=resolved.retrieval_rewrite_timeout_ms,
        rerank_candidates=resolved.retrieval_rerank_candidates,
        recall_multiplier=resolved.retrieval_recall_multiplier,
    )


# 精排的预算自洽线：召回 + ACL + 组装在本机实测 150–300ms，留 500ms 才是
# "明显够"。这不是精确阈值，是报警线——低于它就不用再猜为什么指标没变化。
RERANK_RECALL_HEADROOM_MS = 500.0


def rerank_preflight(settings: Settings) -> list[str]:
    """开了精排但配置不配套时，返回可执行的修复提示（空列表 = 没问题）。

    精排是**唯一一个"开了也可能一行都没执行"的开关**：它的超时取
    `min(rerank_timeout, 剩余预算)`，所以预算给小、超时给短，都会让它在
    `asyncio.wait_for` 里被掐掉，然后静默降级回召回顺序。指标看起来等于
    "没开精排"，很容易被读成"精排无效"——真实原因是参数没给够。

    宁可启动时吵一句，也不要让人对着一份 28/28 全降级的报告猜。
    """
    if not settings.retrieval_rerank:
        return []

    problems: list[str] = []
    needed = settings.retrieval_rerank_timeout_ms + RERANK_RECALL_HEADROOM_MS
    if settings.retrieval_budget_ms < needed:
        problems.append(
            f"检索预算 {settings.retrieval_budget_ms:.0f}ms 不够精排用："
            f"精排超时 {settings.retrieval_rerank_timeout_ms:.0f}ms + 召回余量 "
            f"{RERANK_RECALL_HEADROOM_MS:.0f}ms = 至少 {needed:.0f}ms。"
            f"设 RETRIEVAL_BUDGET_MS={needed:.0f}，否则每次精排都会被预算掐断"
        )

    if settings.retrieval_device is None and sys.platform == "darwin":
        problems.append(
            "macOS 上开精排必须设 RETRIEVAL_DEVICE=cpu：cross-encoder 跑在"
            "工作线程里，torch 的 MPS 后端在多个线程各持一个 encoder 时会断言"
            "失败，进程直接退出（不是异常，抓不到也降不了级）"
        )
    return problems


@dataclass
class RagRuntime:
    """装配结果。持有需要显式释放的资源（线程池、gateway 的连接池）。

    `close()` 必须在 lifespan 里调用：`ThreadPoolExecutor` 不关会让进程
    在退出时挂住（等待线程池空闲），容器里表现为"优雅停止超时"。

    `gateway` 只在开了改写时存在。它有自己的 httpx client，所以另给一个
    `aclose()`：`httpx.AsyncClient.aclose()` 是协程，塞不进同步的 `close()`
    里（`asyncio.run` 在已有事件循环中会直接抛 RuntimeError）。
    """

    # 装配好的检索流水线，整个进程复用一份（主体按请求从 run() 传入）。
    pipeline: RetrievalPipeline
    # 底层召回器，持有线程池 / BM25 索引，close() 里必须显式释放。
    retriever: AsyncRetriever
    # 精排器，也持有线程池；未开精排时为 None，close() 里跳过。
    reranker: AsyncReranker | None = None
    # 只在开了查询改写时存在：改写要打 LLM，所以另带一个 httpx 连接池，
    # 必须由异步的 aclose() 释放（同步的 close() 里跑不了协程）。
    gateway: LLMGateway | None = None
    # 挂在 pipeline 上的改写器（multi_query / hyde），与 gateway 成对出现。
    rewriter: QueryRewriter | None = None

    def close(self) -> None:
        """同步释放。gateway 的连接池由 `aclose()` 负责。"""

        if self.reranker is not None:
            self.reranker.close()
        self.retriever.close()

    async def aclose(self) -> None:
        """异步释放：先关 gateway 的连接池，再走同步那部分。"""

        if self.gateway is not None:
            await self.gateway.aclose()
            self.gateway = None
        self.close()


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
        # 配置不自洽时先把话说清楚：精排"配了但不生效"和"没配"在指标上长得
        # 一模一样，只有启动日志能区分这两件事。
        for problem in rerank_preflight(resolved):
            logger.warning("retrieval.rerank_misconfigured: %s", problem)

        sync_reranker = create_reranker(config)
        if sync_reranker is not None:
            reranker = AsyncReranker(sync_reranker)

    # 改写是**唯一需要 LLM 的检索环节**，所以它带一个 gateway 进来。
    # 没开改写就不建 gateway：启动期不实例化 adapter 是 bootstrap 的纪律，
    # 但没必要时连 gateway 对象都不该存在（否则有人会拿它去打别的调用）。
    gateway: LLMGateway | None = None
    rewriter: QueryRewriter | None = None
    if resolved.retrieval_rewrite:
        gateway = build_gateway(resolved)
        rewriter = MultiQueryRewriter(gateway)
        logger.info("retrieval rewrite enabled: alias=%s", rewriter.alias)

    pipeline = build_pipeline(
        retriever, reranker=reranker, rewriter=rewriter, settings=resolved
    )

    logger.info(
        "retrieval ready: mode=%s store=%s rerank=%s rewrite=%s governance=%s budget=%.0fms",
        resolved.retrieval_mode,
        resolved.vector_store,
        # 开精排时直接打出超时值：`rerank=True budget=800ms` 这种组合本身就是
        # 矛盾（预算装不下精排），让日志一眼可见，比多一行 WARNING 更省事。
        (
            f"on/{resolved.retrieval_rerank_timeout_ms:.0f}ms"
            if resolved.retrieval_rerank
            else "off"
        ),
        resolved.retrieval_rewrite,
        # 治理默认开：它只乘权重不删数据，但"开没开"会影响同一 query 的排序，
        # A/B 时必须能从日志确认两边一致。
        "on" if resolved.retrieval_governance else "off",
        resolved.retrieval_budget_ms,
    )
    return RagRuntime(
        pipeline=pipeline,
        retriever=retriever,
        reranker=reranker,
        gateway=gateway,
        rewriter=rewriter,
    )


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
