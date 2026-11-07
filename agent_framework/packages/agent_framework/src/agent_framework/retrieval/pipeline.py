"""检索流水线（W5 D3）：把召回、精排、过滤、治理串成一条可评测的链路。

三条纪律，都是"踩过才知道"的：

1. **只有召回失败才允许抛错。** 召回失败是系统故障（索引缺失、库连不上），
   伪装成空结果会被 W4 的归因当成"语料缺失"，于是人跑去补三天语料，而真实
   原因是索引没建。精排失败只是失去排序增益，回退到召回顺序仍然可用——所以
   它必须降级，不能把整个请求打死。
2. **预算按剩余时间递减，不是每步各给一份。** 三步各配 800ms 时总延迟是
   2.4s，而配置看起来写的是"预算 800ms"。这个差在压测里会被误读成
   "某一步很慢"，实际是三步都在用满自己的额度。
3. **降级必须同时出现在返回值和日志里。** 只写日志，调用方无法把"这次答案
   质量低"归因到"精排超时"；只返回值，事后没人知道降级发生过。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ..knowledge.base import Chunk, ScoredChunk
from .base import QueryRewriter, Reranker, Retriever
from .fusion import reciprocal_rank_fusion, renumber

logger = logging.getLogger(__name__)

# 默认总预算。本地 bge-m3 在 425 条 chunk 上召回约 150–300ms，
# cross-encoder 精排 20 条约 2.5s——所以"开精排"必须显式配预算，
# 用这个默认值跑精排会立刻降级（这是刻意的：默认档不能悄悄拖慢服务）。
DEFAULT_BUDGET_MS = 800.0

# 精排单独的上限。整链路预算再大，单个精排也不该吃掉超过这个数。
# 8000：本机 CPU 上 20 条 cross-encoder p50 4.7s / p95 5.4s。给 3000 的后果
# 不是"慢一点"，而是**每一次都超时降级**——精排代码一行没执行，日志里全是
# rerank_degraded，看起来像"精排没用"。这是配置问题伪装成算法结论。
DEFAULT_RERANK_TIMEOUT_MS = 8000.0

# 改写单独的上限。改写是一次 LLM 调用，比本地模型更容易抖。
DEFAULT_REWRITE_TIMEOUT_MS = 2500.0

DEGRADE_REWRITE = "rewrite_degraded"
DEGRADE_RERANK = "rerank_degraded"
DEGRADE_BUDGET = "budget_exhausted"


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """本次检索发生了什么。给日志、trace 与 A/B 报告用。

    这些字段存在的唯一目的：让"这次没召回/答偏了"能被归因到具体环节，
    而不是笼统地记一笔"检索失败"。
    """

    # 用户原始查询（改写前）。报告里靠它定位"是哪一问出了问题"。
    query: str
    # 实际用于召回的查询（第一条永远是原查询）。改写没生效时长度为 1。
    queries: tuple[str, ...]
    # 融合后的候选数（截断前）。
    candidates: int
    # ACL 过滤掉的条数。**只记数量不记内容**：被滤掉的正是无权查看的文本，
    # 把它写进日志等于用日志绕过权限。
    filtered_out: int
    # 精排是否真的跑成功了。False 有两种含义（没配 reranker / 超时降级回退），
    # 要靠 `degraded` 区分——只看这个布尔值会把"没开精排"误读成"精排失败"。
    reranked: bool
    # 降级标记集合（rewrite_degraded / rerank_degraded / budget_exhausted）。
    # 非空即"本次结果不是完整链路产出的"，A/B 对比时必须先把这些样本剔掉或单列。
    degraded: tuple[str, ...] = ()
    # 本次是否真的执行了 ACL 过滤。`filtered_out=0` 有两种可能（"全都有权限"与
    # "根本没过滤"），只靠条数分不出来，而这两件事在安全审计里差别巨大。
    acl_applied: bool = False
    # 改写耗时单独记：它是一次 LLM 调用，比另外两段贵一个量级。
    # 不记的话，"总耗时长"只能归因到"检索慢"，而真实原因可能是上游在抖。
    rewrite_ms: float = 0.0
    # 召回耗时。多路改写时是并发执行的墙钟时间，不是各路之和。
    recall_ms: float = 0.0
    # 精排耗时。降级时为已消耗的时间（到超时为止），不是 0——
    # "超时花掉的 8 秒"也是延迟的一部分，藏掉它会让 p95 对不上。
    rerank_ms: float = 0.0
    # 整链路墙钟耗时（含预算等待外的所有环节），与 budget_ms 对照看是否超支。
    total_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.degraded


@dataclass(frozen=True)
class RetrievalOutcome:
    # 最终交给上层组装 context 的块，名次已由 renumber 重编为 1..n。
    # 空列表是**合法结果**（没召回 / 全被 ACL 拦住），不是异常，见模块头约定。
    chunks: list[ScoredChunk] = field(default_factory=list)
    # 本次链路诊断。为 None 只用于测试桩；正常 run() 一定带，
    # 因为"降级了但没记录"会让 A/B 报告把低质量答案误归因到算法。
    diagnostics: RetrievalDiagnostics | None = None


class RetrievalPipeline:
    """召回 → 精排 → 过滤 → 治理 → 截断。

    各段都是可选的，装配在 apps 层完成（框架不认识具体后端）。

        pipeline = RetrievalPipeline(
            retriever=async_retriever,          # 必需
            reranker=async_reranker,            # 可选，开精排时传
            rewriter=multi_query_rewriter,      # 可选，开改写时传
            acl_check=make_acl_check(principal),  # 可选，W6
            budget_ms=settings.retrieval_budget_ms,
        )
        outcome = await pipeline.run("品牌调性怎么定", top_k=8)
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
        acl_check: Callable[[Chunk], bool] | None = None,
        governance: Callable[[Sequence[ScoredChunk]], list[ScoredChunk]] | None = None,
        budget_ms: float = DEFAULT_BUDGET_MS,
        rerank_timeout_ms: float = DEFAULT_RERANK_TIMEOUT_MS,
        rewrite_timeout_ms: float = DEFAULT_REWRITE_TIMEOUT_MS,
        recall_multiplier: float = 3.0,
        rerank_candidates: int = 20,
    ) -> None:
        if budget_ms <= 0:
            raise ValueError("budget_ms 必须大于 0；不要用 0 表示'不限'，用一个大数")
        # 必需：唯一真正碰向量库/BM25 的对象，失败要抛（空结果 ≠ 失败）。
        self._retriever = retriever
        # 可选精排器。None = 不开精排，用召回/RRF 顺序；超时或报错只降级不抛。
        self._reranker = reranker
        # 可选改写器。None = 只用原查询；它的成本与降级单独记在 diagnostics。
        self._rewriter = rewriter
        # 构造期的 ACL 过滤器（默认 None）。真正的主体按请求从 run(acl_check=)
        # 传入，因为 pipeline 是单例而每个请求的用户不同。
        self._acl_check = acl_check
        # 治理（质量降权 / 过期降权 / 冲突标记）在精排之后应用：
        # 精排会用 logit 覆盖 score，任何在它之前乘的权重都会被抹掉。
        self._governance = governance
        # 整条链路的总墙钟预算（ms）。所有分段共用同一个递减的 deadline，
        # 不是每步各给一份——否则配置写 800ms、实际可能跑 2.4s。
        self._budget_ms = budget_ms
        # 精排单步上限（默认 8000ms）。本机 CPU 上 20 条 cross-encoder
        # p50≈4.7s，给小了会 100% 超时降级，看起来像"精排没用"。
        self._rerank_timeout_ms = rerank_timeout_ms
        # 改写单步上限（默认 2500ms）。一次 LLM 调用，比本地模型更容易抖；
        # 超时就用原查询继续，绝不让整个请求失败。
        self._rewrite_timeout_ms = rewrite_timeout_ms
        # 召回放大倍数：多取一些补偿 ACL 过滤与去重造成的损耗，
        # 避免"有权限的结果太少、context 填不满"。
        self._recall_multiplier = recall_multiplier
        # 送进精排的候选上限（默认 20）。它直接决定精排耗时，
        # 调大前先看本机 p95，不要只看召回指标。
        self._rerank_candidates = rerank_candidates

    @property
    def name(self) -> str:
        return f"pipeline({self._retriever.name})"

    async def run(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, object] | None = None,
        acl_check: Callable[[Chunk], bool] | None = None,
    ) -> RetrievalOutcome:
        """跑一次检索。

        `acl_check` 是**按请求**覆盖构造期过滤器的唯一入口。ACL 的主体来自调用方
        （每个请求不同），而 pipeline 是启动阶段建一次的单例——所以主体不能进
        构造函数，只能从 run 传。传 None 时沿用构造期配置（可能是"不启用"）。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        started = time.perf_counter()
        deadline = started + self._budget_ms / 1000.0
        degraded: list[str] = []

        rewrite_started = time.perf_counter()
        queries = await self._rewrite(query, deadline=deadline, degraded=degraded)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000.0

        recall_started = time.perf_counter()
        # 多取一些：后面还有 ACL 过滤与去重会砍掉一部分，
        # 只取 top_k 会在"有权限的结果太少"时直接给不满。
        recall_k = max(int(top_k * self._recall_multiplier), top_k)
        candidates = await self._recall(queries, recall_k=recall_k, filters=filters)
        recall_ms = (time.perf_counter() - recall_started) * 1000.0

        rerank_ms = 0.0
        reranked = False
        filtered_out = 0
        # ACL 在**精排之前**。两个理由，第二个比第一个重要得多：
        #
        # 1. 算力：精排 20 条 cross-encoder 在本机要 4.7s。若其中一半是调用方无权
        #    查看的块，这 2 秒是纯浪费——而且把"无权限文档多"变成了"检索变慢"，
        #    运维会去查模型而不是查权限。
        # 2. 侧信道：精排耗时与降级行为会随候选里无权限块的数量变化。放在精排之后，
        #    一个没有任何权限的调用方也能从 p95 变化推断"库里有多少他看不到的文档"。
        #    过滤在前，后面所有环节只看得见可见数据。
        #
        # 代价：精排池可能因为过滤而变小。这由 `recall_multiplier` 的 over-fetch 补偿
        #    （见上面 recall_k 的注释），而不是靠"先精排再过滤"来凑条数。
        checker = acl_check if acl_check is not None else self._acl_check
        if checker is not None:
            kept = [scored for scored in candidates if checker(scored.chunk)]
            filtered_out = len(candidates) - len(kept)
            candidates = kept

        if self._reranker is not None and len(candidates) > 1:
            remaining_ms = (deadline - time.perf_counter()) * 1000.0
            if remaining_ms <= 0:
                degraded.append(DEGRADE_BUDGET)
            else:
                rerank_started = time.perf_counter()
                reranked_result = await self._rerank(candidates, query, deadline)
                rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
                if reranked_result is None:
                    # 回退到候选顺序（也就是 RRF 的顺序）。**不重排、不截断**：
                    # 此时"能给出结果"比"排序最优"重要。
                    degraded.append(DEGRADE_RERANK)
                else:
                    candidates = reranked_result
                    reranked = True

        if self._governance is not None and candidates:
            candidates = list(self._governance(candidates))

        chunks = renumber(candidates[:top_k])

        # 过滤后为空不是故障（空结果 ≠ 失败）：可能是语料里确实没有，
        # 也可能是全被 ACL 拦住。两种都不该抛错，但都必须能被看见。
        if filtered_out and not chunks:
            logger.warning(
                "retrieval.acl_filtered_all filtered_out=%d top_k=%d query=%r",
                filtered_out,
                top_k,
                query[:60],
            )

        diagnostics = RetrievalDiagnostics(
            query=query,
            queries=queries,
            candidates=len(candidates),
            filtered_out=filtered_out,
            reranked=reranked,
            degraded=tuple(degraded),
            acl_applied=checker is not None,
            rewrite_ms=round(rewrite_ms, 2),
            recall_ms=round(recall_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            total_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        if degraded:
            logger.warning(
                "retrieval.degraded stages=%s budget_ms=%.0f total_ms=%.0f query=%r",
                ",".join(degraded),
                self._budget_ms,
                diagnostics.total_ms,
                query[:60],
            )
        return RetrievalOutcome(chunks=chunks, diagnostics=diagnostics)

    async def _rewrite(
        self,
        query: str,
        *,
        deadline: float,
        degraded: list[str],
    ) -> tuple[str, ...]:
        if self._rewriter is None:
            return (query,)

        timeout_s = min(
            self._rewrite_timeout_ms / 1000.0,
            max(deadline - time.perf_counter(), 0.0),
        )
        if timeout_s <= 0:
            degraded.append(DEGRADE_BUDGET)
            return (query,)

        try:
            generated = await asyncio.wait_for(
                self._rewriter.rewrite(query), timeout=timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - 改写任何失败都降级
            # 改写是"锦上添花"：失败要用原查询继续，而不是让整个请求失败。
            # 这里不区分超时与报错，因为两者的处理方式完全相同，
            # 区分只会在日志里多一个分支、少一分辨识度。
            logger.warning("retrieval.rewrite_failed err=%r query=%r", exc, query[:60])
            degraded.append(DEGRADE_REWRITE)
            return (query,)

        # 协议要求第一条是原查询；实现没遵守时在这里补上，
        # 保证"改写变差"永远不会比"不改写"更糟。
        ordered: list[str] = [query]
        for item in generated:
            text = item.strip()
            if text and text not in ordered:
                ordered.append(text)
        return tuple(ordered)

    async def _recall(
        self,
        queries: Sequence[str],
        *,
        recall_k: int,
        filters: Mapping[str, object] | None,
    ) -> list[ScoredChunk]:
        """多查询并发召回 + 融合。

        并发不是优化而是必要性：改写出 3 条查询后串行执行，延迟直接 ×3，
        而这三路之间没有任何依赖。
        """
        rankings = await asyncio.gather(
            *(
                self._retriever.retrieve(query, top_k=recall_k, filters=filters)
                for query in queries
            )
        )
        if len(rankings) == 1:
            # 单路不必过 RRF：RRF 会把原始分数换成 1/(k+rank)，
            # 白丢掉"差多少"的信息，而下游精排/阈值可能要它。
            return list(rankings[0])
        return reciprocal_rank_fusion(rankings)

    async def _rerank(
        self,
        candidates: Sequence[ScoredChunk],
        query: str,
        deadline: float,
    ) -> list[ScoredChunk] | None:
        """精排。返回 None 表示降级（超时或报错），调用方保留候选顺序。"""
        assert self._reranker is not None
        pool = list(candidates[: max(self._rerank_candidates, 1)])
        timeout_s = min(
            self._rerank_timeout_ms / 1000.0,
            max(deadline - time.perf_counter(), 0.0),
        )
        if timeout_s <= 0:
            return None
        try:
            return list(
                await asyncio.wait_for(
                    self._reranker.rerank(query, pool, top_k=len(pool)),
                    timeout=timeout_s,
                )
            )
        except TimeoutError:
            # 超时只保证调用方脱身：线程池里那批 cross-encoder 计算仍在跑完，
            # 所以"降级了但 CPU 仍然满"是预期现象，不是 bug。
            logger.warning("retrieval.rerank_timeout timeout_s=%.2f", timeout_s)
            return None
        except Exception as exc:  # noqa: BLE001 - 精排任何失败都降级
            logger.warning("retrieval.rerank_failed err=%r", exc)
            return None
