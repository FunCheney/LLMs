"""W5 D2 · 检索质量指标：Recall@K / MRR / nDCG@K，**口径走 `RetrievalPipeline`**。

为什么不能直接测 `Retriever.retrieve()`
--------------------------------------
召回层的 top-k 不是用户看到的 top-k。生产链路上还夹着：超取
（`recall_multiplier`）→ ACL 过滤 → 精排（只重排前 `rerank_candidates` 条，
剩下的**丢弃**）→ 截断 `top_k`。直接测召回层算出的是"召回上界"，
用户拿到的顺序可能差很多。

这不是理论洁癖。本周就因为口径不一致得出过一个错结论：在 rag_mvp 里
`RerankedRetriever` 是 `candidate_k = top_k×10 = 100` 条**全局重排**，而框架
pipeline 是 24 条里重排前 20 条、丢掉 4 条。两套口径下"精排好不好"是两个问题，
混着讲就会把某套实现的缺陷记到"精排"这个技术选项头上。

三条口径纪律
------------
1. **指标只从 `outcome.chunks` 算**，不看 retriever 的原始输出。
2. **延迟只用 `outcome.diagnostics`**（`recall_ms` / `rerank_ms` / `total_ms`），
   不在这里 `time.perf_counter()`——自己计时会长出第二条延迟口径，W2 的
   `first_token_ms` 曲线立刻对不上。
3. **降级的样本不能当正常样本统计**。预算不足时精排会降级回召回顺序，
   此时"精排档"的指标其实等于"无精排档"——报告里会标 `!` 并计数，避免
   把"预算没给够"读成"精排没用"。

用法
----
    # 生产口径（Settings 里的预算/超取/候选数），三档对比
    uv run python evals/scripts/retrieval_metrics.py --configs dense,hybrid,dense+rerank

    # 复现 rag_mvp 的评测口径（top_k=10 / 超取 10 / 全局重排 100 条）
    uv run python evals/scripts/retrieval_metrics.py \
        --preset ragmvp --configs dense,dense+rerank --report /tmp/ragmvp_4way.json

    # 只跑纯数学自检（不加载任何模型，毫秒级）
    uv run python evals/scripts/retrieval_metrics.py --self-test

退出码：0 完成 / 1 有配置出现降级（数字不可信，需调预算重跑）/ 2 环境不满足。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # `evals.metrics` 与 `growth_agent.*` 都靠它。放在文件顶部而不是 main 里，
    # 因为 `_load_queries` 在模块导入后才被调用，但 sys.path 越早越不容易漏。
    sys.path.insert(0, str(REPO_ROOT))

from evals.metrics import percentile  # noqa: E402  (线性插值分位，与 W1 基线同算法)

# 报告哪些 Recall 档位。1/5/10 是 W5 检查点用的，3 是 rag_mvp 历史报告用的，
# 保留它才能和 `data/eval/reports/golden_w5_4way.json` 直接对照。
K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

DEFAULT_TOP_K = 10  # 算 Recall@10 / nDCG@10 的前提；生产 top_k 是 8，评测要 10

# 精排一个 20 条候选的 cross-encoder，本机约 2.5s。生产预算 800ms 必降级，
# 所以"测精排质量"时必须显式调大预算——这不是放宽标准，是让被测对象真的运行。
RERANK_BUDGET_MS = 6000.0

# rag_mvp 的历史口径，用来和已存报告对齐（见模块 docstring 第二条纪律的反例）
RAGMVP_PRESET = {
    "top_k": 10,
    "recall_multiplier": 10.0,
    "rerank_candidates": 100,
    "budget_ms": 120_000.0,
}


# --------------------------------------------------------------------------- #
# 纯数学（无依赖、可单测）
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class QueryOutcome:
    """一条正例题在某个配置下的检索结果。"""

    query_id: str
    question: str
    gold_count: int
    first_rank: int | None
    reciprocal_rank: float
    ndcg_at_10: float
    hits_at_k: dict[str, bool] = field(default_factory=dict)
    top_chunk_ids: list[str] = field(default_factory=list)
    total_ms: float = 0.0
    rerank_ms: float = 0.0
    rewrite_ms: float = 0.0
    degraded: tuple[str, ...] = ()


def first_gold_rank(chunk_ids: Sequence[str], gold_ids: set[str]) -> int | None:
    """首个命中 gold 的名次（1 起）；未命中返回 None。

    返回 None 而不是 0 或 -1：后面要区分"没命中"与"排在第 0 位"，
    用一个哨兵值混进算术里，是最容易写出 1/0 的地方。
    """

    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id in gold_ids:
            return rank
    return None


def ndcg_at_k(chunk_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """二值相关度下的 nDCG@K（命中记 1，否则 0）。

    与 rag_mvp `golden_runner._ndcg_at_k` 同一算法：DCG 用 `1/log2(rank+1)`，
    IDCG 假设所有 gold 排在最前。换算法就等于换口径，两边数字不可比。
    """

    if k <= 0 or not gold_ids:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(chunk_ids[:k], start=1)
        if chunk_id in gold_ids
    )
    ideal_count = min(k, len(gold_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at(rows: Sequence[QueryOutcome], k: int) -> float:
    """Recall@K：top-K 命中任一 gold 的题目占比。

    分母是 `len(rows)`——调用方只传正例题。负例题（无 gold，期望拒答）
    混进来会把"拒答正确"算成"召回失败"，那是另一个指标的事。
    """

    if not rows:
        return 0.0
    key = str(k)
    return sum(1 for row in rows if row.hits_at_k.get(key, False)) / len(rows)


def mean_reciprocal_rank(rows: Sequence[QueryOutcome]) -> float:
    """MRR：首个 gold 名次倒数的均值。未命中记 0（不是丢样本）。"""

    if not rows:
        return 0.0
    return sum(row.reciprocal_rank for row in rows) / len(rows)


def mean_ndcg_at_10(rows: Sequence[QueryOutcome]) -> float:
    """平均 nDCG@10。

    只做 @10：`QueryOutcome` 只缓存了这一个档位。硬做成"任意 K"要每次重算
    top-k 序列，而 W5 的检查点只关心 @10——写一个假装通用的签名，
    比直接叫 `@10` 更容易被误用。
    """

    if not rows:
        return 0.0
    return sum(row.ndcg_at_10 for row in rows) / len(rows)


def make_outcome(
    *,
    query_id: str,
    question: str,
    chunk_ids: Sequence[str],
    gold_ids: set[str],
    top_k: int,
    total_ms: float = 0.0,
    rerank_ms: float = 0.0,
    rewrite_ms: float = 0.0,
    degraded: tuple[str, ...] = (),
) -> QueryOutcome:
    """把"一次检索的 chunk_id 序列"折算成一条 `QueryOutcome`。"""

    rank = first_gold_rank(chunk_ids, gold_ids)
    return QueryOutcome(
        query_id=query_id,
        question=question,
        gold_count=len(gold_ids),
        first_rank=rank,
        reciprocal_rank=(1.0 / rank) if rank else 0.0,
        ndcg_at_10=ndcg_at_k(chunk_ids, gold_ids, 10),
        hits_at_k={
            str(k): rank is not None and rank <= k for k in K_VALUES if k <= top_k
        },
        top_chunk_ids=list(chunk_ids),
        total_ms=total_ms,
        rerank_ms=rerank_ms,
        rewrite_ms=rewrite_ms,
        degraded=degraded,
    )


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Config:
    """一个被测档位：`mode[+rerank][+rewrite]`。

    用标签字符串而不是一堆布尔参数，是因为报告里每一行的键必须能自解释——
    `dense+rerank` 比 `{"mode":"dense","rerank":true}` 在 console 表里更好读。
    """

    mode: str
    rerank: bool = False
    rewrite: bool = False

    @property
    def key(self) -> str:
        parts = [self.mode]
        if self.rerank:
            parts.append("rerank")
        if self.rewrite:
            parts.append("rewrite")
        return "+".join(parts)


def parse_config(label: str) -> Config:
    """`"dense+rerank"` → `Config(mode="dense", rerank=True)`。

    未知标签直接报错：拼错一个字母（`denes`）如果被静默忽略，
    报告里会出现一行看起来正常、实际跑的是别的档位的数据。
    """

    tokens = [token.strip().lower() for token in label.split("+") if token.strip()]
    if not tokens:
        raise ValueError("空的配置标签")

    mode = tokens[0]
    if mode not in {"dense", "sparse", "hybrid"}:
        raise ValueError(f"未知检索模式 {mode!r}；支持 dense / sparse / hybrid")

    unknown = [token for token in tokens[1:] if token not in {"rerank", "rewrite"}]
    if unknown:
        raise ValueError(f"未知开关 {unknown}；只支持 +rerank / +rewrite")

    return Config(mode=mode, rerank="rerank" in tokens, rewrite="rewrite" in tokens)


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ConfigReport:
    """一个档位的汇总。字段命名与 rag_mvp 的报告保持一致，方便 diff。"""

    config: str
    mode: str
    rerank: bool
    rewrite: bool
    top_k: int
    recall_multiplier: float
    rerank_candidates: int
    budget_ms: float
    total: int
    negative_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    rerank_p50_ms: float | None
    rewrite_p50_ms: float | None
    degraded_count: int
    # 改写的成本单独记（Gate「成本可查」）：改写是每次请求都花钱的环节，
    # 没有这一列，A/B 报告回答不了"这点召回提升值不值这个钱"。
    rewrite_calls: int = 0
    rewrite_input_tokens: int = 0
    rewrite_output_tokens: int = 0
    rewrite_cost: float = 0.0
    rewrite_currency: str = ""
    results: list[QueryOutcome] = field(default_factory=list)


def _summarize(
    config: Config,
    rows: list[QueryOutcome],
    *,
    top_k: int,
    recall_multiplier: float,
    rerank_candidates: int,
    budget_ms: float,
    negative_count: int,
) -> ConfigReport:
    totals = [row.total_ms for row in rows]
    rerank_times = [row.rerank_ms for row in rows if row.rerank_ms > 0]
    rewrite_times = [row.rewrite_ms for row in rows if row.rewrite_ms > 0]
    return ConfigReport(
        config=config.key,
        mode=config.mode,
        rerank=config.rerank,
        rewrite=config.rewrite,
        top_k=top_k,
        recall_multiplier=recall_multiplier,
        rerank_candidates=rerank_candidates,
        budget_ms=budget_ms,
        total=len(rows),
        negative_count=negative_count,
        recall_at_1=recall_at(rows, 1),
        recall_at_3=recall_at(rows, 3),
        recall_at_5=recall_at(rows, 5),
        recall_at_10=recall_at(rows, 10),
        mrr=mean_reciprocal_rank(rows),
        ndcg_at_10=mean_ndcg_at_10(rows),
        # p95 至少 5 个样本才算（`percentile` 的 min_samples），3 个点算 p95
        # 只是把最大值改了个名字。
        latency_p50_ms=percentile(totals, 0.50, min_samples=5),
        latency_p95_ms=percentile(totals, 0.95, min_samples=5),
        rerank_p50_ms=percentile(rerank_times, 0.50, min_samples=5),
        rewrite_p50_ms=percentile(rewrite_times, 0.50, min_samples=5),
        degraded_count=sum(1 for row in rows if row.degraded),
        results=rows,
    )


def _settings_for(
    config: Config,
    *,
    top_k: int,
    budget_ms: float | None,
    recall_multiplier: float | None,
    rerank_candidates: int | None,
    rerank_timeout_ms: float | None = None,
    rewrite_timeout_ms: float | None = None,
    device: str | None = None,
) -> Any:
    """按档位派生一份 `Settings` 覆盖。**只改 Settings，不碰 pipeline 内部。**

    这样装配仍然只有 `growth_agent.rag.build_pipeline` 一处，
    评测不会长出自己的第二套参数（改了那边忘了这边，报告数字就不再是生产数字）。
    """

    from agent_framework.config import Settings

    overrides: dict[str, Any] = {
        "retrieval_mode": config.mode,
        "retrieval_rerank": config.rerank,
        "retrieval_rewrite": config.rewrite,
        "retrieval_top_k": top_k,
    }
    if budget_ms is not None:
        overrides["retrieval_budget_ms"] = budget_ms
    if recall_multiplier is not None:
        overrides["retrieval_recall_multiplier"] = recall_multiplier
    if rerank_candidates is not None:
        overrides["retrieval_rerank_candidates"] = rerank_candidates
    if rerank_timeout_ms is not None:
        overrides["retrieval_rerank_timeout_ms"] = rerank_timeout_ms
    if rewrite_timeout_ms is not None:
        overrides["retrieval_rewrite_timeout_ms"] = rewrite_timeout_ms
    if device is not None:
        overrides["retrieval_device"] = device

    return Settings(**overrides)


async def _run_config(
    config: Config,
    queries: Sequence[Any],
    negative_count: int,
    *,
    top_k: int,
    budget_ms: float | None,
    recall_multiplier: float | None,
    rerank_candidates: int | None,
    rerank_timeout_ms: float | None,
    rewrite_timeout_ms: float | None,
    device: str | None,
    warmup: bool,
) -> ConfigReport:
    """跑一个档位。串行执行，不并发——并发会把 p95 变成调度噪声。"""

    from growth_agent.rag import build_runtime

    settings = _settings_for(
        config,
        top_k=top_k,
        budget_ms=budget_ms,
        recall_multiplier=recall_multiplier,
        rerank_candidates=rerank_candidates,
        rerank_timeout_ms=rerank_timeout_ms,
        rewrite_timeout_ms=rewrite_timeout_ms,
        device=device,
    )
    # 建一次 runtime：模型加载 18–20s，每档重建等于把评测时间拖成 ×档位数。
    runtime = build_runtime(settings)
    try:
        pipeline = runtime.pipeline
        if warmup and queries:
            # 首条查询会带上 tokenizer / 线程池的惰性初始化，混进 p95 就是一行
            # "第一条特别慢"的假数据。丢掉它，并在报告里说明丢过。
            await pipeline.run(queries[0].question, top_k=top_k)

        rows: list[QueryOutcome] = []
        for query in queries:
            gold_ids = set(query.gold_chunk_ids)
            outcome = await pipeline.run(query.question, top_k=top_k)
            diagnostics = outcome.diagnostics
            rows.append(
                make_outcome(
                    query_id=query.id,
                    question=query.question,
                    chunk_ids=[scored.chunk.chunk_id for scored in outcome.chunks],
                    gold_ids=gold_ids,
                    top_k=top_k,
                    total_ms=(diagnostics.total_ms if diagnostics else 0.0),
                    rerank_ms=(diagnostics.rerank_ms if diagnostics else 0.0),
                    rewrite_ms=(diagnostics.rewrite_ms if diagnostics else 0.0),
                    degraded=(diagnostics.degraded if diagnostics else ()),
                )
            )

        resolved = settings
        report = _summarize(
            config,
            rows,
            top_k=top_k,
            recall_multiplier=resolved.retrieval_recall_multiplier,
            rerank_candidates=resolved.retrieval_rerank_candidates,
            budget_ms=resolved.retrieval_budget_ms,
            negative_count=negative_count,
        )
        # 改写器的用量走鸭子类型：只有开了改写、且实现带 `usage` 才有值。
        # 不给 `QueryRewriter` 协议加这个属性——协议要能被最简单的实现满足。
        usage = getattr(getattr(runtime, "rewriter", None), "usage", None)
        if usage is not None:
            report.rewrite_calls = usage.calls
            report.rewrite_input_tokens = usage.input_tokens
            report.rewrite_output_tokens = usage.output_tokens
            report.rewrite_cost = usage.cost
            report.rewrite_currency = usage.currency
        return report
    finally:
        runtime.close()


def _load_queries(dataset_path: Path) -> tuple[Any, list[Any], list[Any]]:
    """读 golden 数据集。惰性导入：`--self-test` 不该被拖进 yaml/pydantic。"""

    from rag_mvp.eval.golden_schema import load_golden_dataset

    dataset = load_golden_dataset(dataset_path)
    return dataset, dataset.positive_queries, dataset.queries


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def format_table(reports: Sequence[ConfigReport]) -> str:
    header = (
        f"{'config':<16} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} "
        f"{'MRR':>6} {'nDCG':>6} {'p95ms':>7} {'deg':>4}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        p95 = f"{report.latency_p95_ms:>7.0f}" if report.latency_p95_ms else "      -"
        deg = f"{report.degraded_count:>4}" if report.degraded_count else "   ."
        lines.append(
            f"{report.config:<16} "
            f"{report.recall_at_1:>6.3f} {report.recall_at_3:>6.3f} "
            f"{report.recall_at_5:>6.3f} {report.recall_at_10:>6.3f} "
            f"{report.mrr:>6.3f} {report.ndcg_at_10:>6.3f} {p95} {deg}"
        )
    return "\n".join(lines)


def _report_payload(
    dataset: Any, reports: Sequence[ConfigReport], *, top_k: int, warmed: bool
) -> dict[str, Any]:
    warning = None
    degraded = [report.config for report in reports if report.degraded_count]
    if degraded:
        warning = (
            f"以下档位存在降级样本：{degraded}。降级时精排/改写会被跳过，"
            "指标反映的是退路而不是该组件本身——调大 --budget-ms 重跑再下结论。"
        )

    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version,
        "timestamp": datetime.now(UTC).isoformat(),
        "top_k": top_k,
        "warmup_discarded": warmed,
        "corpus_expected_chunks": (
            dataset.corpus.expected_chunk_count if dataset.corpus else None
        ),
        "warning": warning,
        "configs": {report.config: asdict(report) for report in reports},
    }


def _default_dataset_path() -> Path:
    """语料在隔壁仓库（`06_rag/rag_mvp`）。先看 RAG_ROOT，再按仓库布局推导。"""

    env = os.getenv("RAG_ROOT")
    root = Path(env).expanduser() if env else REPO_ROOT.parent / "06_rag" / "rag_mvp"
    return root / "data" / "eval" / "golden_queries.yaml"


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #


def self_test() -> int:
    """纯数学自检，不加载模型。写在这里是因为本文件的错误几乎全在公式里。"""

    gold = {"a"}
    assert first_gold_rank(["x", "a", "y"], gold) == 2
    assert first_gold_rank(["x", "y"], gold) is None

    row_hit = make_outcome(
        query_id="q1", question="", chunk_ids=["a"], gold_ids=gold, top_k=10
    )
    assert row_hit.reciprocal_rank == 1.0
    assert row_hit.ndcg_at_10 == 1.0
    assert row_hit.hits_at_k == {"1": True, "3": True, "5": True, "10": True}

    row_miss = make_outcome(
        query_id="q2", question="", chunk_ids=["x"], gold_ids=gold, top_k=10
    )
    assert row_miss.reciprocal_rank == 0.0
    assert recall_at([row_hit, row_miss], 1) == 0.5
    assert mean_reciprocal_rank([row_hit, row_miss]) == 0.5

    # 多 gold：位置 1 命中一个，位置 3 命中另一个 → nDCG 应高于单命中
    multi = make_outcome(
        query_id="q3",
        question="",
        chunk_ids=["a", "x", "b"],
        gold_ids={"a", "b"},
        top_k=10,
    )
    single = make_outcome(
        query_id="q4", question="", chunk_ids=["a", "x", "y"], gold_ids={"a", "b"}, top_k=10
    )
    assert multi.ndcg_at_10 > single.ndcg_at_10

    # top_k=5 时不该出现 "10" 这个键（否则 Recall@10 会用到没检索的档位）
    short = make_outcome(
        query_id="q5", question="", chunk_ids=["a"], gold_ids=gold, top_k=5
    )
    assert "10" not in short.hits_at_k

    assert parse_config("dense+rerank").rerank is True
    assert parse_config("hybrid").key == "hybrid"
    assert parse_config("dense+rerank+rewrite").key == "dense+rerank+rewrite"

    for bad in ("denes", "dense+banana", ""):
        try:
            parse_config(bad)
        except ValueError:
            continue
        raise AssertionError(f"非法标签 {bad!r} 应报错")

    print("self-test ok")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索质量指标（Recall@K / MRR / nDCG）")
    parser.add_argument(
        "--configs",
        default="dense,hybrid,dense+rerank",
        help="逗号分隔：mode[+rerank][+rewrite]，如 dense,dense+rerank",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=["production", "ragmvp"],
        default="production",
        help="production=用 Settings 里的预算/超取；ragmvp=复现 rag_mvp 的历史口径",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=None,
        help="整链路预算。测精排质量时必须调大（默认 800ms 会让精排每次都降级）",
    )
    parser.add_argument("--recall-multiplier", type=float, default=None)
    parser.add_argument("--rerank-candidates", type=int, default=None)
    parser.add_argument(
        "--rerank-timeout-ms",
        type=float,
        default=None,
        help=(
            "单次精排的上限。默认 3000ms 只够 GPU 或 ≤5 条候选；本机 CPU 上"
            "20 条要约 3.2s，会**每次都超时降级**（测出来等于没开精排）"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help=(
            "本地模型设备。测精排时在 macOS 上**必须传 cpu**：cross-encoder 跑在"
            "pipeline 的工作线程里，MPS 后端两个线程各持一个 encoder 会断言失败"
            "（进程直接死，不抛异常、无法降级）"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "evals" / "history" / "retrieval_w5.json",
    )
    parser.add_argument(
        "--rewrite-timeout-ms",
        type=float,
        default=None,
        help=(
            "单次改写的上限（一次 LLM 调用）。默认 2500ms 在本机跑 qwen 会**每次超时**，"
            "测出来的是降级路径。实测需要 8–12s 才能拿到真实改写结果"
        ),
    )
    parser.add_argument("--no-warmup", action="store_true", help="不丢弃首条查询")
    parser.add_argument("--self-test", action="store_true", help="只跑纯数学自检")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        return self_test()

    configs = [parse_config(label) for label in args.configs.split(",") if label.strip()]
    if not configs:
        print("没有可跑的配置", file=sys.stderr)
        return 2

    dataset_path = args.dataset or _default_dataset_path()
    if not dataset_path.exists():
        print(f"找不到 golden 数据集：{dataset_path}", file=sys.stderr)
        print("用 --dataset 指定，或设 RAG_ROOT 指向 rag_mvp 仓库根", file=sys.stderr)
        return 2

    preset = RAGMVP_PRESET if args.preset == "ragmvp" else {}
    top_k = args.top_k or preset.get("top_k") or DEFAULT_TOP_K
    recall_multiplier = args.recall_multiplier or preset.get("recall_multiplier")
    rerank_candidates = args.rerank_candidates or preset.get("rerank_candidates")
    budget_ms = args.budget_ms or preset.get("budget_ms")
    if budget_ms is None and any(config.rerank for config in configs):
        # 不主动改预算（那会掩盖"生产预算下精排跑不起来"这个事实），
        # 只提示一句——报告里的 deg 列会把真相摆在明面上。
        print(
            f"提示：精排档未指定 --budget-ms，将用 Settings 的默认预算。"
            f"若出现降级（deg>0），请用 --budget-ms {RERANK_BUDGET_MS:.0f} 重跑。",
            file=sys.stderr,
        )

    dataset, positive, all_queries = _load_queries(dataset_path)
    negative_count = len(all_queries) - len(positive)
    print(
        f"dataset={dataset.dataset_id} 正例={len(positive)} 负例={negative_count} "
        f"top_k={top_k} preset={args.preset}"
    )

    reports: list[ConfigReport] = []
    started = time.perf_counter()
    for config in configs:
        print(f"  running {config.key} ...", file=sys.stderr)
        report = asyncio.run(
            _run_config(
                config,
                positive,
                negative_count,
                top_k=top_k,
                budget_ms=budget_ms,
                recall_multiplier=recall_multiplier,
                rerank_candidates=rerank_candidates,
                rerank_timeout_ms=args.rerank_timeout_ms,
                rewrite_timeout_ms=args.rewrite_timeout_ms,
                device=args.device,
                warmup=not args.no_warmup,
            )
        )
        reports.append(report)

    elapsed = time.perf_counter() - started
    print()
    print(format_table(reports))

    payload = _report_payload(
        dataset, reports, top_k=top_k, warmed=not args.no_warmup
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已写入：{args.report}（总耗时 {elapsed:.1f}s）")

    if payload["warning"]:
        print(f"\n[!] {payload['warning']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
