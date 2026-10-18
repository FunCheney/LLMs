"""
Golden 检索评测执行器（Phase 4 正式形式）。

对数据集里的每条正例题，在每种检索模式（dense/sparse/hybrid）下检索 Top-K，
用 gold_chunk_ids 做精确判定，汇总出排序质量指标：

- Recall@K：Top-K 里至少命中一个 gold chunk 的题目占比（K ∈ {1,3,5,10}）；
- MRR：首个 gold 命中排名的倒数均值（衡量「首个正确答案排得多靠前」）；
- nDCG@K：二值相关度下的归一化折损累计增益（衡量多 gold 时的整体排序质量）。

负例题（expect_insufficient / 无 gold）不参与检索指标，只计数，留给生成层拒答评测。

产出：
- GoldenEvaluationReport（可序列化为 JSON 存档，支持「上次 vs 本次」diff 做回归）；
- 控制台可读报告（各模式指标表 + 未命中用例明细）。
"""

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from eval.golden_schema import GoldenDataset, GoldenQuery, load_golden_dataset
from retrieval.types import RetrievalResult

# 要报告的 Recall@K 档位
K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


@dataclass
class GoldenQueryResult:
    """单条正例题的检索结果。"""
    id: str
    question: str
    gold_count: int
    hit: bool                  # Top-K 是否命中任一 gold
    first_rank: int | None     # 首个 gold 的名次（1 起），未命中为 None
    reciprocal_rank: float     # 1/first_rank，未命中为 0
    ndcg_at_10: float          # 该题的二值相关度 nDCG@10
    hits_at_k: dict[str, bool] = field(default_factory=dict)  # 各 K 档是否命中
    top_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GoldenModeReport:
    """某一检索模式下的指标汇总。"""
    mode: str
    top_k: int
    total: int                 # 正例题数（参与指标计算）
    negative_count: int        # 负例题数（不参与指标）
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    results: list[GoldenQueryResult] = field(default_factory=list)


@dataclass
class GoldenEvaluationReport:
    """一次完整评测（含所有模式）的报告。"""
    dataset_id: str
    timestamp: str
    modes: dict[str, GoldenModeReport]


def _first_gold_rank(hits: list[RetrievalResult], gold_ids: set[str]) -> int | None:
    """返回首个命中 gold 的名次（1 起），未命中返回 None。"""
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk_id in gold_ids:
            return rank
    return None


def _ndcg_at_k(hits: list[RetrievalResult], gold_ids: set[str], k: int) -> float:
    """二值相关度下的 nDCG@K：命中 gold 记为 1，否则 0。"""
    if k <= 0 or not gold_ids:
        return 0.0

    # DCG@K：只累加前 k 位的相关度
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if hit.chunk_id in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # IDCG@K：理想排序把所有 gold 排在最前
    ideal_count = min(k, len(gold_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))  # i=0 → rank 1

    return dcg / idcg if idcg > 0 else 0.0


def _evaluate_query(
    query: GoldenQuery,
    hits: list[RetrievalResult],
    top_k: int,
) -> GoldenQueryResult:
    """对单条正例题计算命中情况、各档 Recall 标记与 nDCG@10。"""
    gold_ids = set(query.gold_chunk_ids)
    first_rank = _first_gold_rank(hits, gold_ids)

    hits_at_k = {
        str(k): first_rank is not None and first_rank <= k
        for k in K_VALUES
        if k <= top_k
    }

    return GoldenQueryResult(
        id=query.id,
        question=query.question,
        gold_count=len(gold_ids),
        hit=first_rank is not None,
        first_rank=first_rank,
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
        ndcg_at_10=_ndcg_at_k(hits, gold_ids, 10),
        hits_at_k=hits_at_k,
        top_chunk_ids=[hit.chunk_id for hit in hits],
    )


def _recall_at(results: list[GoldenQueryResult], k: int) -> float:
    """Recall@K：Top-K 命中任一 gold 的题目占比。"""
    if not results:
        return 0.0
    key = str(k)
    hits = sum(1 for r in results if r.hits_at_k.get(key, False))
    return hits / len(results)


def _mrr(results: list[GoldenQueryResult]) -> float:
    """Mean Reciprocal Rank：首个 gold 排名倒数的均值。"""
    if not results:
        return 0.0
    return sum(r.reciprocal_rank for r in results) / len(results)


def _validate_corpus(dataset: GoldenDataset) -> list[str]:
    """校验语料文件是否存在、chunk 数是否与数据集声明一致。"""
    warnings: list[str] = []
    if dataset.corpus is None:
        return warnings

    chunks_path = dataset.corpus.chunks_path
    if not chunks_path.exists():
        warnings.append(f"Corpus chunks file not found: {chunks_path}")
        return warnings

    if dataset.corpus.expected_chunk_count is not None:
        line_count = sum(
            1
            for line in chunks_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        if line_count != dataset.corpus.expected_chunk_count:
            warnings.append(
                "Chunk count mismatch: "
                f"expected {dataset.corpus.expected_chunk_count}, got {line_count}"
            )
    return warnings


def run_golden_evaluation(
    dataset_path: str | Path,
    modes: list[str] | None = None,
    top_k: int | None = None,
    dense_index_path: Path = Path("data/index/local.pkl"),
    bm25_index_path: Path = Path("data/index/bm25.pkl"),
    device: str | None = None,
    use_reranker: bool = False,
    rerank_model: str | None = None,
) -> tuple[GoldenEvaluationReport, list[str]]:
    """
    执行完整 Golden 检索评测，返回 (报告, 告警列表)。

    modes / top_k 为 None 时使用数据集 defaults 中的配置。
    实际检索条数取 max(top_k, 10)，保证 Recall@10 / nDCG@10 有足够样本。
    use_reranker=True 时，每种模式额外跑一组「召回放大 + BGE 重排」的对比
    （报告键为 "<mode>+rerank"），用于量化重排对排序质量的收益。
    """
    dataset = load_golden_dataset(dataset_path)
    warnings = _validate_corpus(dataset)

    # 惰性导入：create_retriever 会连带加载 sentence-transformers/torch（很重），
    # 仅在真正执行检索时引入，避免「仅加载 schema / 跑纯函数」也拖慢几十秒。
    from retrieval.factory import RetrieverConfig, create_retriever

    selected_modes = modes or dataset.defaults.modes
    selected_top_k = top_k or dataset.defaults.top_k
    # 至少检索 10 条，才能算 Recall@10 / nDCG@10
    retrieve_top_k = max(selected_top_k, max(K_VALUES))

    positive = dataset.positive_queries
    negative_count = len(dataset.queries) - len(positive)

    # (报告键, 检索模式, 是否重排)；重排模式下报告键加后缀，指标独立汇总
    eval_plans: list[tuple[str, str, bool]] = [(m, m, False) for m in selected_modes]
    if use_reranker:
        eval_plans += [(f"{m}+rerank", m, True) for m in selected_modes]

    mode_reports: dict[str, GoldenModeReport] = {}

    for report_key, mode, with_rerank in eval_plans:
        retriever = create_retriever(
            RetrieverConfig(
                mode=mode,
                dense_index_path=dense_index_path,
                bm25_index_path=bm25_index_path,
                device=device,
                reranker="bge" if with_rerank else None,
                rerank_model=rerank_model,
            )
        )

        query_results: list[GoldenQueryResult] = []
        for query in positive:
            # 每道题只检索一次，从同一批 hits 里同时算 Recall@K 与 nDCG@10
            hits = retriever.retrieve(query.question, top_k=retrieve_top_k)
            query_results.append(_evaluate_query(query, hits, retrieve_top_k))

        mode_reports[report_key] = GoldenModeReport(
            mode=report_key,
            top_k=retrieve_top_k,
            total=len(query_results),
            negative_count=negative_count,
            recall_at_1=_recall_at(query_results, 1),
            recall_at_3=_recall_at(query_results, 3),
            recall_at_5=_recall_at(query_results, 5),
            recall_at_10=_recall_at(query_results, 10),
            mrr=_mrr(query_results),
            ndcg_at_10=(
                sum(r.ndcg_at_10 for r in query_results) / len(query_results)
                if query_results
                else 0.0
            ),
            results=query_results,
        )

    report = GoldenEvaluationReport(
        dataset_id=dataset.dataset_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        modes=mode_reports,
    )
    return report, warnings


def report_to_dict(report: GoldenEvaluationReport) -> dict:
    """报告转 dict（用于 JSON 序列化）。"""
    return asdict(report)


def save_report(report: GoldenEvaluationReport, path: str | Path) -> None:
    """把报告写成 JSON 文件（自动创建父目录，中文不转义）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_console_report(report: GoldenEvaluationReport) -> str:
    """把报告格式化为便于阅读的纯文本（指标表 + 未命中用例明细）。"""
    lines: list[str] = []
    lines.append(f"Dataset: {report.dataset_id}")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append("")

    header = (
        f"{'mode':<8} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} "
        f"{'MRR':>7} {'nDCG@10':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for mode_report in report.modes.values():
        lines.append(
            f"{mode_report.mode:<8} "
            f"{mode_report.recall_at_1:>6.1%} "
            f"{mode_report.recall_at_3:>6.1%} "
            f"{mode_report.recall_at_5:>6.1%} "
            f"{mode_report.recall_at_10:>6.1%} "
            f"{mode_report.mrr:>7.3f} "
            f"{mode_report.ndcg_at_10:>8.3f}"
        )
        if mode_report.negative_count:
            lines.append(
                f"    (正例题 {mode_report.total} 条，负例题 {mode_report.negative_count} 条不参与指标)"
            )

    # 逐模式展开「Recall@10 仍未命中」的用例（这些是检索的硬伤，最值得看）
    for mode_report in report.modes.values():
        misses = [
            r for r in mode_report.results
            if not r.hits_at_k.get("10", False)
        ]
        if not misses:
            continue
        lines.append("")
        lines.append(f"[{mode_report.mode}] Recall@10 未命中的用例:")
        for r in misses:
            lines.append(f"  MISS {r.id}: {r.question}")
            lines.append(f"    gold={r.gold_count}  top_ids={r.top_chunk_ids[:5]}")

    return "\n".join(lines).rstrip()
