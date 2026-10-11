"""
Smoke 检索评测执行器（Phase 4 早期形式）。

对数据集里的每条问题，在每种检索模式（dense/sparse/hybrid）下各跑一遍，
按「是否命中期望章节/关键词」判定通过，汇总成结构化报告。

产出：
- SmokeEvaluationReport（可序列化为 JSON 存档到 data/eval/reports/）；
- 控制台可读报告（含失败用例、按 tag 的通过率）。

通过率 = 命中（或按 expect_hit 正确未命中）的问题数 / 总问题数，
是判断检索方案优劣的直接指标。
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from eval.matchers import find_first_matching_rank, hit_matches_rules
from eval.smoke_schema import SmokeDataset, SmokeQuery, load_smoke_dataset
from retrieval.factory import RetrieverConfig, create_retriever
from retrieval.types import RetrievalResult


@dataclass
class SmokeQueryResult:
    """单条问题的评测结果。"""
    id: str
    question: str
    expect_hit: bool
    passed: bool                 # 是否符合预期（期望命中且命中 / 期望不命中且确实未命中）
    hit: bool                    # 是否命中匹配规则
    rank_first_hit: int | None   # 首个命中结果的名次
    tags: list[str]
    top_chunk_ids: list[str] = field(default_factory=list)
    top_section_titles: list[str] = field(default_factory=list)


@dataclass
class SmokeModeReport:
    """某一检索模式下的汇总报告。"""
    mode: str
    top_k: int
    total: int
    passed: int
    hit_rate: float
    results: list[SmokeQueryResult] = field(default_factory=list)
    by_tag: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class SmokeEvaluationReport:
    """一次完整评测（含所有模式）的报告。"""
    dataset_id: str
    timestamp: str
    modes: dict[str, SmokeModeReport]

    def all_passed(self) -> bool:
        """所有模式的所有问题是否全部通过（供 CLI 决定退出码）。"""
        return all(report.passed == report.total for report in self.modes.values())


def _evaluate_query(
    query: SmokeQuery,
    hits: list[RetrievalResult],
) -> SmokeQueryResult:
    """对单条问题执行命中判定并计算是否通过。"""
    matched_hits = [hit for hit in hits if hit_matches_rules(hit, query.match)]
    hit = bool(matched_hits)
    rank_first_hit = find_first_matching_rank(hits, query.match)

    # expect_hit=True：命中才算通过；expect_hit=False：不命中才算通过
    if query.expect_hit:
        passed = hit
    else:
        passed = not hit

    return SmokeQueryResult(
        id=query.id,
        question=query.question,
        expect_hit=query.expect_hit,
        passed=passed,
        hit=hit,
        rank_first_hit=rank_first_hit,
        tags=query.tags,
        top_chunk_ids=[hit.chunk_id for hit in hits],
        top_section_titles=[hit.chunk.section_title for hit in hits],
    )


def _build_tag_summary(results: list[SmokeQueryResult]) -> dict[str, dict[str, int]]:
    """按 tag 汇总通过数 / 总数（未打 tag 的归入 "untagged"）。"""
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        for tag in result.tags or ["untagged"]:
            bucket = summary.setdefault(tag, {"passed": 0, "total": 0})
            bucket["total"] += 1
            if result.passed:
                bucket["passed"] += 1
    return summary


def _validate_corpus(dataset: SmokeDataset) -> list[str]:
    """
    校验语料文件是否存在、chunk 数是否与数据集声明一致，返回告警列表。

    如果索引是用旧语料构建的，评测结果会失真，这个检查能提前发现。
    """
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


def run_smoke_evaluation(
    dataset_path: str | Path,
    modes: list[str] | None = None,
    top_k: int | None = None,
    dense_index_path: Path = Path("data/index/local.pkl"),
    bm25_index_path: Path = Path("data/index/bm25.pkl"),
    device: str | None = None,
) -> tuple[SmokeEvaluationReport, list[str]]:
    """
    执行完整 smoke 评测，返回 (报告, 告警列表)。

    modes / top_k 为 None 时使用数据集 defaults 中的配置。
    """
    dataset = load_smoke_dataset(dataset_path)
    warnings = _validate_corpus(dataset)

    selected_modes = modes or dataset.defaults.modes
    selected_top_k = top_k or dataset.defaults.top_k

    mode_reports: dict[str, SmokeModeReport] = {}

    for mode in selected_modes:
        # 每个模式各建一个检索器，跑完全部问题
        retriever = create_retriever(
            RetrieverConfig(
                mode=mode,
                dense_index_path=dense_index_path,
                bm25_index_path=bm25_index_path,
                device=device,
            )
        )

        query_results: list[SmokeQueryResult] = []
        for query in dataset.queries:
            hits = retriever.retrieve(query.question, top_k=selected_top_k)
            query_results.append(_evaluate_query(query, hits))

        passed = sum(1 for result in query_results if result.passed)
        mode_reports[mode] = SmokeModeReport(
            mode=mode,
            top_k=selected_top_k,
            total=len(query_results),
            passed=passed,
            hit_rate=passed / len(query_results) if query_results else 0.0,
            results=query_results,
            by_tag=_build_tag_summary(query_results),
        )

    report = SmokeEvaluationReport(
        dataset_id=dataset.dataset_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        modes=mode_reports,
    )
    return report, warnings


def report_to_dict(report: SmokeEvaluationReport) -> dict:
    """报告转 dict（用于 JSON 序列化）。"""
    return asdict(report)


def save_report(report: SmokeEvaluationReport, path: str | Path) -> None:
    """把报告写成 JSON 文件（自动创建父目录，中文不转义）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_console_report(report: SmokeEvaluationReport) -> str:
    """把报告格式化为便于阅读的纯文本（只详列失败用例）。"""
    lines: list[str] = []
    lines.append(f"Dataset: {report.dataset_id}")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append("")

    for mode, mode_report in report.modes.items():
        lines.append(
            f"Mode: {mode}  k={mode_report.top_k}  "
            f"passed={mode_report.passed}/{mode_report.total}  "
            f"rate={mode_report.hit_rate:.1%}"
        )
        # 按 tag 的通过情况
        for tag, stats in sorted(mode_report.by_tag.items()):
            lines.append(
                f"  tag {tag}: {stats['passed']}/{stats['total']}"
            )

        # 只展开失败用例，避免输出过长
        for result in mode_report.results:
            if result.passed:
                continue
            lines.append("")
            lines.append(f"  FAIL {result.id}: {result.question}")
            lines.append(
                f"    expect_hit={result.expect_hit}  hit={result.hit}  "
                f"rank_first_hit={result.rank_first_hit}"
            )
            lines.append(f"    top_sections={result.top_section_titles[:3]}")
            lines.append(f"    top_chunk_ids={result.top_chunk_ids[:3]}")
        lines.append("")

    return "\n".join(lines).rstrip()
