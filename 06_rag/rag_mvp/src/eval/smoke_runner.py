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
    id: str
    question: str
    expect_hit: bool
    passed: bool
    hit: bool
    rank_first_hit: int | None
    tags: list[str]
    top_chunk_ids: list[str] = field(default_factory=list)
    top_section_titles: list[str] = field(default_factory=list)


@dataclass
class SmokeModeReport:
    mode: str
    top_k: int
    total: int
    passed: int
    hit_rate: float
    results: list[SmokeQueryResult] = field(default_factory=list)
    by_tag: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class SmokeEvaluationReport:
    dataset_id: str
    timestamp: str
    modes: dict[str, SmokeModeReport]

    def all_passed(self) -> bool:
        return all(report.passed == report.total for report in self.modes.values())


def _evaluate_query(
    query: SmokeQuery,
    hits: list[RetrievalResult],
) -> SmokeQueryResult:
    matched_hits = [hit for hit in hits if hit_matches_rules(hit, query.match)]
    hit = bool(matched_hits)
    rank_first_hit = find_first_matching_rank(hits, query.match)

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
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        for tag in result.tags or ["untagged"]:
            bucket = summary.setdefault(tag, {"passed": 0, "total": 0})
            bucket["total"] += 1
            if result.passed:
                bucket["passed"] += 1
    return summary


def _validate_corpus(dataset: SmokeDataset) -> list[str]:
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
    dataset = load_smoke_dataset(dataset_path)
    warnings = _validate_corpus(dataset)

    selected_modes = modes or dataset.defaults.modes
    selected_top_k = top_k or dataset.defaults.top_k

    mode_reports: dict[str, SmokeModeReport] = {}

    for mode in selected_modes:
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
    return asdict(report)


def save_report(report: SmokeEvaluationReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_console_report(report: SmokeEvaluationReport) -> str:
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
        for tag, stats in sorted(mode_report.by_tag.items()):
            lines.append(
                f"  tag {tag}: {stats['passed']}/{stats['total']}"
            )

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
