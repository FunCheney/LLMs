from dataclasses import dataclass, field

from schemas import DocumentChunk


@dataclass
class ValidationIssue:
    level: str
    code: str
    message: str
    chunk_id: str | None = None


@dataclass
class ValidationReport:
    chunk_count: int
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_chunks(
    chunks: list[DocumentChunk] | list[dict],
    *,
    chunk_size: int = 500,
    max_page_span: int = 8,
    max_unknown_ratio: float = 0.35,
) -> ValidationReport:
    report = ValidationReport(chunk_count=len(chunks))
    if not chunks:
        report.errors.append(
            ValidationIssue("error", "empty_corpus", "没有生成任何 Chunk")
        )
        return report

    records = [
        chunk.to_dict() if hasattr(chunk, "to_dict") else chunk
        for chunk in chunks
    ]

    seen_ids: set[str] = set()
    lengths: list[int] = []
    page_spans: list[int] = []
    unknown = 0
    missing_source = 0
    oversized = 0

    for record in records:
        chunk_id = record.get("chunk_id")
        content = (record.get("content") or "").strip()
        page_start = int(record.get("page_start") or 0)
        page_end = int(record.get("page_end") or 0)
        section_title = record.get("section_title") or "Unknown"

        if not chunk_id:
            report.errors.append(
                ValidationIssue("error", "missing_chunk_id", "Chunk 缺少 chunk_id")
            )
        elif chunk_id in seen_ids:
            report.errors.append(
                ValidationIssue("error", "duplicate_chunk_id", "chunk_id 重复", chunk_id)
            )
        else:
            seen_ids.add(chunk_id)

        if not content:
            report.errors.append(
                ValidationIssue("error", "empty_content", "Chunk 正文为空", chunk_id)
            )

        if page_start <= 0 or page_end <= 0 or page_start > page_end:
            report.errors.append(
                ValidationIssue(
                    "error",
                    "invalid_pages",
                    f"页码非法: {page_start}-{page_end}",
                    chunk_id,
                )
            )

        length = len(content)
        lengths.append(length)
        span = page_end - page_start
        page_spans.append(span)

        if length > chunk_size:
            oversized += 1
            report.warnings.append(
                ValidationIssue(
                    "warning",
                    "oversize",
                    f"正文长度 {length} 超过 chunk_size={chunk_size}",
                    chunk_id,
                )
            )

        if span > max_page_span:
            report.warnings.append(
                ValidationIssue(
                    "warning",
                    "wide_page_span",
                    f"页码跨度 {page_start}-{page_end} 超过 {max_page_span} 页",
                    chunk_id,
                )
            )

        if section_title in {"Unknown", "UNKNOW", ""}:
            unknown += 1

        if not record.get("source_uri"):
            missing_source += 1

    unknown_ratio = unknown / len(records)
    if unknown_ratio > max_unknown_ratio:
        report.warnings.append(
            ValidationIssue(
                "warning",
                "unknown_section_ratio",
                f"未知章节占比 {unknown_ratio:.1%} 超过 {max_unknown_ratio:.0%}",
            )
        )

    if missing_source:
        report.warnings.append(
            ValidationIssue(
                "warning",
                "missing_source_uri",
                f"{missing_source} 个 Chunk 缺少 source_uri",
            )
        )

    report.stats = {
        "chunk_count": len(records),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_page_span": round(sum(page_spans) / len(page_spans), 2),
        "max_page_span": max(page_spans),
        "unknown_section_ratio": round(unknown_ratio, 3),
        "oversized_chunks": oversized,
        "unique_sections": len(
            {
                tuple(record.get("section_path") or [record.get("section_title")])
                for record in records
            }
        ),
    }
    return report
