"""
Chunk 质量校验：分块完成后自动体检，尽早暴露解析/切分问题。

校验分两级：
- errors（致命）：chunk_id 缺失或重复、正文为空、页码非法 —— 只要有一条就判定不合格；
- warnings（提示）：正文超长、页码跨度过大、未知章节占比过高、缺 source_uri。

同时产出统计信息（长度分布、页码跨度、唯一章节数等），写入 ValidationReport.stats，
供 build_chunks.py 打印，也用于人工抽检时快速判断整体质量。
"""

from dataclasses import dataclass, field

from schemas import DocumentChunk


@dataclass
class ValidationIssue:
    """一条校验问题：level 为 error/warning，code 为机器可读标识。"""
    level: str
    code: str
    message: str
    chunk_id: str | None = None


@dataclass
class ValidationReport:
    """一次校验的汇总报告。"""
    chunk_count: int
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """是否存在致命错误（无 error 即视为通过）。"""
        return not self.errors


def validate_chunks(
    chunks: list[DocumentChunk] | list[dict],
    *,
    chunk_size: int = 500,
    max_page_span: int = 8,
    max_unknown_ratio: float = 0.35,
) -> ValidationReport:
    """
    校验 Chunk 列表，返回 ValidationReport。

    参数：
    - chunks：DocumentChunk 列表或已完成 to_dict 的 dict 列表（两种都支持）；
    - chunk_size：判定「超长」的阈值；
    - max_page_span：允许的最大页码跨度，超过则告警；
    - max_unknown_ratio：允许的「未知章节」占比上限，超过则告警。
    """
    report = ValidationReport(chunk_count=len(chunks))
    if not chunks:
        report.errors.append(
            ValidationIssue("error", "empty_corpus", "没有生成任何 Chunk")
        )
        return report

    # 统一转成 dict，兼容两种入参形式
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

        # —— 致命错误：chunk_id ——
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

        # —— 致命错误：空正文 ——
        if not content:
            report.errors.append(
                ValidationIssue("error", "empty_content", "Chunk 正文为空", chunk_id)
            )

        # —— 致命错误：页码非法 ——
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

        # —— 告警：正文超长 ——
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

        # —— 告警：页码跨度过大（可能把不相关内容合并了）——
        if span > max_page_span:
            report.warnings.append(
                ValidationIssue(
                    "warning",
                    "wide_page_span",
                    f"页码跨度 {page_start}-{page_end} 超过 {max_page_span} 页",
                    chunk_id,
                )
            )

        # 统计未知章节与来源缺失
        if section_title in {"Unknown", "UNKNOW", ""}:
            unknown += 1

        if not record.get("source_uri"):
            missing_source += 1

    # —— 告警：未知章节占比过高（说明标题识别效果差）——
    unknown_ratio = unknown / len(records)
    if unknown_ratio > max_unknown_ratio:
        report.warnings.append(
            ValidationIssue(
                "warning",
                "unknown_section_ratio",
                f"未知章节占比 {unknown_ratio:.1%} 超过 {max_unknown_ratio:.0%}",
            )
        )

    # —— 告警：存在缺少来源的 Chunk ——
    if missing_source:
        report.warnings.append(
            ValidationIssue(
                "warning",
                "missing_source_uri",
                f"{missing_source} 个 Chunk 缺少 source_uri",
            )
        )

    # 汇总统计指标
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
