import re

from ingestion.heading_detector import SectionTracker, detect_heading
from schemas import StructuredParagraph

NOISE_LINE_PATTERN = re.compile(r"^(?:\d{1,3}|[A-Z]|P+)$")


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return NOISE_LINE_PATTERN.fullmatch(stripped) is not None


def parse_structured_paragraphs(pages: list[dict]) -> list[StructuredParagraph]:
    """
    按页面顺序解析标题和段落，允许段落跨页，不允许标题被写进正文。
    """
    tracker = SectionTracker()
    paragraphs: list[StructuredParagraph] = []
    current_lines: list[tuple[str, int]] = []

    def flush_paragraph() -> None:
        if not current_lines:
            return

        content = "\n".join(text for text, _ in current_lines).strip()
        if not content:
            current_lines.clear()
            return

        page_spans: list[tuple[int, int, int]] = []
        offset = 0
        for index, (text, page_number) in enumerate(current_lines):
            if index:
                offset += 1
            start = offset
            offset += len(text)
            page_spans.append((start, offset, page_number))

        pages_used = [page for _, page in current_lines]
        paragraphs.append(
            StructuredParagraph(
                content=content,
                page_start=min(pages_used),
                page_end=max(pages_used),
                section_title=tracker.get_current_title(),
                section_path=tracker.get_path(),
                page_spans=page_spans,
            )
        )
        current_lines.clear()

    for page in pages:
        page_number = page["page_number"]
        for raw_line in page.get("text", "").splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                continue
            if is_noise_line(line):
                continue

            heading = detect_heading(line)
            if heading is not None:
                flush_paragraph()
                tracker.update(heading)
                continue

            current_lines.append((line, page_number))

    flush_paragraph()
    return paragraphs
