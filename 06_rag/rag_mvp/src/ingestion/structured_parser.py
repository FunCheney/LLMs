from ingestion.heading_detector import (
    SectionTracker,
    detect_heading,
)
from src.ingestion.structured_units import (
    StructuredParagraph,
)


def parse_structured_paragraphs(
    pages: list[dict],
) -> list[StructuredParagraph]:
    """
    按页面顺序解析标题和段落。

    这是基础版本：
    - 按空行划分段落
    - 按行识别标题
    - 维护章节路径
    """
    tracker = SectionTracker()
    paragraphs = []

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        lines = text.splitlines()
        current_lines: list[str] = []

        def flush_paragraph():
            if not current_lines:
                return

            content = "\n".join(
                current_lines
            ).strip()

            if not content:
                return

            paragraphs.append(
                StructuredParagraph(
                    content=content,
                    page_start=page_number,
                    page_end=page_number,
                    section_title=(
                        tracker.get_current_title()
                    ),
                    section_path=tracker.get_path(),
                )
            )

            current_lines.clear()

        for line in lines:
            heading = detect_heading(line)

            if heading is not None:
                flush_paragraph()
                tracker.update(heading)
                continue

            if not line.strip():
                flush_paragraph()
                continue

            current_lines.append(line.strip())

        flush_paragraph()

    return paragraphs