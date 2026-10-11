"""
结构化段落解析：把逐页文本整理成「带章节信息」的段落列表。

处理流程（逐行扫描所有页面）：
1. 空行 → 结束当前段落（flush）；
2. 噪声行（纯页码 / 单字母等）→ 直接丢弃；
3. 标题行 → 结束当前段落，并更新 SectionTracker 的章节路径；
4. 普通行 → 累加到当前段落。

产出的每个 StructuredParagraph 都带有：
- content：段落正文
- page_start / page_end：段落跨越的页码
- section_title / section_path：段落所属章节
- page_spans：正文中每段对应的 (start, end, page_number)，供后续精确回填页码

这是「保留结构」的关键一步：标题绝不会被混进正文，段落也不被人为按字符切断。
"""

import re

from ingestion.heading_detector import SectionTracker, detect_heading
from schemas import StructuredParagraph

# 噪声行：纯页码（1~3 位数字）、单个大写字母、连续 P（PDF 页脚常见）
NOISE_LINE_PATTERN = re.compile(r"^(?:\d{1,3}|[A-Z]|P+)$")


def is_noise_line(line: str) -> bool:
    """判断一行是否为页眉/页脚之类的噪声（应被丢弃）。"""
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
    # 当前正在累积的段落：[(行文本, 行所属页码), ...]
    current_lines: list[tuple[str, int]] = []

    def flush_paragraph() -> None:
        """把 current_lines 收尾成一个 StructuredParagraph 并清空缓冲。"""
        if not current_lines:
            return

        # 段落正文由多行拼接而成（行间用单个换行）
        content = "\n".join(text for text, _ in current_lines).strip()
        if not content:
            current_lines.clear()
            return

        # 记录「正文中的字符区间 → 页码」的映射，供上层做跨页切分时回填页码
        page_spans: list[tuple[int, int, int]] = []
        offset = 0
        for index, (text, page_number) in enumerate(current_lines):
            if index:
                offset += 1  # 计入行间那个 "\n"
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
                # 空行 = 段落边界
                flush_paragraph()
                continue
            if is_noise_line(line):
                # 页码等噪声行直接忽略
                continue

            heading = detect_heading(line)
            if heading is not None:
                # 标题不进入正文：先收尾上一个段落，再更新章节路径
                flush_paragraph()
                tracker.update(heading)
                continue

            current_lines.append((line, page_number))

    # 文件末尾残留的最后一段
    flush_paragraph()
    return paragraphs
