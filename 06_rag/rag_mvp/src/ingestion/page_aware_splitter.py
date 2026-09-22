import re
from dataclasses import dataclass

from ingestion.text_splitter import split_text


@dataclass
class ParagraphUnit:
    content: str
    page_start: int
    page_end: int


def combine_pages(
    pages: list[dict],
) -> tuple[str, list[tuple[int, int, int]]]:
    """
    将页面按顺序合并为一个字符串。

    返回：
    - combined_text：合并后的文本
    - page_ranges：每一页在 combined_text 中的范围

    page_ranges 中的元素：
    (start_offset, end_offset, page_number)
    """
    text_parts = []
    page_ranges = []

    current_offset = 0

    for index, page in enumerate(pages):
        # print(page)
        page_number = page["page_number"]
        page_text = page["text"].strip()

        if not page_text:
            continue

        # 页面之间只添加一个换行符，避免人为制造段落边界。
        # 这只是一个启发式方案。PDF 提取的文本可能包含页眉、页脚、断行和排版噪声，因此后续仍然需要清洗和评估。
        if text_parts:
            separator = "\n"
            text_parts.append(separator)
            current_offset += len(separator)

        start_offset = current_offset

        text_parts.append(page_text)
        current_offset += len(page_text)

        end_offset = current_offset

        page_ranges.append(
            (
                start_offset,
                end_offset,
                page_number,
            )
        )

    return "".join(text_parts), page_ranges


def _find_pages_for_range(
        start_offset: int,
        end_offset: int,
        page_ranges: list[tuple[int, int, int]]
) -> tuple[int, int]:
    """
    根据文本字符范围，计算对应的起止页码。
    """
    matched_pages = []
    for page_start, page_end, page_number in page_ranges:
        # 两个区间存在交集
        has_overlap = (start_offset < page_end and end_offset > page_start)
        if has_overlap:
            matched_pages.append(page_number)


    if not matched_pages:
        raise ValueError("Could not map text range to a PDF page")

    return (
        min(matched_pages),
        max(matched_pages),
    )

def split_paragraph_units(pages: list[dict]) -> list[ParagraphUnit]:
    """
    将 pdf 页面转换为带页码的段落信息
     PDF 文本提取并不一定保留真实的段落结构。如果原始 PDF 没有空行，或者 PDF 的换行来自排版而不是语义，正则表达式无法完全恢复原始段落。
     所以：页码映射可以做到比较确定，但段落识别仍然属于启发式处理，需要通过真实 PDF 检查结果。
    """
    combined_text, page_ranges = combine_pages(pages)
    if not combined_text:
        return []

    # 以非空白字符开始。
    # 中间包含任意文本。
    # 以非空白字符结束。
    # 直到空行或文本末尾。
    paragraph_matches = re.finditer(
        r"\S(?:.*?\S)?(?=\n\s*\n|\Z)",
        combined_text,
        flags=re.DOTALL,
    )
    units = []
    for match in paragraph_matches:
        content = match.group(0).strip()
        if not content:
            continue

        page_start, page_end = _find_pages_for_range(match.start(), match.end(), page_ranges)
        units.append(
            ParagraphUnit(
                content=content,
                page_start=page_start,
                page_end=page_end,
            )
        )

    return units


def build_page_aware_chunks(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    将带页码的段落单元合并成跨页 Chunk。

    返回的每个字典包含：
    - content
    - page_start
    - page_end
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must satisfy "
            "0 <= overlap < chunk_size"
        )

    units = split_paragraph_units(pages)

    chunks = []
    current_units = []
    current_length = 0

    def flush_current_units() -> None:
        nonlocal current_units
        nonlocal current_length

        if not current_units:
            return

        content = "\n\n".join(
            unit.content
            for unit in current_units
        )

        chunks.append({
            "content": content,
            "page_start": min(
                unit.page_start
                for unit in current_units
            ),
            "page_end": max(
                unit.page_end
                for unit in current_units
            ),
        })

        current_units = []
        current_length = 0

    for unit in units:
        content = unit.content

        # 超长段落单独进行字符级切分
        if len(content) > chunk_size:
            flush_current_units()

            long_chunks = split_text(
                text=content,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for long_chunk in long_chunks:
                chunks.append({
                    "content": long_chunk,
                    "page_start": unit.page_start,
                    "page_end": unit.page_end,
                })

            continue

        additional_length = len(content)

        if current_units:
            additional_length += 2  # "\n\n"

        # 合并后不会超过 chunk_size
        if (
            current_units
            and current_length + additional_length
            <= chunk_size
        ):
            current_units.append(unit)
            current_length += additional_length
            continue

        # 当前 Chunk 为空，直接加入
        if not current_units:
            current_units.append(unit)
            current_length = len(content)
            continue

        # 合并后超出限制，先保存旧 Chunk
        flush_current_units()

        # 当前段落开始新的 Chunk
        current_units.append(unit)
        current_length = len(content)

    flush_current_units()

    return chunks




