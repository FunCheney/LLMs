"""
分块器（生产版）：把「结构化段落」合并成最终写入 chunks.jsonl 的 DocumentChunk。

核心策略：
1. 先调用 parse_structured_paragraphs 得到带章节/页码的段落；
2. 把同一章节下、总长不超过 chunk_size 的相邻段落合并进同一个 Chunk；
3. 段落本身超过 chunk_size 时，退化为字符级滑动窗口切分，并回填页码；
4. 一旦章节发生变化，立即开启新 Chunk（保证一个 Chunk 不跨章节）。

每个 Chunk 都带完整溯源信息：document_id / section_title / section_path /
page_start / page_end / source_uri / source_type，chunk_id 由这些信息拼接而成。
"""

from ingestion.structured_parser import parse_structured_paragraphs
from ingestion.text_splitter import split_text_windows
from schemas import DocumentChunk, StructuredParagraph


def _pages_for_range(
    start_offset: int,
    end_offset: int,
    page_spans: list[tuple[int, int, int]],
    fallback_start: int,
    fallback_end: int,
) -> tuple[int, int]:
    """
    把段落内的字符区间映射回真实页码。

    对每个 (span_start, span_end, page) 判断是否与 [start_offset, end_offset) 有交集，
    取交集页面的最小/最大页码；若完全没有交集则回退到段落自身的页码范围。
    """
    matched = [
        page
        for span_start, span_end, page in page_spans
        if start_offset < span_end and end_offset > span_start
    ]
    if not matched:
        return fallback_start, fallback_end
    return min(matched), max(matched)


def _flush_group(
    group: list[StructuredParagraph],
) -> tuple[str, int, int, str, list[str]]:
    """
    把一组同章节段落合并成一个 Chunk 的原始字段。

    返回：(正文, 起始页, 结束页, 章节标题, 章节路径)。
    正文用空行连接；页码取组内段落的最小起始页与最大结束页。
    """
    content = "\n\n".join(paragraph.content for paragraph in group)
    return (
        content,
        min(paragraph.page_start for paragraph in group),
        max(paragraph.page_end for paragraph in group),
        group[0].section_title,
        list(group[0].section_path),
    )


def _make_chunk(
    *,
    document_id: str,
    document_title: str,
    chunk_index: int,
    page_start: int,
    page_end: int,
    content: str,
    section_title: str,
    section_path: list[str],
    source_uri: str,
    source_type: str,
) -> DocumentChunk:
    """根据字段构造 DocumentChunk，并生成可读且唯一的 chunk_id。"""
    return DocumentChunk(
        # 形如 claude-code-0001-p001-p002-c003，
        # 用零填充保证同一文档内按字符串排序即得到自然顺序
        chunk_id=(
            f"{document_id}-"
            f"p{page_start:03d}-"
            f"p{page_end:03d}-"
            f"c{chunk_index:03d}"
        ),
        document_id=document_id,
        document_title=document_title,
        section_title=section_title,
        section_path=section_path,
        page_start=page_start,
        page_end=page_end,
        content=content,
        source_uri=source_uri,
        source_type=source_type,
    )


def build_chunks(
    pages: list[dict],
    document_id: str,
    document_title: str,
    chunk_size: int = 500,
    overlap: int = 50,
    source_uri: str = "",
    source_type: str = "pdf",
) -> list[DocumentChunk]:
    """
    将解析后的页面文本切成带完整元数据的 DocumentChunk 列表。

    参数：
    - pages：parse_pdf / parse_html 的输出（逐页文本）
    - chunk_size：单个 Chunk 的最大字符数
    - overlap：超长段落做字符级切分时的重叠长度
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    paragraphs = parse_structured_paragraphs(pages)
    chunks: list[DocumentChunk] = []
    current_group: list[StructuredParagraph] = []  # 当前累积的同章节段落
    current_length = 0

    def emit(content: str, page_start: int, page_end: int, section_title: str, section_path: list[str]) -> None:
        """生成一个 Chunk 并追加到结果（chunk_index 用当前已产出数量）。"""
        chunks.append(
            _make_chunk(
                document_id=document_id,
                document_title=document_title,
                chunk_index=len(chunks),
                page_start=page_start,
                page_end=page_end,
                content=content,
                section_title=section_title,
                section_path=section_path,
                source_uri=source_uri,
                source_type=source_type,
            )
        )

    def flush_group() -> None:
        """把当前累积的段落合并输出为一个 Chunk，并清空缓冲。"""
        nonlocal current_group, current_length
        if not current_group:
            return
        content, page_start, page_end, section_title, section_path = _flush_group(current_group)
        emit(content, page_start, page_end, section_title, section_path)
        current_group = []
        current_length = 0

    for paragraph in paragraphs:
        content = paragraph.content
        # 超长段落：先收尾已有 Chunk，再按字符窗口强切
        if len(content) > chunk_size:
            flush_group()
            for piece, start, end in split_text_windows(content, chunk_size, overlap):
                page_start, page_end = _pages_for_range(
                    start,
                    end,
                    paragraph.page_spans,
                    paragraph.page_start,
                    paragraph.page_end,
                )
                emit(piece, page_start, page_end, paragraph.section_title, paragraph.section_path)
            continue

        # 只有「同章节」且「合并后不超长」才能并入当前 Chunk
        same_section = (
            current_group
            and current_group[0].section_path == paragraph.section_path
        )
        additional_length = len(content) + (2 if current_group else 0)  # 计入 "\n\n"

        if same_section and current_length + additional_length <= chunk_size:
            current_group.append(paragraph)
            current_length += additional_length
            continue

        # 否则结束旧 Chunk，以当前段落开启新 Chunk
        flush_group()
        current_group.append(paragraph)
        current_length = len(content)

    # 处理文件末尾残留的段落
    flush_group()
    return chunks
