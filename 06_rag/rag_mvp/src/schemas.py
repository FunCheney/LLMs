"""
全局核心数据结构（贯穿 ingestion → 检索 → 生成 全链路）。

- DocumentChunk：分块的最终产物，既是 chunks.jsonl 的一行，也是检索命中的载体；
  字段包含 chunk_id 与完整溯源信息（章节、页码、来源）。
- StructuredParagraph：解析阶段的中间产物，带 page_spans 以便精确回填页码。
"""

from dataclasses import asdict, dataclass, field


@dataclass
class DocumentChunk:
    """一个可被检索/引用的文档分块。"""
    chunk_id: str          # 形如 <document_id>-p001-p002-c003，全局唯一
    document_id: str       # 所属文档标识
    document_title: str    # 文档标题

    section_title: str     # 所属章节标题
    section_path: list[str]  # 从顶层到当前的完整章节路径

    page_start: int        # 起始页（1 起）
    page_end: int          # 结束页

    content: str           # 正文
    source_uri: str = ""   # 来源文件路径
    source_type: str = "pdf"  # 来源类型：pdf / html

    def to_dict(self) -> dict:
        """转为 dict（写 JSONL 时使用）。"""
        return asdict(self)


@dataclass
class StructuredParagraph:
    """结构化解析得到的段落（分块的输入）。"""
    content: str
    page_start: int
    page_end: int
    section_title: str
    section_path: list[str]
    # (start_offset, end_offset, page_number) within content
    # 正文内的字符区间 → 页码，供超长段落切分后回填准确页码
    page_spans: list[tuple[int, int, int]] = field(default_factory=list)
