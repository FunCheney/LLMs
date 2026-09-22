from dataclasses import dataclass, asdict
from typing import List


# 表示最终用于检索的文本块
@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_title: str

    section_title: str # 当前 Chunk 所属的直接章节标题。
    section_path: List[str] # 从上层章节到当前章节的完整路径。

    page_start: int
    page_end: int

    content: str

    def to_dict(self) -> dict:
        return asdict(self)


# 表示一个带章节和页码信息的文本单元
@dataclass
class StructuredParagraph:
    content: str
    page_start: int
    page_end: int
    section_title: str
    section_path: list[str]

