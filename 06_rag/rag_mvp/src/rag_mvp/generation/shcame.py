"""
生成层的数据结构（文件名 shcame 应为 scheme，为兼容保留）。

- Citation：一条引用，与上下文里的 [n] 编号一一对应，携带章节/页码/来源/摘录；
- RAGResponse：一次问答的完整结果，包含答案、引用、证据是否充分等。
"""

from dataclasses import dataclass, asdict


@dataclass
class Citation:
    """引用条目，与上下文中的 [n] 编号对应。"""

    index: int
    chunk_id: str
    document_title: str
    section_title: str
    section_path: list[str]
    page_start: int
    page_end: int
    source_uri: str
    quote: str          # 引用文本摘录（截断展示用）


@dataclass
class RAGResponse:
    """一次 RAG 问答的结果。"""
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool     # 是否因证据不足而拒答
    retrieved_chunk_ids: list[str]  # 本次检索命中的 chunk（便于追溯）
    query: str
    retrieval_mode: str

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典（ask_cli --json 使用）。"""
        return asdict(self)


