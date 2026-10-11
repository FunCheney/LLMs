from dataclasses import dataclass,asdict

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
    quote: str


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool
    retrieved_chunk_ids: list[str]
    query: str
    retrieval_mode: str

    def to_dict(self) -> dict:
        return asdict(self)


