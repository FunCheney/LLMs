from dataclasses import asdict, dataclass, field


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_title: str

    section_title: str
    section_path: list[str]

    page_start: int
    page_end: int

    content: str
    source_uri: str = ""
    source_type: str = "pdf"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructuredParagraph:
    content: str
    page_start: int
    page_end: int
    section_title: str
    section_path: list[str]
    # (start_offset, end_offset, page_number) within content
    page_spans: list[tuple[int, int, int]] = field(default_factory=list)
