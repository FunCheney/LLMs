from dataclasses import dataclass


@dataclass
class StructuredParagraph:
    content: str
    page_start: int
    page_end: int

    section_title: str
    section_path: list[str]