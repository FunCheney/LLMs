import re

from retrieval.types import RetrievalResult
from generation.shcame import Citation


def build_context_blocks(hits: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = (
            f"[{index}] chunk_id={hit.chunk_id} | "
            f"section={chunk.section_title} | "
            f"pages={chunk.page_start}-{chunk.page_end}"
        )
        blocks.append(f"{header}\n{chunk.content}")
    return "\n\n".join(blocks)


def hits_to_citations(
    hits: list[RetrievalResult],
    quote_max_len: int = 240,
) -> list[Citation]:
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        quote = chunk.content.replace("\n", " ").strip()
        if len(quote) > quote_max_len:
            quote = quote[:quote_max_len] + "..."
        citations.append(
            Citation(
                index=index,
                chunk_id=hit.chunk_id,
                document_title=chunk.document_title,
                section_title=chunk.section_title,
                section_path=list(chunk.section_path),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                source_uri=chunk.source_uri,
                quote=quote,
            )
        )
    return citations


def citations_referenced_in_answer(
    answer: str,
    citations: list[Citation],
) -> list[Citation]:
    referenced_indexes = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", answer)
    }
    if not referenced_indexes:
        return citations
    return [c for c in citations if c.index in referenced_indexes]
