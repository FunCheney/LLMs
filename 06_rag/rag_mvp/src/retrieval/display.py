from retrieval.types import RetrievalResult


def format_section_path(section_path: list[str]) -> str:
    if not section_path:
        return "-"
    return " > ".join(section_path)


def format_hit(hit: RetrievalResult, snippet: int = 200) -> str:
    content = hit.chunk.content.replace("\n", " ")
    preview = content[:snippet]
    if len(content) > snippet:
        preview += "..."

    lines = [
        f"#{hit.rank}  score={hit.score:.4f}  source={hit.source}",
        f"chunk_id: {hit.chunk_id}",
        f"section: {hit.chunk.section_title or '-'}",
        f"path: {format_section_path(hit.chunk.section_path)}",
        (
            f"pages: {hit.chunk.page_start}-{hit.chunk.page_end}  "
            f"source: {hit.chunk.source_uri or '-'}"
        ),
        f"content: {preview}",
    ]
    return "\n".join(lines)


def print_hits(hits: list[RetrievalResult], snippet: int = 200) -> None:
    for hit in hits:
        print("=" * 72)
        print(format_hit(hit, snippet=snippet))
