"""
检索结果的控制台展示工具（search_cli 与调试时使用）。
"""

from retrieval.types import RetrievalResult


def format_section_path(section_path: list[str]) -> str:
    """把章节路径列表拼成 "A > B > C" 形式；为空时显示占位符 "-"。"""
    if not section_path:
        return "-"
    return " > ".join(section_path)


def format_hit(hit: RetrievalResult, snippet: int = 200) -> str:
    """把单条命中格式化为多行文本，正文超长时截断并加省略号。"""
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
    """逐条打印命中结果，条目之间用分隔线隔开。"""
    for hit in hits:
        print("=" * 72)
        print(format_hit(hit, snippet=snippet))
