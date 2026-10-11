"""
上下文与引用的组装（检索结果 → 送给 LLM 的上下文 / 给用户看的引用）。

- build_context_blocks：把命中结果编号成 [1] [2] ...，附带章节/页码，拼成上下文；
- hits_to_citations：同步生成 Citation 列表（编号与上下文一一对应）；
- citations_referenced_in_answer：从答案中解析 [n]，只保留真正被引用的条目。
"""

import re

from retrieval.types import RetrievalResult
from generation.shcame import Citation


def build_context_blocks(hits: list[RetrievalResult]) -> str:
    """
    把命中的 Chunk 拼成带编号的上下文块。

    每条格式：[n] chunk_id=... | section=... | pages=...
              <正文>
    编号与提示词里的引用编号约定一致。
    """
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
    """把命中结果转成 Citation 列表（编号与 build_context_blocks 保持一致）。"""
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
    """
    筛选出答案里真正引用到的 Citation（按答案中的 [n] 编号）。

    若答案完全没有引用标记，则退回返回全部 citations（避免丢失来源）。
    """
    referenced_indexes = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", answer)
    }
    if not referenced_indexes:
        return citations
    return [c for c in citations if c.index in referenced_indexes]
