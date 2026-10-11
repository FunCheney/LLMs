from retrieval.base import Retriever
from retrieval.types import RetrievalResult
from generation.context import (
    build_context_blocks,
    citations_referenced_in_answer,
    hits_to_citations,
)
from generation.generator import generate_answer
from generation.shcame import RAGResponse

INSUFFICIENT_ANSWER = "当前资料中没有足够依据，无法可靠回答该问题。"


"""
# ✅ 正确
answer_question("问题", retriever, retrieval_mode="hybrid")
answer_question("问题", retriever, retrieval_mode="hybrid", top_k=3)

# ❌ 错误：位置参数不能越过 *
answer_question("问题", retriever, "hybrid")          # TypeError
answer_question("问题", retriever, "hybrid", 3, True) # TypeError
"""
def answer_question(
    query: str,  # 位置或关键字都行
    retriever: Retriever, # 位置或关键字都行
    *,# 分隔符
    retrieval_mode: str, # 只能用关键字
    top_k: int = 5, # 只能用关键字
    dry_run: bool = False,
    model: str | None = None,
    temperature: float = 0.0,
) -> RAGResponse:
    query = query.strip()
    hits: list[RetrievalResult] = retriever.retrieve(query, top_k=top_k)
    citations = hits_to_citations(hits)
    chunk_ids = [hit.chunk_id for hit in hits]

    if not query or not hits:
        return RAGResponse(
            answer=INSUFFICIENT_ANSWER,
            citations=[],
            insufficient_evidence=True,
            retrieved_chunk_ids=chunk_ids,
            query=query,
            retrieval_mode=retrieval_mode,
        )

    context = build_context_blocks(hits)

    if dry_run:
        return RAGResponse(
            answer=f"[dry-run] Retrieved {len(hits)} chunks.\n\n{context}",
            citations=citations,
            insufficient_evidence=False,
            retrieved_chunk_ids=chunk_ids,
            query=query,
            retrieval_mode=retrieval_mode,
        )

    answer = generate_answer(
        question=query,
        context=context,
        model=model,
        temperature=temperature,
    )

    insufficient = INSUFFICIENT_ANSWER in answer or "没有足够依据" in answer
    used_citations = citations_referenced_in_answer(answer, citations)

    return RAGResponse(
        answer=answer,
        citations=used_citations if used_citations else citations,
        insufficient_evidence=insufficient,
        retrieved_chunk_ids=chunk_ids,
        query=query,
        retrieval_mode=retrieval_mode,
    )
