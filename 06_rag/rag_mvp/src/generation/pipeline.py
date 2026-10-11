"""
RAG 问答主流程（Phase 3 的核心编排）。

一次问答的完整链路：

    query
      → retriever.retrieve()            检索 Top-K
      → hits_to_citations()             生成引用
      → build_context_blocks()          拼装上下文
      → generate_answer()               调 LLM 生成答案
      → citations_referenced_in_answer() 过滤出真正被引用的来源
      → RAGResponse

特殊分支：
- query 为空或无命中 → 直接返回「证据不足」，不调用 LLM；
- dry_run=True → 只检索并返回上下文，不调用 LLM（便于验证检索质量）。
"""

from retrieval.base import Retriever
from retrieval.types import RetrievalResult
from generation.context import (
    build_context_blocks,
    citations_referenced_in_answer,
    hits_to_citations,
)
from generation.generator import generate_answer
from generation.shcame import RAGResponse

# 统一的「证据不足」话术（与提示词中的约定保持一致，用于判定拒答）
INSUFFICIENT_ANSWER = "当前资料中没有足够依据，无法可靠回答该问题。"


"""
# ✅ 正确：retrieval_mode / top_k 等必须用关键字参数
answer_question("问题", retriever, retrieval_mode="hybrid")
answer_question("问题", retriever, retrieval_mode="hybrid", top_k=3)

# ❌ 错误：位置参数不能越过 *（分隔符之后的参数只能用关键字传）
answer_question("问题", retriever, "hybrid")          # TypeError
answer_question("问题", retriever, "hybrid", 3, True) # TypeError
"""
def answer_question(
    query: str,  # 位置或关键字都行
    retriever: Retriever,  # 位置或关键字都行
    *,  # 分隔符：其后的参数只能通过关键字传入，避免调用方传错位置
    retrieval_mode: str,  # 只能用关键字
    top_k: int = 5,  # 只能用关键字
    dry_run: bool = False,
    model: str | None = None,
    temperature: float = 0.0,
) -> RAGResponse:
    """
    检索 + 生成，返回带引用的 RAGResponse。

    dry_run=True 时跳过 LLM，仅返回检索到的上下文，用于调试检索。
    """
    query = query.strip()
    hits: list[RetrievalResult] = retriever.retrieve(query, top_k=top_k)
    citations = hits_to_citations(hits)
    chunk_ids = [hit.chunk_id for hit in hits]

    # 空查询或无命中：直接拒答，不浪费一次 LLM 调用
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

    # dry-run：只回显上下文，验证检索是否命中
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

    # 判定模型是否明确表示资料不足；并只保留答案中真正引用到的来源
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
