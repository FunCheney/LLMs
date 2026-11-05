"""
generation 包：RAG 问答生成（上下文组装、提示词、LLM 调用、引用映射）。
"""

from rag_mvp.generation.pipeline import answer_question
from rag_mvp.generation.prompts import SYSTEM_PROMPT

__all__ = ["answer_question", "SYSTEM_PROMPT"]
