"""检索抽象（W5 D1）。

核心约定：**空结果不等于失败。**

- 返回 `[]`：链路正常，但没召回。这是内容问题——W4 归因走 retrieval 层，
  该补语料或改查询。
- 抛异常：链路坏了（索引不存在、向量库连不上、精排超时且无降级）。这是系统
  问题——该重试或降级。

两者混成"返回空"，W4 的根因归因会把系统故障误判成语料缺失，于是你去补了三天
语料，而真实原因是 rerank 一直在超时。这条约定要落实到每个实现里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from ..knowledge.base import ScoredChunk


@runtime_checkable
class Retriever(Protocol):
    """召回抽象。单一职责：给定查询，返回带名次的结果。"""

    @property
    def name(self) -> str: ...

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """返回按名次排好的结果；无召回返回空列表，**不要返回 None**。"""


@runtime_checkable
class Reranker(Protocol):
    """精排。

    **必须带超时**——这是生产 RAG 与 demo RAG 的分水岭。精排是同步阻塞调用，
    模型加载慢、输入超长、GPU 排队都会让整条链路从"稍慢"变成"卡死"。
    实现方要么内部设超时，要么由调用方用 `asyncio.wait_for` 兜住；
    协议不提供"无超时"的实现。
    """

    @property
    def name(self) -> str: ...

    async def rerank(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
        *,
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """返回重排后的结果。重排只改顺序与截断，**不允许丢弃原文**——
        被截掉的块在 W6 的引用校验里仍然需要，用于判断"是不是漏引了"。"""


@runtime_checkable
class QueryRewriter(Protocol):
    """查询改写 / 多查询展开。

    用途是提高召回：把"品牌调性怎么定"展开成"品牌 tone""品牌声音 定义"等。
    注意与归一化（NFKC、去空白）区分——归一化是格式清洗，改写是语义扩展，
    rag_mvp 的 `pre_query/normalizer.py` 只是前者，不能当改写用。
    """

    @property
    def name(self) -> str: ...

    async def rewrite(self, query: str, *, n: int = 3) -> list[str]:
        """返回 n 条候选查询，**第一条必须是原查询**（保证改写变差时不倒退）。"""
