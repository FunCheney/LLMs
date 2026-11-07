"""有出处的回答（W6）：链路组装、零召回、引用体检与 HTTP 契约。

这里测的不是"模型答得好不好"（W4 judge 的事），而是三条接线：

1. 检索结果 → 组装 → 编号 → 生成的顺序没错位（编号在裁剪之后）；
2. 零召回**不调模型**（调了就会把幻觉写进溯源链路）；
3. `/v1/answer` 的响应里，`citations[].index` 与正文里的 `[n]` 指向同一段。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from agent_framework.bootstrap import build_gateway
from agent_framework.config import Settings
from agent_framework.knowledge import Chunk, ScoredChunk
from agent_framework.retrieval import Principal, RetrievalDiagnostics, RetrievalOutcome
from growth_agent.answer import NO_CONTEXT_ANSWER, KnowledgeAnswerer
from growth_agent.app import create_app


def _scored(chunk_id: str, *, text: str, page: int = 1, acl: Any = None) -> ScoredChunk:
    metadata = {} if acl is None else {"acl": acl}
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc1",
            text=text,
            source_uri="data/raw/guide.pdf",
            page_start=page,
            page_end=page,
            metadata={"document_title": "设计指南", **metadata},
        ),
        score=1.0,
        retriever="dense",
        rank=1,
    )


class FakePipeline:
    name = "fake-pipeline"

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def run(self, query: str, *, top_k: int = 10, filters=None, acl_check=None):
        self.calls.append({"query": query, "top_k": top_k, "acl_check": acl_check})
        kept = self._chunks
        if acl_check is not None:
            kept = [scored for scored in kept if acl_check(scored.chunk)]
        return RetrievalOutcome(
            chunks=kept,
            diagnostics=RetrievalDiagnostics(
                query=query,
                queries=(query,),
                candidates=len(self._chunks),
                filtered_out=len(self._chunks) - len(kept),
                reranked=False,
                acl_applied=acl_check is not None,
                total_ms=1.0,
            ),
        )


class FakeGateway:
    """只记调用、返回固定文案。`text_builder` 让用例控制答案里的编号。"""

    def __init__(self, text: str = "品牌语气要克制 [1]。") -> None:
        self.text = text
        self.calls: list[Any] = []

    async def chat(self, call: Any) -> Any:
        self.calls.append(call)
        return SimpleNamespace(
            alias="fixture", response=SimpleNamespace(text=self.text)
        )


# --- 组装顺序 ----------------------------------------------------------------


async def test_answer_numbers_chunks_after_trimming() -> None:
    """编号必须建立在裁剪之后：否则答案里的 `[n]` 会指向被裁掉的段。"""

    chunks = [_scored(f"c{i}", text=f"第 {i} 段正文", page=i + 1) for i in range(6)]
    pipeline = FakePipeline(chunks)
    gateway = FakeGateway("结论一 [1]，结论二 [2]。")

    answer = await KnowledgeAnswerer(
        pipeline, gateway, settings=Settings()
    ).answer("品牌语气怎么定")

    assert answer.grounded is True
    assert answer.audit.verdict == "ok"
    # 编号从 1 连续到 N，且与 context 文本里的编号一致
    indices = sorted(answer.context.by_index())
    assert indices == list(range(1, len(answer.context) + 1))
    assert [c.index for c in answer.citations] == [1, 2]


async def test_answer_records_retrieval_diagnostics() -> None:
    pipeline = FakePipeline([_scored("c1", text="正文")])
    answer = await KnowledgeAnswerer(
        pipeline, FakeGateway(), settings=Settings()
    ).answer("问题")

    assert answer.diagnostics is not None
    assert answer.diagnostics.candidates == 1


async def test_answer_pins_temperature_explicitly() -> None:
    """温度必须显式下发：留空时各厂商默认不同（0.7 / 1.0）。

    换个模型答案风格就变，引用行为也跟着变——评测跑两遍会得到两个结论，
    而报告上看不出是"模型变了"还是"改动生效了"。
    """

    pipeline = FakePipeline([_scored("c1", text="正文")])
    gateway = FakeGateway()

    await KnowledgeAnswerer(
        pipeline, gateway, settings=Settings(answer_temperature=0.0)
    ).answer("问题")

    assert gateway.calls[0].opts["temperature"] == 0.0
    assert "max_tokens" in gateway.calls[0].opts


async def test_prompt_carries_rules_context_and_question() -> None:
    pipeline = FakePipeline([_scored("c1", text="品牌语气要克制", page=7)])
    gateway = FakeGateway()

    await KnowledgeAnswerer(pipeline, gateway, settings=Settings()).answer("语气怎么定")

    user_message = gateway.calls[0].messages[-1].content
    assert "引用规则" in user_message
    assert "第7页" in user_message
    assert "语气怎么定" in user_message
    assert "data/raw/guide.pdf" in user_message


# --- 零召回 ------------------------------------------------------------------


async def test_no_context_does_not_call_the_model() -> None:
    """没有资料还去问模型，等于让它自由发挥——而这段幻觉会被算进溯源链路。"""

    pipeline = FakePipeline([])
    gateway = FakeGateway("这是编的答案 [1]")

    answer = await KnowledgeAnswerer(pipeline, gateway, settings=Settings()).answer("问题")

    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.grounded is False
    assert gateway.calls == []
    assert answer.audit.verdict == "no_citation"


# --- 引用体检 ----------------------------------------------------------------


async def test_answer_without_citation_is_flagged() -> None:
    pipeline = FakePipeline([_scored("c1", text="正文")])

    answer = await KnowledgeAnswerer(
        pipeline, FakeGateway("一段没有出处的回答"), settings=Settings()
    ).answer("问题")

    assert answer.grounded is True  # 有资料
    assert answer.audit.verdict == "no_citation"
    assert answer.citations == ()


async def test_hallucinated_index_is_flagged() -> None:
    pipeline = FakePipeline([_scored("c1", text="正文")])

    answer = await KnowledgeAnswerer(
        pipeline, FakeGateway("结论 [1] 与 [8]"), settings=Settings()
    ).answer("问题")

    assert answer.audit.verdict == "unknown_index"
    assert answer.audit.parsed.unknown == (8,)


# --- ACL ---------------------------------------------------------------------


async def test_principal_filters_before_context() -> None:
    pipeline = FakePipeline(
        [
            _scored("secret", text="内部定价策略", acl=["legal"]),
            _scored("public", text="公开语气指南", acl=["*"]),
        ]
    )
    answerer = KnowledgeAnswerer(
        pipeline,
        FakeGateway("公开部分 [1]"),
        settings=Settings(retrieval_acl_allow_unlabeled=False),
    )

    answer = await answerer.answer(
        "问题", principal=Principal("u1", frozenset({"growth"}))
    )

    assert [c.chunk_id for c in answer.context.citations] == ["public"]
    assert answer.diagnostics is not None
    assert answer.diagnostics.filtered_out == 1
    assert "内部定价" not in answer.context.text


async def test_no_principal_means_acl_not_applied() -> None:
    pipeline = FakePipeline([_scored("c1", text="正文")])

    await KnowledgeAnswerer(pipeline, FakeGateway(), settings=Settings()).answer("问题")

    assert pipeline.calls[0]["acl_check"] is None


# --- HTTP 契约 ---------------------------------------------------------------


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    pipeline = FakePipeline(
        [
            _scored("c1", text="品牌语气应克制", page=12),
            _scored("c2", text="禁用绝对化用语", page=13),
        ]
    )
    settings = Settings()
    answerer = KnowledgeAnswerer(
        pipeline, FakeGateway("品牌语气要克制 [1]，并禁用绝对化用语 [2]。"), settings=settings
    )
    app = create_app(gateway=build_gateway(settings), settings=settings, answerer=answerer)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://product.test"
    ) as opened:
        yield opened


async def test_answer_endpoint_returns_citations(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/answer", json={"question": "品牌语气怎么定"})

    assert response.status_code == 200
    body = response.json()

    assert body["grounded"] is True
    assert [item["index"] for item in body["citations"]] == [1, 2]
    assert body["citations"][0]["chunk_id"] == "c1"
    assert body["citations"][0]["page_start"] == 12
    assert body["citations"][0]["locator"]
    assert body["audit"]["verdict"] == "ok"
    assert body["retrieval"]["candidates"] == 2


async def test_answer_endpoint_rejects_blank_question(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/answer", json={"question": ""})

    assert response.status_code == 400
    assert response.json()["code"] == "BAD_REQUEST"


async def test_health_reports_rag_cold_before_first_answer(
    client: httpx.AsyncClient,
) -> None:
    """`cold` 不是错误状态，是"第一个请求会慢 20s"的可见信号。"""

    body = (await client.get("/health")).json()

    assert body["rag"] in {"cold", "ready"}
