"""W10 D1：SEO 流水线的四段与三道硬门。

这个文件里的用例分三类，动机完全不同：

- **接线**（四段各调一次、trace 记全、prompt 里注入了品牌与规则）：
  防止"改了 prompt 但某一段没接上"这种静默失效。
- **硬门**（无证据拒绝成文、悬空编号拒绝、事实句漏标拒绝）：
  这三条是 W10 的核心承诺，每一条都配了反向用例（能过的写法确实能过），
  否则把它们写成"总是抛错"也能让测试变绿。
- **宽容**（关键词/大纲/meta 解析失败时的退化）：
  这些位置不该让整条链路 500，得证明退化后的产物是**可用的**而不是空壳。
"""

from __future__ import annotations

import pytest
from agent_framework.knowledge.citation import Citation, CitationContext
from growth_agent.pipeline import (
    BrandProfile,
    SEOError,
    SEOPipeline,
    unpublished_claims,
)
from scripted import ScriptedGateway, upstream_error

# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #

EVIDENCE = CitationContext(
    text=(
        "[1] 内容营销白皮书 | https://example.test/whitepaper | 第 12 页\n"
        "2025 年 B2B 内容营销投入同比增长 30%，其中长视频与深度报告占比最高。\n"
        "[2] 增长手册 | https://example.test/handbook | 第 3 页\n"
        "内容资产的复利来自持续更新，而非一次性生产。"
    ),
    citations=(
        Citation(
            index=1,
            chunk_id="c1",
            document_id="d1",
            source_uri="https://example.test/whitepaper",
            title="内容营销白皮书",
            page_start=12,
        ),
        Citation(
            index=2,
            chunk_id="c2",
            document_id="d2",
            source_uri="https://example.test/handbook",
        ),
    ),
)

KEYWORDS = "primary: 内容营销\nsecondary: 内容营销策略, B2B 内容\nintent: informational"
OUTLINE = "- 为什么内容营销的投入在上升\n- 内容资产的复利\n- 怎么开始"
# 第一句带数字与标号（合规），第二句是过渡句（不需要标号）。
BODY = "2025 年 B2B 内容营销投入同比增长 30%[1]。因此内容资产的复利更值得关注[2]。"
META = (
    "description: 从投入趋势到资产复利，讲清内容营销该怎么起步\n"
    "slug: content-marketing-2025\n"
    "tags: 内容营销, B2B, 增长"
)



class FakeEvidence:
    """记录 query 的假证据源。空 `citations` 用来构造"零证据"。"""

    def __init__(self, context: CitationContext) -> None:
        self._context = context
        self.queries: list[str] = []

    async def gather(self, query: str) -> CitationContext:
        self.queries.append(query)
        return self._context


def make_pipeline(
    gateway: ScriptedGateway,
    *,
    evidence: FakeEvidence | None = None,
    brand: BrandProfile | None = None,
    strict: bool = True,
) -> SEOPipeline:
    return SEOPipeline(
        gateway,  # type: ignore[arg-type] - 只依赖 chat 方法签名
        evidence=evidence or FakeEvidence(EVIDENCE),
        brand=brand,
        strict_citations=strict,
    )


def healthy_gateway(body: str = BODY) -> ScriptedGateway:
    return ScriptedGateway([KEYWORDS, OUTLINE, body, META])


# --------------------------------------------------------------------------- #
# 接线
# --------------------------------------------------------------------------- #


async def test_four_stages_each_call_once() -> None:
    gateway = healthy_gateway()
    draft = await make_pipeline(gateway).run("内容营销怎么做")

    assert gateway.count == 4
    assert [trace.name for trace in draft.stages] == ["keyword", "outline", "draft", "meta"]
    assert all(trace.alias == "scripted" for trace in draft.stages)


async def test_article_carries_resolved_citations() -> None:
    draft = await make_pipeline(healthy_gateway()).run("内容营销怎么做")

    assert draft.coverage.ok is True
    assert draft.coverage.markers == (1, 2)
    assert draft.coverage.resolved == (1, 2)
    assert draft.coverage.dangling == ()
    assert draft.coverage.candidates == 2
    # 引用不是"编号对得上"就算，出处要能落到原文
    assert {c.source_uri for c in draft.citations} == {
        "https://example.test/whitepaper",
        "https://example.test/handbook",
    }


async def test_draft_prompt_contains_rules_brand_and_context() -> None:
    gateway = healthy_gateway()
    brand = BrandProfile(tone="克制、给数据", forbidden=("颠覆",))
    await make_pipeline(gateway, brand=brand).run("内容营销怎么做")

    draft_prompt = gateway.prompts()[2]
    # 规则：必须交代"无出处不写"与编号格式
    assert "每个陈述事实的句子后面紧跟引用标号" in draft_prompt
    # 品牌：调性与禁用词都进 prompt（否则"注入品牌记忆"只是文档里的一句话）
    assert "克制、给数据" in draft_prompt
    assert "颠覆" in draft_prompt
    # 证据：检索到的原文要真进 prompt，而不是只把数量传进去
    assert "同比增长 30%" in draft_prompt


async def test_evidence_query_appends_two_tail_keywords() -> None:
    evidence = FakeEvidence(EVIDENCE)
    await make_pipeline(healthy_gateway(), evidence=evidence).run("内容营销怎么做")

    assert evidence.queries == ["内容营销怎么做 内容营销策略 B2B 内容"]


async def test_stage_trace_carries_tokens() -> None:
    draft = await make_pipeline(healthy_gateway()).run("内容营销怎么做")

    for trace in draft.stages:
        assert trace.prompt_tokens == 100
        assert trace.completion_tokens == 50
        assert trace.ms >= 0


# --------------------------------------------------------------------------- #
# 硬门 1：没有证据不成文
# --------------------------------------------------------------------------- #


async def test_no_evidence_refuses_to_write() -> None:
    """零召回时**不能**让模型凭常识写一篇看起来完整的稿子。"""

    empty = CitationContext(text="", citations=())
    gateway = healthy_gateway()
    pipeline = make_pipeline(gateway, evidence=FakeEvidence(empty))

    with pytest.raises(SEOError) as caught:
        await pipeline.run("内容营销怎么做")

    assert caught.value.stage == "evidence"
    # 关键：连一次模型调用都不该发生（省下的不只是钱，还有"看起来成功"的假象）
    assert gateway.count == 2  # 只跑了 keyword + outline


# --------------------------------------------------------------------------- #
# 硬门 2：悬空编号
# --------------------------------------------------------------------------- #


async def test_dangling_marker_is_rejected() -> None:
    body = "2025 年投入同比增长 30%[7]。"
    pipeline = make_pipeline(healthy_gateway(body))

    with pytest.raises(SEOError) as caught:
        await pipeline.run("内容营销怎么做")

    assert caught.value.stage == "draft"
    assert caught.value.detail["dangling"] == [7]


async def test_body_without_any_marker_is_rejected() -> None:
    body = "内容营销很重要，应该长期投入。"
    pipeline = make_pipeline(healthy_gateway(body))

    with pytest.raises(SEOError) as caught:
        await pipeline.run("内容营销怎么做")

    assert "没有任何引用标号" in str(caught.value)


# --------------------------------------------------------------------------- #
# 硬门 3：事实句漏标（含反向用例）
# --------------------------------------------------------------------------- #


async def test_fact_sentence_without_marker_is_rejected() -> None:
    body = "2025 年投入同比增长 30%[1]。据行业统计，长视频的占比最高。"
    pipeline = make_pipeline(healthy_gateway(body))

    with pytest.raises(SEOError) as caught:
        await pipeline.run("内容营销怎么做")

    assert caught.value.stage == "draft"
    assert len(caught.value.detail["unpublished"]) == 1


async def test_transition_sentence_needs_no_marker() -> None:
    """反向用例：观点句/过渡句不许被误判，否则这条检查会被运营关掉。"""

    body = "内容资产的复利更值得关注[2]。因此我们建议先从小规模更新做起。"
    draft = await make_pipeline(healthy_gateway(body)).run("内容营销怎么做")

    assert draft.coverage.unpublished == ()
    assert draft.coverage.ok is True


async def test_non_strict_mode_reports_instead_of_raising() -> None:
    """调 prompt 时用宽档：漏标的句子变成 `coverage` 里的一行，而不是 500。

    注意它**仍然**会把 `coverage.ok` 判成 False——宽档只是不拦断，
    不是把体检结果改好。这两件事混在一起，就会出现"报告说通过、稿子其实没出处"。
    """

    body = "2025 年投入增长 30%[1]。据行业统计，长视频占比最高。"
    draft = await make_pipeline(healthy_gateway(body), strict=False).run("内容营销怎么做")

    assert len(draft.coverage.unpublished) == 1
    assert draft.coverage.ok is False


def test_unpublished_claims_detects_signals() -> None:
    assert unpublished_claims("据报告显示，转化率提升了 2 倍。") != ()
    assert unpublished_claims("2025 年市场规模达到 100 亿元。") != ()
    assert unpublished_claims("我们认为这条路值得走。") == ()


# --------------------------------------------------------------------------- #
# 品牌
# --------------------------------------------------------------------------- #


async def test_brand_forbidden_word_blocks_draft() -> None:
    body = "2025 年投入增长 30%[1]，这是一次颠覆式变化。"
    brand = BrandProfile(forbidden=("颠覆",))
    pipeline = make_pipeline(healthy_gateway(body), brand=brand)

    with pytest.raises(SEOError) as caught:
        await pipeline.run("内容营销怎么做")

    assert caught.value.stage == "draft"
    assert caught.value.detail["hits"] == ["颠覆"]


# --------------------------------------------------------------------------- #
# 宽容：解析退化
# --------------------------------------------------------------------------- #


async def test_unparseable_keywords_fall_back_to_topic() -> None:
    gateway = ScriptedGateway(["我觉得这个问题挺有意思的。", OUTLINE, BODY, META])
    draft = await make_pipeline(gateway).run("内容营销怎么做")

    assert draft.keywords.primary == "内容营销怎么做"
    assert draft.keywords.secondary == ()
    # 退化后仍然要能成文
    assert draft.coverage.ok is True


async def test_outline_must_have_bullets() -> None:
    gateway = ScriptedGateway([KEYWORDS, "第一章 第二章", BODY, META])
    pipeline = make_pipeline(gateway)

    with pytest.raises(SEOError) as caught:
        await pipeline.run("内容营销怎么做")

    assert caught.value.stage == "outline"


async def test_meta_without_slug_derives_one() -> None:
    gateway = ScriptedGateway([KEYWORDS, OUTLINE, BODY, "description: Growth basics\ntags: a, b"])
    draft = await make_pipeline(gateway).run("内容营销怎么做")

    assert draft.meta["slug"] == "growth-basics"
    assert draft.meta["tags"] == ["a", "b"]
    assert draft.meta["json_ld"]["@type"] == "Article"
    # 没有作者/日期就别写，写空串比不写更糟
    assert "datePublished" not in draft.meta["json_ld"]


async def test_upstream_error_is_wrapped_with_stage() -> None:
    gateway = ScriptedGateway([KEYWORDS, upstream_error("上游 503")])
    pipeline = make_pipeline(gateway)

    with pytest.raises(SEOError) as caught:
        await pipeline.run("内容营销怎么做")

    assert caught.value.stage == "outline"
    assert "调用失败" in caught.value.message


async def test_empty_topic_is_rejected_before_any_call() -> None:
    gateway = healthy_gateway()
    with pytest.raises(SEOError) as caught:
        await make_pipeline(gateway).run("   ")

    assert caught.value.stage == "input"
    assert gateway.count == 0
