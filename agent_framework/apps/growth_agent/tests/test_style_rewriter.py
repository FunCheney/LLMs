"""W10 D2：多平台改写的约束、修复与失败隔离。

三条边界值得单独测，因为它们各自对应一种线上事故：

1. **约束是数据**：`PlatformSpec` 改了，`validate` 立刻跟着变
   （不是"改了配置但代码里还硬编码着旧上限"）。
2. **修复优先于报错**：超长/禁用词是确定性可修的；"缺话题标签"修不了就报错。
   两种处置混在一起，运营会以为校验只会拦人。
3. **失败隔离**：三平台里坏一个，另外两个的产物必须还在——
   否则一次失败把两次成功调用的钱一起扔掉。
"""

from __future__ import annotations

import pytest
from growth_agent.pipeline import (
    PLATFORMS,
    BrandProfile,
    PlatformFailure,
    PlatformSpec,
    StyleError,
    StyleRewriter,
)
from scripted import ScriptedGateway

ARTICLE_BODY = "2025 年 B2B 内容营销投入同比增长 30%[1]。内容资产的复利来自持续更新[2]。"


def make_article():  # noqa: ANN201 - 测试内小对象，标注类型反而看不清
    from growth_agent.pipeline import ArticleDraft, KeywordPlan

    return ArticleDraft(
        topic="内容营销怎么做",
        title="内容营销怎么做：为什么投入在上升",
        outline=("为什么投入在上升", "内容资产的复利"),
        body=ARTICLE_BODY,
        keywords=KeywordPlan(primary="内容营销"),
    )


def make_rewriter(gateway: ScriptedGateway, *, brand: BrandProfile | None = None) -> StyleRewriter:
    return StyleRewriter(gateway, brand=brand)  # type: ignore[arg-type]


TINY = PlatformSpec(name="tiny", max_chars=12, tone="短")
NO_BREAK = PlatformSpec(name="nobreak", max_chars=200, tone="长", required_elements=("\n\n",))


# --------------------------------------------------------------------------- #
# 约束（纯函数）
# --------------------------------------------------------------------------- #


def test_validate_flags_over_length() -> None:
    rewriter = make_rewriter(ScriptedGateway(["x"]))
    violations = rewriter.validate("一" * 20, TINY)

    assert any("超出平台上限" in item for item in violations)


def test_validate_flags_missing_required_element() -> None:
    rewriter = make_rewriter(ScriptedGateway(["x"]))
    violations = rewriter.validate("一段没有空行的长文", NO_BREAK)

    assert any("缺少必须元素" in item for item in violations)
    # 元素名里的换行要显示成 \n，否则报错信息里是两个空行，读不出是什么
    assert "\\n" in violations[0]


def test_validate_flags_platform_forbidden() -> None:
    rewriter = make_rewriter(ScriptedGateway(["x"]))
    violations = rewriter.validate("加V领取资料", PLATFORMS["xiaohongshu"])

    assert any("微信" in item or "加V" in item for item in violations)


def test_brand_words_are_merged_not_overridden() -> None:
    """平台词与品牌词是两条独立的线，合并不覆盖。"""

    rewriter = make_rewriter(
        ScriptedGateway(["x"]), brand=BrandProfile(forbidden=("最强",))
    )
    forbidden = rewriter._forbidden_for(PLATFORMS["xiaohongshu"])  # noqa: SLF001

    assert "微信" in forbidden  # 平台
    assert "最强" in forbidden  # 品牌


# --------------------------------------------------------------------------- #
# 修复（确定性）
# --------------------------------------------------------------------------- #


def test_repair_truncates_at_sentence_end() -> None:
    rewriter = make_rewriter(ScriptedGateway(["x"]))
    text = "第一句很短。第二句稍微长一点。第三句还没写完"
    repaired, repairs = rewriter.repair(text, PlatformSpec(name="t", max_chars=16, tone=""))

    assert repaired == "第一句很短。第二句稍微长一点。"
    assert repairs and "按句末截断" in repairs[0]


def test_repair_hard_truncates_when_no_sentence_end() -> None:
    rewriter = make_rewriter(ScriptedGateway(["x"]))
    repaired, repairs = rewriter.repair("一" * 30, PlatformSpec(name="t", max_chars=10, tone=""))

    assert len(repaired) == 10
    assert repairs and "硬截断" in repairs[0]


def test_repair_strips_links_for_x() -> None:
    """X 对外链降权——这条是投放约束，删链接比被降权好。"""

    rewriter = make_rewriter(ScriptedGateway(["x"]))
    text = "看看这篇 https://example.test/post 很有用"
    repaired, repairs = rewriter.repair(text, PLATFORMS["x"])

    assert "https://" not in repaired
    assert "https://" in repairs[0]


def test_repair_strips_brand_word() -> None:
    rewriter = make_rewriter(ScriptedGateway(["x"]), brand=BrandProfile(forbidden=("最强",)))
    repaired, repairs = rewriter.repair("这是最强的内容策略", TINY)

    # TINY 的上限 12 会让它同时被截断，这里只关心禁词被删掉
    assert "最强" not in repaired
    assert "最强" in repairs[0]


# --------------------------------------------------------------------------- #
# 端到端三个平台
# --------------------------------------------------------------------------- #

GOOD = {
    "xiaohongshu": "做内容营销最怕一次性生产。\n\n我们试了三个月持续更新，复利看得见。#内容营销",
    "x": "内容营销的复利来自持续更新，不是一次性生产。#内容营销",
    "linkedin": "内容营销的复利来自持续更新。\n\n我们试了三个月，复利看得见。#内容营销",
}


async def test_three_platforms_preserve_requested_order() -> None:
    gateway = ScriptedGateway([GOOD["x"], GOOD["xiaohongshu"], GOOD["linkedin"]])
    results = await make_rewriter(gateway).rewrite_all(
        make_article(), ("x", "xiaohongshu", "linkedin")
    )

    # 顺序必须跟请求一致：按完成顺序会让响应里的三段随机变，前端没法 diff
    assert [r.platform for r in results] == ["x", "xiaohongshu", "linkedin"]  # type: ignore[union-attr]
    assert all(isinstance(r, object) and not isinstance(r, PlatformFailure) for r in results)


async def test_platform_prompt_drops_citation_markers_instruction() -> None:
    gateway = ScriptedGateway([GOOD["x"]])
    await make_rewriter(gateway).rewrite(make_article(), PLATFORMS["x"])

    prompt = gateway.prompts()[0]
    assert "引用标号 [n] 全部去掉" in prompt
    assert "不要新增原文没有的事实" in prompt
    # 平台硬上限必须进 prompt，否则模型不知道 280 字是什么概念
    assert "280" in prompt


async def test_missing_required_element_fails_that_platform() -> None:
    gateway = ScriptedGateway(["一段没有空行的文案 #tag"])
    rewriter = make_rewriter(gateway)

    with pytest.raises(StyleError) as caught:
        await rewriter.rewrite(make_article(), PLATFORMS["linkedin"])

    assert caught.value.platform == "linkedin"
    assert caught.value.violations


async def test_one_failure_does_not_kill_the_others() -> None:
    """三份产物里坏一份，另外两份必须还在。"""

    gateway = ScriptedGateway(
        [
            GOOD["xiaohongshu"],
            GOOD["x"],
            "没有空行的领英文案 #tag",  # linkedin 缺 \n\n → 失败
        ]
    )
    results = await make_rewriter(gateway).rewrite_all(make_article())

    ok = [r for r in results if not isinstance(r, PlatformFailure)]
    bad = [r for r in results if isinstance(r, PlatformFailure)]

    assert [r.platform for r in ok] == ["xiaohongshu", "x"]
    assert [r.platform for r in bad] == ["linkedin"]
    assert bad[0].violations  # 失败必须带原因，否则运营只能重试同一个 prompt


async def test_unknown_platform_raises_before_any_call() -> None:
    gateway = ScriptedGateway(["never used"])
    with pytest.raises(StyleError):
        await make_rewriter(gateway).rewrite_all(make_article(), ("weibo",))

    assert gateway.count == 0


async def test_empty_reply_is_a_failure_not_an_empty_post() -> None:
    gateway = ScriptedGateway(["   "])
    with pytest.raises(StyleError) as caught:
        await make_rewriter(gateway).rewrite(make_article(), PLATFORMS["x"])

    assert "为空" in caught.value.message


async def test_repairs_are_reported_so_quality_is_distinguishable() -> None:
    """靠截断才过 vs 一次过，是两种质量——报告里必须能区分。"""

    long_text = "内容营销的复利来自持续更新。" * 20 + "#内容营销"
    gateway = ScriptedGateway([long_text])
    result = await make_rewriter(gateway).rewrite(make_article(), PLATFORMS["x"])

    assert result.repairs
    assert result.chars <= PLATFORMS["x"].max_chars
