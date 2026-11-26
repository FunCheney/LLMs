"""W8 D3：完成标准校验（第 6 层的执行者）。

核心断言是这条：**没有证据的"已完成"必须被拦下**，而且拦下之后
只能重试一次，第二次仍不满足就如实报"未达标 + 缺什么"，绝不补分。
"""

from __future__ import annotations

from agent_framework.skills import Check, SkillSpec, SkillValidator


def _skill(*checks: Check) -> SkillSpec:
    return SkillSpec(
        name="demo",
        description="演示 Skill。",
        mission="负责演示；不负责生产。",
        principles=("不要编数据",),
        instructions="1. 写一段带引用的结果。",
        parameters={"type": "object", "properties": {}},
        entrypoint="",
        permissions=frozenset(),
        sandbox="none",
        stages=frozenset({"draft"}),
        done_criteria=checks,
    )


def test_contains_pass_and_fail() -> None:
    skill = _skill(Check(kind="contains", target="结论", hint="要写结论"))
    validator = SkillValidator()

    assert validator.verify(skill, "这是结论：A")[0] == []
    unmet, details = validator.verify(skill, "这是结果：A")
    assert unmet == ["要写结论"]
    assert details == ["contains:missing:结论"]


def test_cites_counts_markers() -> None:
    skill = _skill(Check(kind="cites", count=2, hint="至少两个引用"))
    validator = SkillValidator()

    assert validator.verify(skill, "A[1] B[2]")[0] == []
    unmet, details = validator.verify(skill, "A[1] B")
    assert unmet == ["至少两个引用"]
    assert details[0].startswith("cites:1<2")


def test_min_items_counts_bullets() -> None:
    skill = _skill(Check(kind="min_items", count=3, hint="至少三条"))
    validator = SkillValidator()

    assert validator.verify(skill, "- a\n- b\n1. c")[0] == []
    assert validator.verify(skill, "- a\n- b")[0] == ["至少三条"]


def test_regex_kind() -> None:
    skill = _skill(Check(kind="regex", target=r"[0-9a-f]{8}", hint="要给 post_id"))
    validator = SkillValidator()

    assert validator.verify(skill, "post_id: 1a2b3c4d")[0] == []
    assert validator.verify(skill, "post_id: none")[0] == ["要给 post_id"]


async def test_enforce_passes_first_time_without_regenerating() -> None:
    skill = _skill(Check(kind="contains", target="结论", hint="要写结论"))
    calls: list[str] = []

    async def regenerate(feedback: str) -> str:
        calls.append(feedback)
        return "不该被调用"

    result = await SkillValidator().enforce(skill, "结论在此", regenerate=regenerate)

    assert result.passed
    assert result.attempts == 1
    assert calls == []


async def test_enforce_retries_once_and_accepts_second_output() -> None:
    skill = _skill(Check(kind="contains", target="结论", hint="要写结论"))
    feedbacks: list[str] = []

    async def regenerate(feedback: str) -> str:
        feedbacks.append(feedback)
        return "结论：A"

    result = await SkillValidator().enforce(skill, "只有过程", regenerate=regenerate)

    assert result.passed
    assert result.attempts == 2
    assert result.output == "结论：A"
    # 回喂内容必须说清"缺什么"，不能只说"请重试"。
    assert "要写结论" in feedbacks[0]


async def test_enforce_never_passes_on_second_failure() -> None:
    """第二次仍不满足就如实报未达标 —— 和 W4 judge "解析失败不补分"同一条原则。"""

    skill = _skill(Check(kind="contains", target="结论", hint="要写结论"))

    async def regenerate(feedback: str) -> str:
        return "还是没有"

    result = await SkillValidator().enforce(skill, "只有过程", regenerate=regenerate)

    assert not result.passed
    assert result.attempts == 2
    assert result.unmet == ("要写结论",)


async def test_enforce_without_regenerate_reports_failure_immediately() -> None:
    skill = _skill(Check(kind="contains", target="结论", hint="要写结论"))

    result = await SkillValidator().enforce(skill, "只有过程")

    assert not result.passed
    assert result.attempts == 1


async def test_no_criteria_means_not_verified_rather_than_passed() -> None:
    """没有标准 ≠ 通过。`checked=0` 让报告能数出"多少次完成没被验证过"。"""

    result = await SkillValidator().enforce(_skill(), "任何输出都是对的")

    assert result.passed
    assert result.checked == 0
    assert result.details == ()


def test_multiple_checks_all_reported() -> None:
    skill = _skill(
        Check(kind="contains", target="结论", hint="要写结论"),
        Check(kind="cites", hint="要引用"),
    )

    unmet, _ = SkillValidator().verify(skill, "过程描述，无引用")

    assert len(unmet) == 2


def test_contains_can_be_fooled_by_negation() -> None:
    """已知弱点：`contains` 是子串匹配，否定句会误命中。

    "没有结论" 里包含 "结论" —— 这条断言**故意**写下这个行为，
    而不是假装它不存在。关键标准（如"已发布"/post_id）应优先用 `regex`，
    并在评审时按"模型可能怎么绕过去"想一遍；`skill-spec.md` 的已知缺口
    里记了这条。
    """

    skill = _skill(Check(kind="contains", target="结论", hint="要写结论"))

    unmet, _ = SkillValidator().verify(skill, "这一段没有结论，只是过程记录")

    assert unmet == []


def test_builtin_publish_skill_catches_output_without_post_id() -> None:
    """内置的 publish_post：没有 post_id 的输出不能算"已发布"。"""

    from agent_framework.skills import get_builtin

    skill = get_builtin("publish_post")
    unmet, _ = SkillValidator().verify(skill, "已经把文章发出去了，请放心。")

    assert unmet  # 既没 post_id 也没"已发布"
    assert any("post_id" in item for item in unmet)
