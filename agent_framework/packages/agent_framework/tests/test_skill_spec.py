"""W8 D3：Skill 六层规范与 A14 参数校验。

规范的价值全在"不合法就别加载"：一个字段写错的 Skill 会以一个看起来
正常的方式失败，而那时已经很难分辨是模型不听话还是规范没写清。
"""

from __future__ import annotations

import pytest
from agent_framework.skills import (
    Check,
    SkillSpec,
    SkillSpecError,
    builtin_skills,
    validate_arguments,
)


def _spec(**overrides: object) -> SkillSpec:
    base: dict[str, object] = {
        "name": "demo_skill",
        "description": "一个用来测规范的 Skill。",
        "mission": "负责演示；不负责真的干活。",
        "principles": ("不要编数据",),
        "instructions": "1. 做一件事。",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        "entrypoint": "search.search_web",
        "permissions": frozenset({"read"}),
        "sandbox": "none",
        "stages": frozenset({"research"}),
        "done_criteria": (Check(kind="contains", target="完成", hint="要写完成"),),
    }
    base.update(overrides)
    return SkillSpec(**base)  # type: ignore[arg-type]


def test_builtin_library_passes_validation_and_lint() -> None:
    """内置库自己都过不了校验，说明规范或库有问题。"""

    specs = builtin_skills()

    assert len(specs) == 6
    for spec in specs:
        spec.validate()
        assert spec.lint() == (), f"{spec.name} 有 lint 提示：{spec.lint()}"


def test_missing_mission_is_rejected() -> None:
    with pytest.raises(SkillSpecError, match="mission"):
        _spec(mission="  ").validate()


def test_missing_instructions_is_rejected() -> None:
    with pytest.raises(SkillSpecError, match="instructions"):
        _spec(instructions="").validate()


def test_unknown_permission_is_a_load_error_not_a_silent_miss() -> None:
    """拼错权限词必须立刻报错：否则这个 Skill 永远不可见，而没人知道为什么。"""

    with pytest.raises(SkillSpecError, match="未知权限"):
        _spec(permissions=frozenset({"writes"})).validate()


def test_entrypoint_without_permissions_is_rejected() -> None:
    """有入口却不声明权限 = 想绕过白名单。"""

    with pytest.raises(SkillSpecError, match="permissions"):
        _spec(permissions=frozenset()).validate()


def test_write_permission_requires_sandbox() -> None:
    """有写/执行能力就不能 sandbox=none。"""

    with pytest.raises(SkillSpecError, match="sandbox"):
        _spec(permissions=frozenset({"write"}), sandbox="none").validate()


def test_dangerous_without_done_criteria_is_rejected() -> None:
    """没有完成标准的危险动作，事后无法验证"到底发没发出去"。"""

    with pytest.raises(SkillSpecError, match="done_criteria"):
        _spec(dangerous=True, permissions=frozenset({"publish"}), sandbox="subprocess",
              done_criteria=()).validate()


def test_description_over_limit_is_rejected() -> None:
    """描述是渐进式加载里**唯一**被注入的部分，太长就直接进成本。"""

    with pytest.raises(SkillSpecError, match="300"):
        _spec(description="长" * 301).validate()


def test_unknown_field_in_dict_is_rejected() -> None:
    """拼错的键被静默忽略 = 给了一个没有权限限制的 Skill。"""

    raw = {
        "name": "x_skill",
        "description": "x",
        "mission": "负责 x；不负责 y",
        "instructions": "1. x",
        "parameters": {"type": "object", "properties": {}},
        "permisions": ["read"],
    }
    with pytest.raises(SkillSpecError, match="未知字段"):
        SkillSpec.from_dict(raw)


def test_check_requires_hint() -> None:
    with pytest.raises(SkillSpecError, match="hint"):
        Check(kind="contains", target="x", hint="")


def test_check_rejects_bad_regex() -> None:
    with pytest.raises(SkillSpecError, match="正则"):
        Check(kind="regex", target="[", hint="补")


def test_check_kind_is_closed_set() -> None:
    with pytest.raises(SkillSpecError, match="kind"):
        Check(kind="quality", target="x", hint="补")  # type: ignore[arg-type]


def test_to_tool_spec_carries_boundary_into_description() -> None:
    """模型判断"该不该用"最需要的一句话，是使命边界。"""

    tool = builtin_skills()[3]  # publish_post
    spec = tool.to_tool_spec()

    assert spec.name == "publish_post"
    assert "不负责" in spec.description
    assert spec.parameters["required"] == ["title", "body"]


def test_verify_requires_write_or_publish() -> None:
    """只读动作没有"生效"可验，声明 verify 是配置错误。"""

    with pytest.raises(SkillSpecError, match="verify"):
        _spec(verify=__import__("agent_framework.skills", fromlist=["VerifySpec"]).VerifySpec(
            entrypoint="workspace.read_file"
        )).validate()


def test_stages_must_be_known() -> None:
    with pytest.raises(SkillSpecError, match="未知阶段"):
        _spec(stages=frozenset({"deploy"})).validate()


# --------------------------------------------------------------------------- #
# A14 参数校验（工具与 Skill 共用的唯一实现）
# --------------------------------------------------------------------------- #

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 2, "maxLength": 10},
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
        "mode": {"type": "string", "enum": ["fast", "deep"]},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def test_valid_arguments_pass() -> None:
    assert validate_arguments(SCHEMA, {"query": "增长", "count": 3, "mode": "fast"}) == []


def test_missing_required_is_reported() -> None:
    problems = validate_arguments(SCHEMA, {"count": 2})
    assert any("必填" in item for item in problems)


def test_unknown_argument_is_reported_when_additional_properties_false() -> None:
    problems = validate_arguments(SCHEMA, {"query": "增长", "site": "x"})
    assert any("schema 之外" in item for item in problems)


def test_unknown_argument_allowed_by_default() -> None:
    """JSON Schema 的默认语义是允许额外字段——不能替调用方改成严格。"""

    loose = dict(SCHEMA)
    loose.pop("additionalProperties")
    assert validate_arguments(loose, {"query": "增长", "site": "x"}) == []


def test_type_errors_are_reported() -> None:
    problems = validate_arguments(SCHEMA, {"query": "增长", "count": "3"})
    assert any("integer" in item for item in problems)


def test_bool_is_not_an_integer() -> None:
    """Python 里 `True` 是 `int` 的子类，工具边界必须显式堵住。"""

    problems = validate_arguments(SCHEMA, {"query": "增长", "count": True})
    assert any("integer" in item for item in problems)


def test_range_and_enum_are_enforced() -> None:
    problems = validate_arguments(SCHEMA, {"query": "增长", "count": 99, "mode": "fastest"})
    assert any("上限" in item for item in problems)
    assert any("enum" in item or "之一" in item for item in problems)


def test_length_limits_are_enforced() -> None:
    problems = validate_arguments(SCHEMA, {"query": "x" * 20})
    assert any("长度" in item for item in problems)


def test_all_problems_reported_at_once() -> None:
    """逐个报错会让调用方重试三次才凑齐一份完整报告。"""

    problems = validate_arguments(SCHEMA, {"query": 1, "count": 9, "mode": "x"})
    assert len(problems) >= 3


def test_empty_schema_rejects_everything_when_strict() -> None:
    """server 没声明 schema 时我们给的是严格空 schema：不猜它能接什么。"""

    strict_empty = {"type": "object", "properties": {}, "additionalProperties": False}
    assert validate_arguments(strict_empty, {"anything": 1}) != []
