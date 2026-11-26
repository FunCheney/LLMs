"""W8 D4：最小模型可见面 + 注册表（动态加载 / 热重载 / 渐进式加载）。

Gate 8 #8 的判据是"任一 (principal, stage) 可见工具 < 全集 60%"。
这条判据不能只测一个组合：那只是"有一个组合达标"，不是"收敛生效"。
所以下面用矩阵遍历。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_framework.retrieval import Principal
from agent_framework.skills import (
    HIDDEN_BY_PRINCIPAL,
    HIDDEN_BY_STAGE,
    SkillRegistry,
    SkillSpecError,
    TaskStage,
    VisibilityPolicy,
    builtin_skills,
)

ALL = builtin_skills()

ANONYMOUS = Principal.anonymous()
VIEWER = Principal(subject_id="v1", groups=frozenset({"viewer"}))
EDITOR = Principal(subject_id="e1", groups=frozenset({"editor"}))
PUBLISHER = Principal(subject_id="p1", groups=frozenset({"publisher", "editor"}))
OPS = Principal(subject_id="o1", groups=frozenset({"ops"}))

PRINCIPALS = [ANONYMOUS, VIEWER, EDITOR, PUBLISHER, OPS]
STAGES = list(TaskStage)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("principal", PRINCIPALS)
def test_visible_surface_is_under_60_percent(stage: TaskStage, principal: Principal) -> None:
    """任一 (principal, stage) 组合下，可见工具必须少于全集的 60%。"""

    decision = VisibilityPolicy().surface(ALL, stage=stage, principal=principal)

    assert decision.total == len(ALL)
    assert decision.ratio < 0.6, f"{stage} × {principal.subject_id} 可见 {decision.names()}"


def test_research_stage_hides_publish_and_shell_tools() -> None:
    decision = VisibilityPolicy().surface(ALL, stage=TaskStage.RESEARCH, principal=EDITOR)

    assert set(decision.names()) == {"web_search", "web_fetch"}
    assert decision.reason("publish_post") == HIDDEN_BY_STAGE
    assert decision.reason("run_checks") == HIDDEN_BY_STAGE


def test_publish_stage_hides_shell_tools() -> None:
    decision = VisibilityPolicy().surface(ALL, stage=TaskStage.PUBLISH, principal=PUBLISHER)

    assert "run_checks" not in decision.names()
    assert "publish_post" in decision.names()


def test_publish_tool_is_hidden_from_principal_without_publish_permission() -> None:
    """粗粒度主体裁剪：看得见但 100% 会失败的工具不该出现在清单里。"""

    decision = VisibilityPolicy().surface(ALL, stage=TaskStage.PUBLISH, principal=VIEWER)

    assert "publish_post" not in decision.names()
    assert decision.reason("publish_post") == HIDDEN_BY_PRINCIPAL


def test_write_tool_stays_visible_for_viewer_so_denial_is_explainable() -> None:
    """`write` 不参与粗筛：这条让 `permission_denied` 这个码仍然可达。

    如果可见面把 write 也裁掉，"模型尝试越权写"就永远表现为
    `tool_not_visible`，两类完全不同的工单会被并到同一个队列里。
    """

    decision = VisibilityPolicy().surface(ALL, stage=TaskStage.DRAFT, principal=VIEWER)

    assert "write_brief" in decision.names()


def test_unknown_reason_for_nonexistent_skill() -> None:
    decision = VisibilityPolicy().surface(ALL, stage=TaskStage.RESEARCH)

    assert decision.reason("no_such_skill") == "unknown_skill"


def test_stage_can_be_passed_as_string() -> None:
    decision = VisibilityPolicy().surface(ALL, stage="research")

    assert decision.stage == "research"


def test_describe_all_is_much_cheaper_than_full_instructions() -> None:
    """Gate 8 #8 的第二半：渐进式加载要真的省 token（< 全量的 20%）。

    这是"30 个 Skill 也放得下"的前提：全量注入 instructions 是线性增长，
    只注入描述是常数级增长。
    """

    registry = SkillRegistry()
    decision = VisibilityPolicy().surface(ALL, stage=TaskStage.RESEARCH)

    progressive = decision.describe_all()
    full = registry.instructions_of(decision.names())

    assert len(progressive) < 0.2 * len(full), (len(progressive), len(full))


def test_progressive_description_contains_name_only_not_instructions() -> None:
    decision = VisibilityPolicy().surface(ALL, stage=TaskStage.RESEARCH)
    text = decision.describe_all()

    assert "web_search" in text
    # instructions 里的步骤文案不该出现在描述层。
    assert "把用户问题拆成" not in text


def test_visibility_as_dict_is_loggable() -> None:
    payload = VisibilityPolicy().surface(ALL, stage="draft", principal=VIEWER).as_dict()

    assert payload["stage"] == "draft"
    assert isinstance(payload["hidden"], list)
    assert 0.0 <= payload["ratio"] <= 1.0


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #

_SKILL_YAML = """
name: {name}
description: 由测试动态写入的 Skill。
mission: 负责演示动态加载；不负责真实业务。
principles:
  - 不要编数据
instructions: |
  1. 调 echo 打一行字。
parameters:
  type: object
  properties:
    command:
      type: string
  required: [command]
  additionalProperties: false
entrypoint: workspace.run_command
permissions: [read, shell]
sandbox: subprocess
stages: [maintenance]
done_criteria: []
"""


def test_builtins_are_registered() -> None:
    registry = SkillRegistry()

    assert registry.names() == tuple(
        sorted(spec.name for spec in ALL)
    )
    assert registry.get("web_search") is not None
    assert registry.get("nope") is None


def test_discover_new_file_then_call_without_touching_main_flow(tmp_path: Path) -> None:
    """检查点：「新增 Skill 文件 → reload 后可被 list 到并调用，主流程零改动」。"""

    (tmp_path / "echo_skill.yaml").write_text(_SKILL_YAML.format(name="echo_skill"), "utf-8")
    registry = SkillRegistry()

    loaded = registry.discover(tmp_path)

    assert [spec.name for spec in loaded] == ["echo_skill"]
    assert registry.has("echo_skill")
    assert "echo_skill" in registry.describe_all()


def test_invalid_file_is_skipped_and_reported(tmp_path: Path) -> None:
    """一个 YAML 写错不该整表清空——否则没人敢加东西。"""

    (tmp_path / "good.yaml").write_text(_SKILL_YAML.format(name="good_skill"), "utf-8")
    (tmp_path / "bad.yaml").write_text("name: bad\n", "utf-8")
    registry = SkillRegistry()

    loaded = registry.discover(tmp_path)

    assert [spec.name for spec in loaded] == ["good_skill"]
    assert len(registry.errors) == 1
    assert "bad.yaml" in registry.errors[0]


def test_duplicate_name_is_an_error_not_a_silent_override(tmp_path: Path) -> None:
    """静默覆盖会产生"你以为在调 A，实际调的是 B，而日志里只有一个名字"。"""

    (tmp_path / "dup.yaml").write_text(_SKILL_YAML.format(name="web_search"), "utf-8")
    registry = SkillRegistry()

    loaded = registry.discover(tmp_path)

    assert loaded == []
    assert any("重名" in error for error in registry.errors)


def test_register_validates(tmp_path: Path) -> None:
    registry = SkillRegistry(with_builtins=False)
    spec = builtin_skills()[0]

    registry.register(spec)

    assert registry.names() == (spec.name,)


def test_reload_picks_up_changes(tmp_path: Path) -> None:
    path = tmp_path / "dynamic.yaml"
    path.write_text(_SKILL_YAML.format(name="dynamic_skill"), "utf-8")
    registry = SkillRegistry()
    registry.discover(tmp_path)
    generation = registry.generation

    path.write_text(
        _SKILL_YAML.format(name="dynamic_skill").replace(
            "由测试动态写入的 Skill。", "改过描述的动态 Skill。"
        ),
        "utf-8",
    )

    import asyncio

    asyncio.run(registry.reload())

    assert registry.get("dynamic_skill").description.startswith("改过描述")
    assert registry.generation > generation


def test_reload_keeps_previous_skills_if_file_breaks(tmp_path: Path) -> None:
    """重载失败不该把已经能用的工具弄没。"""

    path = tmp_path / "dynamic.yaml"
    path.write_text(_SKILL_YAML.format(name="dynamic_skill"), "utf-8")
    registry = SkillRegistry()
    registry.discover(tmp_path)

    path.write_text("name: dynamic_skill\n", "utf-8")

    import asyncio

    asyncio.run(registry.reload())

    # 文件坏了：旧版本仍在（没被清空），并且错误被记录。
    assert registry.has("dynamic_skill")
    assert registry.errors


def test_unknown_skill_file_suffix_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("not a skill", "utf-8")
    registry = SkillRegistry()

    assert registry.discover(tmp_path) == []


def test_skill_file_with_batch_definitions_is_rejected(tmp_path: Path) -> None:
    """批量定义会让"哪个文件对应哪个 Skill"无法回答。"""

    (tmp_path / "batch.yaml").write_text("skills: []\n", "utf-8")
    registry = SkillRegistry()
    registry.discover(tmp_path)

    assert registry.errors
    assert "只能定义一个" in registry.errors[0]


def test_registry_snapshot_shape() -> None:
    from agent_framework.skills import registry_snapshot

    row = registry_snapshot(ALL)[0]

    assert set(row) == {
        "name",
        "description",
        "entrypoint",
        "permissions",
        "sandbox",
        "stages",
        "dangerous",
        "source",
        "checks",
    }


def test_skill_spec_errors_are_value_errors() -> None:
    """加载器按 `SkillSpecError` 收口，它必须是 ValueError 的子类，
    否则外层 `except ValueError` 会漏掉它。"""

    assert issubclass(SkillSpecError, ValueError)
