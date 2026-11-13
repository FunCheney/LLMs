"""资产版本（W11 D5）。

只给代码打 tag，等于只能回滚 1/6。这组测试守的是"六类都要有版本、
缺了会显形"——因为少一类回滚路径这件事，在复盘那天才会被发现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_framework.runtime.assets import (
    ASSET_KINDS,
    AssetSnapshot,
    AssetVersion,
    capture,
    content_version,
    rollback_table,
    skill_drift,
    skill_fingerprint,
)
from gateway_helpers import make_catalog, make_settings


@dataclass
class FakeSkill:
    name: str = "seo"
    version: str = "1"
    description: str = "写 SEO 文章"
    mission: str = "让文章被搜到"
    principles: tuple[str, ...] = ("先结论后论证",)
    instructions: str = "按平台规范改写"
    parameters: dict[str, Any] = field(default_factory=dict)
    entrypoint: str = "run"
    permissions: tuple[str, ...] = ("read",)


def version(kind: str, name: str = "x", value: str = "1") -> AssetVersion:
    return AssetVersion(kind=kind, name=name, version=value, source="", rollback="")


# --------------------------------------------------------------------------- #
# 指纹
# --------------------------------------------------------------------------- #


def test_content_version_ignores_key_order() -> None:
    """同一份配置换个键顺序不算改版。"""

    assert content_version({"a": 1, "b": 2}) == content_version({"b": 2, "a": 1})


def test_content_version_changes_when_content_changes() -> None:
    assert content_version({"a": 1}) != content_version({"a": 2})


def test_content_version_handles_non_ascii_without_collapsing() -> None:
    assert content_version("中文") != content_version("英文")


# --------------------------------------------------------------------------- #
# 六类资产
# --------------------------------------------------------------------------- #


def test_capture_covers_all_six_kinds() -> None:
    snapshot = capture(
        make_settings(),
        catalog=make_catalog(["primary"]),
        skills=[FakeSkill()],
    )

    assert snapshot.missing_kinds == ()
    assert set(snapshot.by_kind()) == set(ASSET_KINDS)


def test_missing_skill_kind_is_reported_not_papered_over() -> None:
    """没传 Skills 时就该显形，而不是编一个空版本号。"""

    snapshot = capture(make_settings(), catalog=make_catalog(["primary"]))

    assert snapshot.missing_kinds == ("skill",)


def test_digest_is_stable_and_changes_with_models() -> None:
    settings = make_settings()

    one = capture(settings, catalog=make_catalog(["a", "b"]), skills=[FakeSkill()])
    again = capture(settings, catalog=make_catalog(["a", "b"]), skills=[FakeSkill()])
    other = capture(settings, catalog=make_catalog(["a"]), skills=[FakeSkill()])

    assert one.digest == again.digest
    assert one.digest != other.digest


def test_model_version_records_the_wire_name_not_the_file_hash() -> None:
    """alias 指向才是行为：models.yaml 里改一行注释不该算改版。"""

    snapshot = capture(make_settings(), catalog=make_catalog(["primary"]), skills=[FakeSkill()])
    models = snapshot.by_kind()["model"]

    assert models[0].name == "primary"
    assert models[0].version == "openai:primary-wire"


def test_rag_and_memory_versions_follow_settings() -> None:
    dense = capture(make_settings(retrieval_top_k=8), catalog=make_catalog(["a"]))
    wide = capture(make_settings(retrieval_top_k=20), catalog=make_catalog(["a"]))
    governed = capture(make_settings(memory="governed"), catalog=make_catalog(["a"]))

    assert dense.by_kind()["rag"][0].version != wide.by_kind()["rag"][0].version
    assert dense.by_kind()["memory"][0].version == "off"
    assert governed.by_kind()["memory"][0].version == "governed"


def test_code_version_falls_back_to_dev_without_git_sha(monkeypatch: Any) -> None:
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("GIT_COMMIT", raising=False)

    snapshot = capture(make_settings(), catalog=make_catalog(["a"]))

    assert snapshot.by_kind()["code"][0].version == "dev"


# --------------------------------------------------------------------------- #
# 对比与回滚表
# --------------------------------------------------------------------------- #


def test_diff_reports_added_removed_and_changed() -> None:
    before = AssetSnapshot(versions=(version("model", "a", "v1"), version("model", "gone", "v1")))
    after = AssetSnapshot(versions=(version("model", "a", "v2"), version("model", "new", "v1")))

    changed = after.diff(before)

    assert any("a: v1 → v2" in item for item in changed)
    assert any("gone: v1 → —" in item for item in changed)
    assert any("new: — → v1" in item for item in changed)


def test_diff_of_identical_snapshots_is_empty() -> None:
    snapshot = AssetSnapshot(versions=(version("code"),))

    assert snapshot.diff(snapshot) == []


def test_rollback_table_has_one_entry_per_present_kind() -> None:
    snapshot = AssetSnapshot(versions=(
        AssetVersion("code", "image", "abc", "GIT_SHA", "回退镜像 tag"),
        AssetVersion("model", "a", "openai:x", "alias", "改 alias"),
        AssetVersion("model", "b", "openai:y", "alias", "改 alias"),
    ))

    table = rollback_table(snapshot)

    assert table["code"] == "回退镜像 tag"
    assert table["model"] == "改 alias"  # 同类只留一条，不重复六行
    assert "rag" not in table


def test_as_dict_is_json_ready() -> None:
    payload = capture(make_settings(), catalog=make_catalog(["a"])).as_dict()

    assert isinstance(payload["digest"], str)
    assert payload["missing_kinds"] == ["skill"]
    assert isinstance(payload["versions"], list)


# --------------------------------------------------------------------------- #
# Skill 漂移
# --------------------------------------------------------------------------- #


def test_skill_fingerprint_changes_when_content_changes() -> None:
    assert skill_fingerprint(FakeSkill()) != skill_fingerprint(
        FakeSkill(instructions="按平台规范改写，并保留数字")
    )


def test_skill_drift_flags_content_changed_without_version_bump() -> None:
    """手填版本号一定会忘——但忘的时候要有人说出来。"""

    drifted = skill_drift([
        FakeSkill(instructions="旧指令"),
        FakeSkill(instructions="新指令", version="1"),  # 内容变了、version 没变
    ])

    assert drifted and "seo" in drifted[0]


def test_skill_drift_is_silent_when_version_moves_with_content() -> None:
    assert skill_drift([
        FakeSkill(instructions="旧指令", version="1"),
        FakeSkill(instructions="新指令", version="2"),
    ]) == []


def test_skill_drift_is_silent_for_a_single_spec() -> None:
    assert skill_drift([FakeSkill()]) == []
