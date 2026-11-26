"""Feature Flag（W11 D5）。

这组测试守的是三条：
1. **每个 flag 的默认值都是旧行为**——默认值是新行为，回滚就得改代码发版；
2. **灰度按主体稳定分桶**——随机分桶会让同一用户在新旧行为之间来回跳；
3. **拼错的开关名必须炸**——静默忽略等于"以为开了其实没开"。
"""

from __future__ import annotations

from datetime import date

import pytest
from agent_framework.runtime.flags import FLAGS, FlagSet, bucket, parse_features
from gateway_helpers import make_settings

# --------------------------------------------------------------------------- #
# 默认值 = 旧行为
# --------------------------------------------------------------------------- #


def test_every_flag_defaults_to_the_old_behaviour() -> None:
    for flag in FLAGS.values():
        assert flag.default is False, f"{flag.name} 的默认值变成了新行为"


def test_unconfigured_flag_reports_old_behaviour_with_a_reason() -> None:
    state = FlagSet().state("semantic_cache")

    assert state.enabled is False
    assert "默认" in state.reason


def test_empty_features_means_everything_off() -> None:
    flags = FlagSet.from_settings(make_settings())

    assert all(not flags.enabled(name) for name in FLAGS)


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #


def test_parse_accepts_on_off_and_percent() -> None:
    parsed = parse_features(" semantic_cache=ON , difficulty_router=10% ")

    assert parsed == {"semantic_cache": "on", "difficulty_router": "10%"}


def test_parse_ignores_empty_chunks() -> None:
    assert parse_features(",,  ,") == {}
    assert parse_features(None) == {}


@pytest.mark.parametrize("raw", ["a=10", "a=yes", "a=on=off", "a=", "a=101%", "a=-5%"])
def test_parse_rejects_values_it_cannot_honour(raw: str) -> None:
    """`10` 忘了 `%` 如果被当成 `on`，就是 100% 放量。"""

    with pytest.raises(ValueError, match="FEATURES"):
        parse_features(raw)


def test_parse_rejects_entry_without_equals() -> None:
    with pytest.raises(ValueError, match="缺少 '='"):
        parse_features("semantic_cache")


def test_unregistered_flag_name_fails_at_startup_not_silently() -> None:
    with pytest.raises(ValueError, match="未登记"):
        FlagSet.from_settings(make_settings(features="semantic_cach=on"))


def test_unregistered_name_raises_on_lookup_too() -> None:
    with pytest.raises(KeyError, match="未登记"):
        FlagSet().enabled("no_such_flag")


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #


def test_explicit_on_and_off() -> None:
    flags = FlagSet(values={"semantic_cache": "on", "difficulty_router": "off"})

    assert flags.state("semantic_cache").enabled is True
    assert flags.state("difficulty_router").enabled is False
    assert flags.state("difficulty_router").reason == "显式关闭"


def test_zero_percent_is_off_for_everyone_and_hundred_is_on_for_everyone() -> None:
    none = FlagSet(values={"semantic_cache": "0%"})
    all_of_them = FlagSet(values={"semantic_cache": "100%"})

    for principal in ("alice", "bob", "carol"):
        assert none.enabled("semantic_cache", principal=principal) is False
        assert all_of_them.enabled("semantic_cache", principal=principal) is True


def test_bucket_is_stable_for_the_same_principal() -> None:
    assert bucket("semantic_cache", "alice") == bucket("semantic_cache", "alice")


def test_buckets_are_independent_between_flags() -> None:
    """不能出现"进了 A 就必然进 B"——那是把两个实验绑成了一个。"""

    pairs = [(bucket("semantic_cache", f"u{i}"), bucket("difficulty_router", f"u{i}"))
             for i in range(200)]

    assert len({(a, b) for a, b in pairs}) > 100


def test_rollout_is_by_principal_not_per_request() -> None:
    """同一主体反复判定必须得到同一个答案，否则用户体验像在抽风。"""

    flags = FlagSet(values={"semantic_cache": "30%"})
    results = {flags.enabled("semantic_cache", principal="alice") for _ in range(50)}

    assert len(results) == 1


def test_ten_percent_is_roughly_ten_percent() -> None:
    flags = FlagSet(values={"semantic_cache": "10%"})
    hit = sum(flags.enabled("semantic_cache", principal=f"u{i}") for i in range(2000))

    # 2000 个主体、10% 档：允许 140–260，超出就不是"分桶"而是"随机数写错了"
    assert 140 <= hit <= 260, f"实测 {hit}/2000"


def test_percent_without_principal_stays_off() -> None:
    """匿名流量做灰度没有意义：它无法被"同一批人持续走新逻辑"覆盖。"""

    state = FlagSet(values={"semantic_cache": "50%"}).state("semantic_cache")

    assert state.enabled is False
    assert "主体未知" in state.reason


def test_state_carries_the_numbers_needed_to_explain_itself() -> None:
    state = FlagSet(values={"semantic_cache": "10%"}).state(
        "semantic_cache", principal="alice"
    )

    assert state.bucket is not None
    assert state.rollout == 0.1
    assert "/10000" in state.reason


# --------------------------------------------------------------------------- #
# 运维
# --------------------------------------------------------------------------- #


def test_stale_lists_flags_past_their_removal_deadline() -> None:
    flags = FlagSet()

    assert flags.stale(today=date(2027, 6, 1)) == sorted(
        name for name, flag in FLAGS.items()
        if flag.removal_deadline is not None
    )


def test_flags_without_a_deadline_are_never_stale() -> None:
    """常态保护不是技术债——把它列进 stale 会让真正的过期项淹掉。"""

    assert all(
        flag.removal_deadline is not None
        for name, flag in FLAGS.items()
        if name in FlagSet().stale(today=date(2099, 1, 1))
    )


def test_snapshot_only_reports_configured_flags() -> None:
    snapshot = FlagSet(values={"semantic_cache": "on"}).snapshot()

    assert snapshot["values"] == {"semantic_cache": "on"}
    assert snapshot["effective"] == {"semantic_cache": 1.0}
