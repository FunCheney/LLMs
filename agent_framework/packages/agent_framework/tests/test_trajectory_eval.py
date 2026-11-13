"""W9 D5：Trajectory（过程）层指标的口径。

这个文件的作用不是"证明指标能算"，而是**把口径钉死**。过程指标最容易出两类错：

1. "算不出来"被悄悄折成 1.0 或 0（于是虚高/虚低，而读者不知道）；
2. 分母选错（比如把"没声明期望工具"也算进准确率的分母）。

每条用例都对着一个具体的口径决定，改口径必然弄红一条。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.scripts.trajectory_eval import (
    TrajectoryMetrics,
    aggregate_trajectories,
    invalid_call_rate,
    lcs_length,
    load_cases,
    main,
    path_repeat_rate,
    steps_used,
    tool_selection_acc,
)


def _rec(tool: str, digest: str = "d", status: str = "ok", code: str = "") -> dict:
    return {"tool": tool, "args_digest": digest, "status": status, "code": code}


# --------------------------------------------------------------------------- #
# 工具选择准确率
# --------------------------------------------------------------------------- #


class TestToolSelectionAcc:
    def test_perfect_match(self) -> None:
        records = [_rec("web_search", "a"), _rec("fetch_url", "b")]
        assert tool_selection_acc(records, ["web_search", "fetch_url"]) == 1.0

    def test_missing_tool_costs_proportionally(self) -> None:
        records = [_rec("web_search", "a")]
        assert tool_selection_acc(records, ["web_search", "fetch_url"]) == 0.5

    def test_reversed_order_keeps_half(self) -> None:
        """LCS 对**完全换序**只保留一半。这是它的性质，不是 bug。

        多数任务的工具顺序有语义（先 search 再 fetch），所以"换序要扣分"是
        想要的行为。需要"只看集合、不管顺序"的场景应该另算集合覆盖率，
        而不是把这条口径偷偷改掉——口径改了，历史数字就不可比了。
        """

        records = [_rec("fetch_url", "b"), _rec("web_search", "a")]
        assert tool_selection_acc(records, ["web_search", "fetch_url"]) == 0.5

    def test_extra_calls_are_not_penalised(self) -> None:
        """多调了不算选错——这是刻意的。

        "多调了**错的**"由 `invalid_call_rate` 抓。两个指标分工明确，
        否则一个数会被读成两种含义。
        """

        records = [_rec("web_search_v2"), _rec("web_search", "a")]
        assert tool_selection_acc(records, ["web_search"]) == 1.0

    def test_no_expected_tools_is_none_not_one(self) -> None:
        """没声明期望 → `None`。折成 1.0 会让准确率虚高。"""

        assert tool_selection_acc([_rec("web_search")], None) is None
        assert tool_selection_acc([_rec("web_search")], []) is None

    def test_lcs_helper(self) -> None:
        assert lcs_length([], ["a"]) == 0
        assert lcs_length(["a", "b", "c"], ["a", "c"]) == 2


# --------------------------------------------------------------------------- #
# 无效调用率
# --------------------------------------------------------------------------- #


class TestInvalidCallRate:
    def test_only_our_gates_count(self) -> None:
        """只算我们自己的门挡住的，不含对方系统报错。

        前者该改契约（schema/可见面），后者该等对方修——混在一起就分不清。
        """

        records = [
            _rec("ghost", code="tool_not_found", status="denied"),
            _rec("web_search", code="upstream_500", status="error"),
            _rec("web_search", "b"),
        ]
        # 1 / 3：上游 500 不算
        assert invalid_call_rate(records) == 1 / 3

    def test_no_calls_is_none(self) -> None:
        """没调用 ≠ 没有无效调用。"""

        assert invalid_call_rate([]) is None

    def test_sandbox_denied_counts(self) -> None:
        records = [_rec("run_checks", code="sandbox_denied", status="denied")]
        assert invalid_call_rate(records) == 1.0


# --------------------------------------------------------------------------- #
# 步数与路径重复
# --------------------------------------------------------------------------- #


class TestStepsAndRepeat:
    def test_steps_used_counts_tool_executions(self) -> None:
        assert steps_used([_rec("a"), _rec("b")]) == 2
        # 0 是真实观测：纯对话任务本来就不该调工具
        assert steps_used([]) == 0

    def test_repeat_rate_is_zero_when_all_distinct(self) -> None:
        assert path_repeat_rate([_rec("a", "1"), _rec("a", "2")]) == 0.0

    def test_repeat_rate_catches_same_tool_same_args(self) -> None:
        """同工具同参数出现两次 = 在原地打转。"""

        assert path_repeat_rate([_rec("a", "1"), _rec("a", "1")]) == 0.5

    def test_repeat_rate_none_without_calls(self) -> None:
        assert path_repeat_rate([]) is None


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #


class TestAggregate:
    def test_means_skip_none_and_report_samples(self) -> None:
        rows = [
            ([_rec("web_search", "1")], ["web_search"]),
            ([_rec("web_search", "2")], None),  # 没声明期望 → 不进准确率分母
            ([], ["web_search"]),  # 没调用 → 不进无效调用分母
        ]
        metrics = aggregate_trajectories(rows)

        assert metrics.cases == 3
        # 第 1 条 1.0（调对了），第 3 条 0.0（一个都没调）
        assert metrics.tool_selection_acc == 0.5
        assert metrics.tool_selection_samples == 2
        assert metrics.invalid_call_samples == 2
        # 步数三个样本都算得出（含"0 次调用"那条），均值 = (1+1+0)/3
        assert metrics.steps_samples == 3
        assert metrics.steps_used == pytest.approx(2 / 3)
        assert isinstance(metrics, TrajectoryMetrics)

    def test_empty_batch(self) -> None:
        metrics = aggregate_trajectories([])
        assert metrics.cases == 0
        assert metrics.steps_used is None
        assert metrics.tool_selection_acc is None

    def test_as_dict_has_all_fields(self) -> None:
        payload = aggregate_trajectories([]).as_dict()
        for key in (
            "tool_selection_acc",
            "invalid_call_rate",
            "steps_used",
            "path_repeat_rate",
            "cases",
        ):
            assert key in payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCli:
    def test_load_and_group(self, tmp_path: Path, capsys) -> None:
        path = tmp_path / "traces.jsonl"
        lines = [
            "// 注释行要被跳过",
            json.dumps(
                {
                    "id": "c1",
                    "expected_tools": ["web_search"],
                    "records": [_rec("web_search", "1")],
                    "config": "react",
                    "difficulty": 3,
                }
            ),
            json.dumps(
                {
                    "id": "c2",
                    "expected_tools": ["web_search"],
                    "records": [_rec("web_search", "1"), _rec("web_search", "1")],
                    "config": "multi",
                    "difficulty": 3,
                }
            ),
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

        cases = load_cases(path)
        assert [case.id for case in cases] == ["c1", "c2"]

        assert main(["--input", str(path), "--group-by", "config", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"react", "multi"}
        assert payload["react"]["path_repeat_rate"] == 0.0
        assert payload["multi"]["path_repeat_rate"] == 0.5

    def test_empty_file_returns_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("// 只有注释\n", encoding="utf-8")
        assert main(["--input", str(path)]) == 1
