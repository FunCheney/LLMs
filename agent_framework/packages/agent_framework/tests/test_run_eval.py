"""批量 runner 的行为测试（W3 D3 / Gate 3、4、5）。

三条必须证明的事，缺一条整套评测就不可信：

1. **一条失败不中断整批**（Gate 5）——反例是"一条 502 毁掉整晚评测"；
2. **温度钉死 0**（Gate 4）——反例是"两次跑出两个数，无法复现"；
3. **离线结果不写进 quality**（ADR-W3-01 D2）——反例是"夹具回放的 100% 被当成质量结论"。

进程内测 `evaluate`/`snapshot`，子进程测退出码（退出码才是 CI 的判据）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from agent_framework.llm.client import ChatOutcome
from agent_framework.llm.types import FinishReason, LLMResponse, Message, Usage

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "evals" / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_eval  # noqa: E402
from bench_baseline import LogCapture  # noqa: E402

from evals.metrics import CaseStatus  # noqa: E402

GOLDEN = REPO_ROOT / "evals" / "golden" / "golden_v1.jsonl"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "evals/scripts/run_eval.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


# --------------------------------------------------------------------------- #
# 子进程：退出码
# --------------------------------------------------------------------------- #


def test_offline_eval_exits_zero() -> None:
    result = _run("--no-write")

    assert result.returncode == 0, result.stderr
    assert "success_rate 100.0%" in result.stdout


def test_single_failure_does_not_abort_batch() -> None:
    """改坏一条判据：它必须红，其余照跑，退出码非 0。"""

    result = _run("--no-write", "--break-case", "gold-c03")

    assert result.returncode == 1
    assert "gold-c03" in result.stdout
    # 总数仍是 54，说明没有在第一条失败处退出。
    assert "54 条" in result.stdout
    assert "passed 49" in result.stdout


def test_only_filter_runs_selected_case() -> None:
    result = _run("--no-write", "--only", "gold-t01")

    assert result.returncode == 0
    assert "1 条" in result.stdout


def test_unknown_only_id_exits_two() -> None:
    """输入问题用 2（环境/参数），不要和"评测失败"的 1 混起来。"""

    assert _run("--no-write", "--only", "gold-不存在").returncode == 2


# --------------------------------------------------------------------------- #
# 进程内：温度钉死与观测
# --------------------------------------------------------------------------- #


class _RecordingGateway:
    """只记录 call 并返回固定响应的假网关。"""

    def __init__(self) -> None:
        self.calls: list[object] = []

    async def chat(self, call: object) -> ChatOutcome:
        self.calls.append(call)
        return ChatOutcome(
            response=LLMResponse(
                message=Message(role="assistant", content="MCP 相关回答"),
                finish_reason=FinishReason.STOP,
                usage=Usage(input_tokens=5, output_tokens=2),
            ),
            alias="fixture",
        )


async def test_temperature_is_forced_to_zero() -> None:
    """case 里没写温度时也必须注入 0，否则默认值漂移没人发现。"""

    case = run_eval._build_case(
        {
            "id": "gold-c01",
            "kind": "chat",
            "difficulty": 1,
            "tags": ["chat"],
            "task": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "expect": {"finish_reason": "stop"},
        }
    )
    gateway = _RecordingGateway()
    capture = LogCapture()

    await run_eval.evaluate(case, gateway, capture, case.expect, alias="fixture")

    call = gateway.calls[0]
    assert call.opts == {"temperature": 0}
    assert call.model == "fixture"


async def test_passed_case_has_no_failure_categories() -> None:
    case = run_eval._build_case(
        {
            "id": "gold-c01",
            "kind": "chat",
            "difficulty": 1,
            "tags": ["chat"],
            "messages": [{"role": "user", "content": "hi"}],
            "expect": {"text_contains": ["MCP"]},
        }
    )
    gateway = _RecordingGateway()

    # W4 起 `evaluate` 返回 (EvalResult, JudgeResult|None)：judge 结果与指标分开走
    result, judged = await run_eval.evaluate(
        case, gateway, LogCapture(), case.expect, alias="fixture"
    )

    assert judged is None
    assert result.status is CaseStatus.PASSED
    assert result.categories == ()


# --------------------------------------------------------------------------- #
# 落盘结构
# --------------------------------------------------------------------------- #


def _results() -> list:
    return [
        _eval_result("a", CaseStatus.PASSED, latency=100.0, cost=0.002),
        _eval_result("b", CaseStatus.FAILED, latency=200.0, cost=0.004, categories=("format",)),
        _eval_result("c", CaseStatus.SKIPPED),
    ]


def _eval_result(
    case_id: str,
    status: CaseStatus,
    *,
    latency: float | None = None,
    cost: float | None = None,
    categories: tuple[str, ...] = (),
) -> object:
    from evals.metrics import CaseMetrics, EvalResult

    return EvalResult(
        case_id=case_id,
        status=status,
        tags=("tag",),
        metrics=CaseMetrics(latency_ms=latency, cost=cost, currency="CNY"),
        categories=categories,
    )


def test_offline_snapshot_keeps_quality_null() -> None:
    """离线跑通 ≠ 系统好：quality 必须是 null，数字进 pipeline。"""

    from evals.metrics import aggregate

    results = _results()
    payload = run_eval.snapshot(
        results,
        aggregate(results),
        mode="offline",
        week="2026-W46",
        golden_path=GOLDEN,
        golden_digest="deadbeef",
    )

    assert payload["quality"]["success_rate"] is None
    assert payload["quality"]["judge_score"] is None
    assert payload["quality"]["status"] == "pending-online"
    assert payload["pipeline"]["total"] == 3
    assert payload["pipeline"]["skipped"] == 1


def test_online_snapshot_writes_quality_block() -> None:
    from evals.metrics import aggregate

    results = _results()
    payload = run_eval.snapshot(
        results,
        aggregate(results),
        mode="online",
        week="2026-W46",
        golden_path=GOLDEN,
        golden_digest="deadbeef",
    )

    assert payload["quality"]["success_rate"] == pytest.approx(0.5)
    assert "pipeline" not in payload


def test_snapshot_keeps_failure_categories() -> None:
    from evals.metrics import aggregate

    results = _results()
    payload = run_eval.snapshot(
        results,
        aggregate(results),
        mode="offline",
        week="2026-W46",
        golden_path=GOLDEN,
        golden_digest="deadbeef",
    )

    assert payload["failures"] == {"format": 1}
    assert payload["cases"][1]["categories"] == ["format"]


def test_snapshot_metrics_are_json_serializable() -> None:
    """slots dataclass 不是 dict，直接用 `__dict__` 会在落盘时炸。"""

    import json

    from evals.metrics import aggregate

    results = _results()
    payload = run_eval.snapshot(
        results,
        aggregate(results),
        mode="offline",
        week="2026-W46",
        golden_path=GOLDEN,
        golden_digest="deadbeef",
    )

    json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def test_markdown_has_four_blocks_and_offline_disclaimer() -> None:
    from evals.metrics import aggregate

    results = _results()
    text = run_eval.render_markdown(
        results,
        aggregate(results),
        mode="offline",
        week="2026-W46",
        golden_digest="deadbeef",
    )

    for heading in ("## 1. 分层指标", "## 2. 难度与标签分布", "## 3. 趋势", "## 4. 失败分类"):
        assert heading in text

    assert "离线档声明" in text
    assert "judge_score（1–5）" in text
    assert "null（未开 judge）" in text
    assert "faithfulness" in text
