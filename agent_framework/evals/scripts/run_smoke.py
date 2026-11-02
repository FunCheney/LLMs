"""W2 D5 · 10 条 smoke eval 的最小 runner（CI 门禁的执行体）。

它只回答一个问题：**没坏吗？**（"多好"是 W3 golden set + W4 judge 的事）

三条纪律
--------
1. **判据复用 conformance 的风格**：`expect` 里写的是"必须成立的断言"，
   不是"差不多就行"。另写一套宽松判据 = 给 CI 装了一个永远绿的门。
2. **指标只从 `RequestLog` 读**（复用 `bench_baseline.LogCapture`）：
   自己 `time.monotonic()` 会造出第二套 `first_token_ms`，W11 曲线立刻对不上。
3. **离线档绝不联网**：`mock` 回放夹具，`model=fixture`。真机档必须显式 `--live`，
   否则每次提交都在烧钱。

用法
----
    uv run python evals/scripts/run_smoke.py                       # 离线档（默认）
    uv run python evals/scripts/run_smoke.py --live                # 真机档（需 .env.local）
    uv run python evals/scripts/run_smoke.py --only smoke-07 -v    # 单条排查
    uv run python evals/scripts/run_smoke.py --json-out out.json   # 给 W11 曲线留数据

退出码：0 全过 / 1 有失败（CI 门禁就靠它）/ 2 环境不满足（例如 --live 没 key）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "evals" / "golden" / "smoke_v1.jsonl"
FIXTURES_ROOT = REPO_ROOT / "packages" / "agent_framework" / "tests" / "fixtures"

# 复用 W1 的口径实现（同一个 handler、同一份字段解析）。复制一份的代价是
# "两条命令读同一份日志却给出不同 p50"。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_framework.llm.client import ChatCall, LLMGateway  # noqa: E402
from agent_framework.llm.errors import (  # noqa: E402
    LLMError,
    error_code_for,
    http_status_for,
)
from agent_framework.llm.types import (  # noqa: E402
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolSpec,
)
from agent_framework.observability import get_request_logger  # noqa: E402
from bench_baseline import LogCapture  # noqa: E402
from offline_fixtures import build_offline_gateway  # noqa: E402

SMOKE_TOOLS = [
    ToolSpec(
        name="search_web",
        description="检索网页",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
]

_FIXTURE_STATUS: dict[tuple[str, str], int] = {}


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Case:
    id: str
    kind: str
    messages: list[Message]
    expect: dict[str, Any]
    fixture: dict[str, str] = field(default_factory=dict)
    model: str | None = None

    @property
    def provider(self) -> str:
        return self.fixture.get("provider", "openai")

    @property
    def fixture_case(self) -> str:
        return self.fixture.get("case", "text_basic")


@dataclass(slots=True)
class CaseResult:
    case: Case
    passed: bool
    failures: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []

    for line_no, raw in enumerate(path.read_text("utf-8").splitlines(), start=1):
        line = raw.strip()

        if not line or line.startswith("//"):
            continue

        try:
            payload = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc

        cases.append(
            Case(
                id=str(payload["id"]),
                kind=str(payload["kind"]),
                messages=[
                    Message(role=MessageRole(item["role"]), content=item.get("content"))
                    for item in payload["messages"]
                ],
                expect=dict(payload.get("expect", {})),
                fixture=dict(payload.get("fixture", {})),
                model=payload.get("model"),
            )
        )

    if not cases:
        raise SystemExit(f"{path} 里没有任何 case")

    return cases


# --------------------------------------------------------------------------- #
# 网关构造：离线档与真机档走同一个 LLMGateway，只换 adapter
# --------------------------------------------------------------------------- #


def _offline_gateway(case: Case) -> LLMGateway:
    """Case → 工厂参数。

    工厂本身在 `offline_fixtures.py`（W3 抽出，与 `run_eval.py` 共用）：
    两份实现漂移时，两条命令会对同一批夹具给出不同的重试次数，而报告看起来都正常。
    """

    return build_offline_gateway(
        provider=case.provider,
        fixture_case=case.fixture_case,
        expect_error=case.kind == "error",
    )


# --------------------------------------------------------------------------- #
# 断言
# --------------------------------------------------------------------------- #


def _check_text(text: str | None, case: Case) -> list[str]:
    failures: list[str] = []

    for needle in case.expect.get("text_contains", []):
        if not text or needle not in text:
            failures.append(f"文本里没有 {needle!r}（实际 {text!r}）")

    return failures


def _check_common(finish_reason: str | None, usage_estimated: bool | None, case: Case) -> list[str]:
    failures: list[str] = []
    expected_reason = case.expect.get("finish_reason")

    if expected_reason and finish_reason != expected_reason:
        failures.append(f"finish_reason 期望 {expected_reason}，实际 {finish_reason}")

    expected_estimated = case.expect.get("usage_estimated")

    if (
        expected_estimated is not None
        and usage_estimated is not None
        and usage_estimated != expected_estimated
    ):
        failures.append(
            f"usage_estimated 期望 {expected_estimated}，实际 {usage_estimated}"
        )

    return failures


async def _run_case(case: Case, gateway: LLMGateway, capture: LogCapture) -> CaseResult:
    capture.records.clear()
    call = ChatCall(messages=case.messages, tools=SMOKE_TOOLS)

    if case.kind == "error":
        try:
            await gateway.chat(call)
        except LLMError as exc:
            code = error_code_for(exc)
            status = http_status_for(exc)
            failures: list[str] = []

            expected_code = case.expect.get("error_code")
            expected_status = case.expect.get("http_status")

            if expected_code and code != expected_code:
                failures.append(f"error_code 期望 {expected_code}，实际 {code}")

            if expected_status and status != expected_status:
                failures.append(f"http_status 期望 {expected_status}，实际 {status}")

            return CaseResult(
                case=case,
                passed=not failures,
                failures=failures,
                metrics={"error_code": code, "http_status": status},
                error=str(exc),
            )
        else:
            return CaseResult(
                case=case,
                passed=False,
                failures=["期望抛错，但请求成功了（错误映射没被覆盖到）"],
            )

    if case.kind == "stream":
        types: list[str] = []
        chunks: list[str] = []
        finish_reason: str | None = None
        estimated: bool | None = None

        async for event in gateway.stream(call):
            types.append(event.type.value)
            chunks.append(_event_text(event))

            if event.type is StreamEventType.MESSAGE_END:
                finish_reason = str(event.finish_reason) if event.finish_reason else None
                estimated = event.usage.estimated if event.usage else None

        failures = _check_text("".join(chunks), case)
        failures += _check_common(finish_reason, estimated, case)
        failures += _check_frames(types, case)

        return CaseResult(
            case=case,
            passed=not failures,
            failures=failures,
            metrics=_metrics(capture, frames=len(types)),
        )

    outcome = await gateway.chat(call)
    response = outcome.response

    failures = _check_text(response.text, case)
    failures += _check_common(
        str(response.finish_reason), response.usage.estimated, case
    )

    expected_tool = case.expect.get("tool_name")

    if expected_tool:
        names = [item.name for item in response.tool_calls]

        if expected_tool not in names:
            failures.append(f"tool_calls 里没有 {expected_tool!r}（实际 {names}）")

    return CaseResult(
        case=case,
        passed=not failures,
        failures=failures,
        metrics=_metrics(capture),
    )


def _event_text(event: StreamEvent) -> str:
    if event.type is StreamEventType.TEXT_DELTA:
        return event.text_delta or ""

    return ""


def _check_frames(types: list[str], case: Case) -> list[str]:
    expected = case.expect.get("frame_types", [])

    if not expected:
        return []

    if types != expected:
        return [f"帧序列期望 {expected}，实际 {types}"]

    return []


def _metrics(capture: LogCapture, **extra: Any) -> dict[str, Any]:
    latest = capture.latest() or {}
    payload = {
        "latency_ms": latest.get("latency_ms"),
        "first_token_ms": latest.get("first_token_ms"),
        "prompt_tokens": latest.get("prompt_tokens"),
        "completion_tokens": latest.get("completion_tokens"),
        "cost": latest.get("cost"),
        "currency": latest.get("currency"),
        "retry_count": latest.get("retry_count"),
        "usage_estimated": latest.get("usage_estimated"),
        "trace_id": latest.get("trace_id"),
    }
    payload.update(extra)

    return payload


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def _print_report(results: list[CaseResult], *, mode: str, started: float) -> None:
    header = (
        f"{'id':<12}{'kind':<10}{'result':<8}{'first_token':>12}"
        f"{'latency':>10}{'tokens':>9}{'cost':>10}"
    )
    print(header)
    print("-" * len(header))

    for item in results:
        metrics = item.metrics
        tokens = int(metrics.get("prompt_tokens") or 0) + int(
            metrics.get("completion_tokens") or 0
        )
        first_token = metrics.get("first_token_ms")
        first_token_text = (
            f"{first_token:.1f}" if isinstance(first_token, float) else "-"
        )
        latency = metrics.get("latency_ms")
        latency_text = f"{latency:.1f}" if isinstance(latency, float) else "-"
        cost = metrics.get("cost")
        cost_text = f"{cost:.6f}" if isinstance(cost, float) else "-"

        print(
            f"{item.case.id:<12}{item.case.kind:<10}"
            f"{'PASS' if item.passed else 'FAIL':<8}"
            f"{first_token_text:>12}{latency_text:>10}{tokens:>9}{cost_text:>10}"
        )

        for failure in item.failures:
            print(f"    - {failure}")

    passed = sum(1 for item in results if item.passed)
    elapsed = time.perf_counter() - started

    print(
        f"\n{len(results)} cases: {passed} passed, {len(results) - passed} failed"
        f"（mode={mode}, {elapsed:.2f}s）"
    )

    if mode != "offline":
        print("提醒：真机档的延迟/成本受时段影响（DeepSeek 峰谷差 2 倍），对比时必须带时间戳")


def _write_json(path: Path, results: list[CaseResult], *, mode: str) -> None:
    payload = {
        "mode": mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cases": [
            {
                "id": item.case.id,
                "kind": item.case.kind,
                "passed": item.passed,
                "failures": item.failures,
                "metrics": item.metrics,
            }
            for item in results
        ],
        "summary": {
            "total": len(results),
            "passed": sum(1 for item in results if item.passed),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


async def _run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.golden))

    if args.only:
        cases = [case for case in cases if case.id in set(args.only)]

    if not cases:
        print("没有匹配的 case", file=sys.stderr)
        return 2

    capture = LogCapture()
    logger = get_request_logger()
    # 直接挂 handler（不改 level）：库不该改宿主 logging 配置，runner 自己负责看得见。
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)

    started = time.perf_counter()
    results: list[CaseResult] = []

    try:
        for case in cases:
            gateway = _offline_gateway(case)
            results.append(await _run_case(case, gateway, capture))
            await gateway.aclose()
    finally:
        logger.removeHandler(capture)

    _print_report(results, mode="offline", started=started)

    if args.verbose:
        for item in results:
            print(f"\n[{item.case.id}] {json.dumps(item.metrics, ensure_ascii=False)}")

    if args.json_out:
        _write_json(Path(args.json_out), results, mode="offline")

    return 0 if all(item.passed for item in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W2 smoke eval runner（默认离线档）")
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("--only", nargs="*", help="只跑指定 id")
    parser.add_argument("--json-out", default=None, help="把结果写成 JSON（给曲线用）")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="真机档：需要 .env.local 的 key（本周只留缝，默认不跑）",
    )
    args = parser.parse_args(argv)

    if args.live:
        print(
            "真机档本周不做：smoke 只回答'没坏'，真机成本/延迟归 "
            "evals/scripts/bench_baseline.py 与 -m live 用例。",
            file=sys.stderr,
        )
        return 2

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
