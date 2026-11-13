#!/usr/bin/env python
"""W11 D4：一个消融（Ablation）+ 一个分叉对照（Fork）。

    python evals/scripts/w11_fork_ablation.py            # 人读
    python evals/scripts/w11_fork_ablation.py --json     # 机器可读

## 实验 A（消融）：护栏检测前要不要先"解封皮"

W10 的真实事故：外层 `GuardrailPipeline` 把用户输入隔离成
`<untrusted source="user">…</untrusted>`，内层 `SEOPipeline` 的 `fake_delimiter`
规则把这段封皮判成"伪造边界"，于是 `/v1/article` 第一次跑就 500。

消融臂：
- **A1 不隔离**：`guard.screen(raw)`（单层护栏的裸行为）
- **A2 隔离后检测、不解封皮**：修复前的那条路径
- **A3 隔离后检测、先解封皮**：修复后的行为

量的是两个数：`expect: blocked` 的拦截率、`expect: allowed` 的放行率。
只看拦截率的话，A2 看起来"和 A3 一样好"——而它把正常输入全拦了。

## 实验 B（分叉）：`max_attempts` 取 1 还是 3

同一批请求（固定随机种子，50% 上游失败率），只改重试次数：

- 成功率能提到多少；
- 上游调用量会放大多少；
- **每成功一次要打几次上游**（这个数才是成本报告该看的）。

候选链在这里被钉成单模型（`call.model`），否则"重试"的效果会和"换候选"混在一起，
结论没法归因。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework.config import Settings, load_catalog  # noqa: E402
from agent_framework.guardrails.injection import InjectionGuard, isolate  # noqa: E402
from agent_framework.llm.client import ChatCall, LLMGateway  # noqa: E402
from agent_framework.llm.errors import LLMError  # noqa: E402
from agent_framework.llm.registry import Registry  # noqa: E402
from agent_framework.llm.router import TaskRouter  # noqa: E402
from agent_framework.llm.types import Message, MessageRole  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "golden" / "redteam_v1.jsonl"
ALIAS = "gpt-4o-mini"

os.environ.setdefault("OPENAI_API_KEY", "ablation-not-a-real-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "ablation-not-a-real-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "ablation-not-a-real-key")


# --------------------------------------------------------------------------- #
# 实验 A：解封皮消融
# --------------------------------------------------------------------------- #


@dataclass
class Arm:
    name: str
    blocked_total: int = 0
    blocked_caught: int = 0
    allowed_total: int = 0
    allowed_passed: int = 0
    false_positive_rules: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.false_positive_rules is None:
            self.false_positive_rules = {}

    @property
    def recall(self) -> float:
        return self.blocked_caught / self.blocked_total if self.blocked_total else 0.0

    @property
    def pass_rate(self) -> float:
        return self.allowed_passed / self.allowed_total if self.allowed_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.name,
            "injection_recall": round(self.recall, 4),
            "safe_pass_rate": round(self.pass_rate, 4),
            "false_positive_rules": dict(self.false_positive_rules),
        }


def load_cases() -> list[dict[str, Any]]:
    cases = []
    for line in CASES.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def run_ablation() -> list[Arm]:
    guard = InjectionGuard()
    cases = load_cases()

    arms = {
        "A1 不隔离": Arm("A1 不隔离"),
        "A2 隔离后不解封皮": Arm("A2 隔离后不解封皮"),
        "A3 隔离后解封皮（修复后）": Arm("A3 隔离后解封皮（修复后）"),
    }

    for case in cases:
        text, source = case["text"], case["source"]
        expect_block = case["expect"] == "blocked"
        wrapped = isolate(text, source=source)

        # A1：不隔离
        verdict_a1 = guard.screen(text, source=source)
        # A2：修复前的路径——直接拿隔离后的整段去跑规则（不先解封皮）
        hits_a2 = guard._rule_hits(wrapped, source=source)  # noqa: SLF001 - 这就是事故现场
        # A3：修复后
        verdict_a3 = guard.screen(wrapped, source=source)

        for arm, blocked, rules in (
            (arms["A1 不隔离"], not verdict_a1.allowed, verdict_a1.hits),
            (arms["A2 隔离后不解封皮"], bool(hits_a2), hits_a2),
            (arms["A3 隔离后解封皮（修复后）"], not verdict_a3.allowed, verdict_a3.hits),
        ):
            if expect_block:
                arm.blocked_total += 1
                arm.blocked_caught += int(blocked)
            else:
                arm.allowed_total += 1
                arm.allowed_passed += int(not blocked)
                for hit in rules:
                    arm.false_positive_rules[hit.rule] = (
                        arm.false_positive_rules.get(hit.rule, 0) + 1
                    )

    return list(arms.values())


def assert_unwrap_is_the_fix(arms: list[Arm]) -> None:
    """消融的结论必须能被断言：漏了 A3 的修复，A2 的误报是致命的。"""

    by_name = {arm.name: arm for arm in arms}
    a2 = by_name["A2 隔离后不解封皮"]
    a3 = by_name["A3 隔离后解封皮（修复后）"]

    assert a3.recall == 1.0, f"修复后拦截率应仍是 100%，实际 {a3.recall}"
    assert a3.pass_rate == 1.0, f"修复后正常输入应全放行，实际 {a3.pass_rate}"
    assert a2.pass_rate < 0.5, (
        f"消融没复现出事故：不解封皮时正常输入只被拦了 {1 - a2.pass_rate:.0%}"
    )
    assert a2.recall >= a3.recall, "A2 的拦截率不该比 A3 低（它拦得更多，只是拦错了）"


# --------------------------------------------------------------------------- #
# 实验 B：重试次数分叉
# --------------------------------------------------------------------------- #


@dataclass
class ForkResult:
    max_attempts: int
    requests: int
    succeeded: int
    upstream_calls: int
    latencies_ms: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.latencies_ms is None:
            self.latencies_ms = []

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.requests if self.requests else 0.0

    @property
    def calls_per_success(self) -> float:
        return self.upstream_calls / self.succeeded if self.succeeded else float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "requests": self.requests,
            "success_rate": round(self.success_rate, 4),
            "upstream_calls": self.upstream_calls,
            "calls_per_success": round(self.calls_per_success, 3),
            "note": "延迟不在本实验口径内：注入层为内存 MockTransport",
        }


def build_flaky_gateway(
    *, max_attempts: int, failure_rate: float, seed: int
) -> tuple[LLMGateway, list[int]]:
    """单候选 + 按种子随机的失败率。候选链钉死，否则重试与换候选混在一起。"""

    rng = random.Random(seed)
    counter = [0]
    settings = Settings(max_attempts=max_attempts, backoff_base_s=0.0, request_timeout_s=5.0)
    catalog = load_catalog(settings.models_yaml)

    def handler(request: httpx.Request) -> httpx.Response:
        counter[0] += 1
        if rng.random() < failure_rate:
            return httpx.Response(500, json={"error": {"message": "injected 500"}})
        return httpx.Response(
            200,
            json={
                "id": "ablation",
                "model": ALIAS,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    gateway = LLMGateway(
        registry=Registry(catalog, settings, transport=httpx.MockTransport(handler)),
        router=TaskRouter(catalog),
        catalog=catalog,
        settings=settings,
    )
    return gateway, counter


def call() -> ChatCall:
    # `model=` 把候选链钉成一条：这里量的是重试，不是 fallback。
    return ChatCall(
        messages=[Message(role=MessageRole.USER, content="ping")],
        task="chat",
        model=ALIAS,
        opts={"max_tokens": 8, "temperature": 0.0},
    )


async def run_fork(
    max_attempts: int, *, requests: int, failure_rate: float, seed: int
) -> ForkResult:
    gateway, counter = build_flaky_gateway(
        max_attempts=max_attempts, failure_rate=failure_rate, seed=seed
    )
    result = ForkResult(max_attempts=max_attempts, requests=requests, succeeded=0, upstream_calls=0)

    for _ in range(requests):
        try:
            outcome = await gateway.chat(call())
        except LLMError:
            result.latencies_ms.append(0.0)
            continue
        result.succeeded += 1
        result.latencies_ms.append(outcome.response.latency_ms or 0.0)

    result.upstream_calls = counter[0]
    await gateway.aclose()
    return result


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


async def main() -> int:
    parser = argparse.ArgumentParser(description="W11 D4 消融与分叉")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    arms = run_ablation()
    assert_unwrap_is_the_fix(arms)

    # 同一随机种子 → 三种重试次数面对的是**同一串**成功/失败序列，可比。
    forks = [
        await run_fork(attempts, requests=200, failure_rate=0.5, seed=20261113)
        for attempts in (1, 3)
    ]
    forks[0].latencies_ms = forks[0].latencies_ms or []
    assert forks[1].success_rate >= forks[0].success_rate, "重试没有提高成功率，实验无效"
    assert forks[1].upstream_calls > forks[0].upstream_calls, "重试没有放大流量，实验无效"

    if args.json:
        print(
            json.dumps(
                {
                    "ablation": [arm.as_dict() for arm in arms],
                    "fork": [item.as_dict() for item in forks],
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0

    print("== 实验 A（消融）：检测前要不要先解封皮 ==\n")
    header = f"{'臂':<26}{'注入拦截率':>12}{'正常放行率':>12}   误报规则"
    print(header)
    print("-" * 74)
    for arm in arms:
        rules = ", ".join(f"{k}×{v}" for k, v in sorted(arm.false_positive_rules.items())) or "-"
        print(f"{arm.name:<26}{arm.recall:>11.0%}{arm.pass_rate:>12.0%}   {rules}")

    print("\n== 实验 B（分叉）：max_attempts 取 1 还是 3 ==\n")
    # 不报延迟：注入层是内存 MockTransport，量出来的毫秒数没有参考意义，
    # 印在报告里会被当成"重试让 p95 变差了四倍"。
    print(f"{'max_attempts':<14}{'成功率':>9}{'上游调用':>10}{'每成功一次调用':>16}")
    print("-" * 50)
    for item in forks:
        print(
            f"{item.max_attempts:<14}{item.success_rate:>8.1%}"
            f"{item.upstream_calls:>10}{item.calls_per_success:>16.2f}"
        )

    print("\n断言全绿：修复后 100% 拦截 + 100% 放行；重试提高成功率的同时能算出成本代价")
    return 0


if __name__ == "__main__":
    raise SystemExit(__import__("asyncio").run(main()))
