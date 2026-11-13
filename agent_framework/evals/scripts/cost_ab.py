#!/usr/bin/env python
"""成本三手段单变量 A/B（W11 D5 / Gate 11 #10）。

    python evals/scripts/cost_ab.py [--json out.json]

三个实验，每次只动一个变量：

| 实验 | 变量 | 度量 |
|---|---|---|
| 1 语义缓存 | `semantic_cache` off / on | 上游调用数、成本、命中率（逐字 / 改写分开数） |
| 2 难度分流 | `difficulty_router` off / on，**两种基线** | alias 分布、误分流率、成本 |
| 3 批处理 | 串行 / 并发 | 墙钟、调用数（成本不变） |

## 三个口径必须先说清，否则数字会被读成别的意思

1. **token 是估算的**（`字符数/2`，与生产 `usage.estimated=True` 同一口径）。
   所以成本结论是**相对差**，不是账单金额。
2. **价格是真的**：取自 `models.yaml` 的公开报价（gpt-4o-mini 0.15/0.60、
   antml sonnet 3.00/15.00 USD per 1M），因此"贵 20 倍"这个量级是真的。
3. **实验 3 的 200ms 是模拟延迟**：这里要证明的是"并发确实并行"，
   不是"我们的上游多快"。真实上游延迟请看 `ops/slo_report.py`。

## 实验 2 为什么有两个基线

当前 `models.yaml` 是 `chat → cheap`。在这个基线上，难度分流**只会让成本上升**
（把硬请求升级到强档）——它是"加钱买质量"，不是省钱。
省钱的情形是基线本来就把所有请求都发给强档（质量优先的产品常见）。
两个方向都测、都报，是因为只报后者就是挑数字。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_framework.bootstrap import build_cache, build_router  # noqa: E402
from agent_framework.config import (  # noqa: E402
    Catalog,
    ModelConfig,
    Pricing,
    ProviderConfig,
    RouteConfig,
    Settings,
)
from agent_framework.llm.base import Capability, LLMClient  # noqa: E402
from agent_framework.llm.client import ChatCall, LLMGateway  # noqa: E402
from agent_framework.llm.types import (  # noqa: E402
    FinishReason,
    LLMResponse,
    Message,
    MessageRole,
    ToolChoice,
    ToolSpec,
    Usage,
)
from agent_framework.runtime.flags import FlagSet  # noqa: E402

CHEAP = "cheap-a"
STRONG = "strong-a"


# --------------------------------------------------------------------------- #
# 造一条"价格是真的、token 是估的"链路
# --------------------------------------------------------------------------- #


def build_catalog(*, chat_tier: str = "cheap") -> Catalog:
    """两端模型。价格对齐 models.yaml：cheap 约 0.15/0.60，strong 约 3.00/15.00。"""

    provider = ProviderConfig(type="openai_compat", base_url="https://example.test/v1")
    caps = [Capability.STREAM, Capability.TOOLS]

    return Catalog(
        providers={"p": provider},
        models={
            CHEAP: ModelConfig(
                provider="p", model="gpt-4o-mini", tier="cheap", capabilities=caps,
                pricing=Pricing(input=0.15, output=0.60, currency="USD"),
            ),
            STRONG: ModelConfig(
                provider="p", model="claude-sonnet-4-6", tier="strong", capabilities=caps,
                pricing=Pricing(input=3.00, output=15.00, currency="USD"),
            ),
        },
        routes={
            "default": RouteConfig(primary=CHEAP, fallback=[]),
            "cheap": RouteConfig(primary=CHEAP, fallback=[]),
            "strong": RouteConfig(primary=STRONG, fallback=[]),
        },
        tasks={"chat": chat_tier, "research": "strong"},
    )


class CostedClient(LLMClient):
    """按 prompt 长度产 usage 的假 adapter。`estimated=True` 与生产同口径。"""

    def __init__(self, alias: str, *, delay_s: float = 0.0) -> None:
        self._alias = alias
        self._delay = delay_s
        self.calls = 0

    @property
    def name(self) -> str:
        return self._alias

    @property
    def model(self) -> str:
        return f"{self._alias}-wire"

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.STREAM, Capability.TOOLS})

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: object,
    ) -> LLMResponse:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)

        prompt = "".join(str(item.content) for item in messages)
        output_chars = 120  # 一个固定长度的回答，让两侧可比

        return LLMResponse(
            id=f"{self._alias}-resp",
            model=self.model,
            provider=self._alias,
            message=Message(role=MessageRole.ASSISTANT, content="x" * output_chars),
            finish_reason=FinishReason.STOP,
            raw_finish_reason="stop",
            usage=Usage(
                input_tokens=max(1, len(prompt) // 2),
                output_tokens=output_chars // 2,
                estimated=True,  # 与生产同一口径：token 是估的
            ),
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: object,
    ) -> Any:
        raise NotImplementedError("本脚本只压非流式路径")
        yield  # pragma: no cover

    async def aclose(self) -> None:
        return None


class StaticRegistry:
    """替身：真 Registry 的 `get()` 会去读 api_key，而这里不该碰密钥。"""

    def __init__(self, catalog: Catalog, clients: dict[str, CostedClient]) -> None:
        self._catalog = catalog
        self._clients = clients

    def get(self, alias: str) -> LLMClient:
        return self._clients[alias]

    def get_model(self, alias: str) -> ModelConfig:
        return self._catalog.get_model(alias)

    async def aclose(self) -> None:
        return None


class CollectLogs(logging.Handler):
    """收集生产代码真发出来的 RequestLog——不自己算成本，避免第二套口径。"""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "agent_framework.request":
            self.records.append(json.loads(record.getMessage()))


def build_gateway(
    settings: Settings,
    *,
    chat_tier: str = "cheap",
    delay_s: float = 0.0,
) -> tuple[LLMGateway, dict[str, CostedClient]]:
    catalog = build_catalog(chat_tier=chat_tier)
    flags = FlagSet.from_settings(settings)
    clients = {alias: CostedClient(alias, delay_s=delay_s) for alias in (CHEAP, STRONG)}

    gateway = LLMGateway(
        registry=StaticRegistry(catalog, clients),  # type: ignore[arg-type]
        router=build_router(catalog, settings, flags=flags),
        catalog=catalog,
        settings=settings,
        breaker=None,  # 熔断与本次实验无关，开着会让"上游全挂"的样本串到下一个实验
        cache=build_cache(catalog, settings, flags=flags),
    )
    return gateway, clients


def settings_for(**overrides: Any) -> Settings:
    params: dict[str, Any] = {
        "_env_file": None,
        "max_attempts": 1,  # A/B 只测一次成败，重试会把调用数翻倍、掩盖变量
        "backoff_base_s": 0.0,
    }
    params.update(overrides)

    return Settings(**params)


async def run_calls(gateway: LLMGateway, calls: list[ChatCall]) -> list[dict[str, Any]]:
    collector = CollectLogs()
    logger = logging.getLogger("agent_framework.request")
    logger.addHandler(collector)
    previous_level = logger.level
    logger.setLevel(logging.INFO)

    try:
        for call in calls:
            await gateway.chat(call)
    finally:
        logger.removeHandler(collector)
        logger.setLevel(previous_level)

    return collector.records


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    alias_counter = Counter(str(item["model"]) for item in records)
    hits = [item for item in records if item.get("cache_hit")]

    return {
        "requests": len(records),
        "cost": round(sum(float(item["cost"]) for item in records), 6),
        "cache_hits": len(hits),
        # 命中行的成本必须恰好是 0：成本报告里"缓存省的钱"要靠它可核对
        "hit_cost": round(sum(float(item["cost"]) for item in hits), 6),
        "hit_tokens": sum(
            int(item["prompt_tokens"]) + int(item["completion_tokens"]) for item in hits
        ),
        "by_alias": dict(sorted(alias_counter.items())),
    }


def ask(text: str, *, tools: list[ToolSpec] | None = None) -> ChatCall:
    return ChatCall(messages=[Message(role=MessageRole.USER, content=text)], tools=tools)


# --------------------------------------------------------------------------- #
# 实验 1：语义缓存
# --------------------------------------------------------------------------- #

# 8 个问题，每题问两次：一半逐字重复（一定命中），一半轻度改写（看阈值表现）。
BASE_QUESTIONS: tuple[str, ...] = (
    "如何给 MCP server 配置调用超时",
    "团队周报应该写哪几段内容",
    "向量检索的 top_k 取多少合适",
    "灰度发布怎么按用户分桶",
    "为什么熔断器不能自己再重试一次",
    "结构化输出失败率高该先改什么",
    "长上下文里模型丢约束怎么办",
    "断点续跑时哪些步骤必须重跑",
)

# 只改写末尾一个词——这是"用户换个问法"最常见的样子，也是缓存该命中的样子。
PARAPHRASE_SUFFIX: dict[int, str] = {
    0: "参数",
    1: "结构",
    2: "档位",
    3: "比例",
}


def cache_corpus() -> list[ChatCall]:
    calls: list[ChatCall] = [ask(text) for text in BASE_QUESTIONS]

    for index, text in enumerate(BASE_QUESTIONS):
        suffix = PARAPHRASE_SUFFIX.get(index)
        # 后四个问题逐字重复，前四个换一个词。
        calls.append(ask(text if suffix is None else f"{text}{suffix}"))

    return calls


async def experiment_cache() -> dict[str, Any]:
    corpus = cache_corpus()

    off_gateway, off_clients = build_gateway(settings_for())
    off_logs = await run_calls(off_gateway, corpus)

    on_gateway, on_clients = build_gateway(settings_for(features="semantic_cache=on"))
    on_logs = await run_calls(on_gateway, corpus)

    off = summarise(off_logs)
    on = summarise(on_logs)
    upstream = sum(client.calls for client in on_clients.values())

    return {
        "corpus_size": len(corpus),
        "off": {**off, "upstream_calls": sum(c.calls for c in off_clients.values())},
        "on": {**on, "upstream_calls": upstream},
        "hit_rate": round(on["cache_hits"] / len(corpus), 4),
        "saved_ratio": round(1 - on["cost"] / off["cost"], 4) if off["cost"] else 0.0,
    }


def assert_cache_is_a_real_saving(result: dict[str, Any]) -> None:
    """断言的是**机制**，不是某个具体百分比（百分比会随语料变）。"""

    assert result["off"]["upstream_calls"] == result["corpus_size"], "关闭时不该有命中"
    assert result["on"]["upstream_calls"] < result["off"]["upstream_calls"], \
        "开启后上游调用必须变少"
    assert result["on"]["cache_hits"] > 0, "一条都没命中，说明缓存没接上"
    assert result["on"]["cost"] < result["off"]["cost"], "调用少了但成本没降，说明路径不对"
    # 命中行必须是 cost=0 且 token=0，否则"缓存省了多少钱"在账上无从核对
    assert result["on"]["hit_cost"] == 0.0, "命中行记了钱，成本看板会重复计费"
    assert result["on"]["hit_tokens"] == 0, "命中行记了 token，用量统计会被虚增"
    assert result["on"]["cost"] > 0, "全都命中说明语料没有区分度，实验无意义"


# --------------------------------------------------------------------------- #
# 实验 2：难度分流
# --------------------------------------------------------------------------- #

SIMPLE_QUESTIONS: tuple[str, ...] = (
    "把这句话翻成英文：今天开会",
    "杭州的区号是多少",
    "写一句产品 slogan",
    "这段文案有没有错别字",
    "给这个函数起个名字",
    "把日期改成年月日格式",
)

HARD_QUESTIONS: tuple[str, ...] = (
    "为什么这两套检索方案的成本差这么多？请对比并输出 json",
    "规划一个三步的灰度方案，并论证每一步的回滚条件",
    "评估现有 prompt 的失败模式，给出改进方案",
    "对比 dense 与 hybrid 的取舍，输出 markdown 表格",
    "设计一套幂等键规则，并说明为什么不能用随机数",
    "推理一下：续跑时如果重跑了已完成步会怎样",
)


def router_corpus() -> list[ChatCall]:
    return [ask(text) for text in (*SIMPLE_QUESTIONS, *HARD_QUESTIONS)]


async def run_router_arm(*, chat_tier: str, feature_on: bool) -> dict[str, Any]:
    settings = settings_for(features="difficulty_router=on" if feature_on else "")
    gateway, clients = build_gateway(settings, chat_tier=chat_tier)
    logs = await run_calls(gateway, router_corpus())

    aliases = [str(item["model"]) for item in logs]
    simple_hits = aliases[: len(SIMPLE_QUESTIONS)]
    hard_hits = aliases[len(SIMPLE_QUESTIONS) :]

    return {
        **summarise(logs),
        "upstream_calls": sum(client.calls for client in clients.values()),
        "cheap_share": round(aliases.count(CHEAP) / len(aliases), 4),
        # 误分流 = 简单问题被送去强档（多花钱）；过设计 = 硬问题留在便宜档（省错钱）
        "simple_to_strong": simple_hits.count(STRONG),
        "hard_to_cheap": hard_hits.count(CHEAP),
    }


async def experiment_router() -> dict[str, Any]:
    quality_first = await run_router_arm(chat_tier="strong", feature_on=False)
    quality_first_routed = await run_router_arm(chat_tier="strong", feature_on=True)
    current = await run_router_arm(chat_tier="cheap", feature_on=False)
    current_routed = await run_router_arm(chat_tier="cheap", feature_on=True)

    return {
        "corpus": {"simple": len(SIMPLE_QUESTIONS), "hard": len(HARD_QUESTIONS)},
        "baseline_quality_first": {"off": quality_first, "on": quality_first_routed},
        "baseline_current_yaml": {"off": current, "on": current_routed},
    }


def assert_router_separates_by_difficulty(result: dict[str, Any]) -> None:
    routed = result["baseline_quality_first"]["on"]

    assert routed["hard_to_cheap"] == 0, "硬请求落在便宜档，等于没分流"
    assert routed["simple_to_strong"] == 0, "简单请求仍走强档，分流没生效"
    assert routed["cheap_share"] == round(
        len(SIMPLE_QUESTIONS) / (len(SIMPLE_QUESTIONS) + len(HARD_QUESTIONS)), 4
    ), "cheap 占比应等于简单样本占比"


def assert_router_saves_only_on_a_strong_baseline(result: dict[str, Any]) -> None:
    """这一条是本次实验最反直觉、也最该写进报告的结论。"""

    quality_first = result["baseline_quality_first"]
    current = result["baseline_current_yaml"]

    assert quality_first["on"]["cost"] < quality_first["off"]["cost"], \
        "质量优先基线下分流必须省钱"
    assert current["on"]["cost"] > current["off"]["cost"], \
        "当前 yaml（chat→cheap）下分流应更贵——它是加钱买质量"


# --------------------------------------------------------------------------- #
# 实验 3：批处理（串行 vs 并发）
# --------------------------------------------------------------------------- #

REWRITE_TARGETS: tuple[str, ...] = ("xiaohongshu", "x", "linkedin")
SIMULATED_UPSTREAM_DELAY_S = 0.2


async def experiment_batching() -> dict[str, Any]:
    calls = [ask(f"把这篇稿子改写成 {platform} 风格") for platform in REWRITE_TARGETS]

    gateway, clients = build_gateway(settings_for(), delay_s=SIMULATED_UPSTREAM_DELAY_S)

    started = time.perf_counter()
    await run_calls(gateway, calls)
    serial_s = time.perf_counter() - started
    serial_calls = sum(client.calls for client in clients.values())

    for client in clients.values():
        client.calls = 0

    gateway, clients = build_gateway(settings_for(), delay_s=SIMULATED_UPSTREAM_DELAY_S)

    started = time.perf_counter()
    await asyncio.gather(*(gateway.chat(call) for call in calls))
    concurrent_s = time.perf_counter() - started
    concurrent_calls = sum(client.calls for client in clients.values())

    return {
        "targets": len(REWRITE_TARGETS),
        "delay_s": SIMULATED_UPSTREAM_DELAY_S,
        "serial": {"wall_s": round(serial_s, 3), "calls": serial_calls},
        "concurrent": {"wall_s": round(concurrent_s, 3), "calls": concurrent_calls},
        "speedup": round(serial_s / concurrent_s, 2) if concurrent_s else 0.0,
    }


def assert_batching_saves_latency_not_money(result: dict[str, Any]) -> None:
    assert result["serial"]["calls"] == result["concurrent"]["calls"], \
        "并发不该改变调用数——改变的是墙钟"
    assert result["speedup"] > 2.0, f"并发没有真的并行：加速比 {result['speedup']}"


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def print_report(cache: dict[str, Any], router: dict[str, Any], batch: dict[str, Any]) -> None:
    print("=" * 72)
    print("实验 1 · 语义缓存（单变量：semantic_cache）")
    print("-" * 72)
    print(f"{'':10}{'请求':>6}{'上游调用':>10}{'命中':>6}{'成本(USD,估)':>14}")
    for arm in ("off", "on"):
        row = cache[arm]
        print(
            f"{arm:10}{row['requests']:>6}{row['upstream_calls']:>10}"
            f"{row['cache_hits']:>6}{row['cost']:>14.6f}"
        )
    print(f"  → 命中率 {cache['hit_rate']:.1%}，成本下降 {cache['saved_ratio']:.1%}")

    print("=" * 72)
    print("实验 2 · 难度分流（单变量：difficulty_router，两种基线）")
    print("-" * 72)
    for label in ("baseline_quality_first", "baseline_current_yaml"):
        arm = router[label]
        print(f"  基线 {label}")
        print(f"{'':6}{'成本(USD,估)':>14}{'cheap 占比':>12}{'简单→强档':>11}{'硬→便宜':>9}")
        for flag in ("off", "on"):
            row = arm[flag]
            print(
                f"{flag:6}{row['cost']:>14.6f}{row['cheap_share']:>12.1%}"
                f"{row['simple_to_strong']:>11}{row['hard_to_cheap']:>9}"
            )

    print("=" * 72)
    print(f"实验 3 · 批处理（{batch['targets']} 次独立改写，模拟上游 {batch['delay_s']}s）")
    print("-" * 72)
    for arm in ("serial", "concurrent"):
        row = batch[arm]
        print(f"  {arm:12} 墙钟 {row['wall_s']:.3f}s   调用 {row['calls']}")
    print(f"  → 加速比 {batch['speedup']}×，调用数不变（省的是延迟，不是钱）")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="W11 D5 成本三手段单变量 A/B")
    parser.add_argument("--json", type=Path, default=None, help="把结果写成 JSON")
    args = parser.parse_args()

    cache = asyncio.run(experiment_cache())
    router = asyncio.run(experiment_router())
    batch = asyncio.run(experiment_batching())

    print_report(cache, router, batch)

    assert_cache_is_a_real_saving(cache)
    assert_router_separates_by_difficulty(router)
    assert_router_saves_only_on_a_strong_baseline(router)
    assert_batching_saves_latency_not_money(batch)
    print("断言全部通过（机制成立，不是数字好看）。")

    if args.json is not None:
        payload = {"cache": cache, "router": router, "batching": batch}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        print(f"已写入 {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
