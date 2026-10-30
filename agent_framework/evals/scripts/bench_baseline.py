"""Day5 · 首 token / 延迟 / token / 成本的基线对比表。

这张表为什么重要
----------------
它是 W2 压测、W11 成本优化、W12 作品集里的"改动前"那一列。没有基线，
后面所有"优化了 30%"都没有参照物；而不写清测量口径，基线本身也不可复现。

口径（唯一、且必须复现）
------------------------
**全部指标都从 `RequestLog` 里读，不自己计时。**

自己 `time.monotonic()` 包一层会造出第二套定义：`first_token_ms` 在日志里是
"首个 text_delta / tool_call 事件的时间"（reasoning_delta 不算），自己计时很容易
把首字节、首 reasoning、首 content 混成一个数——三个数差好几倍，结论会完全反过来。

用法
----
    # 1) 不联网，先验证脚本与输出格式（mock 回放，延迟无意义）
    uv run python evals/scripts/bench_baseline.py --dry-run --models fixture

    # 2) 真机基线（需 .env.local）
    uv run python evals/scripts/bench_baseline.py --models qwen3.7-plus,deepseek-flash

    # 3) 只测流式首 token
    uv run python evals/scripts/bench_baseline.py --modes stream --repeat 5

产出
----
`evals/history/<week>-w1-baseline.md`（人读）+ 同名 `.json`（W2 起做曲线要读它）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "packages" / "agent_framework" / "tests"
HISTORY_DIR = REPO_ROOT / "evals" / "history"

sys.path.insert(0, str(TESTS_DIR))

from agent_framework.bootstrap import build_gateway  # noqa: E402
from agent_framework.config import Settings, load_catalog  # noqa: E402
from agent_framework.llm.client import ChatCall, LLMGateway  # noqa: E402
from agent_framework.llm.errors import LLMError, error_code_for  # noqa: E402
from agent_framework.observability import get_request_logger  # noqa: E402
from canonical_input import MAX_TOKENS, canonical_messages, canonical_tools  # noqa: E402

CASES: dict[str, bool] = {"text_basic": True, "tool_call_basic": True}
MODES = ("stream", "blocking")


# --------------------------------------------------------------------------- #
# 日志捕获：唯一指标来源
# --------------------------------------------------------------------------- #


class LogCapture(logging.Handler):
    """把 `agent_framework.request` 的 JSON 行收进列表。

    **刻意不用 dataclass**：`logging.Handler.__init__` 会建锁与级别，
    dataclass 生成的 `__init__` 不调父类构造，`emit` 里用 `self.lock` 直接炸。
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = json.loads(record.getMessage())
        except ValueError:
            return

        if isinstance(payload, dict):
            self.records.append(payload)

    def latest(self) -> dict[str, Any] | None:
        return self.records[-1] if self.records else None


# --------------------------------------------------------------------------- #
# 单次采样
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Sample:
    model: str
    case: str
    mode: str
    first_token_ms: float | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    currency: str = ""
    retry_count: int = 0
    usage_estimated: bool = False
    finish_reason: str = ""
    error: str | None = None


async def run_once(
    gateway: LLMGateway,
    capture: LogCapture,
    *,
    model: str,
    case: str,
    mode: str,
) -> Sample:
    call = ChatCall(
        messages=canonical_messages(),
        task="chat",
        tools=canonical_tools(),
        model=model,
        opts={"max_tokens": MAX_TOKENS},
    )

    sample = Sample(model=model, case=case, mode=mode)
    capture.records.clear()

    try:
        if mode == "stream":
            async for _event in gateway.stream(call):
                pass
        else:
            await gateway.chat(call)
    except LLMError as exc:
        sample.error = error_code_for(exc)
        return sample

    log = capture.latest()

    if log is None:
        sample.error = "NO_REQUEST_LOG"
        return sample

    sample.first_token_ms = log.get("first_token_ms")
    sample.latency_ms = float(log.get("latency_ms") or 0.0)
    sample.prompt_tokens = int(log.get("prompt_tokens") or 0)
    sample.completion_tokens = int(log.get("completion_tokens") or 0)
    sample.cost = float(log.get("cost") or 0.0)
    sample.currency = str(log.get("currency") or "")
    sample.retry_count = int(log.get("retry_count") or 0)
    sample.usage_estimated = bool(log.get("usage_estimated"))
    sample.finish_reason = str(log.get("finish_reason") or "")

    return sample


def percentile(values: list[float], ratio: float) -> float:
    """线性插值分位数。样本少时不要用统计库的默认分位定义（n<2 直接抛）。"""

    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * ratio
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low

    return ordered[low] * (1 - weight) + ordered[high] * weight


# --------------------------------------------------------------------------- #
# 矩阵执行
# --------------------------------------------------------------------------- #


async def run_matrix(
    *,
    models: list[str],
    cases: list[str],
    modes: list[str],
    repeat: int,
    warmup: int,
    settings: Settings,
    on_sample: Any = None,
) -> list[Sample]:
    """返回计入统计的样本（warmup 只用于预热连接，不参与聚合）。"""

    gateway = build_gateway(settings)

    logger = get_request_logger()
    capture = LogCapture()
    previous_level = logger.level
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)

    measured: list[Sample] = []

    try:
        for model in models:
            for case in cases:
                for mode in modes:
                    for _ in range(warmup):
                        # 预热：建连 + TLS 握手 + 厂商侧冷启动都算在第一次请求里，
                        # 不预热的话第一个样本会稳定偏高，p50 直接被拉歪。
                        await run_once(gateway, capture, model=model, case=case, mode=mode)

                    for _ in range(repeat):
                        sample = await run_once(
                            gateway, capture, model=model, case=case, mode=mode
                        )
                        measured.append(sample)

                        if on_sample is not None:
                            on_sample(sample)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_level)
        await gateway.aclose()

    return measured


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #

HEADER = (
    "| 模型 (alias) | provider | case | 模式 | 次数 | 首 token p50 | 首 token p95 "
    "| 总延迟 p50 | 输入 tok | 输出 tok | 单次成本 | 币种 | 备注 |"
)
SEPARATOR = "|" + "---|" * 13


def group_rows(samples: list[Sample], providers: dict[str, str]) -> list[str]:
    """按 (model, case, mode) 聚合，输出 markdown 行。"""

    groups: dict[tuple[str, str, str], list[Sample]] = {}

    for sample in samples:
        groups.setdefault((sample.model, sample.case, sample.mode), []).append(sample)

    rows: list[str] = []

    for (model, case, mode), items in groups.items():
        ok = [item for item in items if item.error is None]
        errors = [item for item in items if item.error is not None]

        if not ok:
            rows.append(
                f"| {model} | {providers.get(model, '?')} | {case} | {mode} | 0 | - | - "
                f"| - | - | - | - | - | 全部失败：{errors[0].error} |"
            )
            continue

        first_tokens = [item.first_token_ms for item in ok if item.first_token_ms is not None]
        latencies = [item.latency_ms for item in ok]

        notes: list[str] = []

        if errors:
            notes.append(f"{len(errors)}/{len(items)} 次失败：{errors[0].error}")

        if any(item.usage_estimated for item in ok):
            notes.append("usage 为估算值")

        if any(item.retry_count for item in ok):
            notes.append(f"含重试（最多 {max(item.retry_count for item in ok)} 次）")

        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                model,
                providers.get(model, "?"),
                case,
                mode,
                len(ok),
                f"{statistics.fmean(first_tokens):.0f} ms" if first_tokens else "-",
                f"{percentile(first_tokens, 0.95):.0f} ms" if first_tokens else "-",
                f"{statistics.median(latencies):.0f} ms",
                f"{statistics.fmean(i.prompt_tokens for i in ok):.0f}",
                f"{statistics.fmean(i.completion_tokens for i in ok):.0f}",
                f"{statistics.fmean(i.cost for i in ok):.6f}",
                ok[0].currency or "-",
                "；".join(notes) or "-",
            )
        )

    return rows


def build_report(
    samples: list[Sample],
    *,
    mode: str,
    settings: Settings,
    providers: dict[str, str],
    repeat: int,
    warmup: int,
) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    week = datetime.now(UTC).strftime("%G-W%V")

    lines = [
        f"# W1 基线对比表（{week}）",
        "",
        f"- 生成时间：`{now}`（UTC）",
        f"- 采样口径：每格 `{repeat}` 次，前 `{warmup}` 次 warmup 不计入",
        f"- 运行模式：`{mode}`"
        + (
            "　**mock 回放，延迟与成本无意义，仅用于验证脚本与表格式**"
            if mode == "dry-run"
            else ""
        ),
        "- 指标来源：`agent_framework.request` 的结构化日志（`first_token_ms` 只认"
        "首个 `text_delta` / `tool_call`，`reasoning_delta` 不算）",
        f"- 生效配置：`first_token_timeout_s={settings.first_token_timeout_s}`、"
        f"`max_attempts={settings.max_attempts}`",
        "- 成本币种按 `models.yaml` 的 `pricing.currency`，**跨币种不可直接相加**",
        "",
        HEADER,
        SEPARATOR,
        *group_rows(samples, providers),
        "",
        "> 峰谷/阶梯计价提醒：DeepSeek 高峰时段（工作日 09–12、14–18 北京时间）单价翻倍；",
        "> Qwen 输入超过 256K 后单价跳到 6/24。跨时段对比要在同一时段内做。",
        "",
    ]

    return "\n".join(lines)


def build_payload(
    samples: list[Sample],
    *,
    mode: str,
    settings: Settings,
    providers: dict[str, str],
) -> dict[str, Any]:
    return {
        "week": datetime.now(UTC).strftime("%G-W%V"),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "settings": {
            "first_token_timeout_s": settings.first_token_timeout_s,
            "max_attempts": settings.max_attempts,
        },
        "rows": [
            {
                "model": sample.model,
                "provider": providers.get(sample.model),
                "case": sample.case,
                "mode": sample.mode,
                "first_token_ms": sample.first_token_ms,
                "latency_ms": round(sample.latency_ms, 3),
                "prompt_tokens": sample.prompt_tokens,
                "completion_tokens": sample.completion_tokens,
                "cost": sample.cost,
                "currency": sample.currency,
                "retry_count": sample.retry_count,
                "usage_estimated": sample.usage_estimated,
                "finish_reason": sample.finish_reason,
                "error": sample.error,
            }
            for sample in samples
        ],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def resolve_targets(catalog: Any, requested: list[str] | None) -> tuple[list[str], dict[str, str]]:
    """决定跑哪些 alias，并给出 alias → provider 映射。

    默认排除 mock：基线表里混进回放延迟，会让"首 token p50"这个最关键的
    指标彻底失去意义（一个 0.2ms 的数字会把均值拉到看不见真机差异）。
    """

    available = {
        alias: model.provider
        for alias, model in catalog.models.items()
        if model.provider != "mock"
    }

    if requested:
        unknown = [alias for alias in requested if alias not in catalog.models]
        if unknown:
            raise SystemExit(f"未知 alias：{unknown}；可用：{sorted(catalog.models)}")
        chosen = requested
    else:
        chosen = list(available)

    return chosen, {alias: catalog.models[alias].provider for alias in chosen}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W1 基线对比表")
    parser.add_argument("--models", default="", help="逗号分隔 alias；默认全部非 mock 模型")
    parser.add_argument("--cases", default="text_basic,tool_call_basic")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="用 mock 回放跑通流程")
    parser.add_argument("--out", default="", help="markdown 输出路径")
    args = parser.parse_args(argv)

    settings = Settings()
    catalog = load_catalog(settings.models_yaml)

    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    models, providers = resolve_targets(catalog, requested or None)

    if args.dry_run:
        models = ["fixture"]
        providers = {"fixture": "mock"}

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    invalid = set(modes) - set(MODES)

    if invalid:
        raise SystemExit(f"未知模式：{sorted(invalid)}；只支持 {MODES}")

    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown_case = set(cases) - set(CASES)

    if unknown_case:
        raise SystemExit(f"未知 case：{sorted(unknown_case)}")

    def on_sample(sample: Sample) -> None:
        flag = "FAIL " if sample.error else "ok   "
        first = (
            f"{sample.first_token_ms:.0f}ms"
            if sample.first_token_ms is not None
            else "-"
        )
        print(
            f"{flag}{sample.model:<18} {sample.case:<16} {sample.mode:<8} "
            f"first={first:<8} total={sample.latency_ms:.0f}ms "
            f"tok={sample.prompt_tokens}/{sample.completion_tokens} "
            f"cost={sample.cost:.6f}"
        )

    measured = asyncio.run(
        run_matrix(
            models=models,
            cases=cases,
            modes=modes,
            repeat=args.repeat,
            warmup=args.warmup,
            settings=settings,
            on_sample=on_sample,
        )
    )

    mode = "dry-run" if args.dry_run else "live"
    report = build_report(
        measured,
        mode=mode,
        settings=settings,
        providers=providers,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    payload = build_payload(measured, mode=mode, settings=settings, providers=providers)

    if args.dry_run and not args.out:
        # dry-run 绝不落进 evals/history：那里是"真机基线"的档案库，
        # 混进一份 mock 延迟的表，W11 做成本曲线时会把它当真数据读。
        print("\n" + report)
        print("dry-run 不写盘（要落盘请显式给 --out）")

        return 0

    week = datetime.now(UTC).strftime("%G-W%V")
    md_path = Path(args.out) if args.out else HISTORY_DIR / f"{week}-w1-baseline.md"
    json_path = md_path.with_suffix(".json")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report, "utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )

    print(f"\n已写入：{md_path}\n已写入：{json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
