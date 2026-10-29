"""基线脚本的离线自测。

两件事必须被证明，否则这张表不能作为"改动前"的参照物：

1. **指标确实来自 `RequestLog`**，不是脚本自己计时（否则口径会与生产日志分叉）；
2. **异常样本不会让表变成空白**：失败要显式出现在"备注"列，不能静默丢掉。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from agent_framework.config import Settings, load_catalog

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "evals" / "scripts" / "bench_baseline.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_baseline", SCRIPT_PATH)

    assert spec and spec.loader

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


bench = _load_script()


async def test_run_matrix_reads_metrics_from_request_log() -> None:
    """用 mock 回放跑真 gateway：能拿到字段就说明链路通、日志通、口径统一。"""

    samples = await bench.run_matrix(
        models=["fixture"],
        cases=["text_basic"],
        modes=["stream"],
        repeat=2,
        warmup=0,
        settings=Settings(),
    )

    assert len(samples) == 2
    assert all(sample.error is None for sample in samples)
    assert all(sample.first_token_ms is not None for sample in samples)
    # 夹具里写死的 58/22：能对上就证明数字确实来自日志而不是编的
    assert samples[0].prompt_tokens == 58
    assert samples[0].completion_tokens == 22


async def test_blocking_mode_has_no_first_token() -> None:
    """非流式没有"首 token"这回事，表里必须是 `-` 而不是 0。

    写 0 会被后续聚合当成"极快"，把 p50 拉到失真。
    """

    samples = await bench.run_matrix(
        models=["fixture"],
        cases=["text_basic"],
        modes=["blocking"],
        repeat=1,
        warmup=0,
        settings=Settings(),
    )

    assert samples[0].first_token_ms is None
    assert samples[0].latency_ms > 0


def test_percentile_handles_degenerate_samples() -> None:
    assert bench.percentile([], 0.5) == 0.0
    assert bench.percentile([42.0], 0.95) == 42.0

    values = [10.0, 20.0, 30.0]

    assert bench.percentile(values, 0.5) == 20.0
    assert bench.percentile(values, 0.95) == 29.0


def test_group_rows_surfaces_failure_instead_of_dropping_it() -> None:
    samples = [
        bench.Sample(model="m1", case="text_basic", mode="stream", error="RATE_LIMIT"),
        bench.Sample(model="m1", case="text_basic", mode="stream", error="RATE_LIMIT"),
    ]

    rows = bench.group_rows(samples, {"m1": "openai"})

    assert len(rows) == 1
    assert "全部失败：RATE_LIMIT" in rows[0]


def test_group_rows_marks_estimated_usage_and_retries() -> None:
    samples = [
        bench.Sample(
            model="m1",
            case="text_basic",
            mode="stream",
            first_token_ms=100.0,
            latency_ms=500.0,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.0001,
            currency="CNY",
            usage_estimated=True,
            retry_count=1,
        )
    ]

    row = bench.group_rows(samples, {"m1": "qwen"})[0]

    assert "usage 为估算值" in row
    assert "含重试" in row
    assert "100 ms" in row


def test_resolve_targets_excludes_mock_by_default() -> None:
    catalog = load_catalog()

    models, providers = bench.resolve_targets(catalog, None)

    assert "fixture" not in models
    assert set(models) == {"qwen3.7-plus", "deepseek-flash", "gpt-4o-mini", "claude-sonnet-4-6"}
    assert providers["deepseek-flash"] == "deepseek"


def test_resolve_targets_rejects_unknown_alias() -> None:
    catalog = load_catalog()

    try:
        bench.resolve_targets(catalog, ["ghost"])
    except SystemExit as exc:
        assert "未知 alias" in str(exc)
    else:  # pragma: no cover - 走到这里说明校验没生效
        raise AssertionError("未知 alias 应该直接退出")


def test_build_payload_is_json_serialisable() -> None:
    samples: list[Any] = [
        bench.Sample(
            model="m1",
            case="text_basic",
            mode="stream",
            first_token_ms=1.5,
            latency_ms=9.0,
            cost=0.000123,
            currency="USD",
        )
    ]

    payload = bench.build_payload(
        samples, mode="dry-run", settings=Settings(), providers={"m1": "openai"}
    )

    assert payload["mode"] == "dry-run"
    assert payload["rows"][0]["provider"] == "openai"
    assert "week" in payload
