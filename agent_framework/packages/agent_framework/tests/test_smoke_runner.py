"""smoke runner 的门禁能力测试（W2 D5 / Gate「反向验证能拦住」）。

两条：
1. 当前 10 条离线全绿（否则门禁永远红，大家学会忽略）；
2. **故意改坏一条 expect 必须变红**——不变红的门禁等于没有门禁。

用子进程跑（而不是 import）：runner 是"可执行入口"，它的退出码才是 CI 的判据；
在进程内调 `main()` 测不到 `SystemExit`/环境变量这类真实行为。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = "evals/scripts/run_smoke.py"
GOLDEN = REPO_ROOT / "evals" / "golden" / "smoke_v1.jsonl"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, RUNNER, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_smoke_set_has_ten_cases() -> None:
    cases = [
        json.loads(line)
        for line in GOLDEN.read_text("utf-8").splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]

    assert len(cases) == 10
    assert {case["kind"] for case in cases} == {"chat", "tool_call", "error", "stream"}


def test_offline_smoke_passes() -> None:
    result = _run()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "10 passed, 0 failed" in result.stdout


def test_broken_expectation_fails_the_gate(tmp_path: Path) -> None:
    broken = tmp_path / "smoke_broken.jsonl"
    lines = GOLDEN.read_text("utf-8").splitlines()
    mutated: list[str] = []

    for line in lines:
        if '"smoke-01"' in line:
            # 把"必须包含 MCP"改成永远不可能成立的断言
            mutated.append(
                line.replace(
                    '"text_contains": ["MCP"]',
                    '"text_contains": ["绝对不会出现的文案"]',
                )
            )
        else:
            mutated.append(line)

    broken.write_text("\n".join(mutated) + "\n", "utf-8")

    result = _run("--golden", str(broken))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed" in result.stdout
