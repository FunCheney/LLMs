"""公共 API 契约测试（W12 D1）。

这里测的不是"功能对不对"，而是**框架对外的承诺是否成立**：

1. `__all__` 里每个名字都真的能取到（写进 `__all__` 就是承诺，取不到就是撒谎）；
2. 重的东西不因为 `import agent_framework.anything` 就被拉起来（包初始化代价）；
3. 产品在用的符号能从子包**浅路径**导入（否则"公共面在子包"只是文档说法）；
4. 框架永不 import 产品（ADR-004 单向依赖）。

第 3 条直接复用 `evals/scripts/api_audit.py` 的判定 —— 让审计脚本和 CI 用**同一份
规则**，而不是脚本报一套、测试守另一套。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import agent_framework
import pytest

REPO = Path(__file__).resolve().parents[4]
AUDIT_SCRIPT = REPO / "evals/scripts/w12/api_audit.py"


# ─────────────────────────── 1. 顶层导出 ───────────────────────────


def test_every_name_in_all_is_actually_reachable() -> None:
    """`__all__` 是承诺面。名字在里面但 `getattr` 抛错，等于承诺了不存在的东西。

    惰性导出（`__getattr__`）最容易在这里出事：写进 `__all__` 却忘了加进
    `_LAZY`，`from agent_framework import X` 就在用户第一行代码上炸。
    """
    missing = [name for name in agent_framework.__all__ if not hasattr(agent_framework, name)]

    assert not missing, f"__all__ 声明了但取不到：{missing}"


def test_dir_matches_all() -> None:
    assert sorted(agent_framework.__all__) == dir(agent_framework)


def test_top_level_is_deliberately_small() -> None:
    """顶层是「起步面」不是「全集索引」。

    数量本身不是目标，但它是"克制"最直接的代理指标：框架有 179 个产品在用的
    符号，顶层只该放"起一个服务、发一条 chat"所需的那十几个。哪天有人想
    `from agent_framework import *` 拿到全部，这条会先红。
    """
    assert len(agent_framework.__all__) <= 20, (
        f"顶层导出了 {len(agent_framework.__all__)} 个名字，"
        "超出「起步面」的合理规模——请确认它们都是发一条 chat 所必需的"
    )


def test_subpackages_are_not_re_exported_at_top_level() -> None:
    """子包按需访问，不堆到顶层。

    `agent_framework.retrieval` / `memory` / `planner` 这些是能力包，
    把它们塞进顶层会让"顶层导出"变成目录页。
    """
    capability_packages = {
        "retrieval",
        "memory",
        "skills",
        "planner",
        "multi_agent",
        "runtime",
        "context",
        "guardrails",
        "knowledge",
    }

    assert not (capability_packages & set(agent_framework.__all__))


# ─────────────────────────── 2. 初始化代价 ───────────────────────────


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """在**子进程**里跑一段代码，避免当前测试进程的 sys.modules 污染结论。"""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


def _run_script(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


def test_importing_config_does_not_pull_httpx() -> None:
    """读一行配置不该付 HTTP 客户端的代价。

    这不是洁癖：脚手架生成的工具里，"读配置校验一下"是最常见的第一个动作，
    而 `import agent_framework` 曾经会连带拉起 bootstrap → registry → httpx。
    """
    result = _run(
        "import sys, agent_framework.config\n"
        "print('client' if 'agent_framework.llm.client' in sys.modules else '-')\n"
        "print('httpx' if 'httpx' in sys.modules else '-')\n"
        "print('bootstrap' if 'agent_framework.bootstrap' in sys.modules else '-')\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["-", "-", "-"], (
        f"import agent_framework.config 拉起了不该拉起的东西：{result.stdout}"
    )


def test_importing_llm_base_does_not_pull_client() -> None:
    """`llm/__init__` 的 `ChatCall` 走惰性。

    直接 `from .client import ...` 会造出 config ↔ client 的真环：
    `config` 需要 `llm.base.Capability`，`llm.client` 需要 `config.Catalog`。
    """
    result = _run(
        "import sys, agent_framework.llm.base\n"
        "print('client' if 'agent_framework.llm.client' in sys.modules else '-')\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "-"


def test_lazy_exports_still_work_through_the_public_path() -> None:
    """惰性归惰性，`from agent_framework import ChatCall` 必须照常可用。"""
    from agent_framework import ChatCall, ChatOutcome, LLMGateway  # noqa: F401

    assert ChatCall is not None


# ─────────────────────────── 3. 子包公共面 ───────────────────────────


def test_audit_script_passes() -> None:
    """把审计结论固化进 CI。

    跑的是同一个脚本，所以"文档说公共面完整"和"CI 说公共面完整"不会各说各话。
    """
    result = _run_script(AUDIT_SCRIPT, "--check")

    assert result.returncode == 0, (
        "公共 API 审计未通过：\n" + result.stdout + result.stderr
    )


@pytest.mark.parametrize("package", ["agent_framework.llm", "agent_framework.server"])
def test_subpackage_all_is_backed_by_real_attributes(package: str) -> None:
    """子包 `__all__` 里的每个名字都要能取到。"""
    module = __import__(package, fromlist=["__all__"])

    missing = [name for name in module.__all__ if not hasattr(module, name)]

    assert not missing, f"{package} 的 __all__ 声明了但取不到：{missing}"


# ─────────────────────────── 4. 依赖方向 ───────────────────────────


def test_framework_never_imports_product() -> None:
    """ADR-004：`apps/* -> agent_framework` 单向。反过来就是架构坏了。

    这条如果失守，框架就绑死在 growth_agent 上，"脚手架给第三方用"直接不成立。
    """
    framework_src = REPO / "packages/agent_framework/src"
    offenders: list[str] = []

    for path in framework_src.rglob("*.py"):
        text = path.read_text("utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import growth_agent", "from growth_agent")):
                offenders.append(f"{path.relative_to(REPO)}: {stripped}")

    assert not offenders, "框架 import 了产品：\n" + "\n".join(offenders)
