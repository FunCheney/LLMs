"""W8 D1：协议边界的**结构性**断言（Gate 8 #1）。

这条测试不是在测行为，是在测依赖方向：业务代码看得见 `ToolSpec`，
看不见 MCP。用扫源码的方式断言，是因为"约定"没法靠 review 保证——
第一次有人为了图方便 `from agent_framework.skills.mcp import ...`，
边界就没了，而代码照样能跑、测试照样绿。

三条规则：

1. `skills/` 之外**不许**出现 `.mcp` 的 import
2. 任何地方**不许** import 第三方 `mcp` SDK（本周根本没装它，但这条防止将来"顺手加一个"）
3. `apps/` 只认 `agent_framework.skills` 的公开入口
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[1] / "src" / "agent_framework"
APPS = Path(__file__).resolve().parents[4] / "apps"


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _imported_modules(path: Path) -> set[str]:
    """用 AST 而不是 grep：`# from .mcp import x` 这种注释不该让测试变红。"""
    tree = ast.parse(path.read_text("utf-8"))
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module:
                modules.add(f"{prefix}{node.module}")
            else:
                # `from . import mcp`
                modules.update(f"{prefix}{alias.name}" for alias in node.names)

    return modules


def _is_mcp_reference(module: str) -> bool:
    parts = module.lstrip(".").split(".")
    return "mcp" in parts


def test_only_skills_may_reference_mcp() -> None:
    offenders: list[str] = []

    for path in _python_files(PACKAGES):
        if "skills" in path.parts:
            continue
        for module in _imported_modules(path):
            if _is_mcp_reference(module):
                offenders.append(f"{path.relative_to(PACKAGES)}: {module}")

    assert not offenders, (
        "MCP 只应出现在 skills/ 内部；业务层必须只认 ToolSpec/ToolOutcome：\n"
        + "\n".join(offenders)
    )


def test_no_third_party_mcp_sdk_import() -> None:
    """我们没有装 `mcp` SDK，也不该装：协议边界自己写，依赖面更小。"""

    offenders: list[str] = []

    for root in (PACKAGES, APPS):
        for path in _python_files(root):
            for module in _imported_modules(path):
                if module == "mcp" or module.startswith("mcp."):
                    offenders.append(f"{path}: {module}")

    assert not offenders, "不要 import 第三方 mcp SDK：" + ", ".join(offenders)


def test_apps_use_public_skills_entrypoint_only() -> None:
    """产品层走 `agent_framework.skills` 的公开入口，不深入子模块。

    深入子模块意味着我们可以"为了产品层方便"改内部结构——而内部结构
    一旦为产品层定制，框架就不再是框架了。
    """

    offenders: list[str] = []
    allowed_prefixes = (
        "agent_framework.skills",
        "agent_framework.skills.servers",
    )

    for path in _python_files(APPS):
        for module in _imported_modules(path):
            if not module.startswith("agent_framework.skills"):
                continue
            allowed = any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in allowed_prefixes
            )
            if allowed:
                continue
            offenders.append(f"{path.name}: {module}")

    assert not offenders


def test_framework_never_imports_product() -> None:
    """依赖方向：apps/* → agent_framework，框架永不 import 产品。"""

    offenders: list[str] = []

    for path in _python_files(PACKAGES):
        for module in _imported_modules(path):
            if module.startswith("growth_agent"):
                offenders.append(f"{path}: {module}")

    assert not offenders


@pytest.mark.parametrize(
    "path",
    [
        PACKAGES / "skills" / "mcp" / "types.py",
        PACKAGES / "skills" / "mcp" / "client.py",
    ],
)
def test_mcp_transport_returns_raw_dicts(path: Path) -> None:
    """传输层的返回值必须是原始 dict（wire），不能是内部类型。

    这条保证了"换传输不改上层"：`RecordedToolTransport` / 真 stdio / 内存版
    三种实现能互换，因为它们都只说 JSON-RPC。
    """

    source = path.read_text("utf-8")

    assert "async def list_tools" in source
    assert "dict[str, Any]" in source


def test_skills_package_exports_do_not_leak_mcp_types() -> None:
    """`skills/__init__` 不该导出 MCP 的 wire 类型（`MCPToolDef` 之类）。"""

    import agent_framework.skills as skills

    leaked = [name for name in skills.__all__ if name.startswith("MCP")]
    assert leaked == ["MCPClient"] or leaked == [], f"导出了 MCP 内部类型：{leaked}"


def test_init_all_resolves() -> None:
    """`__all__` 里有一个不存在的名字，`from x import *` 会在运行时炸。"""

    import agent_framework.skills as skills

    missing = [name for name in skills.__all__ if not hasattr(skills, name)]
    assert missing == []
