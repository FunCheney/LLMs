"""公共 API 审计：谁在用框架的哪个符号。

## 为什么需要这个脚本

W12 D1 的判据是「只提取被 `apps/` 真实使用过的抽象」。这句话如果没有可复现的
扫描，它就只是我的印象——而印象在两种情况下都会错：

- **高估**：把 W9 的实验产物当成"公共能力"暴露出去，第三方拿去用半年后发现
  我要改签名；
- **低估**：把产品天天在用的 `ChatCall` 漏在 internal 里，第三方 import 不到。

所以先把「谁在用」变成数据，再决定 `__init__.py` 里放什么。

## 使用域怎么分

| 域 | 路径 | 含义 |
|---|---|---|
| `product` | `apps/*/src` | **真金白银在用**——只有这一档够格进公共 API |
| `product_test` | `apps/*/tests` | 产品测试在用，说明是产品侧的契约 |
| `evals` | `evals/scripts` | 评测脚本在用，通常是内部探针 |
| `framework_test` | `packages/*/tests` | 框架自测在用 |

分四档而不是两档，是因为「产品测试在用但产品源码不用」和「产品源码在用」
是两件不同的事：前者说明这个符号是**测试脚手架**，暴露它等于把测试实现
变成公共契约。

## 相对 import 必须还原

框架内部互相 import 用的是 `from ..resilience.circuit_breaker import CircuitBreaker`，
`node.module` 是 `resilience.circuit_breaker`、`node.level` 是 2。不还原成绝对名，
内部使用就会被全部漏掉，`never_used` 里会塞满其实天天在用的符号。

用法：

    uv run python evals/scripts/w12/api_audit.py            # 打印报告
    uv run python evals/scripts/w12/api_audit.py --check    # 断言（CI 用）
    uv run python evals/scripts/w12/api_audit.py --json     # 机器可读
"""

from __future__ import annotations

import argparse
import ast
import json
import pkgutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
FRAMEWORK_PKG = "agent_framework"
FRAMEWORK_SRC = REPO / "packages/agent_framework/src"

# 使用域 → 路径。顺序即报告里的展示顺序，也是「档位」从高到低。
#
# `framework` 这一档只用来回答「这个模块是不是死代码」，不参与公共 API 判定：
# `server/routes_chat.py` 只被 `server/app.py` 用，它是**内部实现**而不是空壳；
# 不扫 src 内部就会把它列进「可以删」，而删了 `/v1/chat` 直接 500。
DOMAINS: tuple[tuple[str, str], ...] = (
    ("product", "apps/growth_agent/src"),
    ("product_test", "apps/growth_agent/tests"),
    ("evals", "evals/scripts"),
    ("framework", "packages/agent_framework/src"),
    ("framework_test", "packages/agent_framework/tests"),
)

# 进了公共入口就算承诺稳定。只有 product / product_test 两档够格。
PROMOTABLE = frozenset({"product", "product_test"})


@dataclass
class ModuleUsage:
    """一个框架模块被谁用、用了哪些符号。"""

    module: str
    # 符号 → 使用域集合
    symbols: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # 整模块 import（`import agent_framework.memory.long_term`）也算使用，
    # 但拿不到具体符号——记在模块级，避免误判成「没人用」。
    module_level: set[str] = field(default_factory=set)
    # 产品 import 了下划线开头的符号：框架把实现细节漏出去了
    leaks: list[str] = field(default_factory=list)

    @property
    def all_domains(self) -> set[str]:
        out = set(self.module_level)
        for domains in self.symbols.values():
            out |= domains
        return out

    def public_domains(self) -> set[str]:
        return {d for d in self.all_domains if d in PROMOTABLE}


def _resolve(node: ast.ImportFrom, path: Path) -> str | None:
    """把（可能是相对的）import 还原成绝对模块名。"""
    if node.level == 0:
        return node.module

    # 相对 import：从当前文件所在 package 往上走 level-1 层
    inside = path.relative_to(FRAMEWORK_SRC / FRAMEWORK_PKG).parts[:-1]
    up = node.level - 1
    if up > len(inside):
        return None  # 越过包根，非法但不用在这里报
    base = inside[: len(inside) - up] if up else inside
    parts = (FRAMEWORK_PKG, *base)
    if node.module:
        parts = (*parts, *node.module.split("."))
    return ".".join(parts)


def build_reexport_map() -> dict[tuple[str, str], str]:
    """`(包, 符号) → 真正定义它的模块`。

    ## 为什么非有不可

    产品写的是 `from agent_framework.context import ContextBuilder`，而
    `ContextBuilder` 是 `context/builder.py` 里定义的、由 `context/__init__.py`
    re-export 出来的。不做穿透，`context.builder` 会显示"没人 import"，
    而 D1 的动作恰恰是「删掉没人用的抽象」——**误删的正是产品天天在用的那些**。

    一个包可以复用另一个包的 re-export（`agent_framework` → `llm` → `llm.client`），
    所以这里迭代到不动点，而不是只走一层。
    """
    mapping: dict[tuple[str, str], str] = {}

    for path in sorted(FRAMEWORK_SRC.rglob("__init__.py")):
        inside = path.relative_to(FRAMEWORK_SRC / FRAMEWORK_PKG).parts[:-1]
        pkg = ".".join((FRAMEWORK_PKG, *inside))
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            src = _resolve(node, path)
            if not src or src.split(".")[0] != FRAMEWORK_PKG:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                mapping[(pkg, local)] = src

    # 传递闭包：`__init__` 从 `__init__` 再 re-export 的情况
    changed = True
    while changed:
        changed = False
        for key, module in list(mapping.items()):
            nxt = mapping.get((module, key[1]))
            if nxt is not None and nxt != module:
                mapping[key] = nxt
                changed = True
    return mapping


def scan_domain(name: str, rel_root: str) -> dict[str, ModuleUsage]:
    """扫一个使用域，返回「模块 → 使用情况」。"""
    root = REPO / rel_root
    usage: dict[str, ModuleUsage] = {}

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = _resolve(node, path) if path.is_relative_to(FRAMEWORK_SRC) else node.module
                if not module or module.split(".")[0] != FRAMEWORK_PKG:
                    continue
                entry = usage.setdefault(module, ModuleUsage(module))
                for alias in node.names:
                    if alias.name == "*":
                        entry.module_level.add(name)
                        continue
                    entry.symbols[alias.name].add(name)
                    # 私有符号被外部 import —— 这是 API 泄漏，不是「用法」。
                    # dunder（`__version__`/`__all__`）不算：那是标准协议的一部分，
                    # 把它报成泄漏会让这条规则第一批就跑出假阳性，然后没人再看它。
                    if (
                        alias.name.startswith("_")
                        and not alias.name.startswith("__")
                        and name in PROMOTABLE
                    ):
                        entry.leaks.append(f"{module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] != FRAMEWORK_PKG:
                        continue
                    entry = usage.setdefault(alias.name, ModuleUsage(alias.name))
                    entry.module_level.add(name)

    return usage


def walk_framework_modules() -> list[str]:
    """枚举框架下所有可 import 的模块（跳过 providers 之类的实现细节？不跳——
    跳过就等于替读者决定，报告里列出来让人自己看）。"""
    import agent_framework

    modules = [FRAMEWORK_PKG]
    for info in pkgutil.walk_packages(
        agent_framework.__path__, prefix=f"{FRAMEWORK_PKG}."
    ):
        modules.append(info.name)
    return sorted(modules)


def module_path(module: str) -> Path | None:
    """模块名 → 源码文件。不 import，避免审计触发副作用（连库、读 env）。"""
    base = FRAMEWORK_SRC.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def declared_symbols(module: str) -> set[str]:
    """模块自己定义的公开符号。

    用 AST 静态解析，不用运行时 `vars()`。运行时方案有个致命盲点：
    判断「这是不是本模块的东西」只能看 `value.__module__`，而
    `router = APIRouter()` 这种**实例**的 `__module__` 是 `fastapi.routing`，
    于是 `server/routes_chat.py` 的公开符号集算成空——这个模块就会被报成
    「没人用的空壳」，而它正是 `/v1/chat` 的实现。

    优先信 `__all__`（作者显式承诺的面）；没有就取顶层 def/class/赋值。
    """
    path = module_path(module)
    if path is None:
        return set()

    try:
        tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    except SyntaxError:  # pragma: no cover - 源码坏了该由 ruff/mypy 先报
        return set()

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            continue
        try:
            declared = ast.literal_eval(node.value)
        except ValueError:
            break
        return {str(item) for item in declared}

    public: set[str] = set()
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        public.update(name for name in names if not name.startswith("_"))
    return public


def package_of(module: str) -> str | None:
    """module 所属的**包**（模块自己就是包时返回自己）。

    `agent_framework.llm.client` → `agent_framework.llm`；
    `agent_framework.observability`（单文件模块）→ 父级 `agent_framework` 虽是包，
    但它不是「这个符号的命名空间」，返回 None 让调用方跳过。
    """
    own = module_path(module)
    if own is not None and own.name == "__init__.py":
        return module
    if "." not in module:
        return None
    parent = module.rsplit(".", 1)[0]
    parent_path = module_path(parent)
    if parent_path is not None and parent_path.name == "__init__.py":
        return parent
    return None


def check_public_surface(report: dict[str, Any]) -> list[str]:
    """产品在用的符号，能否从它所在子包**浅路径**导入。

    ## 这条断言的由来

    产品原来的写法是 `from agent_framework.llm.client import ChatCall`——
    因为 `from agent_framework.llm import ChatCall` 根本不通：`llm/__init__.py`
    没导出 `client` 里的那三个类。于是"公共 API 在子包"这件事只是文档上的说法，
    实际每个人都在穿到实现模块里拿东西。

    更糟的是：一旦有人从深层路径 import，`client.py` 的内部重构就变成了
    breaking change，而我们不会知道——因为没有任何东西在守着。

    所以把"子包 `__init__` 覆盖产品所用符号"变成断言。
    """
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()

    for module, rows in report["symbols"].items():
        for row in rows:
            # 产品测试也算：测试用的符号如果导不出来，测试会在 collection 阶段
            # 炸（`ImportError: cannot import name 'AUTH_INVALID'`），
            # 而那是"公共面不完整"的另一种表现，不是"测试写错了"。
            if not (set(row["domains"]) & PROMOTABLE):
                continue
            symbol = row["symbol"]
            origin = report["reexports"].get(f"{module}.{symbol}", module)
            pkg = package_of(origin)
            if pkg is None or (pkg, symbol) in seen:
                continue
            seen.add((pkg, symbol))
            if symbol not in declared_symbols(pkg):
                problems.append(f"{pkg} 未导出 {symbol}（定义在 {origin}）")
    return problems


def build_report() -> dict[str, Any]:
    merged: dict[str, ModuleUsage] = {}
    for domain, rel in DOMAINS:
        usage = scan_domain(domain, rel)
        for module, entry in usage.items():
            target = merged.setdefault(module, ModuleUsage(module))
            for symbol, domains in entry.symbols.items():
                target.symbols[symbol] |= domains
            target.module_level |= entry.module_level
            target.leaks.extend(entry.leaks)

    # ── 穿透 re-export：把「从哪个模块 import」归因到「符号定义在哪个模块」──
    reexports = build_reexport_map()
    defined: dict[tuple[str, str], set[str]] = defaultdict(set)
    for module, entry in merged.items():
        for symbol, domains in entry.symbols.items():
            origin = reexports.get((module, symbol), module)
            defined[(origin, symbol)] |= domains

    def domains_of(module: str, symbol: str) -> set[str]:
        # 使用记录已经按「定义模块」归因，所以查询前也要先解析 re-export。
        # 不解析的话，`agent_framework.context` 这种纯 re-export 包会显示
        # 「它的 `__all__` 里没有一个符号被使用」——而事实是它们全都在用，
        # 只是被记到了 `context.builder` 这类定义模块名下。
        origin = reexports.get((module, symbol), module)
        return defined.get((origin, symbol), set())

    modules = walk_framework_modules()
    unused: list[str] = []
    detail: dict[str, list[dict[str, Any]]] = {}
    for module in modules:
        public = declared_symbols(module)
        entry = merged.get(module, ModuleUsage(module))
        detail[module] = [
            {
                "symbol": symbol,
                "domains": sorted(domains_of(module, symbol)),
                "promotable": bool(domains_of(module, symbol) & PROMOTABLE),
            }
            for symbol in sorted(public)
        ]
        if not public:
            continue
        # 使用方拿整个模块当命名空间用（`import agent_framework.x`），
        # 单个符号没被点名不等于没人用
        if entry.module_level:
            continue
        if not any(domains_of(module, symbol) for symbol in public):
            unused.append(module)

    return {
        "modules_total": len(modules),
        "modules_used_by_product": sorted(
            m for m, e in merged.items() if "product" in e.all_domains
        ),
        "reexports": {
            f"{pkg}.{symbol}": origin
            for (pkg, symbol), origin in sorted(reexports.items())
            if pkg != origin
        },
        "symbols": detail,
        "used_symbols": {
            module: [
                {
                    "symbol": symbol,
                    "domains": sorted(domains),
                    "promotable": bool(domains & PROMOTABLE),
                }
                for symbol, domains in sorted(entry.symbols.items())
            ]
            for module, entry in sorted(merged.items())
        },
        "leaks": sorted({leak for e in merged.values() for leak in e.leaks}),
        "unused_modules": sorted(unused),
    }


def print_report(report: dict[str, Any]) -> None:
    by_module: dict[str, list[dict[str, Any]]] = report["symbols"]

    print("=" * 100)
    print(f"公共 API 审计 · 框架模块 {report['modules_total']} 个")
    print("=" * 100)

    product = {
        module: [row for row in rows if row["promotable"]]
        for module, rows in by_module.items()
    }
    product = {m: rows for m, rows in product.items() if rows}
    total_syms = sum(len(rows) for rows in product.values())
    print(
        f"\n## 产品真实使用（{len(product)} 个模块 / {total_syms} 个符号，"
        "够格进公共 API）\n"
    )
    for module in sorted(product):
        names = sorted(row["symbol"] for row in product[module])
        print(f"  {module}")
        print(f"    {', '.join(names)}")

    internal = {
        module: [row for row in rows if not row["promotable"] and row["domains"]]
        for module, rows in by_module.items()
    }
    internal = {m: rows for m, rows in internal.items() if rows}
    print(f"\n## 仅内部/测试使用（{len(internal)} 个模块，留在 internal）\n")
    for module in sorted(internal):
        names = sorted(
            f"{row['symbol']}[{'/'.join(row['domains'])}]" for row in internal[module]
        )
        print(f"  {module}: {', '.join(names)}")

    if report["leaks"]:
        print("\n## ⚠ 私有符号泄漏（外部 import 了 `_` 开头的符号）\n")
        for leak in report["leaks"]:
            print(f"  {leak}")

    gaps = check_public_surface(report)
    if gaps:
        print(f"\n## ⚠ 子包未导出产品在用的符号（{len(gaps)} 处）\n")
        for gap in gaps:
            print(f"  {gap}")

    if report["unused_modules"]:
        print(f"\n## 任何使用域都没有引用的模块（{len(report['unused_modules'])} 个）\n")
        for module in report["unused_modules"]:
            print(f"  {module}")
    print()


def check(report: dict[str, Any]) -> int:
    """CI 断言。一条失败就退出非零——审计报告的结论必须能被自动化守住。"""
    problems: list[str] = []

    if report["leaks"]:
        problems.append(
            f"{len(report['leaks'])} 处私有符号泄漏：" + ", ".join(report["leaks"])
        )

    unused_public = report["unused_modules"]
    if unused_public:
        problems.append(
            f"{len(unused_public)} 个模块的公开符号无人 import（可能是空壳）："
            + ", ".join(unused_public)
        )

    gaps = check_public_surface(report)
    if gaps:
        problems.append(f"{len(gaps)} 处子包未导出产品在用的符号：" + "; ".join(gaps))

    if problems:
        print("审计未通过：")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("审计通过：无私有符号泄漏，无空壳模块。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="公共 API 审计")
    parser.add_argument("--check", action="store_true", help="只做断言，不打全量报告")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    report = build_report()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.check:
        return check(report)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
