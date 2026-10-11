"""
rag_mvp.cli：`rag-mvp` 命令行统一入口。

pyproject.toml 中声明的 console script：

    [project.scripts]
    rag-mvp = "rag_mvp:main"

它要求 `rag_mvp` 包暴露一个可调用的 ``main()``（见 rag_mvp/__init__.py 的再导出）。
本模块实现这个 main()：把子命令转发给 scripts/ 下已有的脚本，
让 `rag-mvp` 成为一个可用的入口，而不是一执行就 ImportError。

设计要点：
1. 子脚本都有 ``main()``，这里以模块方式加载后直接调用，
   等价于 ``uv run python scripts/xxx.py``，无需把 scripts/ 变成可导入的包。
   （不用 runpy 是因为它会把 sys.argv[0] 覆盖成脚本路径，
   导致子脚本的 usage 显示成 search_cli.py 之类的文件名。）
2. 子脚本内部大量使用 ``data/...`` 这类相对路径（如 data/index/local.pkl），
   因此请在**项目根目录**执行本命令。
3. 子脚本自己会 raise SystemExit（argparse 出错、smoke 未通过等），
   这里不做拦截，让退出码原样向上传递。

用法示例::

    rag-mvp --help
    rag-mvp chunks                    # → scripts/build_chunks.py
    rag-mvp vector-index              # → scripts/build_vector_index.py
    rag-mvp bm25-index                # → scripts/build_bm25_index.py
    rag-mvp search "MCP 是什么" -k 3   # → scripts/search_cli.py
    rag-mvp ask "MCP 是什么" --dry-run # → scripts/ask_cli.py
    rag-mvp smoke --modes hybrid      # → scripts/run_smoke_retrieval.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# 当前文件是 src/rag_mvp/cli.py：
# parents[0]=src/rag_mvp  parents[1]=src  parents[2]=项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# 子命令 → scripts/ 下的脚本文件；一个脚本可以有多个别名
COMMANDS: dict[str, str] = {
    "chunks": "build_chunks.py",
    "build-chunks": "build_chunks.py",
    "vector-index": "build_vector_index.py",
    "index": "build_vector_index.py",
    "bm25-index": "build_bm25_index.py",
    "bm25": "build_bm25_index.py",
    "search": "search_cli.py",
    "ask": "ask_cli.py",
    "smoke": "run_smoke_retrieval.py",
}

# --help 里展示的规范命令与说明（别名不单独列出）
_HELP_ROWS: list[tuple[str, str]] = [
    ("chunks", "解析文档 → data/processed/chunks.jsonl"),
    ("vector-index", "构建稠密向量索引 → data/index/local.pkl（首次会拉 BGE-M3）"),
    ("bm25-index", "构建 BM25 稀疏索引 → data/index/bm25.pkl"),
    ("search", "检索 Top-K 并打印 chunk 与元数据（--mode dense|sparse|hybrid）"),
    ("ask", "RAG 问答并输出引用（需 DEEPSEEK_API_KEY，--dry-run 仅检索）"),
    ("smoke", "运行 smoke 检索评测并写出报告"),
]

# 别名 → 规范命令，仅用于提示
_ALIASES: dict[str, str] = {
    "build-chunks": "chunks",
    "index": "vector-index",
    "bm25": "bm25-index",
}


def _print_help(stream=sys.stdout) -> None:
    width = max(len(name) for name, _ in _HELP_ROWS)
    lines = [
        "usage: rag-mvp <command> [options]",
        "",
        "RAG MVP 统一命令行入口，命令会转发到 scripts/ 下对应脚本。",
        "",
        "commands:",
    ]
    for name, desc in _HELP_ROWS:
        lines.append(f"  {name.ljust(width)}  {desc}")
    lines += [
        "",
        "aliases:",
        "  " + ", ".join(f"{alias} → {target}" for alias, target in _ALIASES.items()),
        "",
        "options:",
        "  -h, --help     显示本帮助",
        "  -V, --version  显示版本",
        "",
        "提示：请在项目根目录执行，子脚本使用 data/ 下的相对路径。",
        "查看某个子命令的参数：rag-mvp <command> --help",
    ]
    print("\n".join(lines), file=stream)


def _version() -> str:
    try:
        from importlib.metadata import version

        return f"rag-mvp {version('rag-mvp')}"
    except Exception:  # 未安装（如直接跑源码）时兜底
        return "rag-mvp (version unknown)"


def _load_script(command: str, script_path: Path) -> ModuleType:
    """把 scripts/ 下的脚本当模块加载（不执行其 main）。"""
    module_name = f"_rag_mvp_cmd_{command.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    """`rag-mvp` 的入口函数，返回进程退出码。"""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    if args[0] in {"-V", "--version", "version"}:
        print(_version())
        return 0

    command, rest = args[0], args[1:]
    script_name = COMMANDS.get(command)
    if script_name is None:
        print(f"rag-mvp: 未知子命令 {command!r}\n", file=sys.stderr)
        _print_help(sys.stderr)
        return 2

    script_path = SCRIPTS_DIR / script_name
    if not script_path.is_file():
        print(f"rag-mvp: 找不到脚本 {script_path}", file=sys.stderr)
        return 1

    # 转发参数：argv[0] 取 "rag-mvp <command>"，让子脚本的 usage 显示更统一；
    # 其余参数原样交给它自己的 argparse。
    saved_argv = sys.argv
    sys.argv = [f"rag-mvp {command}", *rest]
    try:
        module = _load_script(command, script_path)
        entry = getattr(module, "main", None)
        if not callable(entry):
            print(f"rag-mvp: {script_name} 未定义 main()", file=sys.stderr)
            return 1
        result = entry()
    finally:
        sys.argv = saved_argv

    return result if isinstance(result, int) else 0
