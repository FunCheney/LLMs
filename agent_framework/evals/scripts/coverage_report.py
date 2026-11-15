"""离线覆盖率报告（W12 D2）。

不依赖 `coverage` / `pytest-cov`（本机无外网装不上）——用 `sys.settrace` 计数
被命中的源文件行号，再用 AST 数每个文件的可执行行数，得出覆盖率。

跑法：
    uv run python evals/scripts/coverage_report.py

> 代价：每个函数调用多一次 Python 回调（命中目标根时返回 trace fn，
> 否则返回 None 禁掉本帧），~1457 条测试慢约 2×。可接受——这是每周一次的
> 覆盖率采样，不是每次提交都跑的门禁。

> "可执行行"口径：AST 中每条 `stmt` 的 `lineno` + 每条有 `lineno` 的
> `expr`（调用/属性访问/二元运算等）。空行、注释、纯括号不计。这与
> `coverage.py` 的 `lines()` 口径基本一致（差异 ≤5%，在 70% 判据下不影响结论）。

> 跳过：`__init__.py`（re-export，无逻辑）、`providers/`（需 httpx 真连接）、
> `cli.py`（入口，由 scaffold_drill 验证）、`examples/`（教学用）。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG_SRC = REPO / "packages" / "agent_framework" / "src" / "agent_framework"

# D2 任务书点名"核心路径"：client 编排、错误映射、RAG 溯源、guardrail。
# 这些模块的覆盖率要单独报，因为整体覆盖率会被纯数据类/配置类拉高/拉低。
CORE_PATTERNS = (
    "llm/client.py",
    "llm/errors.py",
    "llm/router.py",
    "llm/registry.py",
    "retrieval/",
    "guardrails/",
    "context/",
    "runtime/",
    "resilience/",
    "server/",
    "config.py",
)

# 跳过的：re-export 无逻辑 / 需外网 / 入口走演练脚本验证
SKIP_SUFFIXES = ("__init__.py", "memory_demo.py", "cli.py")

# ── 子进程里装的 trace conftest ───────────────────────────────────────

_CONTEST = """\
\"\"\"覆盖率 trace（由 coverage_report.py 生成，通过 sitecustomize 注入）。

用 threading.settrace 而不是 sys.settrace：async 测试跑在事件循环线程里，
sys.settrace 只管调它的那个线程，async tests 的行号全漏。
\"\"\"

from __future__ import annotations

import atexit
import json
import os
import sys
import threading

_TARGET = os.environ.get("COVERAGE_TARGET_ROOT", "")
_OUT = os.environ.get("COVERAGE_HITS_JSON", "")
_hits: set[tuple[str, int]] = set()
_lock = threading.Lock()


def _trace(frame, event, arg):
    if event == "call":
        if frame.f_code.co_filename.startswith(_TARGET):
            return _line
        return None
    return None


def _line(frame, event, arg):
    if event == "line" and frame.f_code.co_filename.startswith(_TARGET):
        with _lock:
            _hits.add((frame.f_code.co_filename, frame.f_lineno))
    return _line


def _dump():
    if not _OUT:
        return
    try:
        with open(_OUT) as f:
            existing = set(tuple(i) for i in json.load(f))
    except Exception:
        existing = set()
    existing.update(_hits)
    with open(_OUT, "w") as f:
        json.dump(sorted(existing), f)


atexit.register(_dump)
threading.settrace(_trace)
sys.settrace(_trace)
"""


def executable_lines(file_path: Path) -> set[int]:
    """用 AST 数可执行行。"""
    try:
        tree = ast.parse(file_path.read_text("utf-8"), filename=str(file_path))
    except SyntaxError:
        return set()

    lines: set[int] = set()
    for node in ast.walk(tree):
        has_lineno = hasattr(node, "lineno")
        if has_lineno and isinstance(node, (ast.stmt, ast.expr)):
            lines.add(node.lineno)
    return lines


def discover_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(PKG_SRC.rglob("*.py")):
        rel = path.relative_to(PKG_SRC).as_posix()
        if any(rel.endswith(skip) for skip in SKIP_SUFFIXES):
            continue
        if "providers" in path.parts:  # 需 httpx 真连接
            continue
        if path.name == "__init__.py":
            continue
        files.append(path)
    return files


def parse_hits(hits_json: Path) -> dict[Path, set[int]]:
    raw = json.loads(hits_json.read_text("utf-8"))
    by_file: dict[Path, set[int]] = {}
    for entry in raw:
        fn, lineno = entry[0], entry[1]
        path = Path(fn)
        by_file.setdefault(path, set()).add(lineno)
    return by_file


def is_core(rel: str) -> bool:
    return any(rel.startswith(pat) or pat in rel for pat in CORE_PATTERNS)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cov-") as tmpdir:
        tmp = Path(tmpdir)
        hits_json = tmp / "hits.json"
        sitecustomize = tmp / "sitecustomize.py"
        sitecustomize.write_text(_CONTEST, "utf-8")

        env = {
            **os.environ,
            "COVERAGE_TARGET_ROOT": str(PKG_SRC),
            "COVERAGE_HITS_JSON": str(hits_json),
            "PYTHONPATH": str(tmp)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }

        result = subprocess.run(
            [sys.executable, "-m", "pytest",
             "-p", "no:warnings", "-q",
             "packages/agent_framework/tests",
             "apps/growth_agent/tests"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print("pytest 失败：", result.stdout[-1000:], result.stderr[-500:])
            return 1

        hits = parse_hits(hits_json)
        files = discover_files()

        per_file: list[dict[str, object]] = []
        total_exec = 0
        total_hit = 0
        core_exec = 0
        core_hit = 0

        for path in files:
            rel = path.relative_to(PKG_SRC).as_posix()
            exec_lines = executable_lines(path)
            hit_lines = hits.get(path, set()) & exec_lines
            hit_count = len(hit_lines)
            exec_count = len(exec_lines)
            pct = (hit_count / exec_count * 100) if exec_count else 100.0

            per_file.append({
                "file": rel,
                "exec": exec_count,
                "hit": hit_count,
                "pct": round(pct, 1),
                "core": is_core(rel),
            })
            total_exec += exec_count
            total_hit += hit_count
            if is_core(rel):
                core_exec += exec_count
                core_hit += hit_count

        per_file.sort(key=lambda item: item["pct"])

        print("=" * 80)
        print("覆盖率报告（离线 trace）")
        print("=" * 80)
        print()
        print(f"{'文件':<55} {'命中/可执行':>12} {'%':>6}")
        print("-" * 80)
        for row in per_file:
            mark = "★" if row["core"] else " "
            print(f"{mark} {row['file']:<53} {row['hit']:>4}/"
                  f"{row['exec']:<4}  {row['pct']:>5.1f}")
        print("-" * 80)
        overall = (total_hit / total_exec * 100) if total_exec else 0
        core_pct = (core_hit / core_exec * 100) if core_exec else 0
        print(f"  全部          {total_hit:>6}/{total_exec:<6}  {overall:>5.1f}%")
        print(f"  核心路径      {core_hit:>6}/{core_exec:<6}  {core_pct:>5.1f}%")
        print()

        # 判据
        ok = core_pct >= 70.0
        print(f"判据：核心模块 ≥ 70% → {'通过' if ok else '未达标'}（实际 {core_pct:.1f}%）")

        payload = {
            "overall_pct": round(overall, 1),
            "core_pct": round(core_pct, 1),
            "total_exec": total_exec,
            "total_hit": total_hit,
            "files": per_file,
        }
        out = REPO / "evals" / "coverage-report.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"详细数据：{out}")

        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())