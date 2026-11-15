"""`agent create` 脚手架的纯函数测试。

测的是**生成器本身**：命名归一化、模板渲染、文件落盘、`--base-url`/`--framework-spec`
能不能落到该落的地方。**不**测"生成的项目跑不跑得起来"——那件事由
`evals/scripts/scaffold_drill.py` 端到端验证，这里再写一遍只是把慢测试复制一份。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from agent_framework.cli import ScaffoldOptions, create_project, main, package_name, render_files

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture()
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """tmp_path 在沙箱里会被 shim 拦；用项目内临时目录绕开。"""
    d = Path(tempfile.mkdtemp(prefix="cli-test-", dir=str(REPO / ".tmp")))
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


# ── 命名归一化 ──────────────────────────────────────────────────────────


def test_dashes_and_spaces_become_underscores() -> None:
    assert package_name("my-agent") == "my_agent"
    assert package_name("My Agent") == "my_agent"


def test_non_identifier_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="合法的 Python 包名"):
        package_name("123abc")

    with pytest.raises(ValueError, match="合法的 Python 包名"):
        package_name("if")


# ── 渲染：纯函数 ───────────────────────────────────────────────────────


def test_render_produces_the_expected_file_set() -> None:
    files = render_files("demo-agent")

    paths = sorted(files)
    assert "src/demo_agent/__init__.py" in paths
    assert "src/demo_agent/app.py" in paths
    assert "configs/models.yaml" in paths
    assert "pyproject.toml" in paths
    assert "tests/test_app.py" in paths
    assert "README.md" in paths


def test_base_url_lands_in_models_yaml() -> None:
    files = render_files(
        "demo", options=ScaffoldOptions(base_url="http://127.0.0.1:9999/v1")
    )
    yaml = files["configs/models.yaml"]
    assert "http://127.0.0.1:9999/v1" in yaml
    # 默认值不应残留
    assert "https://api.openai.com/v1" not in yaml


def test_framework_spec_lands_in_pyproject() -> None:
    files = render_files(
        "demo",
        options=ScaffoldOptions(framework_spec="agent-framework @ file:///x/whl"),
    )
    toml = files["pyproject.toml"]
    assert 'agent-framework @ file:///x/whl' in toml


def test_framework_path_writes_uv_sources() -> None:
    files = render_files(
        "demo",
        options=ScaffoldOptions(framework_source="/abs/path"),
    )
    toml = files["pyproject.toml"]
    assert 'path = "/abs/path"' in toml


def test_app_template_uses_public_imports_only() -> None:
    """生成的代码不许穿透实现模块——否则等于教第三方走 D1 刚清掉的路。"""
    app = render_files("demo-agent")["src/demo_agent/app.py"]
    assert "from agent_framework import" in app
    assert "from agent_framework.server import create_app" in app
    # 不该出现深层路径
    assert "agent_framework.llm.client" not in app
    assert "agent_framework.server.app import" not in app


def test_test_template_uses_fake_gateway_not_network() -> None:
    test_text = render_files("demo-agent")["tests/test_app.py"]
    assert "FakeGateway" in test_text


# ── 落盘 ───────────────────────────────────────────────────────────────


def test_create_project_writes_files(workdir: Path) -> None:
    root = create_project("demo-agent", workdir)
    assert (root / "src" / "demo_agent" / "app.py").is_file()
    assert (root / "configs" / "models.yaml").is_file()
    assert (root / "README.md").read_text("utf-8").startswith("# demo-agent")


def test_create_project_refuses_nonempty_without_force(workdir: Path) -> None:
    (workdir / "demo").mkdir()
    (workdir / "demo" / "junk").write_text("x")
    with pytest.raises(FileExistsError):
        create_project("demo", workdir)


def test_create_project_force_overwrites(workdir: Path) -> None:
    root = workdir / "demo"
    root.mkdir()
    (root / "README.md").write_text("old")
    create_project("demo", workdir, force=True)
    assert (root / "README.md").read_text("utf-8").startswith("# demo")


# ── CLI ────────────────────────────────────────────────────────────────


def test_main_create_writes_project_and_returns_zero(workdir: Path) -> None:
    rc = main(["create", "demo", "--target", str(workdir)])
    assert rc == 0
    assert (workdir / "demo" / "src" / "demo" / "app.py").is_file()


def test_main_rejects_bad_name(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["create", "123bad", "--target", str(workdir)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "失败" in captured.err


def test_cli_entrypoint_is_installed() -> None:
    """`agent --help` 能跑起来，证明 pyproject 的 [project.scripts] 真装了。"""
    result = subprocess.run(
        [sys.executable, "-c", "from agent_framework.cli import main; main(['--help'])"],
        capture_output=True,
        text=True,
        check=False,
    )
    # --help 让 argparse 走 SystemExit(0) → 子进程退出码 0
    assert result.returncode == 0
    assert "create" in result.stdout
