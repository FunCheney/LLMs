"""工作区 MCP Server（W8 D4/D5）：带副作用的工具，用来把边界与验真跑实。

三个工具正好对应三个机制：

| 工具 | 机制 | 在哪一层拦 |
|---|---|---|
| `write_file` | 路径边界 | `sandbox.check_path`（副作用之前） |
| `run_command` | 命令边界 + 超时 + 输出上限 | `sandbox.run_command` |
| `publish_post` + `get_post` | 验真（回读真实状态） | `action` 的 verify 层 |

**为什么 server 自己也校验一遍**（执行器已经校验过）：server 是独立信任域，
将来可能跑在别的进程/别的机器上。假设调用方一定干净，等于把边界放在
一个可以被绕过的地方。重复校验的成本是一次字符串比较，收益是"这一层
自己就是安全的"，不是"因为它上游校验过了，所以它是安全的"。

`publish_post` 有个 `fail_soft` 开关：打开后返回 HTTP 层面的成功
（MCP 不报错）但响应体 `{"ok": false}`。这不是玩具——真实 CMS 里
"接口 200 + 审核未通过"是常态，而它正是 W8 三个 badcase 里最难抓的那个
（前两个 badcase 会报错，这个不会）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..mcp.client import MCPClient
from ..mcp.inmemory import InMemoryMCPTransport
from ..sandbox import SandboxPolicy

__all__ = ["WORKSPACE_TOOLS", "WorkspaceServer", "workspace_client"]

SERVER_NAME = "workspace"

_RUN_COMMAND_TOOL: dict[str, Any] = {
    "name": "run_command",
    "description": "在工作区执行一条白名单内的只读命令（ruff / pytest / git status）。",
    "inputSchema": {
        "type": "object",
        "properties": {"command": {"type": "string", "minLength": 3, "maxLength": 200}},
        "required": ["command"],
        "additionalProperties": False,
    },
}

_WRITE_FILE_TOOL: dict[str, Any] = {
    "name": "write_file",
    "description": "在工作区内写入一个文本文件（父目录自动创建）。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 5, "maxLength": 200},
            "content": {"type": "string", "minLength": 1, "maxLength": 20000},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
}

_READ_FILE_TOOL: dict[str, Any] = {
    "name": "read_file",
    "description": "读取工作区内的文本文件（验真回读用）。",
    "inputSchema": {
        "type": "object",
        "properties": {"id": {"type": "string", "minLength": 1, "maxLength": 200}},
        "required": ["id"],
        "additionalProperties": False,
    },
}

_PUBLISH_TOOL: dict[str, Any] = {
    "name": "publish_post",
    "description": "把内容发布为对外文章。不可撤销，必须先拿到人工审批。",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 2, "maxLength": 120},
            "body": {"type": "string", "minLength": 50, "maxLength": 20000},
        },
        "required": ["title", "body"],
        "additionalProperties": False,
    },
}

_GET_POST_TOOL: dict[str, Any] = {
    "name": "get_post",
    "description": "按 post_id 回读文章状态（验真用，只读）。",
    "inputSchema": {
        "type": "object",
        "properties": {"id": {"type": "string", "minLength": 1, "maxLength": 64}},
        "required": ["id"],
        "additionalProperties": False,
    },
}

WORKSPACE_TOOLS: tuple[dict[str, Any], ...] = (
    _RUN_COMMAND_TOOL,
    _WRITE_FILE_TOOL,
    _READ_FILE_TOOL,
    _PUBLISH_TOOL,
    _GET_POST_TOOL,
)


class WorkspaceServer:
    """工作区能力。`root` 是**唯一**允许写入的目录（沙箱的 write_roots）。"""

    def __init__(
        self,
        *,
        root: str | Path,
        sandbox: SandboxPolicy | None = None,
        fail_soft: bool = False,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        # 没传沙箱就用一个只允许本目录的默认策略。**不**用"无沙箱"：
        # 默认放开是这个模块最容易犯的错。
        self._sandbox = sandbox or SandboxPolicy(write_roots=(self._root,))
        # 帖子库（真系统里是 CMS）。进程内存储足够支撑验真与 badcase。
        self._posts: dict[str, dict[str, Any]] = {}
        # 打开后 publish_post 会返回 `{"ok": false}` 的假成功（badcase #3）。
        self.fail_soft = fail_soft

    # -- 工具实现 ---------------------------------------------------------- #

    async def handle(self, params: dict[str, Any]) -> dict[str, Any]:
        """按工具名分发。名字从 params 里取而不是靠闭包分装：
        一个 handler 对应多种工具时，`add_tool` 才不用再造一个函数。"""
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}

        if name == "write_file":
            return self._write_file(arguments)
        if name == "read_file":
            return self._read_file(arguments)
        if name == "run_command":
            return await self._run_command(arguments)
        if name == "publish_post":
            return self._publish_post(arguments)
        if name == "get_post":
            return self._get_post(arguments)
        return {"content": [{"type": "text", "text": f"未知工具 {name}"}], "isError": True}

    def _write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("path") or "")
        content = str(arguments.get("content") or "")
        target = self._resolve(relative)

        # 副作用之前判（纪律 1）。这里再判一次的理由见模块 docstring。
        decision = self._sandbox.check_path(target)
        if decision.denied:
            return _denied(decision.code, decision.detail)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, "utf-8")
        except OSError as exc:
            return {"content": [{"type": "text", "text": f"写入失败：{exc}"}], "isError": True}

        return _ok(
            {
                "ok": True,
                "path": relative,
                "bytes": len(content.encode()),
                "status": "ok",
            }
        )

    def _read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = str(arguments.get("id") or arguments.get("path") or "")
        target = self._resolve(relative)
        decision = self._sandbox.check_path(target, write=False)
        if decision.denied:
            return _denied(decision.code, decision.detail)

        if not target.is_file():
            # 读不到不是错误：验真层要靠 status=missing 判断"没生效"。
            return _ok({"path": relative, "status": "missing", "bytes": 0, "text": ""})

        text = target.read_text("utf-8", errors="replace")
        return _ok(
            {"path": relative, "status": "ok", "bytes": len(text.encode()), "text": text[:4000]}
        )

    async def _run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command") or "")
        result = await self._sandbox.run_command(command, cwd=self._root)

        if result.denied_code:
            return _denied(result.denied_code, result.denied_detail)

        payload = {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "truncated": result.truncated,
            "timed_out": result.timed_out,
        }
        if result.timed_out:
            return {"content": [{"type": "text", "text": "命令超时"}], "isError": True}
        return _ok(payload)

    def _publish_post(self, arguments: dict[str, Any]) -> dict[str, Any]:
        title = str(arguments.get("title") or "")
        body = str(arguments.get("body") or "")

        if self.fail_soft:
            # 假成功：MCP 层不报错，响应体说没生效。验真层必须抓到它。
            return _ok(
                {
                    "ok": False,
                    "post_id": None,
                    "status": "draft",
                    "reason": "审核未通过",
                }
            )

        post_id = hashlib.sha256(f"{title}\x00{body}\x00{time.time()}".encode()).hexdigest()[:8]
        self._posts[post_id] = {
            "post_id": post_id,
            "title": title,
            "body": body,
            "status": "published",
        }
        return _ok({"ok": True, "post_id": post_id, "status": "published"})

    def _get_post(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_id = str(arguments.get("id") or "")
        record = self._posts.get(post_id)
        if record is None:
            return _ok({"post_id": post_id, "status": "missing"})
        return _ok(
            {"post_id": post_id, "title": record["title"], "status": record["status"]}
        )

    def posts(self) -> dict[str, dict[str, Any]]:
        """已发布内容的快照（只读副本）。

        测试用它数"副作用发生了几次"——幂等是否真的生效，只能靠副作用计数
        验证，不能靠返回值（返回值在第二次就已经是缓存的）。
        """
        return {key: dict(value) for key, value in self._posts.items()}

    def _resolve(self, relative: str) -> Path:
        """相对路径 → 绝对路径。绝对路径也接受（沙箱会判它是否在 root 内）。"""
        candidate = Path(relative).expanduser()
        return candidate if candidate.is_absolute() else (self._root / candidate)

    # -- 装配 -------------------------------------------------------------- #

    def transport(self) -> InMemoryMCPTransport:
        return InMemoryMCPTransport(
            list(WORKSPACE_TOOLS), server_name=SERVER_NAME, handlers=_handlers(self)
        )

    def client(
        self, *, timeout_s: float = 15.0, transport: InMemoryMCPTransport | None = None
    ) -> MCPClient:
        return MCPClient(
            transport or self.transport(),
            server_name=SERVER_NAME,
            timeout_s=timeout_s,
            description_suffix="副作用动作，结果需回读验真。",
        )


def _handlers(server: WorkspaceServer) -> dict[str, Any]:
    """每个工具一个 handler 闭包，把工具名绑进去。

    `InMemoryMCPTransport.call_tool` 按工具名找 handler，所以这里必须
    按名分装——用一个共享 handler 再从 params 里取名字也行，
    但那样 transport 层就"知道"了工具名语义，边界会糊。
    """

    async def _dispatch(params: dict[str, Any]) -> dict[str, Any]:
        return await server.handle(params)

    return {str(tool["name"]): _dispatch for tool in WORKSPACE_TOOLS}


def workspace_client(
    *,
    root: str | Path,
    sandbox: SandboxPolicy | None = None,
    fail_soft: bool = False,
    timeout_s: float = 15.0,
) -> MCPClient:
    return WorkspaceServer(root=root, sandbox=sandbox, fail_soft=fail_soft).client(
        timeout_s=timeout_s
    )


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "structuredContent": payload,
        "_meta": {"cost": 0.0},
    }


def _denied(code: str, detail: str) -> dict[str, Any]:
    """边界拒绝的返回形状。

    用 `isError: True` 而不是"成功但内容为空"：拒绝必须是一个**明确结果**，
    否则模型会把"被拒绝"当成"没有结果"，然后换个参数再试——
    而安全上我们恰恰希望它停下来。
    """
    return {
        "content": [{"type": "text", "text": f"越界被拒绝（{code}）：{detail}"}],
        "isError": True,
        "structuredContent": {"error": code, "detail": detail},
    }
