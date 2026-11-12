"""W8 产品层：`/v1/tools` 的接线与两条不变式。

测的是"框架能力真的接到了产品链路上"，而不是框架内部行为（那些在
`packages/agent_framework/tests/test_skill_*.py` 里）。所以这里只断言：

1. `TOOLS=off` 时**报错**而不是返回空工具集（空集会被读成"没加载到"）
2. `TOOLS=governed` 时可见面、执行、审批、热重载都能通过 HTTP 走通
3. 主体从请求头解析（无头 = 匿名，匿名不能写）
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_framework.config import Settings
from fastapi.testclient import TestClient
from growth_agent.app import create_app

BODY = "这是一篇足够长的正文，用来满足发布工具的最小长度限制。" * 2

SKILL_YAML = """
name: hot_skill
description: 热加载进来的演示 Skill。
mission: 负责演示热加载；不负责真实业务。
principles:
  - 不要乱写文件
instructions: |
  1. 写一行。
parameters:
  type: object
  properties:
    path: {type: string}
    content: {type: string}
  required: [path, content]
  additionalProperties: false
entrypoint: workspace.write_file
permissions: [read, write]
sandbox: subprocess
stages: [draft]
"""


def _client(tmp_path: Path, **overrides: object) -> TestClient:
    settings = Settings(
        tools="governed",
        tools_workspace_root=str(tmp_path / "ws"),
        tools_command_allow="echo",
        **overrides,  # type: ignore[arg-type]
    )
    return TestClient(create_app(settings=settings))


def _headers(groups: str = "editor") -> dict[str, str]:
    return {"X-Principal-Id": "u1", "X-Principal-Groups": groups}


def test_tools_off_returns_explicit_error() -> None:
    """没开工具链路时，`/v1/tools` 必须报错。

    返回空列表是更糟的选择：调用方无法区分"没开"与"开了但没有工具"，
    而这两种情况的处置完全不同。
    """

    client = TestClient(create_app(settings=Settings(tools="off")))
    response = client.get("/v1/tools")

    assert response.status_code == 500
    body = response.json()
    # 错误契约是扁平的 `{code, message, request_id}`（框架统一形状），
    # 不是嵌套的 `{"error": {...}}`——两者只差一层，但断言写错就会"以为测到了"。
    assert body["code"] == "CONFIG_ERROR"
    assert "TOOLS" in body["message"]
    assert body["request_id"].startswith("req_")


def test_list_tools_returns_visible_surface(tmp_path: Path) -> None:
    client = _client(tmp_path)

    body = client.get("/v1/tools", params={"stage": "research"}, headers=_headers()).json()

    assert body["stage"] == "research"
    assert "web_search" in body["visible"]
    assert "publish_post" not in body["visible"]
    assert body["ratio"] < 0.6
    # 渐进式加载的注入文本可被上层直接用。
    assert "web_search" in body["describe"]
    assert "把用户问题拆成" not in body["describe"]


def test_anonymous_principal_only_sees_read_tools(tmp_path: Path) -> None:
    client = _client(tmp_path)

    body = client.get("/v1/tools", params={"stage": "publish"}).json()

    assert body["principal"]["subject_id"] == "anonymous"
    assert "publish_post" not in body["visible"]


def test_describe_tool_returns_full_six_layers(tmp_path: Path) -> None:
    client = _client(tmp_path)

    body = client.get("/v1/tools/publish_post").json()

    assert body["mission"]
    assert body["principles"]
    assert body["instructions"]
    assert body["done_criteria"]
    assert body["dangerous"] is True
    assert body["lint"] == []


def test_describe_unknown_tool_is_404(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/v1/tools/nope").status_code == 404


def test_call_read_tool_succeeds(tmp_path: Path) -> None:
    client = _client(tmp_path)

    body = client.post(
        "/v1/tools/call",
        json={"name": "web_search", "arguments": {"query": "增长"}, "stage": "research"},
        headers=_headers(),
    ).json()

    assert body["status"] == "verified"
    # 回注消息要能直接被拼回 messages。
    assert body["tool_message"]["role"] == "tool"
    assert body["tool_message"]["tool_call_id"].startswith("http-")


def test_call_write_tool_as_viewer_is_forbidden(tmp_path: Path) -> None:
    client = _client(tmp_path)

    body = client.post(
        "/v1/tools/call",
        json={
            "name": "write_brief",
            "arguments": {"path": "briefs/a.md", "content": "内容" * 20},
            "stage": "draft",
        },
        headers=_headers("viewer"),
    ).json()

    assert body["status"] == "forbidden"


def test_out_of_stage_call_is_not_visible(tmp_path: Path) -> None:
    client = _client(tmp_path)

    body = client.post(
        "/v1/tools/call",
        json={
            "name": "publish_post",
            "arguments": {"title": "t", "body": BODY},
            "stage": "research",
        },
        headers=_headers("publisher,editor"),
    ).json()

    assert body["status"] == "tool_not_visible"


def test_publish_without_approver_is_pending(tmp_path: Path) -> None:
    """不可撤销动作停在 pending —— 没接 HITL 不等于默认放行。"""

    client = _client(tmp_path)

    body = client.post(
        "/v1/tools/call",
        json={
            "name": "publish_post",
            "arguments": {"title": "t", "body": BODY},
            "stage": "publish",
        },
        headers=_headers("publisher,editor"),
    ).json()

    assert body["status"] == "pending"
    assert body["content"] is None


def test_duplicate_call_is_idempotent(tmp_path: Path) -> None:
    client = _client(tmp_path)
    payload = {
        "name": "write_brief",
        "arguments": {"path": "briefs/a.md", "content": "内容" * 20},
        "stage": "draft",
        "intent_id": "task-1",
    }
    headers = _headers()

    first = client.post("/v1/tools/call", json=payload, headers=headers).json()
    second = client.post("/v1/tools/call", json=payload, headers=headers).json()

    assert first["status"] == "verified"
    assert second["status"] == "duplicate"
    assert second["duplicate_of"] == first["action_id"]


def test_hot_reload_picks_up_new_skill_file(tmp_path: Path) -> None:
    """检查点：新增 Skill 文件 → reload 后可被 list 到，主路由代码未改。"""

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    client = _client(tmp_path, tools_skills_dir=str(skills_dir))

    before = client.get("/v1/tools", params={"stage": "draft"}, headers=_headers()).json()
    assert "hot_skill" not in before["visible"]

    (skills_dir / "hot.yaml").write_text(SKILL_YAML, "utf-8")
    reloaded = client.post("/v1/tools/reload").json()

    assert "hot_skill" in reloaded["skills"]
    after = client.get("/v1/tools", params={"stage": "draft"}, headers=_headers()).json()
    assert "hot_skill" in after["visible"]

    called = client.post(
        "/v1/tools/call",
        json={
            "name": "hot_skill",
            "arguments": {"path": "hot/a.md", "content": "你好"},
            "stage": "draft",
        },
        headers=_headers(),
    ).json()
    assert called["status"] == "verified"


def test_sandbox_denial_over_http(tmp_path: Path) -> None:
    """越界写入在 HTTP 面也是结构化拒绝，而不是 500。"""

    client = _client(tmp_path)

    body = client.post(
        "/v1/tools/call",
        json={
            "name": "write_brief",
            "arguments": {"path": "../../escape.md", "content": "内容" * 20},
            "stage": "draft",
        },
        headers=_headers(),
    ).json()

    assert body["status"] == "failed"
    assert "sandbox" in str(body["notes"]) or "path_denied" in body["detail"]
    assert not (tmp_path / "escape.md").exists()


@pytest.mark.parametrize("stage", ["research", "draft", "publish", "maintenance"])
def test_every_stage_is_under_60_percent(tmp_path: Path, stage: str) -> None:
    """Gate 8 #8 在产品面上也要成立（不只是框架单测）。"""

    client = _client(tmp_path)

    body = client.get("/v1/tools", params={"stage": stage}, headers=_headers()).json()

    assert body["ratio"] < 0.6
