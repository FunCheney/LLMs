"""W10 D4 产品层：`publish_post` 走完整的 HITL 闭环。

这条链路是本周边界最重要的一条，因为它是**安全红线**唯一能被 HTTP 复现的地方：

    调用 → pending（不执行） → 人工批 → 再调用 → 真的执行

三个阶段缺一个都说明门没装上：

- 少了"不执行"：审批门是个装饰；
- 少了"能放行"：审批门是堵墙，产品没法用；
- 少了审计：事后说不清谁批的。
"""

from __future__ import annotations

from pathlib import Path

from agent_framework.config import Settings
from fastapi.testclient import TestClient
from growth_agent.app import create_app

# 50 字以上，满足 publish_post 的 body minLength。
BODY = "这是一篇足够长的正文，用来满足发布工具的最小长度限制。" * 2


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(
        tools="governed",
        tools_workspace_root=str(tmp_path / "ws"),
        tools_command_allow="echo",
    )
    return TestClient(create_app(settings=settings))


def _headers() -> dict[str, str]:
    return {"X-Principal-Id": "u1", "X-Principal-Groups": "publisher,editor"}


def _publish(client: TestClient, intent_id: str) -> dict:
    return client.post(
        "/v1/tools/call",
        json={
            "name": "publish_post",
            "arguments": {"title": "增长周报", "body": BODY},
            "stage": "publish",
            "intent_id": intent_id,
        },
        headers=_headers(),
    ).json()


def test_publish_is_pending_and_listed(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = _publish(client, "intent-1")

    assert body["status"] == "pending"
    assert body["content"] is None

    listed = client.get("/v1/approvals").json()
    assert len(listed["pending"]) == 1
    assert listed["pending"][0]["skill"] == "publish_post"
    # 摘要要能让人判断"是不是我批的那件事"
    assert "publish_post" in listed["pending"][0]["summary"]


def test_approved_publish_actually_runs(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _publish(client, "intent-1")

    pending = client.get("/v1/approvals").json()["pending"][0]
    decided = client.post(
        f"/v1/approvals/{pending['id']}",
        json={"approved": True, "by": "alice"},
    )

    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"

    # 换一个 intent_id：这是一次**新的**发布请求，不是上一次的重放
    second = _publish(client, "intent-2")
    assert second["status"] != "pending"
    assert second["content"]


def test_rejected_publish_never_runs(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _publish(client, "intent-1")

    pending = client.get("/v1/approvals").json()["pending"][0]
    client.post(f"/v1/approvals/{pending['id']}", json={"approved": False, "by": "alice"})

    second = _publish(client, "intent-2")
    assert second["status"] == "pending"
    assert second["content"] is None


def test_decide_twice_is_conflict(tmp_path: Path) -> None:
    """已终态不能再改——**改历史等于没有审计**。"""

    client = _client(tmp_path)
    _publish(client, "intent-1")
    pending = client.get("/v1/approvals").json()["pending"][0]

    first = client.post(f"/v1/approvals/{pending['id']}", json={"approved": True, "by": "a"})
    assert first.status_code == 200

    again = client.post(f"/v1/approvals/{pending['id']}", json={"approved": False, "by": "b"})
    assert again.status_code == 409
    assert again.json()["error"] == "approval_conflict"


def test_unknown_approval_is_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post("/v1/approvals/nope", json={"approved": True, "by": "a"})

    assert response.status_code == 404


def test_audit_shows_who_decided(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _publish(client, "intent-1")
    pending = client.get("/v1/approvals").json()["pending"][0]
    client.post(f"/v1/approvals/{pending['id']}", json={"approved": True, "by": "alice"})

    rows = client.get("/v1/approvals").json()
    approved = [row for row in rows["audit_tail"] if row["action"] == "approved"]
    assert approved and approved[-1]["by"] == "alice"


def test_behavior_snapshot_counts_dangerous_calls(tmp_path: Path) -> None:
    """行为监控接在调用路径上：发布属于 dangerous，必须被计数。"""

    client = _client(tmp_path)
    _publish(client, "intent-1")

    snapshot = client.get("/v1/approvals").json()["behavior"]
    assert snapshot["dangerous_call"]["count"] == 1
    assert snapshot["approval_pending"]["count"] == 1


def test_read_only_tool_does_not_create_approval(tmp_path: Path) -> None:
    """给每次调用都开一条 pending 记录，等于把审计淹掉。"""

    client = _client(tmp_path)
    body = client.post(
        "/v1/tools/call",
        json={"name": "web_search", "arguments": {"query": "MCP 协议", "limit": 2},
              "stage": "research"},
        headers={"X-Principal-Id": "u1", "X-Principal-Groups": ""},
    ).json()

    assert body["status"] != "pending"
    assert client.get("/v1/approvals").json()["pending"] == []
