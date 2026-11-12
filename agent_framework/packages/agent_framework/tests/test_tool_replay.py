"""W8 D2：录制回放。

四件事必须成立，每条都对应一次真实翻车：
1. 未命中明确报错、不联网
2. 参数顺序不同算同一条夹具
3. 落盘前脱敏
4. 回放标记与成本回流（`_meta.replayed` / `cost=0`）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent_framework.skills.replay import (
    RecordedToolTransport,
    ReplayMissError,
    canonical_args,
    fixture_key,
)
from agent_framework.skills.servers import search_client
from agent_framework.skills.servers.search_server import SearchServer


async def test_record_then_replay_returns_same_payload(tmp_path: Path) -> None:
    upstream = SearchServer().transport()
    recorder = RecordedToolTransport(fixtures_root=tmp_path, record=True, upstream=upstream)

    await recorder.list_tools()
    await recorder.call_tool("search_web", {"query": "增长"})

    assert recorder.stats.records == 1
    files = list(tmp_path.glob("search_web/*.json"))
    assert len(files) == 1

    player = RecordedToolTransport(fixtures_root=tmp_path)
    replayed = await player.call_tool("search_web", {"query": "增长"})

    assert player.stats.hits == 1
    assert replayed["structuredContent"]["count"] == 2
    # 回放标记与成本必须回流：报告要能分开统计。
    assert replayed["_meta"]["replayed"] is True
    assert replayed["_meta"]["cost"] == 0.0


async def test_miss_raises_instead_of_calling_network(tmp_path: Path) -> None:
    player = RecordedToolTransport(fixtures_root=tmp_path)  # 没有 upstream，也不该有

    with pytest.raises(ReplayMissError) as excinfo:
        await player.call_tool("search_web", {"query": "从未录过"})

    assert "不会联网" in str(excinfo.value)
    assert player.stats.misses == 1


async def test_missing_tools_fixture_is_loud(tmp_path: Path) -> None:
    player = RecordedToolTransport(fixtures_root=tmp_path)

    with pytest.raises(ReplayMissError):
        await player.list_tools()


async def test_argument_order_does_not_change_fixture_key() -> None:
    """JSON 的键顺序不该决定 CI 是否通过。"""

    assert fixture_key("t", {"a": 1, "b": 2}) == fixture_key("t", {"b": 2, "a": 1})
    assert fixture_key("t", {"a": 1}) != fixture_key("t", {"a": 2})
    # tool 名参与哈希：不同工具收到同样参数不能撞到同一文件。
    assert fixture_key("x", {"a": 1}) != fixture_key("y", {"a": 1})


async def test_replay_hits_regardless_of_key_order(tmp_path: Path) -> None:
    upstream = SearchServer().transport()
    recorder = RecordedToolTransport(fixtures_root=tmp_path, record=True, upstream=upstream)
    await recorder.call_tool("search_web", {"query": "增长", "max_results": 2})

    player = RecordedToolTransport(fixtures_root=tmp_path)
    result = await player.call_tool("search_web", {"max_results": 2, "query": "增长"})

    assert player.stats.hits == 1
    assert result["structuredContent"]["query"] == "增长"


async def test_recorded_fixture_is_sanitized(tmp_path: Path) -> None:
    """夹具会进 git，所以落盘前必须打码。"""

    from agent_framework.skills.mcp.inmemory import InMemoryMCPTransport

    async def leaky(params: dict) -> dict:
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {
                "api_key": "sk-live-abcdefghijklmn",
                "url": "https://api.example.com/v1?token=supersecret&q=x",
                "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6",
                "count": 3,
            },
        }

    transport = InMemoryMCPTransport([{"name": "leaky"}], server_name="s",
                                     handlers={"leaky": leaky})
    recorder = RecordedToolTransport(fixtures_root=tmp_path, record=True, upstream=transport)
    await recorder.call_tool("leaky", {})

    written = next((tmp_path / "leaky").glob("*.json")).read_text("utf-8")
    body = json.loads(written)
    stored = body["result"]["structuredContent"]

    assert stored["api_key"] == "***"
    assert stored["authorization"] == "***"
    assert "supersecret" not in written
    assert "sk-live" not in written
    # 非敏感字段原样保留：全打码会让夹具无法 review。
    assert stored["count"] == 3
    assert "q=x" in stored["url"]


async def test_record_requires_upstream() -> None:
    with pytest.raises(ValueError):
        RecordedToolTransport(fixtures_root="x", record=True)


async def test_oversized_response_is_not_written(tmp_path: Path) -> None:
    from agent_framework.skills.mcp.inmemory import InMemoryMCPTransport

    async def huge(params: dict) -> dict:
        return {"content": [{"type": "text", "text": "x" * 300_000}]}

    transport = InMemoryMCPTransport([{"name": "huge"}], server_name="s",
                                     handlers={"huge": huge})
    recorder = RecordedToolTransport(fixtures_root=tmp_path, record=True, upstream=transport)

    raw = await recorder.call_tool("huge", {})

    # 真调用照样返回，只是不落盘（不报错：抓到一个大页面不该让整任务失败）。
    assert raw["content"][0]["text"]
    assert list(tmp_path.glob("huge/*.json")) == []


async def test_no_tmp_file_left_behind(tmp_path: Path) -> None:
    """原子落盘：中断不能留下半个 JSON。"""

    upstream = SearchServer().transport()
    recorder = RecordedToolTransport(fixtures_root=tmp_path, record=True, upstream=upstream)
    await recorder.list_tools()
    await recorder.call_tool("search_web", {"query": "AI"})

    assert list(tmp_path.rglob("*.tmp")) == []


async def test_future_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "search_web" / f"{fixture_key('search_web', {'query': 'x'})}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"tool": "search_web", "args": {}, "result": {},
                    "meta": {"schema": 99, "synthetic": False}}),
        "utf-8",
    )

    player = RecordedToolTransport(fixtures_root=tmp_path)

    with pytest.raises(ReplayMissError) as excinfo:
        await player.call_tool("search_web", {"query": "x"})
    assert "版本" in str(excinfo.value)


async def test_canonical_args_is_stable_and_readable() -> None:
    text = canonical_args({"b": "中文", "a": 1})

    assert text == '{"a":1,"b":"中文"}'
    # 不转义中文：夹具要能 review。
    assert "\\u" not in text


async def test_search_client_over_replay_returns_structured_payload(tmp_path: Path) -> None:
    """真链路：server → 录制 → 回放 → MCPClient → ToolOutcome。"""

    recorder = RecordedToolTransport(
        fixtures_root=tmp_path, record=True, upstream=SearchServer().transport()
    )
    recording_client = SearchServer().client(transport=recorder)
    await recording_client.list_tools()
    await recording_client.call_tool("search__search_web", {"query": "增长"})

    player = RecordedToolTransport(fixtures_root=tmp_path)
    client = SearchServer().client(transport=player)
    await client.list_tools()
    outcome = await client.call_tool("search__search_web", {"query": "增长"})

    assert outcome.ok
    assert outcome.replayed is True
    assert outcome.cost == 0.0
    assert outcome.payload is not None
    assert outcome.payload["count"] == 2


async def test_stub_search_client_works_offline() -> None:
    """不联网也能拿到结果（离线档的基本盘）。"""

    client = search_client()
    await client.list_tools()
    outcome = await client.call_tool("search__search_web", {"query": "AI 营销"})

    assert outcome.ok
    assert outcome.payload is not None
    assert outcome.payload["count"] >= 1
