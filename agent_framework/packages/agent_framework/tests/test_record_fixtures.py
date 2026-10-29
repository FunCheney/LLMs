"""录制器的离线自测（不联网、不需要任何 key）。

为什么这些测试值得写：录制脚本是整个 Day5 里唯一"只在真机跑一次"的东西，
如果它自己写错了（漏了 stream、覆盖了 request.json、把 key 落盘），
错误会在**夹具层**静默固化——之后所有 conformance 都在测试一份假数据。

做法：把 `RecordingTransport` 的 inner 换成 `httpx.MockTransport`，
让"真机"变成一个可控的假上游。链路其余部分（真 adapter、真 registry）全不动。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from agent_framework.config import Settings
from pydantic import SecretStr

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "evals" / "scripts" / "record_fixtures.py"


def _load_script() -> ModuleType:
    """按路径加载 evals/scripts 下的脚本（它不在任何包里，不能直接 import）。"""

    spec = importlib.util.spec_from_file_location("record_fixtures", SCRIPT_PATH)

    assert spec and spec.loader

    module = importlib.util.module_from_spec(spec)
    # 必须先登记到 sys.modules：脚本里用了 `from __future__ import annotations`，
    # dataclass 解析字符串注解时会按 `cls.__module__` 反查模块，不登记就是 None。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


record = _load_script()


def _settings() -> Settings:
    """注入假 key：录制器要求"能取到密钥"，但不该真的去读 .env.local。"""

    return Settings(openai_api_key=SecretStr("sk-test-key-0000000000"))


def _message(*, tool: bool) -> dict[str, Any]:
    if not tool:
        return {"role": "assistant", "content": "真机回答"}

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_real_1",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": json.dumps({"query": "MCP 最新进展"}),
                },
            }
        ],
    }


def _body(*, tool: bool) -> dict[str, Any]:
    """刻意比手写基线多两个字段。

    `service_tier` / `system_fingerprint` 是真实 OpenAI 响应长期存在、手写夹具里
    却没有的东西——A13 的方向是"允许多、不允许少"，所以真实录制必须能多出字段。
    """

    return {
        "id": "chatcmpl-real",
        "object": "chat.completion",
        "created": 1793000000,
        "model": "gpt-4o-mini-real",
        "service_tier": "default",
        "system_fingerprint": "fp_real",
        "choices": [
            {
                "index": 0,
                "message": _message(tool=tool),
                "finish_reason": "tool_calls" if tool else "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 61,
            "completion_tokens": 24,
            "total_tokens": 85,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


STREAM_TEXT = (
    ": keep-alive\n"
    "\n"
    'data: {"choices":[{"index":0,"delta":{"content":"真机"}}]}\n'
    "\n"
    'data: {"choices":[{"index":0,"delta":{"content":"回答"},"finish_reason":"stop"}]}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)


def _transport(*, tool: bool) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")

        if body.get("stream") is True:
            return httpx.Response(
                200,
                content=STREAM_TEXT.encode("utf-8"),
                headers={"content-type": "text/event-stream"},
            )

        return httpx.Response(
            200,
            json=_body(tool=tool),
            headers={"content-type": "application/json"},
        )

    return httpx.MockTransport(handler)


async def _record(
    tmp_path: Path,
    *,
    cases: tuple[str, ...] = ("text_basic", "tool_call_basic"),
    tool: bool = True,
    **kwargs: Any,
) -> list[str]:
    return await record.record_provider(
        "openai",
        cases,
        transport=_transport(tool=tool),
        settings=_settings(),
        fixtures_root=tmp_path,
        out=lambda *_: None,
        **kwargs,
    )


async def test_record_writes_response_and_meta(tmp_path: Path) -> None:
    await _record(tmp_path, cases=("text_basic",), tool=False, apply=True)

    case_dir = tmp_path / "openai" / "text_basic"
    response = json.loads((case_dir / "response.json").read_text("utf-8"))
    meta = json.loads((case_dir / "meta.json").read_text("utf-8"))

    assert response["id"] == "chatcmpl-real"
    assert meta["synthetic"] is False
    assert meta["recorded_at"]
    assert meta["status_code"] == 200


async def test_record_captures_stream_verbatim(tmp_path: Path) -> None:
    """流式夹具要的是逐字节原文：`: keep-alive` / 空行 / [DONE] 一个都不能少。"""

    await _record(tmp_path, cases=("tool_call_basic",), apply=True)

    stream = (tmp_path / "openai" / "tool_call_basic" / "stream.txt").read_text("utf-8")

    assert stream == STREAM_TEXT


async def test_expect_comes_from_real_parse_result(tmp_path: Path) -> None:
    """expect 必须是"回放要复现的那次真机解析结果"，不能沿用旧的手写期待。"""

    await _record(tmp_path, cases=("tool_call_basic",), apply=True)

    meta = json.loads(
        (tmp_path / "openai" / "tool_call_basic" / "meta.json").read_text("utf-8")
    )

    assert meta["expect"]["finish_reason"] == "tool_calls"
    assert meta["expect"]["tool_calls"] == [
        {"name": "search_web", "arguments": {"query": "MCP 最新进展"}}
    ]
    assert meta["expect"]["usage_estimated"] is False


async def test_record_never_overwrites_request_json(tmp_path: Path) -> None:
    """request.json 是 A1 的断言锚，录制只能写 response/stream/meta。"""

    case_dir = tmp_path / "openai" / "text_basic"
    case_dir.mkdir(parents=True)
    anchor = {"model": "人工核对过的 body"}
    (case_dir / "request.json").write_text(json.dumps(anchor), "utf-8")

    with pytest.raises(SystemExit) as excinfo:
        await _record(tmp_path, cases=("text_basic",), apply=True)

    assert "request.json 不一致" in str(excinfo.value)
    assert json.loads((case_dir / "request.json").read_text("utf-8")) == anchor


async def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    await _record(tmp_path, cases=("text_basic",), tool=False, apply=False)

    assert not (tmp_path / "openai" / "text_basic" / "response.json").exists()


def test_sanitize_redacts_secrets() -> None:
    raw = b'{"error":{"message":"Incorrect API key provided: sk-live-abcdef123456"}}'

    cleaned = record.sanitize(raw, ["", "sk", "sk-live-abcdef123456"]).decode()

    assert "sk-live-abcdef123456" not in cleaned
    assert "REDACTED" in cleaned
    # 短串不能被当秘密：否则空串替换会把整个 body 打碎
    assert "Incorrect API key provided" in cleaned


async def test_bad_key_probe_records_real_auth_error(tmp_path: Path) -> None:
    """鉴权失败必须能录成夹具，否则"401 映射成 502 而不是 401"永远是口头结论。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided",
                    "code": "invalid_api_key",
                }
            },
            headers={"content-type": "application/json"},
        )

    await record.probe_bad_key(
        "openai",
        transport=httpx.MockTransport(handler),
        settings=_settings(),
        apply=True,
        fixtures_root=tmp_path,
        out=lambda *_: None,
    )

    meta = json.loads((tmp_path / "openai" / "error_auth" / "meta.json").read_text("utf-8"))

    assert meta["status_code"] == 401
    assert meta["expect_error"] is True
    assert meta["synthetic"] is False


class _ChunkStream(httpx.AsyncByteStream):
    """未消费的响应流：真实网络路径就是这个形态（body 还没读进内存）。"""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for chunk in self._chunks:
            yield chunk


async def test_tee_captures_unconsumed_stream_chunk_by_chunk() -> None:
    """tee 分支必须单独测：替身返回的"已读"响应走的是另一条路径。

    这条用例的价值在于证明"字节是被流式抄下来的，不是靠事后 aread 补的"——
    后者在真流式下会把整段 SSE 缓存进内存，长流（W2 压测）会直接 OOM。
    """

    chunks = [b"data: {\"a\":1}\n\n", b"data: [DONE]\n\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkStream(chunks),
        )

    recorder = record.RecordingTransport(httpx.MockTransport(handler))

    async with httpx.AsyncClient(base_url="https://api.test", transport=recorder) as client:
        response = await client.post("/chat", json={"stream": True})
        await response.aread()

    assert recorder.captures[0].body == b"".join(chunks)
    assert recorder.captures[0].is_event_stream is True
