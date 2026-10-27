"""共享夹具加载器与 httpx.MockTransport 工厂。

为什么用 MockTransport 而不是 monkeypatch：
monkeypatch 掉 `_post` 只测到 parse，header / URL / body 序列化 / SSE 逐行解析全没测，
也无法断言"发了几次请求"（A8）。MockTransport 按夹具回放原始 bytes，真 HTTP 语义全走一遍。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from agent_framework.llm.base import LLMClient
from agent_framework.llm.providers import AnthropicClient, OpenAICompatClient
from agent_framework.llm.types import Message, MessageRole, ToolSpec

FIXTURES_ROOT = Path(__file__).parent / "fixtures"

PROVIDERS: tuple[str, ...] = ("openai", "deepseek", "qwen", "anthropic")

PROVIDER_MODEL: dict[str, str] = {
    "openai": "gpt-test",
    "deepseek": "deepseek-flash",
    "qwen": "qwen3.7-plus",
    "anthropic": "claude-test",
}

PROVIDER_BASE_URL: dict[str, str] = {
    "openai": "https://api.openai.test/v1",
    "deepseek": "https://api.deepseek.test/v1",
    "qwen": "https://dashscope.test/compatible-mode/v1",
    "anthropic": "https://api.anthropic.test",
}

# 必须与 fixtures/_generate_fixtures.py 的常量一致：夹具是"内部输入"的另一端。
SYSTEM_TEXT = "你是研究助手，只回答可验证的事实。"
USER_TEXT = "帮我查一下 MCP 协议的最新进展"
MAX_TOKENS = 512

TOOL_DESCRIPTION = "当需要最新外部信息时使用；不要用于纯计算或改写。"
TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "检索关键词"}},
    "required": ["query"],
    "additionalProperties": False,
}


def canonical_messages() -> list[Message]:
    """conformance 的统一内部输入。"""

    return [
        Message(role=MessageRole.SYSTEM, content=SYSTEM_TEXT),
        Message(role=MessageRole.USER, content=USER_TEXT),
    ]


def canonical_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="search_web",
            description=TOOL_DESCRIPTION,
            parameters=TOOL_PARAMETERS,
        )
    ]


@dataclass(frozen=True)
class FixtureCase:
    """一个用例目录的全部内容。"""

    provider: str
    name: str
    path: Path
    meta: dict[str, Any]
    request: dict[str, Any]
    response: dict[str, Any]
    stream_text: str | None

    @property
    def status_code(self) -> int:
        return int(self.meta.get("status_code", 200))

    @property
    def expect(self) -> dict[str, Any]:
        value = self.meta.get("expect")

        return value if isinstance(value, dict) else {}

    @property
    def stream_lines(self) -> list[str]:
        if self.stream_text is None:
            return []

        return self.stream_text.splitlines()


def load_case(provider: str, name: str) -> FixtureCase:
    case_dir = FIXTURES_ROOT / provider / name

    if not case_dir.is_dir():
        raise FileNotFoundError(f"missing fixture: {case_dir}")

    meta = json.loads((case_dir / "meta.json").read_text("utf-8"))
    response = json.loads((case_dir / "response.json").read_text("utf-8"))
    request_path = case_dir / "request.json"
    request = json.loads(request_path.read_text("utf-8")) if request_path.is_file() else {}

    stream_path = case_dir / "stream.txt"
    stream_text = stream_path.read_text("utf-8") if stream_path.is_file() else None

    return FixtureCase(
        provider=provider,
        name=name,
        path=case_dir,
        meta=meta,
        request=request,
        response=response,
        stream_text=stream_text,
    )


def make_client(
    provider: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Any:
    """按 provider 构造真 adapter（测试里替换 transport，不发真请求）。"""

    model = PROVIDER_MODEL[provider]
    base_url = PROVIDER_BASE_URL[provider]

    if provider == "anthropic":
        return AnthropicClient(
            name=provider,
            model=model,
            base_url=base_url,
            api_key="sk-ant-test",
            transport=transport,
        )

    return OpenAICompatClient(
        name=provider,
        model=model,
        base_url=base_url,
        api_key="sk-test",
        transport=transport,
    )


@dataclass
class Recorder:
    """按夹具回放，并记录调用次数（A8 靠它）。"""

    case: FixtureCase
    calls: list[httpx.Request] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)

        if self.case.status_code >= 400:
            return httpx.Response(
                self.case.status_code,
                json=self.case.response,
                headers={"content-type": "application/json"},
            )

        # 按请求体的 stream 标记决定回放哪一种形态，
        # 这样同一个 case 既能测非流式也能测流式（A3 需要）。
        try:
            requested_stream = json.loads(request.content or b"{}").get("stream") is True
        except ValueError:
            requested_stream = False

        if requested_stream:
            assert self.case.stream_text is not None, "fixture has no stream.txt"

            return httpx.Response(
                200,
                content=self.case.stream_text.encode("utf-8"),
                headers={"content-type": "text/event-stream"},
            )

        return httpx.Response(
            200,
            json=self.case.response,
            headers={"content-type": "application/json"},
        )

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    @property
    def count(self) -> int:
        return len(self.calls)


def recorded_client(case: FixtureCase) -> tuple[LLMClient, Recorder]:
    """构造"真 adapter + 夹具回放"的客户端。"""

    recorder = Recorder(case=case)

    return make_client(case.provider, transport=recorder.transport()), recorder
