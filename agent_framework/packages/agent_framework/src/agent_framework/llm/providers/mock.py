"""夹具回放器（离线评测与单测的数据源）。

**它不是假 LLM，是 replay 引擎。** 只回放 wire 原文，然后复用真实 adapter 的
parse_response / Assembler —— 如果它自己 `return LLMResponse(...)`，就等于
"自己跟自己对答案"，真机解析 bug 一个都测不出来（假绿的头号来源）。

因此它刻意**不做错误映射**：错误路径由 conformance 用 httpx.MockTransport 走真实薄壳。
W3 的评测量产依赖这里（不烧钱、不受外部波动影响）。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..base import Capability, LLMClient
from ..types import (
    LLMResponse,
    Message,
    StreamEvent,
    StreamEventType,
    ToolChoice,
    ToolSpec,
)
from ._common import SSELineKind, parse_sse_line

__all__ = ["MockLLMClient", "ReplayAdapter"]


@runtime_checkable
class ReplayAssembler(Protocol):
    """真 adapter 的组装器需要暴露的最小接口。"""

    def feed_line(self, raw: str | bytes) -> list[StreamEvent]: ...

    def finalize(self) -> list[StreamEvent]: ...


@runtime_checkable
class ReplayAdapter(Protocol):
    """真 adapter 需要暴露的最小接口（parse 与 assembler 都是纯逻辑）。"""

    @property
    def name(self) -> str: ...

    def parse_response(
        self,
        body: dict[str, Any],
        *,
        prompt_text: str | None = None,
    ) -> LLMResponse: ...

    def make_assembler(self, *, prompt_text: str | None = None) -> ReplayAssembler: ...


class MockLLMClient(LLMClient):
    """按 `<fixtures_root>/<provider>/<case>/` 回放 wire 数据。"""

    def __init__(
        self,
        adapter: ReplayAdapter,
        *,
        fixtures_root: Path,
        case: str = "text_basic",
    ) -> None:
        self._adapter = adapter
        self._case_dir = fixtures_root / adapter.name / case

        if not self._case_dir.is_dir():
            raise FileNotFoundError(f"fixture case not found: {self._case_dir}")

    @property
    def name(self) -> str:
        return self._adapter.name

    @property
    def model(self) -> str:
        return ""

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.STREAM, Capability.TOOLS})

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: Any,
    ) -> LLMResponse:
        body = json.loads((self._case_dir / "response.json").read_text("utf-8"))

        return self._adapter.parse_response(body)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **opts: Any,
    ) -> AsyncIterator[StreamEvent]:
        stream_file = self._case_dir / "stream.txt"

        if not stream_file.is_file():
            raise FileNotFoundError(f"fixture has no stream.txt: {stream_file}")

        assembler = self._adapter.make_assembler()

        for raw in stream_file.read_text("utf-8").splitlines():
            for event in assembler.feed_line(raw):
                # 走一遍真 adapter 的组装器后再吐出，mock 不解释事件含义。
                yield event

        for event in assembler.finalize():
            yield event

    def has_stream(self) -> bool:
        """该用例是否带流式夹具（给测试做参数化筛选）。"""

        stream_file = self._case_dir / "stream.txt"

        if not stream_file.is_file():
            return False

        for raw in stream_file.read_text("utf-8").splitlines():
            if parse_sse_line(raw).kind is SSELineKind.DATA:
                return True

        return False


def event_types(events: list[StreamEvent]) -> list[StreamEventType]:
    """调试辅助：把事件序列压缩成类型列表。"""

    return [event.type for event in events]
