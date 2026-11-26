"""W7 D2 工作记忆的单测：摘要触发 / 幂等 / 20 轮不超限 / 截断可恢复。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_framework.llm.types import Message, MessageRole
from agent_framework.memory import (
    CompressAction,
    SummarizerUsage,
    WorkingMemory,
    make_gateway_summarizer,
)
from agent_framework.session.memory import InMemorySessionStore


def long_turn(i: int, chars: int = 400) -> list[Message]:
    return [
        Message(role=MessageRole.USER, content=f"第{i}轮问题：" + "问" * chars),
        Message(role=MessageRole.ASSISTANT, content=f"第{i}轮回答：" + "答" * chars),
    ]


async def append_turns(
    store: InMemorySessionStore, session: str, n: int, chars: int = 400, *, prefix: str = "r"
) -> None:
    for i in range(1, n + 1):
        await store.append(session, long_turn(i, chars), request_id=f"{prefix}{i}")


class FakeSummarizer:
    """确定性摘要：把输入条数写进摘要（断言能看到它被谁调用）。"""

    def __init__(self) -> None:
        self.calls: list[int] = []  # 每次收到的消息条数

    async def __call__(self, old: str, messages: list[Message]) -> str:
        self.calls.append(len(messages))
        head = f"[摘要] 覆盖 {len(messages)} 条；" + (f"并入旧摘要({len(old)}字)；" if old else "")
        return head + "要点：调性=专业"


class TestCompress:
    async def test_below_threshold_skips(self) -> None:
        store = InMemorySessionStore()
        await append_turns(store, "s1", 2, chars=50)
        memory = WorkingMemory(store, summarizer=FakeSummarizer(), token_threshold=6000)
        result = await memory.compress("s1")
        assert result.action is CompressAction.SKIPPED

    async def test_over_token_threshold_triggers(self) -> None:
        store = InMemorySessionStore()
        await append_turns(store, "s2", 10, chars=400)  # 20 条 × 250 tok ≈ 5000
        memory = WorkingMemory(
            store, summarizer=FakeSummarizer(),
            token_threshold=2000, keep_recent_messages=4,
        )
        result = await memory.compress("s2")
        assert result.action is CompressAction.CREATED
        assert result.compressed_messages == 20 - 4
        assert result.summary_tokens > 0

    async def test_idempotent_double_compress(self) -> None:
        """幂等：同一会话连续 compress 两次，第二次必然 skipped。"""
        store = InMemorySessionStore()
        await append_turns(store, "s3", 10, chars=400)
        fake = FakeSummarizer()
        memory = WorkingMemory(store, summarizer=fake, token_threshold=2000, keep_recent_messages=4)
        first = await memory.compress("s3")
        second = await memory.compress("s3")
        assert first.action is CompressAction.CREATED
        assert second.action is CompressAction.SKIPPED
        assert len(fake.calls) == 1  # 摘要器只被调用一次

    async def test_rolling_summary_merges_old(self) -> None:
        """滚动：新增可压缩消息后再次触发，摘要器收到旧摘要。"""
        store = InMemorySessionStore()
        await append_turns(store, "s4", 8, chars=400)
        fake = FakeSummarizer()
        memory = WorkingMemory(store, summarizer=fake, token_threshold=2000, keep_recent_messages=2)
        await memory.compress("s4")
        # 新 4 轮用新的 request_id 前缀（append 幂等按 request_id 判重）。
        await append_turns(store, "s4", 4, chars=400, prefix="r2-")
        second = await memory.compress("s4")
        assert second.action is CompressAction.UPDATED
        assert len(fake.calls) == 2

    async def test_no_summarizer_degrades(self) -> None:
        """没注入 summarizer：只推进覆盖进度，不炸。"""
        store = InMemorySessionStore()
        await append_turns(store, "s5", 10, chars=400)
        memory = WorkingMemory(store, token_threshold=2000, keep_recent_messages=4)
        result = await memory.compress("s5")
        assert result.action is CompressAction.UPDATED
        assert "未注入" in result.note


class TestView:
    async def test_view_has_labeled_summary_and_recent(self) -> None:
        store = InMemorySessionStore()
        await append_turns(store, "s6", 10, chars=400)
        memory = WorkingMemory(
            store, summarizer=FakeSummarizer(),
            token_threshold=2000, keep_recent_messages=4,
        )
        await memory.compress("s6")
        view = await memory.view("s6")
        assert len(view) == 5  # 1 摘要 + 4 原文
        assert "会话摘要" in (view[0].content or "")
        assert "非用户原话" in (view[0].content or "")
        assert view[1].role is MessageRole.USER

    async def test_view_short_session_unchanged(self) -> None:
        store = InMemorySessionStore()
        await append_turns(store, "s7", 1, chars=10)
        memory = WorkingMemory(store, summarizer=FakeSummarizer())
        view = await memory.view("s7")
        assert len(view) == 2  # 未压缩：原样


class TestTwentyRounds:
    async def test_20_rounds_stay_within_budget(self) -> None:
        """20+ 轮后喂给模型的历史视图始终在预算内（压缩的意义）。

        上界的口径：阈值（2000）+ 一轮未触发的增量（keep_recent=4 条 ≈
        1020 tok）+ 摘要（≈50）。压缩是"超阈值才触发"，不是逐条跟随，
        所以 view 允许短暂超过阈值——但必须远低于不压缩的 12500。
        """
        store = InMemorySessionStore()
        memory = WorkingMemory(
            store,
            summarizer=FakeSummarizer(),
            token_threshold=2000,
            keep_recent_messages=4,
        )
        # 25 轮，每轮约 500 token：不压缩会到 12500 token。
        for i in range(1, 26):
            await store.append("s8", long_turn(i, chars=400), request_id=f"r{i}")
            await memory.compress("s8")
        view = await memory.view("s8")
        tokens = sum(len(m.content or "") / 1.6 for m in view)
        assert tokens < 3200  # 阈值 + keep_recent 余量；远低于 12500
        # 第 1 轮的内容进了摘要（可回忆）。
        assert "要点" in (view[0].content or "")
        # 每一轮都调了 compress，全程序号都在预算内（无爆窗路径）。
        uncompressed = sum(len(m.content or "") / 1.6 for m in await store.get_history("s8"))
        assert tokens < uncompressed / 3


class TestToolResultTruncation:
    def test_short_result_untouched(self) -> None:
        memory = WorkingMemory(InMemorySessionStore())
        assert memory.truncate_tool_result("短结果", handle="h1") == "短结果"

    def test_long_result_truncated_with_handle(self) -> None:
        memory = WorkingMemory(InMemorySessionStore())
        text = "数" * 5000
        truncated = memory.truncate_tool_result(text, handle="tool-handle-1", max_chars=100)
        assert len(truncated) < 300
        assert "tool-handle-1" in truncated
        assert "重新获取" in truncated

    def test_refetch_recovers_full_text(self) -> None:
        """信息没丢：句柄能取回完整结果（可靠性自检）。"""
        memory = WorkingMemory(InMemorySessionStore())
        text = "结果" * 2000
        memory.truncate_tool_result(text, handle="h", max_chars=100)
        assert memory.refetch("h") == text
        assert memory.refetch("nonexistent") is None


class TestGuardrails:
    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            WorkingMemory(InMemorySessionStore(), token_threshold=0)
        with pytest.raises(ValueError):
            WorkingMemory(InMemorySessionStore(), keep_recent_messages=0)


class _Route:
    primary = "cheap-alias"


class _FakeGateway:
    """只够摘要器用：`catalog.routes["cheap"].primary` + 返回带 usage 的响应。"""

    class _Catalog:
        routes = {"cheap": _Route()}

    def __init__(self, input_tokens: int = 120, output_tokens: int = 30) -> None:
        self.catalog = self._Catalog()
        self.calls = 0
        self._in = input_tokens
        self._out = output_tokens

    async def chat(self, call):  # noqa: ANN001, ANN201 - 测试桩，签名无关
        self.calls += 1
        usage = SimpleNamespace(input_tokens=self._in, output_tokens=self._out)
        response = SimpleNamespace(text="[摘要] 要点", usage=usage)
        return SimpleNamespace(response=response, alias="cheap-alias")


class TestSummarizerUsage:
    """压缩不是免费的：摘要用量必须能被记成数（Gate 7 #7 的分母）。"""

    async def test_usage_accumulates_and_reports_per_call_delta(self) -> None:
        store = InMemorySessionStore()
        await append_turns(store, "s9", 10, chars=400)
        usage = SummarizerUsage()
        gateway = _FakeGateway()
        memory = WorkingMemory(
            store,
            summarizer=make_gateway_summarizer(gateway, usage=usage),
            token_threshold=2000,
            keep_recent_messages=4,
            summarizer_usage=usage,
        )
        result = await memory.compress("s9")
        assert result.action is CompressAction.CREATED
        # 累计器与 CompressResult 都要有数——只有一个有数，成本报告就会缺一半。
        assert usage.calls == 1
        assert (usage.input_tokens, usage.output_tokens) == (120, 30)
        assert (result.input_tokens, result.output_tokens) == (120, 30)

    async def test_skipped_does_not_touch_usage(self) -> None:
        store = InMemorySessionStore()
        await append_turns(store, "s10", 2, chars=40)
        usage = SummarizerUsage()
        memory = WorkingMemory(
            store,
            summarizer=make_gateway_summarizer(_FakeGateway(), usage=usage),
            token_threshold=6000,
            summarizer_usage=usage,
        )
        result = await memory.compress("s10")
        assert result.action is CompressAction.SKIPPED
        assert usage.calls == 0
        assert result.input_tokens == 0
