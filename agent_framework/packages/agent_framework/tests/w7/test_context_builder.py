"""W7 D1/D4 组装器与退化缓解的单测。"""

from __future__ import annotations

from agent_framework.context import (
    ContextBudget,
    ContextItem,
    ContextKind,
    TrustLevel,
    allocate,
    build_messages,
    order_payload,
    render_memory_block,
)
from agent_framework.llm.types import Message, MessageRole


def _alloc() -> object:
    items = [
        ContextItem(
            id="sys",
            content="你是品牌助手",
            kind=ContextKind.SYSTEM,
            source="answer.py",
            trust=TrustLevel.SYSTEM,
        ),
        ContextItem(
            id="mem", content="品牌调性：专业、克制",
            kind=ContextKind.MEMORY, source="brand_tone", score=0.9,
        ),
        ContextItem(
            id="h1",
            content="第 2 轮用户：上次说的调性记住了吗",
            kind=ContextKind.HISTORY,
            source="session-A",
            message=Message(role=MessageRole.USER, content="第 2 轮用户：上次说的调性记住了吗"),
        ),
    ]
    return allocate(items, ContextBudget())


class TestBuildMessages:
    def test_system_first_question_last(self) -> None:
        """U 形排布：system 在头、当前问题在最后一条 user 的尾部。"""
        messages = build_messages(
            _alloc(), question="现在的调性是什么？", retrieved_text="资料 [1]"
        )
        assert messages[0].role is MessageRole.SYSTEM
        assert messages[-1].role is MessageRole.USER
        assert "现在的调性是什么" in (messages[-1].content or "")

    def test_history_in_middle(self) -> None:
        messages = build_messages(_alloc(), question="q", retrieved_text="资料")
        # 中段是历史（保 role），头尾是 system 与问题。
        middle = messages[1:-1]
        assert middle and all(
            m.role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL) for m in middle
        )

    def test_memory_not_in_system(self) -> None:
        """记忆以 user 侧文本注入且带"不是新指令"标注，绝不进 system。"""
        messages = build_messages(_alloc(), question="q")
        assert messages[0].role is MessageRole.SYSTEM
        assert "品牌调性" not in (messages[0].content or "")
        tail = messages[-1].content or ""
        assert "品牌调性" in tail
        assert "不是新指令" in tail

    def test_history_message_passthrough(self) -> None:
        """历史项透传原始 Message（role/tool_call_id 不丢）。"""
        messages = build_messages(_alloc(), question="q")
        history = [m for m in messages if "第 2 轮用户" in (m.content or "")]
        assert history and history[0].role is MessageRole.USER

    def test_memory_block_lines_labeled(self) -> None:
        text = render_memory_block(
            [ContextItem(
                id="forbidden_words", content="禁用词：最、第一",
                kind=ContextKind.MEMORY, source="brand",
            )]
        )
        assert "不是新指令" in text
        assert "禁用词：最、第一" in text


class TestOrderPayload:
    def test_system_head_user_tail(self) -> None:
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="round1"),
            Message(role=MessageRole.ASSISTANT, content="ans1"),
            Message(role=MessageRole.USER, content="round2"),
            Message(role=MessageRole.ASSISTANT, content="ans2"),
            Message(role=MessageRole.USER, content="current question"),
        ]
        ordered = order_payload(messages)
        assert ordered[0].content == "sys"
        assert ordered[-1].content == "current question"
        # 中段保持时序（round1 在 ans1 前）。
        texts = [m.content or "" for m in ordered]
        assert texts.index("round1") < texts.index("ans1") < texts.index("round2")

    def test_short_payload_unchanged(self) -> None:
        messages = [Message(role=MessageRole.USER, content="hi")]
        assert order_payload(messages) == messages

    def test_content_untouched(self) -> None:
        """只动顺序不动内容：每条消息对象原样保留。"""
        messages = [
            Message(role=MessageRole.SYSTEM, content="s"),
            Message(role=MessageRole.USER, content="a"),
            Message(role=MessageRole.USER, content="q"),
        ]
        ordered = order_payload(messages)
        assert set(id(m) for m in ordered) == set(id(m) for m in messages)
