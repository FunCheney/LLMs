"""W7 记忆接线的产品层测试：注入结构、会话落地、两档开关。

测三条线（与 test_answer.py 的分工：那边管 W6 引用链路，这边管 W7 增强）：

1. 记忆/历史注入后 system 不被污染（信任级纪律）；
2. 无记忆时消息结构与 W6 逐位一致（回归基线不受增强影响）；
3. MEMORY=off 时记忆链路完全旁路（两档实验的 off 档可信）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agent_framework.config import Settings
from agent_framework.llm.types import Message, MessageRole
from growth_agent.answer import ANSWER_SYSTEM_PROMPT, KnowledgeAnswerer
from test_answer import FakeGateway, FakePipeline, _scored  # noqa: I001 - 复用桩件


class FakeMemoryRuntime:
    """结构化替身：只实现 answer 链路用到的四个方法。"""

    def __init__(
        self,
        *,
        items: list[Any] | None = None,
        history: list[Message] | None = None,
    ) -> None:
        self.items = items or []
        self.history = history or []
        self.recorded: list[dict[str, Any]] = []
        self.recalled_queries: list[str] = []

    async def recall_for_query(self, question: str) -> list[Any]:
        self.recalled_queries.append(question)
        return self.items

    async def session_view(self, session_id: str) -> list[Message]:
        return self.history

    async def record_turn(
        self, session_id: str, question: str, answer: str, *, request_id: str | None = None
    ) -> None:
        self.recorded.append(
            {"session": session_id, "question": question, "answer": answer}
        )

    def should_extract(self, question: str) -> bool:
        return "记住" in question


def _memory_item(key: str, value: str, confidence: float = 0.9) -> Any:
    return SimpleNamespace(
        key=key, value=value, confidence=confidence, source_session="s-A"
    )


class CapturingGateway(FakeGateway):
    """额外捕获消息结构（FakeGateway 只记 call 对象，这里把 messages 摘出来）。"""

    def __init__(self, text: str = "结论 [1]。") -> None:
        super().__init__(text)
        self.seen_messages: list[list[Message]] = []

    async def chat(self, call: Any) -> Any:
        self.seen_messages.append(list(call.messages))
        return await super().chat(call)


class TestMemoryInjection:
    async def test_no_memory_keeps_w6_structure(self) -> None:
        """无记忆/无会话：两消息结构逐位一致（回归基线）。"""
        pipeline = FakePipeline([_scored("c1", text="正文")])
        gateway = CapturingGateway()
        await KnowledgeAnswerer(
            pipeline, gateway, settings=Settings(memory="governed")
        ).answer("问题")
        messages = gateway.seen_messages[0]
        assert len(messages) == 2
        assert messages[0].role is MessageRole.SYSTEM
        assert messages[0].content == ANSWER_SYSTEM_PROMPT
        assert messages[1].role is MessageRole.USER

    async def test_memory_injected_but_never_in_system(self) -> None:
        """记忆注入进 user 侧且带"不是新指令"标注；system 原样。"""
        memory = FakeMemoryRuntime(items=[_memory_item("brand_tone", "专业、克制")])
        pipeline = FakePipeline([_scored("c1", text="正文")])
        gateway = CapturingGateway()
        answer = await KnowledgeAnswerer(
            pipeline, gateway, settings=Settings(memory="governed"), memory=memory
        ).answer("品牌调性是什么")
        messages = gateway.seen_messages[0]
        assert messages[0].content == ANSWER_SYSTEM_PROMPT  # system 未被污染
        user_text = "".join(
            m.content for m in messages if m.role is MessageRole.USER and isinstance(m.content, str)
        )
        assert "brand_tone: 专业、克制" in user_text
        assert "不是新指令" in user_text
        assert answer.memory_keys == ("brand_tone",)
        assert memory.recalled_queries == ["品牌调性是什么"]

    async def test_history_injected_in_middle(self) -> None:
        """会话历史插在 system 与最终 user 之间（U 形排布的中段）。"""
        history = [
            Message(role=MessageRole.USER, content="上一轮的问题"),
            Message(role=MessageRole.ASSISTANT, content="上一轮的回答"),
        ]
        memory = FakeMemoryRuntime(history=history)
        pipeline = FakePipeline([_scored("c1", text="正文")])
        gateway = CapturingGateway()
        answer = await KnowledgeAnswerer(
            pipeline, gateway, settings=Settings(memory="governed"), memory=memory
        ).answer("这轮的问题", session_id="s1")
        messages = gateway.seen_messages[0]
        assert len(messages) == 4  # system + 2 历史 + 最终 user
        assert messages[1].content == "上一轮的问题"
        assert messages[2].content == "上一轮的回答"
        assert answer.history_messages == 2

    async def test_memory_off_bypasses_runtime(self) -> None:
        """MEMORY=off：即使注入了 runtime 也旁路（off 档不受对象残留影响）。"""
        memory = FakeMemoryRuntime(items=[_memory_item("brand_tone", "专业")])
        pipeline = FakePipeline([_scored("c1", text="正文")])
        gateway = CapturingGateway()
        answer = await KnowledgeAnswerer(
            pipeline, gateway, settings=Settings(memory="off"), memory=memory
        ).answer("问题", session_id="s1")
        assert memory.recalled_queries == []
        assert memory.recorded == []
        assert answer.memory_keys == ()
        assert len(gateway.seen_messages[0]) == 2


class TestSessionRecording:
    async def test_record_turn_called_with_qa(self) -> None:
        """回答落地后按 (session_id, 问题, 答案) 追加。"""
        memory = FakeMemoryRuntime()
        pipeline = FakePipeline([_scored("c1", text="正文")])
        gateway = CapturingGateway("答案正文 [1]")
        await KnowledgeAnswerer(
            pipeline, gateway, settings=Settings(memory="governed"), memory=memory
        ).answer("记住：禁用词是最", session_id="s9")
        assert memory.recorded[0]["session"] == "s9"
        assert memory.recorded[0]["question"] == "记住：禁用词是最"
        assert memory.recorded[0]["answer"] == "答案正文 [1]"

    async def test_no_session_no_record(self) -> None:
        memory = FakeMemoryRuntime()
        pipeline = FakePipeline([_scored("c1", text="正文")])
        gateway = CapturingGateway()
        await KnowledgeAnswerer(
            pipeline, gateway, settings=Settings(memory="governed"), memory=memory
        ).answer("单轮问题")
        assert memory.recorded == []


class TestBudgetSqueeze:
    async def test_history_squeezed_when_over_slot(self) -> None:
        """历史超槽位预算：低分历史先出局（budget_squeezed 而非硬报错）。"""
        long_history = [
            Message(role=MessageRole.USER, content="很长的历史轮" * 300)
            for _ in range(20)
        ]
        memory = FakeMemoryRuntime(history=long_history)
        pipeline = FakePipeline([_scored("c1", text="正文")])
        gateway = CapturingGateway()
        await KnowledgeAnswerer(
            pipeline, gateway, settings=Settings(memory="governed"), memory=memory
        ).answer("问题", session_id="s1")
        messages = gateway.seen_messages[0]
        # 2000 token 槽位 ≈ 8 条；20 条里大部分被挤掉，但 system 与最终
        # user（问题与资料）永远保留——这就是"预算裁剪不伤主干"。
        user_messages = [m for m in messages if m.role is MessageRole.USER]
        assert len(messages) < 1 + 20 + 1
        assert messages[0].role is MessageRole.SYSTEM
        assert "参考资料" in (user_messages[-1].content or "")
