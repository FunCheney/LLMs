#!/usr/bin/env python
"""20+ 轮长会话演示：不爆上下文（Gate 7 #2 / W12 demo 素材）。

**零依赖、零密钥、零网络**：不调 LLM，答案用规则合成。这样它可以在任何机器上
跑出同一张表——演示脚本如果依赖密钥，"演示"就变成了"我的机器上能跑"。

它演示三件事：

1. 全量历史会让 payload 线性增长，第 N 轮就撞穿预算；
2. 同一份历史走 `allocate` 后，payload 稳定在上限内，且能说清裁掉了谁；
3. 加上工作记忆（摘要 + 保留最近）后，**裁掉的不是信息，是冗余**——
   早期结论被压进摘要，仍在 prompt 里。

用法：

    .venv/bin/python scripts/demo_long_session.py
    .venv/bin/python scripts/demo_long_session.py --rounds 30 --window 8000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packages" / "agent_framework" / "src"))

from agent_framework.context import (  # noqa: E402
    ContextBudget,
    ContextBuilder,
    ContextItem,
    ContextKind,
    TrustLevel,
    allocate,
)
from agent_framework.llm.types import Message, MessageRole  # noqa: E402
from agent_framework.memory import WorkingMemory  # noqa: E402
from agent_framework.retrieval.context_assemble import estimate_tokens  # noqa: E402
from agent_framework.session.memory import InMemorySessionStore  # noqa: E402

SYSTEM_PROMPT = "你是增长营销团队的文案助手，回答简洁、可执行。"

# 20 轮递进追问。每轮都带材料，让 payload 真实增长（不靠无意义填充）。
QUESTIONS = (
    "梳理保温杯的目标人群。",
    "定 3 个核心卖点。",
    "选一个做主标题。",
    "主标题给 3 个版本。",
    "哪个版本搜索友好？",
    "详情页首屏放什么？",
    "材质卖点怎么写不夸大？",
    "竞品都讲 24 小时保温，怎么差异化？",
    "给一条用户证言方向。",
    "证言几句合适？",
    "小红书标题怎么起？",
    "小红书正文模板给一个。",
    "短视频前 3 秒说什么？",
    "第 15 秒该出现什么？",
    "直播话术最忌讳什么？",
    "售后话术怎么写不敷衍？",
    "复购提醒给一句。",
    "把本周结论整理成 5 条。",
    "挑出最该先做的一条。",
    "用一句话总结核心策略。",
)


def text_of(content: object) -> str:
    """Message.content 是 `str | list[ContentBlock]`（多模态）。本演示只处理文本。"""

    return content if isinstance(content, str) else ""


def fake_answer(index: int, question: str) -> str:
    """规则合成答案。长度接近真实回答（~120 token），让预算压力可信。"""

    return (
        f"第 {index} 轮答复：针对「{question}」给出 3 条可执行建议。"
        f"一、先锁定通勤人群的高频场景（早八通勤、办公室续水），"
        f"文案只讲一个主场景，不铺开。"
        f"二、卖点排序固定为：保温时长 > 一键开盖 > 可拆洗，"
        f"每个卖点配一句不超过 15 字的主张。"
        f"三、所有表述避免绝对化词，用可验证的数字替代形容词。"
        f"补充：本条结论与第 {max(index - 1, 1)} 轮保持一致，"
        f"若后续出现冲突，以最新一轮为准。"
    )


class RuleSummarizer:
    """规则摘要器（演示用）。真实现是 `make_gateway_summarizer`（调 cheap 档）。

    这里压缩到"最早几轮的关键词"就够演示：重点是**视图长度不随轮次增长**，
    而不是摘要写得好不好。
    """

    async def __call__(self, old_summary: str, messages: Sequence[Message]) -> str:
        head = "；".join(
            text_of(message.content).split("。")[0]
            for message in messages
            if message.role is MessageRole.USER
        )
        prefix = f"{old_summary}｜" if old_summary else ""
        return f"[摘要] {prefix}早期结论：{head[:120]}"


def build_payload(
    builder: ContextBuilder,
    budget: ContextBudget,
    system: str,
    history: list[Message],
    question: str,
) -> tuple[list[Message], int, int]:
    """返回 (messages, payload token, 被裁条数)。"""

    items = [
        ContextItem(
            id="system",
            content=system,
            kind=ContextKind.SYSTEM,
            source="demo",
            trust=TrustLevel.SYSTEM,
            score=1.0,
        )
    ]
    total = len(history)
    for index, message in enumerate(history):
        items.append(
            ContextItem(
                id=f"h{index}",
                content=text_of(message.content),
                kind=ContextKind.HISTORY,
                source="session",
                # 越新分越高：裁剪时先裁最旧的那几轮。
                score=(index + 1) / total,
                message=message,
            )
        )
    allocated = allocate(items, budget)
    messages = builder.build(allocated, question=question)
    tokens = sum(estimate_tokens(text_of(m.content)) for m in messages)
    return messages, tokens, len(allocated.exclusions)


async def main() -> int:
    parser = argparse.ArgumentParser(description="20+ 轮长会话预算演示（不调 LLM）")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--window",
        type=int,
        default=2000,
        help="模型输入预算。演示默认 2000 是为了把爆窗点压进 20 轮内"
        "（生产按 8000，约 45–60 轮才撞穿，一次 demo 看不到）",
    )
    parser.add_argument("--history-slot", type=int, default=1000, help="history 槽位预算")
    parser.add_argument("--threshold", type=int, default=1500, help="摘要触发阈值")
    parser.add_argument("--keep-recent", type=int, default=6)
    args = parser.parse_args()

    budget = ContextBudget(
        window=args.window,
        system=400,
        retrieved=0,
        memory=600,
        history=args.history_slot,
        tool_results=0,
    )
    store = InMemorySessionStore()
    working = WorkingMemory(
        store,
        summarizer=RuleSummarizer(),
        token_threshold=args.threshold,
        keep_recent_messages=args.keep_recent,
    )
    builder = ContextBuilder(estimate_tokens)
    session = "demo"

    history: list[Message] = []
    naive_over = 0
    budget_over = 0
    summary_over = 0

    print(
        f"{'轮':>3} {'naive':>7} {'budget':>7} {'summary':>8} "
        f"{'裁掉':>5} {'摘要':>6}  说明"
    )
    print("-" * 78)

    for index, question in enumerate(QUESTIONS[: args.rounds], start=1):
        answer = fake_answer(index, question)
        naive_payload = [*history, Message(role=MessageRole.USER, content=question)]
        naive_tokens = sum(estimate_tokens(text_of(m.content)) for m in naive_payload)
        naive_tokens += estimate_tokens(SYSTEM_PROMPT)

        _, budget_tokens, dropped = build_payload(
            builder, budget, SYSTEM_PROMPT, history, question
        )
        view = await working.view(session)
        _, summary_tokens, _ = build_payload(
            builder, budget, SYSTEM_PROMPT, view, question
        )

        naive_over += naive_tokens > args.window
        budget_over += budget_tokens > args.window
        summary_over += summary_tokens > args.window

        await store.append(
            session,
            [
                Message(role=MessageRole.USER, content=question),
                Message(role=MessageRole.ASSISTANT, content=answer),
            ],
            request_id=f"demo-{index}",
        )
        compressed = await working.compress(session)
        history.extend(
            [
                Message(role=MessageRole.USER, content=question),
                Message(role=MessageRole.ASSISTANT, content=answer),
            ]
        )

        note = {
            "created": f"生成摘要（覆盖 {compressed.compressed_messages} 条）",
            "updated": f"更新摘要（+{compressed.compressed_messages} 条）",
            "skipped": "未达阈值",
        }[compressed.action.value]
        flag = " ⚠ 爆窗" if naive_tokens > args.window else ""
        print(
            f"{index:>3} {naive_tokens:>7} {budget_tokens:>7} {summary_tokens:>8} "
            f"{dropped:>5} {compressed.action.value:>6}  {note}{flag}"
        )

    print("-" * 78)
    print(
        f"爆窗轮数（>{args.window} token）："
        f"naive {naive_over}/{args.rounds}，"
        f"budget {budget_over}/{args.rounds}，"
        f"summary {summary_over}/{args.rounds}"
    )
    print(
        "结论：naive 随轮次线性增长；budget 把总量压在上限内但**丢早期信息**；"
        "summary 既不爆窗，又把早期结论留在摘要里——这就是工作记忆存在的理由。"
    )
    return 0 if budget_over == 0 and summary_over == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
