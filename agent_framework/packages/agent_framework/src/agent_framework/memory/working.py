"""工作记忆（W7 D2）：滚动摘要 + tool result 截断，让 20+ 轮不爆上下文。

两条设计判断：

1. **阈值按 token 数而非轮数**。不同轮长度差一个量级（"好"和 3000 字的
   调研报告都是一轮），按轮数触发会把长会话压得太晚、短会话压得太早。
   `summarize_after` 轮数兜底：极端"轮轮都短"的会话也有上限。
2. **压缩状态放进程内、原文仍在 SessionStore**。SessionStore 是追加式
   （append-only）协议，没有"改写历史"的方法——那是刻意的：历史是审计
   对象，摘要只是**派生视图**。进程重启后第一次 `compress()` 会重建摘要
   （多花一次 cheap 调用），但历史永不丢。这个取舍写进了
   docs/w7/context-budget-spec.md。

幂等：同一会话连续 `compress()` 两次，第二次必然 `skipped`——可压缩区
（keep_recent 之前的消息）没有新增，判定条件天然不成立。不靠"记一个
compressed 标志位"：标志位会在进程重启后丢失，而条件判定不会。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..llm.client import ChatCall, LLMGateway
from ..llm.types import Message, MessageRole
from ..observability import current_request_id
from ..retrieval.context_assemble import estimate_tokens
from ..session.base import SessionStore

__all__ = [
    "SUMMARY_MAX_TOKENS",
    "CompressAction",
    "CompressResult",
    "WorkingMemory",
    "make_gateway_summarizer",
]

# 摘要自身的长度上限（粗估 token）。不设上限的话，摘要会随会话无限增长，
# "压缩"变成"换一种方式存全文"。
SUMMARY_MAX_TOKENS = 300

# 截断提示的模板。句柄必须是"可重新执行的动作"，不是一句道歉——
# 信息丢了能找回，才叫可恢复。
_TRUNCATION_NOTE = (
    "\n\n[结果过长，已截断保留前 {kept} 字。完整结果可重新获取：{handle}]"
)


class CompressAction(StrEnum):
    """一次 compress 实际做了什么。报告与实验脚本按它分组计数。"""

    CREATED = "created"
    UPDATED = "updated"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CompressResult:
    """压缩结果，字段直接进日志/实验报告。"""

    action: CompressAction
    # 摘要覆盖到的消息数（历史列表前缀长度）。
    covered_messages: int
    # 摘要本身的 token 数（粗估）。对照 SUMMARY_MAX_TOKENS 看是否触顶。
    summary_tokens: int
    # 本次被压缩掉的消息数（updated 时是新增被覆盖的条数）。
    compressed_messages: int
    # 摘要调用的 token 用量（0 = 用了注入的 summarizer / skipped）。
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float = 0.0
    # skipped / 降级的原因。空串 = 没什么可解释的。
    note: str = ""


@dataclass
class _SessionCompressState:
    """每个会话的压缩进度。进程内状态，重启即重建（见模块 docstring）。"""

    summary: str = ""
    covered_messages: int = 0
    last_action: str = ""


SummaryFn = Callable[[str, Sequence[Message]], Awaitable[str]]


@dataclass
class SummarizerUsage:
    """摘要调用的累计用量。

    **压缩不是免费的**：摘要是一次真实的 LLM 调用。不把它的用量记成数，
    对照实验就会得出"压缩零成本"的结论——那是把成本藏进了没统计的那一栏。
    """

    # 摘要调用次数（= 触发压缩的次数，不含 skipped）。
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def snapshot(self) -> tuple[int, int, int]:
        """取当前累计值，供调用方做差（compress 需要"本次"用量）。"""
        return (self.calls, self.input_tokens, self.output_tokens)


def make_gateway_summarizer(
    gateway: LLMGateway,
    *,
    max_tokens: int = SUMMARY_MAX_TOKENS,
    usage: SummarizerUsage | None = None,
    extra_opts: dict[str, Any] | None = None,
) -> SummaryFn:
    """构造走 LLMGateway 的摘要函数。单独一个工厂：测试注入假 summarizer
    时不需要假 gateway（依赖面更小）。

    `usage` 是可选的累计器：传进来才会记录用量（生产装配传，测试不必）。
    """

    async def summarize(old_summary: str, messages: Sequence[Message]) -> str:
        transcript = "\n".join(
            f"{msg.role.value}: {msg.content if isinstance(msg.content, str) else ''}"
            for msg in messages
            if isinstance(msg.content, str)
        )
        prompt = (
            "把下面的对话历史压缩成不超过 200 字的要点摘要，保留：任务目标、"
            "已确定的结论、用户给出的偏好与约束。只输出摘要正文。\n"
        )
        if old_summary:
            prompt = f"已有摘要（先把新内容并入它）：\n{old_summary}\n\n{prompt}"
        prompt = f"{prompt}\n对话历史：\n{transcript}"

        # 路由：从路由表取 cheap 档而不是从 models 里挑——路由表才是
        # "谁来干便宜活"的唯一真相（与 QueryRewriter 同一纪律）。
        alias = gateway.catalog.routes["cheap"].primary
        call = ChatCall(
            messages=[Message(role=MessageRole.USER, content=prompt)],
            task="chat",
            model=alias,
            opts={"max_tokens": max_tokens, "temperature": 0.2, **(extra_opts or {})},
        )
        try:
            outcome = await gateway.chat(call)
        except Exception:  # noqa: BLE001 - 摘要失败必须降级（返回旧摘要），
            # 理由：摘要是优化项不是正确性依赖。压不下历史最多贵一点，
            # 回答链路 500 才是真故障。调用方从 note 里看得到这次没压成。
            return old_summary

        if usage is not None:
            # 失败的那次不计：没拿到 usage 就没有可记的数，宁可少记也不编。
            call_usage = getattr(outcome.response, "usage", None)
            usage.calls += 1
            usage.input_tokens += int(getattr(call_usage, "input_tokens", 0) or 0)
            usage.output_tokens += int(getattr(call_usage, "output_tokens", 0) or 0)

        text = (outcome.response.text or "").strip()
        return text or old_summary

    return summarize


class WorkingMemory:
    """会话级工作记忆。构造一次、多会话复用（状态按 session_id 分桶）。"""

    def __init__(
        self,
        store: SessionStore,
        *,
        summarizer: SummaryFn | None = None,
        token_threshold: int = 6000,
        summarize_after: int = 20,
        keep_recent_messages: int = 8,
        tokenizer: Callable[[str], int] = estimate_tokens,
        clock: Callable[[], float] = time.perf_counter,
        summarizer_usage: SummarizerUsage | None = None,
    ) -> None:
        if token_threshold <= 0 or summarize_after < 1 or keep_recent_messages < 1:
            raise ValueError("token_threshold/summarize_after/keep_recent 必须为正")
        self._store = store
        self._summarizer = summarizer
        self._token_threshold = token_threshold
        self._summarize_after = summarize_after
        self._keep_recent_messages = keep_recent_messages
        self._tokenizer = tokenizer
        self._clock = clock
        # 进程内压缩状态：session_id -> state。不做淘汰——会话数受
        # SessionStore TTL 约束，且单条状态只有一份摘要文本。
        self._states: dict[str, _SessionCompressState] = {}
        # tool result 截断的全文本登记处：handle -> full text。
        self._handles: dict[str, str] = {}
        # 摘要用量累计器（与 summarizer 共用同一个对象；None = 不统计）。
        self._summary_usage = summarizer_usage

    # -- 视图 -------------------------------------------------------------- #

    async def view(self, session_id: str) -> list[Message]:
        """喂给模型的历史视图：[摘要] + 最近原文。

        摘要是 role=user 的**带标签**消息（"[会话摘要]..."）——用 system
        角色会被当指令，不带标签会被当用户原话，两者都错。
        """
        history = await self._store.get_history(session_id)
        state = self._states.get(session_id)
        if state is None or state.covered_messages == 0:
            return list(history)

        summary_message = Message(
            role=MessageRole.USER,
            content=(
                f"[会话摘要（系统生成，覆盖前 {state.covered_messages} 条消息，"
                f"非用户原话）]\n{state.summary}"
            ),
        )
        return [summary_message, *history[state.covered_messages :]]

    # -- 压缩 -------------------------------------------------------------- #

    async def compress(self, session_id: str) -> CompressResult:
        """把 `历史[covered : -keep_recent]` 压进滚动摘要。幂等（见模块 docstring）。"""
        started = self._clock()
        history = await self._store.get_history(session_id)
        state = self._states.setdefault(
            session_id, _SessionCompressState()
        )

        # 可压缩区 = 排除"必须保留原文的最近 N 条"之后的尾部。
        compressible_end = max(len(history) - self._keep_recent_messages, 0)
        new_messages = history[state.covered_messages : compressible_end]

        if not new_messages:
            return CompressResult(
                action=CompressAction.SKIPPED,
                covered_messages=state.covered_messages,
                summary_tokens=self._count(state.summary),
                compressed_messages=0,
                note="无可压缩的新消息（keep_recent 之外没有新增）",
                elapsed_ms=(self._clock() - started) * 1000,
            )

        pending_tokens = sum(
            self._count(_text_of(msg)) for msg in new_messages
        )
        uncovered_turns = len(history) - state.covered_messages
        # 触发条件二选一：token 超阈值（主判据），或未压缩消息数超
        # summarize_after（兜底，防"轮轮都短"的会话无限不压缩）。
        over_tokens = pending_tokens + self._count(state.summary) > self._token_threshold
        over_turns = uncovered_turns > self._summarize_after
        if not (over_tokens or over_turns) and state.covered_messages > 0:
            return CompressResult(
                action=CompressAction.SKIPPED,
                covered_messages=state.covered_messages,
                summary_tokens=self._count(state.summary),
                compressed_messages=0,
                note=f"未达阈值（pending={pending_tokens}tok / {uncovered_turns}条）",
                elapsed_ms=(self._clock() - started) * 1000,
            )

        if self._summarizer is None:
            # 没有摘要器（测试/离线）：只推进覆盖进度，摘要置空并说明。
            # 这样 view() 至少不会重复喂"已被判定为可压缩"的消息。
            state.covered_messages = compressible_end
            state.last_action = "no_summarizer"
            return CompressResult(
                action=CompressAction.UPDATED,
                covered_messages=state.covered_messages,
                summary_tokens=0,
                compressed_messages=len(new_messages),
                note="未注入 summarizer，只推进覆盖进度不生成摘要",
                elapsed_ms=(self._clock() - started) * 1000,
            )

        before = (
            self._summary_usage.snapshot() if self._summary_usage is not None else None
        )
        summary = await self._summarizer(state.summary, new_messages)
        action = CompressAction.CREATED if not state.summary else CompressAction.UPDATED
        state.summary = summary
        state.covered_messages = compressible_end
        state.last_action = action.value

        # 本次用量 = 调用后减调用前。摘要可能内部重试或多次调用，取差值
        # 比"假设一次调用"更准。
        delta_in = delta_out = 0
        if before is not None and self._summary_usage is not None:
            after = self._summary_usage.snapshot()
            delta_in = max(after[1] - before[1], 0)
            delta_out = max(after[2] - before[2], 0)

        return CompressResult(
            action=action,
            covered_messages=state.covered_messages,
            summary_tokens=self._count(summary),
            compressed_messages=len(new_messages),
            input_tokens=delta_in,
            output_tokens=delta_out,
            elapsed_ms=(self._clock() - started) * 1000,
        )

    # -- tool result 截断 -------------------------------------------------- #

    def truncate_tool_result(self, text: str, *, handle: str, max_chars: int = 2000) -> str:
        """超长工具结果 → 截断 + 可重新获取的句柄。

        句柄登记在进程内（`refetch` 取回）。不做持久化：工具结果本身
        可重新执行获取，句柄只是"找回的钥匙"而不是"存储"。若 W9 需要
        跨进程句柄，届时把这里换成工具注册表——接口不变。
        """
        if len(text) <= max_chars:
            return text

        self._handles[handle] = text
        kept = text[:max_chars]
        return kept + _TRUNCATION_NOTE.format(kept=max_chars, handle=handle)

    def refetch(self, handle: str) -> str | None:
        """按句柄取回完整结果。None = 句柄不存在（过期/别的进程）。"""
        return self._handles.get(handle)

    def handles(self) -> tuple[str, ...]:
        """已登记句柄（报告用，让"截断了多少"可数）。"""
        return tuple(self._handles)

    # -- 内部 -------------------------------------------------------------- #

    def _count(self, text: str) -> int:
        return self._tokenizer(text) if text else 0


def _text_of(message: Message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return ""


def summarize_usage(result: CompressResult) -> dict[str, Any]:
    """压缩结果 → 日志投影（字段固定，实验脚本读它算摘要成本）。"""
    return {
        "action": result.action.value,
        "covered_messages": result.covered_messages,
        "summary_tokens": result.summary_tokens,
        "compressed_messages": result.compressed_messages,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "request_id": current_request_id() or "-",
    }
