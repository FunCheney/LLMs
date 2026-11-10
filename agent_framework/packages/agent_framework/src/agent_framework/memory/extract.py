"""记忆提取（W7 D3）：从会话消息提取结构化记忆候选。

成本纪律：**提取在会话结束时（或显式触发时）执行一次**，不在每轮做。
一次会话 20 轮 × 每轮一次提取 = 20 次 cheap 调用，而其中 19 次提取的
内容马上就会被下一轮推翻——那不是"记得更牢"，是白花钱。

信任级标注规则（写入管线的污染校验靠它）：

- 用户消息里的显式偏好（"记住：禁用词是'最'和'第一'"）→ trust=user，
  `confirmed_by_user=True`；
- 用户消息里的隐含偏好 → trust=user，confirmed_by_user=False；
- 助手转述 / 工具返回 / 网页内容 → trust=external（默认被 write 拒绝）。

这个规则是**提取侧的标注义务**：write() 只看 trust 字段，不重新判断
内容来源——内容特征判断（"这句话像不像用户说的"）不可靠，来源判断
（"这条消息的 role 是什么"）可靠。按来源拦，不按内容拦。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..llm.client import ChatCall, LLMGateway
from ..llm.types import Message, MessageRole
from .long_term import MemoryItem

__all__ = [
    "EXTRACT_MAX_ITEMS",
    "MemoryExtractor",
    "parse_extraction",
]

_LOG = logging.getLogger("agent_framework.memory.extract")

# 单次提取的候选上限。不设上限的话，一次跑偏的提取能把 key 覆盖面
# 撑爆（同一个 key 写 20 条近似值，冲突闸会把前 19 条全部顶替一遍）。
EXTRACT_MAX_ITEMS = 5

# 允许的 key 白名单。extract 输出别的 key 直接丢弃：受控词表让
# recall 端的 prompt 注入稳定（不用处理任意 key 的语义）。
KNOWN_KEYS = frozenset({"brand_tone", "forbidden_words", "past_topics"})

_EXTRACT_PROMPT = """从下面的对话记录中提取值得长期记住的品牌工作偏好。
只输出 JSON 数组，不要输出其他文字。

每条格式：
{"key": "...", "value": "...", "confidence": 0.0,
 "evidence_role": "user|assistant|tool", "evidence": "..."}

规则：
- key 只能取：brand_tone（品牌调性）、forbidden_words（禁用词）、past_topics（历史选题）
- key 的边界必须守住（同一个 key 的新值会顶替旧值，归错类等于把对的删掉）：
  * brand_tone 只收"品牌整体语气/人设"（如 专业克制、年轻活泼）；
    写作手法（句长、标点、排版）不是调性，**不要归到 brand_tone**
  * forbidden_words 只收具体的词或固定表述（如「最」「第一」「最好用」）
  * past_topics 只收做过的选题/内容主题
- 一次提取里同一个 key 最多一条，取信息量最大的那条
- value 用一句短语概括，保留用户的原始用词
- confidence 0–1：用户明确说"记住/以后要/不要"给 0.9 以上；从对话推断给 0.5–0.7
- evidence_role 是结论依据的那条消息的角色；evidence 摘录原句（30 字内）
- 没有值得记的内容就输出 []

对话记录：
"""

# "记住：xxx" 这类显式指令的识别。confirmed_by_user 的唯一来源——
# 不用 LLM 判断（它会把"看起来像偏好"都标成显式），用字面模式。
_EXPLICIT_MARKERS = re.compile(r"记住|以后(要|请|别|不要)|从今往后|务必|禁止使用")


@dataclass(frozen=True)
class ExtractionResult:
    """一次提取的产物。candidates 按置信度降序、截到 EXTRACT_MAX_ITEMS。"""

    candidates: tuple[MemoryItem, ...]
    # LLM 原始输出是否解析成功（失败 = 0 条候选，但不是错误）。
    parsed: bool
    # 提取调用的用量。
    input_tokens: int = 0
    output_tokens: int = 0


def parse_extraction(
    raw: str, *, source_session: str
) -> list[MemoryItem]:
    """把 LLM 输出解析成候选。**不抛异常**：坏输出记日志返回空——
    提取失败损失的只是"这次没记住"，抛异常损失的是整个回答链路。"""
    text = (raw or "").strip()
    # 容错两类常见包装：```json ... ``` 与前后闲话。
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        _LOG.warning("memory.extract: 输出里找不到 JSON 数组：%r", text[:120])
        return []

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        _LOG.warning("memory.extract: JSON 解析失败：%r", text[:120])
        return []

    if not isinstance(payload, list):
        return []

    items: list[MemoryItem] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key", ""))
        value = str(entry.get("value", ""))
        if key not in KNOWN_KEYS or not value.strip():
            continue
        try:
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence_role = str(entry.get("evidence_role", "user"))
        evidence = str(entry.get("evidence", ""))
        # 信任级由**证据消息的角色**决定（见模块 docstring 的标注规则）。
        trust = "user" if evidence_role == "user" else "external"
        # confirmed 只认字面显式指令，且证据必须是用户原话。
        confirmed = trust == "user" and bool(_EXPLICIT_MARKERS.search(evidence))
        items.append(
            MemoryItem(
                key=key,
                value=value.strip(),
                confidence=min(max(confidence, 0.0), 1.0),
                source_session=source_session,
                trust=trust,
                confirmed_by_user=confirmed,
            )
        )

    items.sort(key=lambda it: -it.confidence)
    return items[:EXTRACT_MAX_ITEMS]


class MemoryExtractor:
    """走 gateway（cheap 档）的会话记忆提取器。构造一次、全程复用。"""

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        max_items: int = EXTRACT_MAX_ITEMS,
        max_tokens: int = 500,
    ) -> None:
        self._gateway = gateway
        self._max_items = max_items
        self._max_tokens = max_tokens

    async def extract(
        self, messages: Sequence[Message], *, source_session: str
    ) -> ExtractionResult:
        transcript = _render_transcript(messages)
        if not transcript:
            return ExtractionResult(candidates=(), parsed=True)

        prompt = f"{_EXTRACT_PROMPT}{transcript}"
        alias = self._gateway.catalog.routes["cheap"].primary
        call = ChatCall(
            messages=[Message(role=MessageRole.USER, content=prompt)],
            task="chat",
            model=alias,
            opts={"max_tokens": self._max_tokens, "temperature": 0.0},
        )
        try:
            outcome = await self._gateway.chat(call)
        except Exception:  # noqa: BLE001 - 提取失败按空结果处理，见 parse 注释
            _LOG.warning("memory.extract: LLM 调用失败，本次不提取", exc_info=True)
            return ExtractionResult(candidates=(), parsed=False)

        raw = outcome.response.text or ""
        items = parse_extraction(raw, source_session=source_session)[: self._max_items]
        usage = getattr(outcome.response, "usage", None)
        return ExtractionResult(
            candidates=tuple(items),
            parsed=True,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


def _render_transcript(messages: Sequence[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        if not isinstance(message.content, str) or not message.content.strip():
            continue
        if message.role not in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL):
            continue
        lines.append(f"{message.role.value}: {message.content}")
        if len(lines) >= 40:  # 只取最近 40 条：提取对象是"最近发生了什么"
            break
    return "\n".join(lines)
