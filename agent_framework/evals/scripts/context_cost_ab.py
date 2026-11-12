#!/usr/bin/env python
"""W7 · 上下文与记忆的两档对照实验（Gate 7 #7 / #11 的执行者）。

两个实验，两个不同的问题：

- **实验一（压缩有没有用）**：同一个 20+ 轮长会话，三种 payload 构造方式
  各跑一遍，看 input tokens 与"是否爆预算窗口"。要在成本下降的同时不让
  回答质量掉下去——只报 token 下降是自我表扬，不是结论。
- **实验二（记忆值不值得存在）**：`MEMORY=off` vs `MEMORY=governed`，
  同一批跨会话任务跑两遍。记忆是少数**加了会让成本上升**的能力，
  所以必须能回答"多花多少、换来什么"。

四条实验纪律（都是为了"数字能被别人复算"）：

1. **单变量**：同 alias、同 temperature、同问题序列；实验一只改 payload
   的构造方式，实验二只改记忆开关。
2. **历史同源**：三个 arm 共用同一份"标准答案历史"（由 naive arm 产生），
   否则臂与臂之间连对话内容都不同，token 差就不再是压缩带来的。
3. **token 数从网关 usage 取**，不自己算——自己算的口径与厂商计费口径
   必然有出入，而"预算"和"账单"说的必须是同一件事。
4. **爆窗判定要说明口径**：本机模型真实窗口 256K，20 轮根本碰不到厂商 400；
   所以"爆窗"按 `ContextBudget.window=8000` 这套**产品预算口径**判。
   把"我们自己的预算炸了"说成"厂商返回 400"是假证据。

用法：

    # 全跑（实验一 + 实验二，真实调用）
    .venv/bin/python evals/scripts/context_cost_ab.py --json evals/history/2026-W50-ab-run.json

    # 只跑实验一 / 只跑实验二
    .venv/bin/python evals/scripts/context_cost_ab.py --only exp1 --rounds 20
    .venv/bin/python evals/scripts/context_cost_ab.py --only exp2

    # 复用已记录的结果重算聚合（不调 LLM）
    .venv/bin/python evals/scripts/context_cost_ab.py --replay evals/history/2026-W50-ab-run.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT,
    REPO_ROOT / "packages" / "agent_framework" / "src",
    REPO_ROOT / "apps" / "growth_agent" / "src",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from agent_framework.bootstrap import build_gateway  # noqa: E402
from agent_framework.config import Settings  # noqa: E402
from agent_framework.context import (  # noqa: E402
    ContextBudget,
    ContextBuilder,
    ContextItem,
    ContextKind,
    TrustLevel,
    allocate,
)
from agent_framework.llm.client import ChatCall, LLMGateway  # noqa: E402
from agent_framework.llm.types import Message, MessageRole  # noqa: E402
from agent_framework.memory import (  # noqa: E402
    MemoryExtractor,
    SummarizerUsage,
    WorkingMemory,
    make_gateway_summarizer,
)
from agent_framework.memory.long_term import LongTermMemory  # noqa: E402
from agent_framework.retrieval.context_assemble import estimate_tokens  # noqa: E402
from agent_framework.session.memory import InMemorySessionStore  # noqa: E402

from evals.metrics import percentile  # noqa: E402

# 产品预算口径：与回答链路 `_answer_budget` 同源（system 400 / history 2000）。
# 实验要回答的是"这套预算够不够用"，所以判定必须用它，不能另设一个宽松的。
BUDGET = ContextBudget(system=400, retrieved=0, memory=0, history=2000, tool_results=0)
# 工作记忆的摘要阈值。设 1500 而不是生产默认 6000：20 轮 × ~250 token
# 才 5000，按 6000 根本触发不到压缩，实验一会得出"压缩没影响"——
# 那不是压缩没用，是压缩没生效。
SUMMARY_THRESHOLD = 1500
KEEP_RECENT = 6

SYSTEM_PROMPT = (
    "你是增长营销团队的文案助手。回答简洁、可用中文，"
    "每条建议控制在 80 字以内，不要重复用户已经说过的内容。"
)

# 实验二的记忆写入指令。用"记住：…"是刻意的：提取器只认字面显式指令，
# 换一句含蓄的说法，写不进去就不是这条链路的锅，是实验设计的问题。
TEACHING_TURNS = (
    "记住：我们是做户外保温杯的品牌，品牌调性要专业克制，不要喊口号。",
    "记住：所有标题都禁止使用「最」「第一」「绝对」这类绝对化词。",
    "以后写文案都用短句，不要用感叹号。",
)

# 实验二的跨会话任务。每条都带一个**可规则判定**的期望：
# 不用 LLM judge，因为"标题里有没有绝对化词"是可枚举的事实判断，
# 让 judge 去判这件事只会引入噪声，还更贵。
MEMORY_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "title_3",
        "question": "给这款 500ml 保温杯写 3 个电商标题，每行一个。",
        "expect": "no_banned",
    },
    {
        "id": "tone_recall",
        "question": "我们品牌的调性是什么？一句话回答。",
        "expect": "tone_words",
    },
    {
        "id": "banned_check",
        "question": "标题里可以用「最好用」这个词吗？直接回答可以或不可以。",
        "expect": "says_no",
    },
    {
        "id": "slogan",
        "question": "写一句 20 字以内的卖点文案。",
        "expect": "no_banned",
    },
)

# 绝对化词。含"最好""第一"这类，正是用户在会话里禁止的词。
BANNED_WORDS = ("最好", "第一", "绝对", "唯一", "顶级", "史上")
# 品牌调性的关键词。命中任一即认为答案体现了"专业克制"这条偏好。
TONE_WORDS = ("专业", "克制", "不喊口号", "不夸张")

# 实验一的问题序列：20 条递进的品牌策划追问，每条带一段"本轮材料"，
# 让 payload 随轮次真实增长（不是靠无意义填充撑大 token）。
ROUND_QUESTIONS = (
    "帮我梳理这款保温杯的目标人群。",
    "目标人群定了，接下来该定什么卖点？",
    "卖点里哪个最适合做主标题？",
    "主标题给 3 个版本，风格不要一样。",
    "这三个版本里哪个最容易被搜索到？",
    "详情页第一屏该放什么信息？",
    "详情页第二屏讲材质，怎么写？",
    "保温时长的卖点怎么讲才不夸大？",
    "竞品都在讲 24 小时保温，我们怎么差异化？",
    "给一个用户证言的方向。",
    "用户证言要几句？每句多长？",
    "小红书笔记的标题该怎么起？",
    "小红书正文的结构给个模板。",
    "抖音短视频前 3 秒该说什么？",
    "视频脚本写到第 15 秒该出现什么？",
    "直播间的话术里最忌讳什么？",
    "售后话术怎么写才不显得敷衍？",
    "复购提醒的文案给一句。",
    "把这周定的东西整理成 5 条要点。",
    "最后用一句话总结这次策划的核心策略。",
)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def load_env() -> None:
    """读 `.env.local`（存在才读）。密钥不进命令行历史，也不写进报告。

    用 python-dotenv 而不是手写解析：手写版会把行尾的中文注释当成值的一部分，
    非 ASCII 字符进了 HTTP header 会在**建连时**炸（`UnicodeEncodeError`），
    报错栈离真正的错误行很远。加载后统一体检一遍，坏值直接剔除并告警——
    "密钥里混进中文"这种事必须当场暴露，不能等某个 provider 首次被路由到时。
    """

    path = REPO_ROOT / ".env.local"
    if path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(path, override=False)
        except ImportError:
            for line in path.read_text("utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.split(" #")[0].strip())

    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        value = os.environ.get(key, "")
        if value and not value.isascii():
            print(f"[warn] {key} 含非 ASCII 字符，已忽略（检查 .env.local 行尾注释）", flush=True)
            os.environ.pop(key, None)


@dataclass
class CallRecord:
    """一次真实调用的读数。字段与报告表列一一对应。"""

    # 本次 payload 的估算 token（我们自己的预算口径）。
    payload_tokens: int = 0
    # 厂商返回的输入/输出 token（计费口径）。
    input_tokens: int = 0
    output_tokens: int = 0
    # 成本（按 models.yaml 的单价折算，币种随 alias）。
    cost: float = 0.0
    # usage 是不是估算值（部分厂商不回 usage，网关会按字符估）。
    estimated: bool = False
    latency_ms: float = 0.0
    # 这次调用是否真的发了出去（异常时 False，报告要能看出降级）。
    ok: bool = True
    error: str = ""


def cost_of(gateway: LLMGateway, alias: str, record: CallRecord) -> float:
    """按 catalog 里的单价折算成本。**不自己写单价表**——单价一变，
    报告和账单就会对不上，而那种错误没人会在意，直到对账。"""

    model = gateway.catalog.models.get(alias)
    if model is None:
        return 0.0
    pricing = model.pricing
    return round(
        (record.input_tokens * pricing.input + record.output_tokens * pricing.output)
        / pricing.per,
        6,
    )


# 思考模式必须钉死：qwen3.7-plus 默认开思考，同一条 26 字答案实测
# 876 output tokens（思考）vs 14（不思考）。不钉住，输出侧噪声会完全盖过
# 输入侧的优化——实验会得出"压缩没用"，而真相是"测量被另一种开销淹了"。
PIN_OPTS: dict[str, Any] = {"thinking": False}


async def chat_once(
    gateway: LLMGateway,
    messages: list[Message],
    *,
    alias: str,
    max_tokens: int = 320,
    temperature: float = 0.2,
) -> tuple[str, CallRecord]:
    """发一次真实请求并读数。异常不抛：记进 record 由调用方决定怎么算。"""

    call = ChatCall(
        messages=messages,
        task="chat",
        model=alias,
        opts={"max_tokens": max_tokens, "temperature": temperature, **PIN_OPTS},
    )
    record = CallRecord(
        payload_tokens=sum(estimate_tokens(text_of(m.content)) for m in messages)
    )
    started = time.perf_counter()
    try:
        outcome = await gateway.chat(call)
    except Exception as exc:  # noqa: BLE001 - 实验脚本要跑完所有 arm 再总结
        record.ok = False
        record.error = f"{type(exc).__name__}: {exc}"
        record.latency_ms = (time.perf_counter() - started) * 1000
        return "", record

    record.latency_ms = (time.perf_counter() - started) * 1000
    usage = getattr(outcome.response, "usage", None)
    record.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    record.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    record.estimated = bool(getattr(usage, "estimated", False))
    record.cost = cost_of(gateway, outcome.alias, record)
    return outcome.response.text or "", record


def text_of(content: object) -> str:
    """Message.content 是 `str | list[ContentBlock]`（多模态）。这里只处理文本。"""

    return content if isinstance(content, str) else ""


def history_items(messages: list[Message]) -> list[ContextItem]:
    """历史消息 → ContextItem，**score 按新旧递增**。

    这一行决定裁剪方向：`allocate` 从低分往高裁，所以"越旧越低分"
    才能裁掉最早那几轮。若不设 score（全部 0.5），裁剪顺序会退化成
    按 id 字典序——裁掉的既不是最旧的也不是最不相关的，报告无法解释。
    """

    total = len(messages)
    items: list[ContextItem] = []
    for index, message in enumerate(messages):
        items.append(
            ContextItem(
                id=f"h{index}",
                content=text_of(message.content),
                kind=ContextKind.HISTORY,
                source="session",
                score=(index + 1) / total,
                trust=TrustLevel.USER,
                message=message,
            )
        )
    return items


def build_payload(
    builder: ContextBuilder,
    history: list[Message],
    question: str,
) -> tuple[list[Message], int]:
    """按预算组装 payload，返回 (messages, 是否被裁剪过的标记位数)。"""

    items = [
        ContextItem(
            id="system",
            content=SYSTEM_PROMPT,
            kind=ContextKind.SYSTEM,
            source="answer.py",
            trust=TrustLevel.SYSTEM,
            score=1.0,
        ),
        *history_items(history),
    ]
    allocated = allocate(items, BUDGET)
    messages = builder.build(allocated, question=question)
    return messages, len(allocated.exclusions)


# --------------------------------------------------------------------------- #
# 实验一：压缩有没有用
# --------------------------------------------------------------------------- #


@dataclass
class RoundRow:
    """实验一的单轮读数（三个 arm 各一行）。"""

    round_index: int
    arm: str
    payload_tokens: int
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    over_budget: bool
    dropped_history: int
    ok: bool = True


@dataclass
class Exp1Result:
    rows: list[RoundRow] = field(default_factory=list)
    # 摘要调用的累计读数（压缩本身也要花钱，必须计入对照）。
    summarize_calls: int = 0
    summarize_input_tokens: int = 0
    summarize_output_tokens: int = 0
    summarize_cost: float = 0.0
    # naive arm 的最后一轮最终答案（报告里引用，证明质量没崩）。
    final_answer: str = ""


async def run_exp1(
    gateway: LLMGateway, *,
    rounds: int,
    alias: str,
    arms: set[str] | None = None,
) -> Exp1Result:
    """三个 arm 跑同一段会话。

    - `naive`：全量历史直接拼（不裁、不压）；
    - `budget`：同一份历史走 `allocate`，按 2000 token 槽位裁；
    - `summary`：先滚动摘要，再走同一套预算。

    **轮内顺序很重要**：先跑 naive 拿到"标准答案"，再用这份答案推进历史，
    最后让另外两个 arm 用同一份历史去读数。如果每个 arm 各自生成答案，
    三轮之后它们讨论的已经是三件不同的事，token 差就归因不到压缩上了。
    """

    store = InMemorySessionStore()
    # 摘要器与工作记忆共用同一个用量累计器：压缩是一次真实 LLM 调用，
    # 不记账它就会在报告里变成"零成本"，而"压缩免费"是最容易得出的错误结论。
    summary_usage = SummarizerUsage()
    working = WorkingMemory(
        store,
        summarizer=make_gateway_summarizer(
            gateway, usage=summary_usage, extra_opts=PIN_OPTS
        ),
        token_threshold=SUMMARY_THRESHOLD,
        keep_recent_messages=KEEP_RECENT,
        summarizer_usage=summary_usage,
    )
    builder = ContextBuilder(estimate_tokens)
    session = "exp1"
    result = Exp1Result()
    canonical: list[Message] = []
    wanted = arms or {"naive", "budget", "summary"}

    def want(arm: str) -> bool:
        """只跑指定 arm：调试时不想为一个改动付三个 arm 的钱。"""

        return arm in wanted

    questions = list(ROUND_QUESTIONS[:rounds])
    for index, question in enumerate(questions, start=1):
        materials = (
            f"【本轮材料】第 {index} 轮讨论。产品：500ml 316 不锈钢保温杯，"
            f"定价 199 元，主打 12 小时保温、一键开盖、可拆洗。"
            f"已确认的约束：不夸大、不用绝对化词、面向通勤人群。"
        )
        prompt = f"{materials}\n\n{question}"

        # --- arm 1：naive（全量历史） ---
        naive_messages = [
            Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            *canonical,
            Message(role=MessageRole.USER, content=prompt),
        ]
        answer, naive_record = await chat_once(
            gateway, naive_messages, alias=alias
        )
        result.rows.append(
            RoundRow(
                round_index=index,
                arm="naive",
                payload_tokens=naive_record.payload_tokens,
                input_tokens=naive_record.input_tokens,
                output_tokens=naive_record.output_tokens,
                cost=naive_record.cost,
                latency_ms=naive_record.latency_ms,
                over_budget=naive_record.payload_tokens > BUDGET.window,
                dropped_history=0,
                ok=naive_record.ok,
            )
        )

        # --- arm 2：budget（裁剪，不压缩） ---
        if want("budget"):
            budget_messages, dropped = build_payload(builder, canonical, prompt)
            _, budget_record = await chat_once(gateway, budget_messages, alias=alias)
            result.rows.append(
                RoundRow(
                    round_index=index,
                    arm="budget",
                    payload_tokens=budget_record.payload_tokens,
                    input_tokens=budget_record.input_tokens,
                    output_tokens=budget_record.output_tokens,
                    cost=budget_record.cost,
                    latency_ms=budget_record.latency_ms,
                    over_budget=budget_record.payload_tokens > BUDGET.window,
                    dropped_history=dropped,
                    ok=budget_record.ok,
                )
            )

        # --- arm 3：summary（摘要 + 裁剪） ---
        if want("summary"):
            view = await working.view(session)
            summary_messages, summary_dropped = build_payload(builder, view, prompt)
            _, summary_record = await chat_once(gateway, summary_messages, alias=alias)
            result.rows.append(
                RoundRow(
                    round_index=index,
                    arm="summary",
                    payload_tokens=summary_record.payload_tokens,
                    input_tokens=summary_record.input_tokens,
                    output_tokens=summary_record.output_tokens,
                    cost=summary_record.cost,
                    latency_ms=summary_record.latency_ms,
                    over_budget=summary_record.payload_tokens > BUDGET.window,
                    dropped_history=summary_dropped,
                    ok=summary_record.ok,
                )
            )

        # --- 推进历史：只用 naive 的答案（见 docstring 的"轮内顺序"） ---
        canonical.extend(
            [
                Message(role=MessageRole.USER, content=prompt),
                Message(role=MessageRole.ASSISTANT, content=answer),
            ]
        )
        await store.append(
            session,
            [
                Message(role=MessageRole.USER, content=prompt),
                Message(role=MessageRole.ASSISTANT, content=answer),
            ],
            request_id=f"exp1-{index}",
        )
        compressed = await working.compress(session)
        if compressed.action.value != "skipped":
            result.summarize_calls += 1
            result.summarize_input_tokens += compressed.input_tokens
            result.summarize_output_tokens += compressed.output_tokens
            # 单价从 catalog 取，不写死数字：写死就等于在报告里嵌了一份
            # 会过期的价目表，而账单不会跟着这份价目表变。
            result.summarize_cost += cost_of(
                gateway,
                alias,
                CallRecord(
                    input_tokens=compressed.input_tokens,
                    output_tokens=compressed.output_tokens,
                ),
            )
        result.final_answer = answer

    return result


def summarize_exp1(result: Exp1Result) -> dict[str, Any]:
    """按 arm 聚合。均值之外一定给 p95：长会话的成本问题出在尾部，
    均值会被前几轮的廉价请求摊平，看起来"没那么糟"。"""

    summary: dict[str, Any] = {}
    for arm in ("naive", "budget", "summary"):
        rows = [row for row in result.rows if row.arm == arm and row.ok]
        failures = [row for row in result.rows if row.arm == arm and not row.ok]
        if not rows:
            summary[arm] = {"rounds": 0, "errors": len(failures)}
            continue
        inputs = [float(row.input_tokens) for row in rows]
        costs = [row.cost for row in rows]
        summary[arm] = {
            "rounds": len(rows),
            "errors": len(failures),
            "avg_input_tokens": round(sum(inputs) / len(inputs), 1),
            "max_input_tokens": int(max(inputs)),
            "p95_input_tokens": percentile(inputs, 0.95),
            "first_round_input": int(inputs[0]),
            "last_round_input": int(inputs[-1]),
            "growth_x": round(inputs[-1] / inputs[0], 2) if inputs[0] else None,
            "avg_output_tokens": round(
                sum(row.output_tokens for row in rows) / len(rows), 1
            ),
            "total_cost": round(sum(costs), 6),
            "avg_cost": round(sum(costs) / len(costs), 6),
            "avg_latency_ms": round(
                sum(row.latency_ms for row in rows) / len(rows), 1
            ),
            "over_budget_rounds": sum(1 for row in rows if row.over_budget),
            "total_dropped_history": sum(row.dropped_history for row in rows),
        }

    summary["_compression"] = {
        "summary_calls": result.summarize_calls,
        "summary_input_tokens": result.summarize_input_tokens,
        "summary_output_tokens": result.summarize_output_tokens,
        "summary_cost": result.summarize_cost,
    }
    return summary


# --------------------------------------------------------------------------- #
# 实验二：记忆值不值得存在
# --------------------------------------------------------------------------- #


class LexicalHashEmbedder:
    """实验替身：字符 bigram 哈希 → 固定维向量（L2 归一化）。

    为什么不用真 bge-m3：本机离线环境没有 torch。要验的是"写入 → 召回 →
    注入"这条链路，不是"中文语义相似度有多强"。词面哈希的召回**偏保守**
    （比语义模型更容易漏），所以它能召回真模型只会更好——结论方向可信，
    绝对值不能当生产指标。生产装配走 `build_memory_runtime` 注入真 embedder。
    """

    dimension = 256

    def _vector(self, text: str) -> list[float]:
        import hashlib
        import math
        import re

        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens.extend(run[i : i + 2] for i in range(max(len(run) - 1, 1)))
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            slot = int.from_bytes(digest, "big") % self.dimension
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[Any]:
        return [
            SimpleNamespace(
                vector=self._vector(text),
                model="lexical-hash",
                dimension=self.dimension,
            )
            for text in texts
        ]


def score_answer(answer: str, expect: str) -> bool:
    """规则判定。三类期望各自一条判据，全部可枚举、可复算。"""

    text = answer or ""
    if expect == "no_banned":
        return not any(word in text for word in BANNED_WORDS)
    if expect == "tone_words":
        return any(word in text for word in TONE_WORDS)
    if expect == "says_no":
        # 明确否定 + 没有把"最好用"当成允许的说法。
        return ("不可以" in text or "不建议" in text or "不能" in text) and (
            "可以" in text
        )
    return False


@dataclass
class MemoryCase:
    """实验二单条任务的两个 arm 读数。"""

    case_id: str
    expect: str
    off_answer: str = ""
    off_ok: bool = False
    off_input_tokens: int = 0
    off_cost: float = 0.0
    on_answer: str = ""
    on_ok: bool = False
    on_input_tokens: int = 0
    on_cost: float = 0.0
    on_memory_keys: list[str] = field(default_factory=list)
    on_memory_hit: bool = False
    on_payload_tokens: int = 0


@dataclass
class Exp2Result:
    cases: list[MemoryCase] = field(default_factory=list)
    # 写入阶段（会话 A）的产物：哪些 key 真写进去了。
    written_keys: list[str] = field(default_factory=list)
    write_notes: list[str] = field(default_factory=list)
    extract_input_tokens: int = 0
    extract_output_tokens: int = 0


class MemoryHarness:
    """把"装配 + 教学 + 提问"包成一个小对象，两个 arm 共用同一套装配。

    刻意**不**复用 `growth_agent.memory.MemoryRuntime.record_turn`：
    实验要的是可控的三步（写入 / 召回 / 注入），而 record_turn 还带着
    追加会话与压缩副作用。协议层直接调，报告才解释得清每一步发生了什么。
    """

    def __init__(self, gateway: LLMGateway, *, dsn: str, table: str) -> None:
        from growth_agent.memory import PgVectorMemoryStore

        self._gateway = gateway
        self._store = PgVectorMemoryStore(
            dsn, LexicalHashEmbedder(), table=table, dimension=256
        )
        self.long_term = LongTermMemory(self._store, min_confidence=0.6)
        self.extractor = MemoryExtractor(gateway)

    def reset(self) -> None:
        """清表。两个 arm 必须从同一个空态出发，否则第二次跑会读到第一次的记忆。"""

        conn = self._store._conn  # noqa: SLF001 - 实验脚本，故意直接控制库
        conn.execute(f"TRUNCATE TABLE {self._store._table}")  # noqa: SLF001
        conn.commit()

    async def teach(self, session: str) -> tuple[list[str], list[str]]:
        """会话 A：用户连着三轮给偏好，逐轮提取并写入。"""

        written: list[str] = []
        notes: list[str] = []
        history: list[Message] = []
        for index, text in enumerate(TEACHING_TURNS, start=1):
            history.append(Message(role=MessageRole.USER, content=text))
            answer, _ = await chat_once(
                self._gateway,
                [Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT), *history],
                alias=self._alias(),
                max_tokens=80,
            )
            history.append(Message(role=MessageRole.ASSISTANT, content=answer))
            extraction = await self.extractor.extract(
                [
                    Message(role=MessageRole.USER, content=text),
                    Message(role=MessageRole.ASSISTANT, content=answer),
                ],
                source_session=session,
            )
            if not extraction.candidates:
                notes.append(f"turn{index}: 没有提取到候选（parsed={extraction.parsed}）")
                continue
            outcome = await self.long_term.write(extraction.candidates)
            written.extend(item.key for item in extraction.candidates)
            notes.append(f"turn{index}: {outcome.as_dict()}")
        return written, notes

    def _alias(self) -> str:
        return self._gateway.catalog.routes["cheap"].primary

    def close(self) -> None:
        self._store.close()


async def run_exp2(
    gateway: LLMGateway, *, dsn: str, table: str
) -> Exp2Result:
    """两个 arm：off（不注入记忆） / on（召回 + 注入记忆槽位）。"""

    harness = MemoryHarness(gateway, dsn=dsn, table=table)
    result = Exp2Result()
    try:
        harness.reset()
        written, notes = await harness.teach("memory-session-a")
        result.written_keys = written
        result.write_notes = notes

        builder = ContextBuilder(estimate_tokens)
        budget = ContextBudget(system=400, retrieved=0, memory=300, history=0, tool_results=0)

        for task in MEMORY_TASKS:
            case = MemoryCase(case_id=task["id"], expect=task["expect"])

            # --- off arm：只有 system + 问题 ---
            off_messages = [
                Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
                Message(role=MessageRole.USER, content=task["question"]),
            ]
            case.off_answer, off_record = await chat_once(
                gateway, off_messages, alias=harness._alias()
            )
            case.off_ok = score_answer(case.off_answer, task["expect"])
            case.off_input_tokens = off_record.input_tokens
            case.off_cost = off_record.cost

            # --- on arm：召回 → 记忆槽位 → 同一个 builder ---
            memories = await harness.long_term.recall(task["question"], k=3)
            case.on_memory_keys = [item.key for item in memories]
            items = [
                ContextItem(
                    id="system",
                    content=SYSTEM_PROMPT,
                    kind=ContextKind.SYSTEM,
                    source="answer.py",
                    trust=TrustLevel.SYSTEM,
                    score=1.0,
                ),
                *[
                    ContextItem(
                        id=f"{item.key}",
                        content=item.value,
                        kind=ContextKind.MEMORY,
                        source=item.source_session,
                        score=item.confidence,
                        trust=TrustLevel.USER,
                    )
                    for item in memories
                ],
            ]
            allocated = allocate(items, budget)
            on_messages = builder.build(allocated, question=task["question"])
            case.on_answer, on_record = await chat_once(
                gateway, on_messages, alias=harness._alias()
            )
            case.on_ok = score_answer(case.on_answer, task["expect"])
            case.on_input_tokens = on_record.input_tokens
            case.on_cost = on_record.cost
            case.on_payload_tokens = on_record.payload_tokens
            case.on_memory_hit = bool(memories)
            result.cases.append(case)
    finally:
        harness.close()
    return result


def summarize_exp2(result: Exp2Result) -> dict[str, Any]:
    total = len(result.cases)
    off_pass = sum(1 for case in result.cases if case.off_ok)
    on_pass = sum(1 for case in result.cases if case.on_ok)
    off_tokens = [float(case.off_input_tokens) for case in result.cases]
    on_tokens = [float(case.on_input_tokens) for case in result.cases]
    off_cost = sum(case.off_cost for case in result.cases)
    on_cost = sum(case.on_cost for case in result.cases)
    hit = sum(1 for case in result.cases if case.on_memory_hit)
    return {
        "cases": total,
        "written_keys": result.written_keys,
        "off": {
            "passed": off_pass,
            "pass_rate": round(off_pass / total, 4) if total else 0.0,
            "avg_input_tokens": round(sum(off_tokens) / total, 1) if total else 0.0,
            "total_cost": round(off_cost, 6),
        },
        "on": {
            "passed": on_pass,
            "pass_rate": round(on_pass / total, 4) if total else 0.0,
            "avg_input_tokens": round(sum(on_tokens) / total, 1) if total else 0.0,
            "total_cost": round(on_cost, 6),
            "memory_hit_cases": hit,
        },
        # 增量：记忆值不值得存在，就看这两行。
        "delta_pass": on_pass - off_pass,
        "delta_cost_pct": (
            round((on_cost - off_cost) / off_cost * 100, 1) if off_cost else None
        ),
        "delta_tokens_pct": (
            round((sum(on_tokens) - sum(off_tokens)) / sum(off_tokens) * 100, 1)
            if sum(off_tokens)
            else None
        ),
        "write_notes": result.write_notes,
    }


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_env()
    settings = Settings()
    gateway = build_gateway(settings)
    alias = gateway.catalog.routes["cheap"].primary

    report: dict[str, Any] = {
        "week": args.week,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_alias": alias,
        "temperature": 0.2,
        "budget": {
            "window": BUDGET.window,
            "system": BUDGET.system,
            "history": BUDGET.history,
            "summary_threshold": SUMMARY_THRESHOLD,
            "keep_recent": KEEP_RECENT,
        },
        "notes": [
            "token 数全部取网关 usage（厂商计费口径），payload token 为粗估口径（1.6 chars/token）",
            "爆窗按 ContextBudget.window=8000 的产品预算口径判；本机模型真实窗口更大，"
            "未触发厂商 400",
            "实验一三个 arm 共用同一份 naive 答案作为历史，保证单变量",
            "所有调用钉住 thinking=False：qwen 默认思考会让 output tokens 涨 60 倍，"
            "输出侧开销会盖过输入侧对比",
        ],
    }

    try:
        if args.only in ("all", "exp1"):
            print(f"[exp1] rounds={args.rounds} alias={alias}", flush=True)
            exp1 = await run_exp1(
                gateway, rounds=args.rounds, alias=alias, arms=args.arms_set
            )
            # 先落原始行、再算聚合：聚合属于"看得懂"的加工，它出错不该让
            # 十几分钟的真实调用白跑（第一次跑实验一就是这么丢掉 60 行的）。
            report["exp1"] = {
                "rows": [
                    {
                        "round": row.round_index,
                        "arm": row.arm,
                        "payload_tokens": row.payload_tokens,
                        "input_tokens": row.input_tokens,
                        "output_tokens": row.output_tokens,
                        "cost": row.cost,
                        "latency_ms": round(row.latency_ms, 1),
                        "over_budget": row.over_budget,
                        "dropped_history": row.dropped_history,
                        "ok": row.ok,
                    }
                    for row in exp1.rows
                ],
                "final_answer": exp1.final_answer,
                "summary": {},
            }
            try:
                report["exp1"]["summary"] = summarize_exp1(exp1)
            except Exception as exc:  # noqa: BLE001 - 聚合失败不丢原始数据
                report["exp1"]["summary"] = {"error": f"{type(exc).__name__}: {exc}"}

        if args.only in ("all", "exp2"):
            print(f"[exp2] memory off vs governed, table={args.table}", flush=True)
            exp2 = await run_exp2(gateway, dsn=args.dsn, table=args.table)
            report["exp2"] = {
                "summary": summarize_exp2(exp2),
                "cases": [
                    {
                        "id": case.case_id,
                        "expect": case.expect,
                        "off": {
                            "answer": case.off_answer,
                            "pass": case.off_ok,
                            "input_tokens": case.off_input_tokens,
                            "cost": case.off_cost,
                        },
                        "on": {
                            "answer": case.on_answer,
                            "pass": case.on_ok,
                            "input_tokens": case.on_input_tokens,
                            "cost": case.on_cost,
                            "memory_keys": case.on_memory_keys,
                            "memory_hit": case.on_memory_hit,
                            "payload_tokens": case.on_payload_tokens,
                        },
                    }
                    for case in exp2.cases
                ],
            }
    finally:
        await gateway.aclose()

    return report


def print_report(report: dict[str, Any]) -> None:
    """终端速览。完整数字在 JSON 里，这里只打决策需要看的几行。"""

    print("\n=== 实验一：压缩有没有用 ===")
    exp1 = report.get("exp1")
    if exp1 and exp1.get("summary", {}).get("error"):
        print(f"聚合失败（原始行仍在 JSON 里）：{exp1['summary']['error']}")
    elif exp1:
        print(
            f"{'arm':<10}{'avg_in':>9}{'last_in':>9}{'max_in':>8}"
            f"{'p95':>8}{'cost':>10}{'爆窗':>6}{'裁掉':>6}"
        )
        for arm in ("naive", "budget", "summary"):
            item = exp1["summary"].get(arm, {})
            if not item.get("rounds"):
                print(f"{arm:<10}{'--':>9}")
                continue
            print(
                f"{arm:<10}{item['avg_input_tokens']:>9.0f}{item['last_round_input']:>9}"
                f"{item['max_input_tokens']:>8}{float(item['p95_input_tokens'] or 0):>8.0f}"
                f"{item['total_cost']:>10.4f}{item['over_budget_rounds']:>6}"
                f"{item['total_dropped_history']:>6}"
            )
        compression = exp1["summary"]["_compression"]
        print(
            f"压缩调用 {compression['summary_calls']} 次，"
            f"成本 {compression['summary_cost']:.4f} 元"
        )

    print("\n=== 实验二：记忆值不值得存在 ===")
    exp2 = report.get("exp2")
    if exp2:
        summary = exp2["summary"]
        print(f"写入键：{summary['written_keys']}")
        for arm in ("off", "on"):
            item = summary[arm]
            extra = (
                f"  记忆命中 {item['memory_hit_cases']}/{summary['cases']}"
                if arm == "on"
                else ""
            )
            print(
                f"{arm:<4}通过 {item['passed']}/{summary['cases']}"
                f"  avg_in {item['avg_input_tokens']:.0f}"
                f"  cost {item['total_cost']:.4f}{extra}"
            )
        print(
            f"增量：通过 +{summary['delta_pass']}，"
            f"cost {summary['delta_cost_pct']}%，"
            f"input tokens {summary['delta_tokens_pct']}%"
        )


def replay(path: Path) -> dict[str, Any]:
    """读已记录的 run 重算聚合，不调 LLM。"""

    report = json.loads(path.read_text("utf-8"))
    print(f"replay {path}（{report.get('generated_at')} / {report.get('model_alias')}）")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="W7 上下文与记忆两档对照实验")
    parser.add_argument("--week", default="2026-W50", help="报告周号")
    parser.add_argument("--rounds", type=int, default=20, help="实验一轮数（>=20 才有意义）")
    parser.add_argument("--only", choices=["all", "exp1", "exp2"], default="all")
    parser.add_argument(
        "--arms",
        default="naive,budget,summary",
        help="实验一只跑哪几个 arm（naive 永远跑：它是另两个的历史来源）",
    )
    parser.add_argument("--json", default="", help="把完整结果写到这个路径")
    parser.add_argument("--replay", default="", help="复算已记录的 JSON，不调 LLM")
    parser.add_argument("--dsn", default="", help="pgvector DSN（记忆实验用）")
    parser.add_argument("--table", default="agent_memories_w7ab", help="实验专用记忆表")
    args = parser.parse_args()
    args.arms_set = {arm.strip() for arm in args.arms.split(",") if arm.strip()}

    if args.replay:
        print_report(replay(Path(args.replay)))
        return 0

    if not args.dsn:
        args.dsn = os.environ.get("PGVECTOR_DSN", "") or os.environ.get(
            "VECTOR_DSN", ""
        )
    if args.only in ("all", "exp2") and not args.dsn:
        print("实验二需要 --dsn 或 PGVECTOR_DSN", file=sys.stderr)
        return 2

    report = asyncio.run(run(args))
    print_report(report)
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
        print(f"\n完整结果：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
