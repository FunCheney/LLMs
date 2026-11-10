"""有出处的回答（W6 D1 的落地点）：检索 → 组装 → 编号 → 生成 → 引用体检。

这个模块是 W6 那句检查点的执行者：**报告里的每条结论都能指出出自哪份文档的哪一段**。
它把四步串起来，顺序不能换：

1. `assemble`     去冗余 / 预算裁剪 / 首尾放置 —— 必须在编号**之前**；
2. `build_context` 编号并产出 `index -> Citation` 映射 —— 唯一决定编号的地方；
3. 调模型生成，prompt 里带引用规则与冲突提示；
4. `audit_citations` 把答案里的 `[n]` 解回 Citation，判它是否指向标注段。

两个刻意的行为：

- **零召回不调模型。** 没有 context 时让模型"自由发挥"，等于把幻觉写进溯源链路，
  而这一段在抽样里还会被算成"引用缺失 1 条"——不如直接返回固定话术并标注
  `grounded=False`，让归因指向检索层。
- **生成失败不吞。** 模型报错走框架的错误契约（502/504），不在这里降级成
  "资料未覆盖"：把上游故障伪装成"没有资料"会让归因系统去补语料。

prompt 由产品层写（这里是增长营销），框架只提供引用**格式**规则（`CITATION_RULES`）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent_framework.config import Settings
from agent_framework.context import (
    ContextBudget,
    ContextBuilder,
    ContextItem,
    ContextKind,
    TrustLevel,
    allocate,
)
from agent_framework.knowledge import (
    Citation,
    CitationAudit,
    CitationContext,
    audit_citations,
    build_context,
    conflict_note,
    render_prompt,
)
from agent_framework.llm.client import ChatCall, LLMGateway
from agent_framework.llm.types import Message, MessageRole
from agent_framework.observability import current_request_id
from agent_framework.retrieval import (
    ACLPolicy,
    AssembledContext,
    Principal,
    RetrievalDiagnostics,
    RetrievalPipeline,
    assemble,
    make_acl_check,
)

logger = logging.getLogger(__name__)

# W7 槽位化组装器。复用框架的 ContextBuilder：排布规则（system 头、
# 历史中段、问题尾）是全局策略，不该每个产品自己写一份。
_builder = ContextBuilder()

# 产品层的说话方式。引用规则不写在这里——那是框架的 `CITATION_RULES`，
# 换产品只需要换这段 system prompt，引用机制不变。
ANSWER_SYSTEM_PROMPT = (
    "你是增长营销团队的资料助手。只依据用户提供的参考资料回答，"
    "回答用中文、结构清晰、控制在 300 字以内；"
    "每个事实性结论后面必须带上来源编号。"
)

# 零召回时的回答。**不要**让模型来写这句话：没有资料时模型很可能顺手编一个。
NO_CONTEXT_ANSWER = "知识库里没有检索到与这个问题相关的资料，无法给出有出处的回答。"


@dataclass(frozen=True)
class KnowledgeAnswer:
    """一次有出处的回答。字段够 `/v1/answer` 直接用，不再二次加工。"""

    # 最终给用户的答案正文（含 [n] 引用标记）。零召回时是固定话术，不是模型生成。
    text: str
    # 本次编号后的 context 全文 + index→Citation 映射。引用面板与复核都读它，
    # 编号只能从这里取，不能由调用方自己按数组下标再算一遍。
    context: CitationContext
    # 引用体检结果（verdict / matched / unmatched / precision）。
    # 与 text 同源产生，避免"答案与体检报告对不上"。
    audit: CitationAudit
    # 组装过程记录：去冗余删了谁、超预算裁了谁、是否首尾重排。
    # W7 回答"这条为什么没进上下文"靠它的 explain()。
    assembled: AssembledContext
    # 检索链路诊断（降级、耗时、ACL 过滤条数）。零召回或测试桩时可能为 None，
    # 所以可空；报告里 None 与"没有降级"要能区分。
    diagnostics: RetrievalDiagnostics | None = None
    # 冲突提示：非 None 表示本次 context 里存在互相矛盾的资料，
    # 答案里必须显式指出（见 governance 模块的取舍说明）。
    conflict: str | None = None
    # 实际命中的模型 alias（不是厂商原始模型名）：换模型导致引用行为变化时，
    # 报告能按它分组，而不是去猜是哪次路由改了。零召回时为空串。
    model: str = ""
    # 本次调用的 token 用量。带出来是为了让"每条答案多少钱"可算——
    # 抽样脚本靠它汇总成本，路由面板也靠它；不带就没法回答"RAG 贵不贵"。
    input_tokens: int = 0
    output_tokens: int = 0
    # 本次注入的记忆 key 列表（W7）。空 = 没开记忆或没召回到；
    # 这两个字段让"记忆有没有真的进 prompt"在响应里可核对，不靠猜。
    memory_keys: tuple[str, ...] = ()
    # 会话历史注入的条数（W7 工作记忆视图：摘要 + 最近原文）。
    history_messages: int = 0

    @property
    def grounded(self) -> bool:
        """有没有真的基于资料回答。零召回时是 False。"""

        return bool(self.context.citations)

    @property
    def citations(self) -> tuple[Citation, ...]:
        """答案实际引用到的段（不是全部候选段）。"""

        return self.audit.parsed.citations


# W7 记忆/历史槽位的固定预算（token，粗估口径同框架 estimate_tokens）。
# memory 600：3 条 × 一句偏好足够；调大前先想清楚"更多记忆"是否真的
# 提升答案——注入的每条记忆都在稀释资料的相对注意力。
MEMORY_SLOT_TOKENS = 600
HISTORY_SLOT_TOKENS = 2000
SYSTEM_SLOT_TOKENS = 400


def _answer_budget(settings: Settings, *, token_budget: int) -> ContextBudget:
    """回答链路的槽位预算。retrieved 跟随请求/配置（可被 token_budget
    覆盖调试），window 取五槽之和——槽位预算加起来必须装进窗口，
    这是 ContextBudget.validate 的前提。

    tool_results=0：回答链路没有工具结果（W9 才会有），显式关掉
    这个槽位，省下 1000 token 给检索。
    """
    retrieved = token_budget or settings.retrieval_token_budget
    return ContextBudget(
        window=SYSTEM_SLOT_TOKENS + retrieved + MEMORY_SLOT_TOKENS + HISTORY_SLOT_TOKENS,
        system=SYSTEM_SLOT_TOKENS,
        retrieved=retrieved,
        memory=MEMORY_SLOT_TOKENS,
        history=HISTORY_SLOT_TOKENS,
        tool_results=0,
    )


class KnowledgeAnswerer:
    """把检索流水线与生成串起来。构造一次、全程复用。"""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        gateway: LLMGateway,
        *,
        settings: Settings | None = None,
        memory: Any | None = None,
    ) -> None:
        # 检索流水线（进程内单例）。按请求传入 principal，主体不进这里。
        self._pipeline = pipeline
        # 生成用的网关，复用它的重试 / fallback / 计费 / trace。
        self._gateway = gateway
        # 产品侧配置（top_k、token_budget、answer_temperature、治理开关等）。
        # 允许注入是为了评测能钉住参数，不靠改全局环境变量。
        self._settings = settings or Settings()
        # W7 记忆运行时（MEMORY=governed 时由 runtime.py 装配注入）。
        # None = 记忆关闭（两档实验的 off 档就是它——不是"坏了"，是对照组）。
        self._memory = memory

    @property
    def settings(self) -> Settings:
        return self._settings

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        principal: Principal | None = None,
        token_budget: int | None = None,
        session_id: str | None = None,
    ) -> KnowledgeAnswer:
        resolved = self._settings
        # ACL 关闭时 principal 一定是 None（路由侧决定），这里不再判断开关：
        # 传了 principal 就是"要过滤"，没传就是"不启用"。两处判断会互相覆盖。
        policy = ACLPolicy(allow_unlabeled=resolved.retrieval_acl_allow_unlabeled)
        acl_check = make_acl_check(principal, policy=policy)

        outcome = await self._pipeline.run(
            question,
            top_k=top_k or resolved.retrieval_top_k,
            acl_check=acl_check,
        )

        assembled = assemble(
            outcome.chunks,
            token_budget=token_budget or resolved.retrieval_token_budget,
            dedupe_threshold=resolved.retrieval_dedupe_threshold,
            attention_ordering=resolved.retrieval_attention_ordering,
            reserve_tokens=resolved.retrieval_context_reserve_tokens,
        )
        context = build_context(
            assembled.chunks,
            max_chars_per_chunk=resolved.retrieval_citation_max_chars,
        )

        if not context.citations:
            return KnowledgeAnswer(
                text=NO_CONTEXT_ANSWER,
                context=context,
                audit=audit_citations(NO_CONTEXT_ANSWER, context),
                assembled=assembled,
                diagnostics=outcome.diagnostics,
            )

        numbers = {citation.chunk_id: citation.index for citation in context.citations}
        note = conflict_note(assembled.chunks, numbers) if resolved.retrieval_governance else None

        user_content = render_prompt(context, question=question)
        if note:
            # 冲突提示放在问题之前：放在最后时模型容易把它当成"另一条资料"，
            # 而它其实是一条指令。
            user_content = f"{user_content}\n\n{note}"

        # -- W7：记忆召回与会话视图（都是可选增强，失败/缺省不挡回答） ----- #
        memory_items: list[Any] = []
        history: list[Message] = []
        memory = self._memory
        if memory is not None and resolved.memory == "governed":
            # 召回按"当前问题"：记忆里最相关的是品牌调性/禁用词这类
            # 偏好项，与检索命中的资料互不抢预算（各自的槽位）。
            memory_items = await memory.recall_for_query(question)
            if session_id:
                history = await memory.session_view(session_id)

        messages = self._build_messages(
            user_content, question, memory_items, history,
            token_budget=token_budget,
        )

        call = ChatCall(
            messages=messages,
            task=resolved.answer_task,
            # 温度显式下发而不是留给厂商默认：默认值各厂商不同（0.7/1.0），
            # 换个模型答案风格就变，引用行为（尤其是"没资料就编编号"）也跟着变。
            opts={
                "max_tokens": resolved.answer_max_tokens,
                "temperature": resolved.answer_temperature,
            },
        )
        chat = await self._gateway.chat(call)
        text = chat.response.text or ""
        # 用 getattr 而不是直接取 `chat.response.usage`：测试里的假网关返回
        # SimpleNamespace，没有 usage 字段。为统计用量而让整个回答 500，
        # 是本末倒置——用量是运营信息，缺了就当 0。
        usage = getattr(chat.response, "usage", None)

        # -- W7：会话落地（追加本轮 → 超阈值时压缩 → 显式指令才提取） ----- #
        if memory is not None and session_id and resolved.memory == "governed":
            await memory.record_turn(
                session_id, question, text, request_id=current_request_id() or None
            )

        return KnowledgeAnswer(
            text=text,
            context=context,
            audit=audit_citations(text, context),
            assembled=assembled,
            diagnostics=outcome.diagnostics,
            conflict=note,
            model=chat.alias,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            memory_keys=tuple(getattr(item, "key", "") for item in memory_items),
            history_messages=len(history),
        )

    def _build_messages(
        self,
        user_content: str,
        question: str,
        memory_items: list[Any],
        history: list[Message],
        *,
        token_budget: int | None = None,
    ) -> list[Message]:
        """无记忆/无历史时**逐位保持 W6 的两消息结构**（回归基线不变）；
        有记忆或历史时走 W7 的槽位化组装（system 头 + 历史中段 + 记忆/资料/问题尾）。

        记忆块的信任级是 user（会话沉淀的偏好），**不进 system**：
        混进 system 会被模型当成系统指令而不是历史信息（W10 注入防护
        的同一条原则，提前在 Context 层立住）。
        """
        if not memory_items and not history:
            return [
                Message(role=MessageRole.SYSTEM, content=ANSWER_SYSTEM_PROMPT),
                Message(role=MessageRole.USER, content=user_content),
            ]

        items: list[ContextItem] = [
            ContextItem(
                id="answer-system",
                content=ANSWER_SYSTEM_PROMPT,
                kind=ContextKind.SYSTEM,
                source="growth_agent/answer.py",
                trust=TrustLevel.SYSTEM,
            ),
        ]
        for item in memory_items:
            items.append(
                ContextItem(
                    id=str(getattr(item, "key", "memory")),
                    content=f"{getattr(item, 'key', '')}: {getattr(item, 'value', '')}",
                    kind=ContextKind.MEMORY,
                    source=f"session:{getattr(item, 'source_session', '')}",
                    # 召回分数归一化 Confidence 直接当相关性用（0–1 同量纲）。
                    score=float(getattr(item, "confidence", 0.5)),
                    trust=TrustLevel.USER,
                )
            )
        for message in history:
            content = message.content if isinstance(message.content, str) else ""
            items.append(
                ContextItem(
                    id=f"history-{len(items)}",
                    content=content,
                    kind=ContextKind.HISTORY,
                    source="session-history",
                    message=message,
                    score=0.4,  # 历史=上下文胶水：低分，超预算时最先让位
                )
            )

        allocated = allocate(
            items,
            _answer_budget(self._settings, token_budget=token_budget or 0),
        )
        # 提问已在 render_prompt 里（user_content 尾部），不再重复追加。
        return _builder.build(allocated, question="", retrieved_text=user_content)
