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

from dataclasses import dataclass

from agent_framework.config import Settings
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
from agent_framework.retrieval import (
    ACLPolicy,
    AssembledContext,
    Principal,
    RetrievalDiagnostics,
    RetrievalPipeline,
    assemble,
    make_acl_check,
)

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

    text: str
    context: CitationContext
    audit: CitationAudit
    assembled: AssembledContext
    diagnostics: RetrievalDiagnostics | None = None
    # 冲突提示：非 None 表示本次 context 里存在互相矛盾的资料，
    # 答案里必须显式指出（见 governance 模块的取舍说明）。
    conflict: str | None = None
    model: str = ""
    # 本次调用的 token 用量。带出来是为了让"每条答案多少钱"可算——
    # 抽样脚本靠它汇总成本，路由面板也靠它；不带就没法回答"RAG 贵不贵"。
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def grounded(self) -> bool:
        """有没有真的基于资料回答。零召回时是 False。"""

        return bool(self.context.citations)

    @property
    def citations(self) -> tuple[Citation, ...]:
        """答案实际引用到的段（不是全部候选段）。"""

        return self.audit.parsed.citations


class KnowledgeAnswerer:
    """把检索流水线与生成串起来。构造一次、全程复用。"""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        gateway: LLMGateway,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._gateway = gateway
        self._settings = settings or Settings()

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

        call = ChatCall(
            messages=[
                Message(role=MessageRole.SYSTEM, content=ANSWER_SYSTEM_PROMPT),
                Message(role=MessageRole.USER, content=user_content),
            ],
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
        )
