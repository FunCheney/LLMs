"""POST /v1/answer —— 有出处的回答（W6）。

与 `/v1/draft` 的分工，不是"重复端点"，是两种产物：

- `/v1/draft` 写文案：调性、长度、不编数据，产物本身不需要逐句出处；
- `/v1/answer` 回答事实：**每条结论必须带 `[n]`**，响应里给出 `[n]` 对应的
  `chunk_id / 文档 / 页码 / 摘录`，可点开原文核对。

把它们合成一个端点，就得让"写 slogan"也背上引用义务（模型会开始给
"更懂你的选择"标出处），或者让"回答事实"可以不标——两种都糟。
capstone 的 SEO 稿要的是：先 `/v1/answer` 拿到带出处的结论，再用 `/v1/draft`
把它们写成文案。

响应里的 `audit` 是**给运维看的**，不是给终端用户的：`verdict != ok` 表示这次
回答的引用不成立（没标、标错、出处缺失），调用方可以据此拒稿或重试。
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import Settings
from agent_framework.knowledge import citation_payload
from agent_framework.observability import current_request_id
from agent_framework.retrieval import Principal
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from growth_agent.runtime import ensure_answerer

router = APIRouter()


class AnswerRequest(BaseModel):
    # 用户原始问题。1–2000 字：空串不必走检索，超长则不该直接塞给模型。
    question: str = Field(min_length=1, max_length=2000, description="要回答的问题")
    # 覆盖默认召回条数（仅调试/评测）。None = 用 Settings.retrieval_top_k。
    top_k: int | None = Field(default=None, ge=1, le=50, description="覆盖默认召回条数")
    # 覆盖默认 context token 预算（调试用）。用于复现"第 6 条为什么没进上下文"。
    token_budget: int | None = Field(
        default=None, ge=200, le=32000, description="覆盖默认 context 预算（调试用）"
    )
    # W7：会话 id。传入后本轮问答会追加进会话历史（工作记忆），
    # 下轮自动带"摘要 + 最近原文"；显式记忆指令（"记住…"）会触发提取。
    # None = 单轮问答（W6 行为，回归基线不受影响）。
    session_id: str | None = Field(
        default=None, min_length=1, max_length=64, description="多轮会话 id（可选）"
    )


class AnswerResponse(BaseModel):
    """手写字段，不 dump 内部 dataclass。理由同 `/v1/draft`：字段一旦出去就是契约。"""

    # 与响应头 X-Request-ID 同源，调用方用它对账"我看到的是哪一次"。
    request_id: str
    # 答案正文（含 [n]）。零召回时是固定的 NO_CONTEXT_ANSWER，不是模型生成。
    text: str
    # 实际命中的模型 alias；零召回时为空串（没调模型）。
    model: str
    # 是否真的基于资料回答（等价于 context 里有没有块）。
    grounded: bool
    # 答案实际引用到的段，citation_payload() 手写投影的字典列表。
    citations: list[dict[str, Any]]
    # 引用体检结果：verdict / reasons / matched / unmatched / precision。
    audit: dict[str, Any]
    # 检索诊断：降级标记、各段耗时、ACL 过滤条数、候选数。
    retrieval: dict[str, Any]
    # 冲突提示文本；None 表示本次 context 里没有互相矛盾的资料。
    conflict: str | None = None
    # W7：本次注入的记忆 key（空 = 未开启记忆或未召回到）与历史条数。
    # 让"记忆有没有真的进 prompt"在响应里可核对——记忆是可关的能力，
    # 关掉时这里恒为空，两档实验（MEMORY=off/governed）看的就是这两列。
    memory_keys: list[str] = Field(default_factory=list)
    history_messages: int = 0


def _principal(request: Request, settings: Settings) -> Principal | None:
    """从请求头取主体。**只有显式开启 ACL 才返回主体**。

    用请求头是因为 W6 只做最小实现，真鉴权在 W10。但"最小"不等于"随便"：
    开启 ACL 后没有头的请求按匿名主体处理（默认拒绝未标注文档），
    而不是当成"没开 ACL"——否则任何人删掉一个 header 就绕过了权限。
    """

    if not settings.retrieval_acl_enabled:
        return None

    subject = (request.headers.get("X-Principal-Id") or "").strip()
    raw_groups = request.headers.get("X-Principal-Groups") or ""
    groups = frozenset(
        item.strip() for item in raw_groups.split(",") if item.strip()
    )
    if not subject:
        return Principal.anonymous()
    return Principal(subject_id=subject, groups=groups)


def _retrieval_payload(diagnostics: Any) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    return {
        "queries": list(diagnostics.queries),
        "candidates": diagnostics.candidates,
        "filtered_out": diagnostics.filtered_out,
        "acl_applied": diagnostics.acl_applied,
        "reranked": diagnostics.reranked,
        "degraded": list(diagnostics.degraded),
        "rewrite_ms": diagnostics.rewrite_ms,
        "recall_ms": diagnostics.recall_ms,
        "rerank_ms": diagnostics.rerank_ms,
        "total_ms": diagnostics.total_ms,
    }


@router.post("/v1/answer")
async def answer(body: AnswerRequest, request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    answerer = await ensure_answerer(request.app)

    result = await answerer.answer(
        body.question,
        top_k=body.top_k,
        token_budget=body.token_budget,
        principal=_principal(request, settings),
        session_id=body.session_id,
    )

    payload = AnswerResponse(
        request_id=current_request_id() or "-",
        text=result.text,
        model=result.model,
        grounded=result.grounded,
        citations=citation_payload(result.citations),
        audit={
            "verdict": result.audit.verdict,
            "reasons": list(result.audit.reasons),
            "cited": list(result.audit.parsed.indices),
            "unknown": list(result.audit.parsed.unknown),
            # `unmatched` 是"引了但没落在标注段上"的编号。运行时没有 ground truth，
            # 所以这里是空的；抽样脚本才会填。留这个字段是为了让两边共用一套
            # 审计结构，而不是让评测另造一个 DTO。
            "unmatched": list(result.audit.unmatched),
            # 候选段总数与最终进 context 的段数：引用数远小于它时，
            # 说明"检索到了但没用上"，与"没检索到"是两种不同的问题。
            "candidates": len(result.context.citations),
            "tokens": result.assembled.tokens,
            "dropped_duplicates": len(result.assembled.duplicate_of),
            "dropped_over_budget": len(result.assembled.dropped_over_budget),
        },
        retrieval=_retrieval_payload(result.diagnostics),
        conflict=result.conflict,
        memory_keys=list(result.memory_keys),
        history_messages=result.history_messages,
    )

    return JSONResponse(payload.model_dump(mode="json"))
