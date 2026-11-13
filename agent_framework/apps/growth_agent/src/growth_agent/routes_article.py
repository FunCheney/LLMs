"""POST /v1/article —— 主题进 → SEO 文章 + 三平台文案出（W10 D1/D2）。

与 `/v1/draft`（写一段文案）和 `/v1/answer`（答一个问题）的分工：
这是**唯一"一条请求跑完四段 + 三平台扇出"的端点**，也是 capstone 的
主线演示路径（W10 Gate #1 就是它）。

两处刻意的取舍：

1. **平台改写失败不让整条请求 500。** 文章本身可能是好的，坏的是某个平台的
   改写（超长、缺话题标签）。把它变成 500 等于让运营丢掉已经生成好的稿子重跑，
   而重跑一次是四次模型调用。所以按平台返回错误、顶层给 `partial`。
   但这**不是**"静默失败"：失败条带 `error` 与 `violations`，
   前端能直接显示"小红书这条没通过：缺话题标签"。

2. **`coverage` 进响应。** 引用体检（引了几个、有没有悬空编号、几句事实没出处）
   在服务端算一次，比让调用方自己正则去数可靠。W6 的 `/v1/answer`
   也是这么做的（`audit` 字段），口径保持一致。

文章本身失败（没证据 / 引用体检不过 / 品牌违规）走 `SEOError`，
由 `install_error_handlers` 之外的**本文件**翻译成 422——它既不是上游故障，
也不是参数校验失败，而"生成结果不合格"，用 422 语义最准（请求合法但无法处理）。
"""

from __future__ import annotations

from typing import Any, Literal

from agent_framework.guardrails import OutputPolicy
from agent_framework.knowledge import citation_payload
from agent_framework.observability import current_request_id
from agent_framework.server.errors import error_body
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from growth_agent.pipeline import (
    PlatformFailure,
    SEOError,
    StyleError,
    as_payload,
    failure_payload,
)
from growth_agent.runtime import ensure_guardrails, ensure_seo

router = APIRouter()

PlatformName = Literal["xiaohongshu", "x", "linkedin"]

# 这些 stage 的失败属于"生成结果不合格"，不是上游故障，也不是参数错。
# 用 422（Unprocessable Content）：请求本身合法，是处理结果不能交付。
_UNPROCESSABLE_STAGES = frozenset({"draft", "evidence", "outline"})


class ArticleRequest(BaseModel):
    # 主题。1–200 字：再长说明调用方没做拆解，应该先过 W9 的调研规划器。
    topic: str = Field(min_length=1, max_length=200, description="文章主题")
    # 要出哪几个平台的文案。空列表 = 只要文章，不扇出（省 3 次调用）。
    platforms: list[PlatformName] = Field(
        default_factory=list, description="要改写的平台；空 = 只出文章"
    )


class ArticleResponse(BaseModel):
    request_id: str
    topic: str
    title: str
    outline: list[str]
    body: str
    # 引用体检：markers / resolved / dangling / unpublished / candidates / ok。
    coverage: dict[str, Any]
    # 编号 → 出处的投影（与 /v1/answer 同一个 citation_payload）。
    citations: list[dict[str, Any]]
    # description / slug / tags / json_ld。
    meta: dict[str, Any]
    keywords: dict[str, Any]
    # 四段的耗时与 token（含各平台的别名与降级来源），供 W11 成本看板拆分。
    stages: list[dict[str, Any]]
    # 每个平台一条。失败条带 error/violations，不带 text。
    platforms: list[dict[str, Any]] = Field(default_factory=list)
    # 任一平台失败即为 True。**不等于请求失败**（HTTP 仍是 200）。
    partial: bool = False


def _coverage_payload(coverage: Any) -> dict[str, Any]:
    return {
        "markers": list(coverage.markers),
        "resolved": list(coverage.resolved),
        "dangling": list(coverage.dangling),
        "unpublished": list(coverage.unpublished),
        "candidates": coverage.candidates,
        "ok": coverage.ok,
    }


def _trace_payload(trace: Any) -> dict[str, Any]:
    return {
        "stage": trace.name,
        "ms": round(trace.ms, 1),
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "alias": trace.alias,
        "fallback_from": trace.fallback_from,
    }


@router.post("/v1/article")
async def article(body: ArticleRequest, request: Request) -> JSONResponse:
    pipeline, rewriter = await ensure_seo(request.app)
    guards = await ensure_guardrails(request.app)

    # 为什么这条链路要过护栏编排器，而不是直接 `pipeline.run(topic)`：
    # 主题是**用户输入**，正文是**要交付的产物**，两头都是 W10 的攻击面。
    # 走 `run()` 之后，注入在花掉四次模型调用之前就被拦，产物在返回前
    # 至少要过"有没有引用标号"这一关。
    #
    # 注意 `execute` 拿到的是**已隔离**的 topic（`<untrusted>` 包裹）。
    # SEO 的 prompt 把它当主题用，隔离标签不会影响关键词抽取——它只是
    # 让模型知道"这部分是数据"。
    try:
        run = await guards.run(
            body.topic,
            execute=pipeline.run,
            policy=OutputPolicy(require_citations=True),
            render=lambda draft: draft.body,
        )
    except SEOError as exc:
        status = 422 if exc.stage in _UNPROCESSABLE_STAGES else 500
        return JSONResponse(
            error_body(
                "CONTENT_UNQUALIFIED" if status == 422 else "ARTICLE_STAGE_FAILED",
                str(exc),
                stage=exc.stage,
                **exc.detail,
            ),
            status_code=status,
        )

    draft = run.value

    platform_payloads: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = [_trace_payload(trace) for trace in draft.stages]
    partial = False

    if body.platforms:
        wanted = tuple(dict.fromkeys(body.platforms))  # 去重但保序
        try:
            outcomes = await rewriter.rewrite_all(draft, wanted)
        except StyleError as exc:
            # 走到这里只有一种情况：平台名不认识（参数错误 → 400）。
            # 单平台的生成失败在 rewrite_all 内部已收成 PlatformFailure。
            return JSONResponse(
                error_body("BAD_REQUEST", str(exc), platform=exc.platform), status_code=400
            )

        for outcome in outcomes:
            if isinstance(outcome, PlatformFailure):
                # 失败也算一条结果：运营要能看到"哪条没出来、为什么"。
                partial = True
                platform_payloads.append(failure_payload(outcome))
            else:
                # `repairs` 非空表示这条是靠截断/删词才过的——产物可用，
                # 但不算"一次过"。它不进 `partial`：`partial` 的语义是
                # "有产物缺失"，把"修过"混进去会让这个字段失去信号。
                platform_payloads.append(as_payload(outcome))

        # 平台阶段的 trace 也要进 stages：成本看板按 stage 聚合，
        # 少了这三条就会把"改写贵"算到"成文贵"头上。
        for outcome in outcomes:
            trace = getattr(outcome, "trace", None)
            if trace is not None:
                traces.append(_trace_payload(trace))

    response = ArticleResponse(
        request_id=current_request_id() or "-",
        topic=draft.topic,
        title=draft.title,
        outline=list(draft.outline),
        body=draft.body,
        coverage=_coverage_payload(draft.coverage),
        citations=citation_payload(draft.citations),
        meta=draft.meta,
        keywords={
            "primary": draft.keywords.primary,
            "secondary": list(draft.keywords.secondary),
            "intent": draft.keywords.intent,
        },
        stages=traces,
        platforms=platform_payloads,
        partial=partial,
    )
    return JSONResponse(response.model_dump(mode="json"))
