"""POST /v1/draft —— 产品层真正消费框架的第一个端点（Day5）。

这个端点存在的目的不是"能生成文案"，而是**证明框架的边界画对了**：

1. 产品层只 import 框架的公开入口（`llm.client` / `server.errors` /
   `server.middleware`），**没有一行** adapter、厂商字段、重试、cost 计算；
2. 换模型只改 `models.yaml`（或请求里钉 alias），**本文件零改动**；
3. 错误契约（6 类错误码 + HTTP 状态）直接复用框架的 handler，
   不在这边复制一份——否则 W12 提取时会出现"同一个上游故障两个服务返回不同 code"。

如果哪天这里被迫 `from agent_framework.llm.providers import ...`，就说明边界画错了，
应当场改，而不是等到 W12。

为什么响应里不带 cost：`cost` 依赖 `models.yaml` 的 pricing，是**运营口径**。
把它放进对外响应，等于把"内部计价表"变成对外契约，改一次价格就要发一次版本。
成本留在 `agent_framework.request` 的结构化日志里，由看板去读（W11）。
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import ConfigError
from agent_framework.llm.client import ChatCall, ChatOutcome, LLMGateway
from agent_framework.llm.types import Message, MessageRole
from agent_framework.observability import current_request_id
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()

# prompt 属于产品层：框架不许知道"增长营销"这四个字。
DRAFT_SYSTEM_PROMPT = (
    "你是增长营销团队的文案助手。根据用户给出的 brief 写一版中文短文案："
    "控制在 80 字以内，不使用夸张承诺，不编造数据，不复述 brief。"
)

DEFAULT_MAX_TOKENS = 512


class DraftRequest(BaseModel):
    brief: str = Field(min_length=1, max_length=2000, description="一句话需求")
    task: str = Field(default="chat", description="路由用的 task（见 models.yaml 的 tasks）")
    model: str | None = Field(
        default=None,
        description="仅调试/评测用：钉住模型 alias；生产环境不要传",
    )
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)


class DraftResponse(BaseModel):
    """对外 DTO：手写字段，不继承任何框架类型。

    这里的 `model_dump()` 是安全的——它本身就是"已经投影干净的响应体"，
    与 Day4 禁止的"直接 dump `LLMResponse`"是两件事。
    """

    request_id: str
    text: str
    model: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    usage_estimated: bool
    latency_ms: float

    @classmethod
    def from_outcome(cls, outcome: ChatOutcome, *, request_id: str) -> DraftResponse:
        response = outcome.response
        # `latency_ms` 在契约层是可选的（adapter 直连时可能没设），产品响应要求必填：
        # 缺了就报 0，而不是让 500 或 None 漏到前端。
        latency = round(response.latency_ms or 0.0, 3)

        return cls(
            request_id=request_id or "-",
            text=response.text or "",
            model=outcome.alias,
            finish_reason=str(response.finish_reason),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            usage_estimated=response.usage.estimated,
            latency_ms=latency,
        )


def _gateway(request: Request) -> LLMGateway:
    gateway: Any = getattr(request.app.state, "gateway", None)

    if not isinstance(gateway, LLMGateway):
        # 复用框架的 CONFIG_ERROR（500）：这不是调用方的错，别伪装成 400。
        raise ConfigError("gateway 未初始化：create_app() 必须注入或构建 gateway")

    return gateway


@router.post("/v1/draft")
async def draft(body: DraftRequest, request: Request) -> JSONResponse:
    gateway = _gateway(request)

    call = ChatCall(
        messages=[
            Message(role=MessageRole.SYSTEM, content=DRAFT_SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=body.brief),
        ],
        task=body.task,
        model=body.model,
        opts={"max_tokens": body.max_tokens},
    )

    outcome = await gateway.chat(call)
    payload = DraftResponse.from_outcome(outcome, request_id=current_request_id())

    return JSONResponse(
        payload.model_dump(mode="json"),
        headers={"X-Model-Alias": outcome.alias},
    )
