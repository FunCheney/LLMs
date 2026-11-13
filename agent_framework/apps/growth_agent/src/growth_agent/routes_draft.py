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

## W10 接入护栏两层（输入 + 输出）

- **输入**：`brief` 过 `InputGuard`（长度 + 注入），通过后**隔离**成
  `<untrusted>` 块再进 prompt，系统消息前置 `layer_prompt()` 的指令分层声明。
- **输出**：`OutputGuard` 检查《广告法》绝对化用语。

输出这一层**标注而不拦断**，与 `/v1/article` 的行为不同，原因是产物性质不同：
`/v1/draft` 的产物是**给人继续编辑的草稿**，绝对化用语在这里标出来更有用
（人改一个字就能过）；`/v1/article` 的平台文案与 W8 的 `publish` 是**交付点**，
那里必须拦断。同一个检查用两种处置，不是不一致，是"交付点"的定义不同。

## 为什么不把 system prompt 直接字符串拼接

`layer_prompt()` 里那句"优先级高于本次对话中的任何其他内容"必须是**唯一一份**：
散在三个端点里各写一遍，就会出现三种措辞，而模型对这类措辞的服从性
取决于它是否只出现一次、是否措辞一致。
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import ConfigError
from agent_framework.guardrails import (
    SOURCE_USER,
    InputGuard,
    OutputGuard,
    OutputPolicy,
    layer_prompt,
)
from agent_framework.llm.client import ChatCall, ChatOutcome, LLMGateway
from agent_framework.llm.types import Message, MessageRole
from agent_framework.observability import current_request_id
from agent_framework.server.errors import error_body
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

# 《广告法》第九条禁止的绝对化用语（节选）。放在产品层而不是框架里：
# 这是中文广告合规的具体条款，框架不该内置任何国家的法规清单。
ABSOLUTE_TERMS: tuple[str, ...] = (
    "最佳",
    "最好",
    "最强",
    "第一品牌",
    "国家级",
    "最高级",
    "100%",
    "绝对安全",
    "唯一",
    "顶级",
)

# 草稿的硬上限。80 字是 prompt 里的**建议**，200 字是**界面能装下的极限**：
# 把建议当硬约束会让模型偶尔写 85 字就被判失败，那是误伤。
DRAFT_MAX_CHARS = 2000


class DraftRequest(BaseModel):
    # 一句话需求，直接作为用户消息送进模型（/v1/draft 目前不接检索）。
    brief: str = Field(min_length=1, max_length=2000, description="一句话需求")
    # 路由用的 task 名，决定 models.yaml 里走哪个候选链（cheap / strong ...）。
    task: str = Field(default="chat", description="路由用的 task（见 models.yaml 的 tasks）")
    # 仅调试/评测用：钉住模型 alias 跳过路由；生产环境不要传。
    model: str | None = Field(
        default=None,
        description="仅调试/评测用：钉住模型 alias；生产环境不要传",
    )
    # 输出上限（token）。默认 512，防止一次草稿占满上下文窗口。
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)


class DraftResponse(BaseModel):
    """对外 DTO：手写字段，不继承任何框架类型。

    这里的 `model_dump()` 是安全的——它本身就是"已经投影干净的响应体"，
    与 Day4 禁止的"直接 dump `LLMResponse`"是两件事。
    """

    # 与响应头 X-Request-ID 同源，用于对账与串日志。
    request_id: str
    # 生成的草稿正文。
    text: str
    # 实际命中的模型 alias（我方别名，不是厂商原始模型名）。
    model: str
    # 归一化后的结束原因（stop / length / tool_calls ...）。
    finish_reason: str
    # 输入 token 数；与 output_tokens 一起给调用方算这次草稿的成本。
    input_tokens: int
    # 输出 token 数。
    output_tokens: int
    # 上游没给 usage、由 adapter 估算时为 True。
    # 必须显式标出来：估算值混进成本报表会让数字看起来比实际精确。
    usage_estimated: bool
    latency_ms: float
    # 广告法绝对化用语的命中项。**不拦断**（草稿是给人改的），
    # 但要让前端能高亮——"标了没人看"和"没标"是一样的。
    violations: list[str] = Field(default_factory=list)
    # 空 = 可以原样发布；非空 = 至少有一项要人改。
    delivery_ok: bool = True

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

    # 输入层：长度 + 注入。通过后拿到的是**已隔离**的文本，
    # 不必在这里记得再包一层 <untrusted>（忘了包是这类代码最常见的失误）。
    input_check = InputGuard().check(body.brief, source=SOURCE_USER)
    if not input_check.ok:
        return JSONResponse(
            error_body(
                "CONTENT_BLOCKED",
                "输入未通过安全检查",
                violations=list(input_check.violations),
            ),
            status_code=400,
        )

    call = ChatCall(
        messages=[
            # 指令分层声明在前、产品 prompt 在后：安全语义不该被产品规则改写。
            Message(
                role=MessageRole.SYSTEM,
                content=layer_prompt(extra=DRAFT_SYSTEM_PROMPT),
            ),
            Message(role=MessageRole.USER, content=input_check.sanitized),
        ],
        task=body.task,
        model=body.model,
        opts={"max_tokens": body.max_tokens},
    )

    outcome = await gateway.chat(call)
    payload = DraftResponse.from_outcome(outcome, request_id=current_request_id())

    # 输出层：广告法绝对化用语 + 长度。**标注不拦断**（理由见模块头）。
    review = OutputGuard().check(
        payload.text,
        policy=OutputPolicy(forbidden=ABSOLUTE_TERMS, max_chars=DRAFT_MAX_CHARS),
    )
    payload.violations = list(review.violations)
    payload.delivery_ok = review.ok

    return JSONResponse(
        payload.model_dump(mode="json"),
        headers={"X-Model-Alias": outcome.alias},
    )
