"""降级策略（W11 D2）：当上游已经救不回来时，给用户什么。

## 降级不是"再试一次"

`LLMGateway` 已经做了重试与候选 fallback（ADR-003）。当它**仍然**抛错时，
说明"再试一次"这条路已经走完了。所以本模块只做三件事，每件最多一次：

| 顺序 | 手段 | 命中时用户看到 | 成本 |
|---|---|---|---|
| 1 | 语义缓存（W11 D5，可注入） | 上一次的答案 + `cache_hit` 标记 | 0 |
| 2 | 兜底话术（可选） | 明确的"当前不可用"说明 | 0 |
| 3 | 原样抛出 | 真实错误 | — |

**不做**"换一个更便宜的档再试"——那是 gateway 的候选链该做的事。
在这里再来一次，就是第二次重试，直接违反 ADR-003。

## 为什么"兜底话术"要显式开关，且默认关

一个"当前请求较多，请稍后再试"的假回答，看起来比 503 友好，
但它把三件事同时毁掉了：

1. 用户以为拿到了答案（其实是空话），不会重试；
2. SLO 的失败计数里少了一条（可用性看起来更好）；
3. 复盘时 trace 里只有一次成功。

所以默认**不**兜底，抛 `503`。要开的话，返回的 outcome 必须带
`degraded=True`——这三个字是下游（日志、SLO、前端）唯一能识别"这不是真答案"的信号。

## 标记优先于措辞

`ChatOutcome.degraded` / `cache_hit` 是**结构化**标记，不靠文案判断。
靠文案判断的降级，在换一次措辞之后就失效了，而失效时没有任何测试会红。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..llm.client import ChatCall, ChatOutcome
from ..llm.errors import LLMError
from ..llm.types import FinishReason, LLMResponse, Message, MessageRole

__all__ = [
    "DegradationPolicy",
    "DegradationResult",
    "DegradedAnswer",
]

# 兜底话术。**只用于"明确告诉用户现在不行"**，不用于伪装成功。
CANNED_MESSAGE = "这个请求暂时没处理成功（上游模型不可用）。稍后重试通常就好了。"


@dataclass(frozen=True)
class DegradedAnswer:
    """一次降级的结果。`path` 是命中的手段，进日志便于统计各手段的真实使用率。"""

    outcome: ChatOutcome
    path: str  # cache | canned

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "alias": self.outcome.alias,
            "degraded": self.outcome.degraded,
            "cache_hit": self.outcome.cache_hit,
        }


@dataclass(frozen=True)
class DegradationResult:
    """`answered=False` 表示"降级也没能给东西"，调用方应当把原始错误抛出去。"""

    answered: bool
    answer: DegradedAnswer | None = None
    reason: str = ""


# 缓存的形态由调用方决定（W11 D5 的 SemanticCache 是异步实现）。
# 这里只约定"给一个 call 拿一个 outcome 或 None"，不绑定具体存储。
CacheLookup = Callable[[ChatCall], Awaitable[ChatOutcome | None]]


@dataclass
class DegradationPolicy:
    """按顺序尝试降级手段。每个手段**只试一次**，失败即进入下一个。"""

    cache: CacheLookup | None = None
    allow_canned: bool = False
    canned_text: str = CANNED_MESSAGE
    alias: str = "degraded"
    # 记录每次降级的路径，给报告用。不落盘也能从日志看，但报告需要计数。
    _hits: dict[str, int] = field(default_factory=dict, init=False)

    @property
    def hits(self) -> dict[str, int]:
        return dict(self._hits)

    async def fallback(self, call: ChatCall, *, error: LLMError | None = None) -> DegradationResult:
        """`error` 只用于日志与决策（例如"鉴权错误不兜底"），不参与重试判定。"""

        if self.cache is not None:
            cached = await self.cache(call)
            if cached is not None:
                # 缓存命中也要复制一份再打标记：直接改传入对象会让同一份缓存
                # 被后续请求共享，`degraded` 字段会串味。
                marked = _mark(cached, degraded=True, cache_hit=True)
                self._hits["cache"] = self._hits.get("cache", 0) + 1
                return DegradationResult(
                    answered=True, answer=DegradedAnswer(outcome=marked, path="cache")
                )

        if self.allow_canned:
            canned = _canned(call, self.alias, self.canned_text, error)
            self._hits["canned"] = self._hits.get("canned", 0) + 1
            return DegradationResult(
                answered=True, answer=DegradedAnswer(outcome=canned, path="canned")
            )

        return DegradationResult(
            answered=False,
            reason="缓存未命中且未开启兜底话术；原错误应抛给调用方",
        )


def _mark(outcome: ChatOutcome, *, degraded: bool, cache_hit: bool) -> ChatOutcome:
    return ChatOutcome(
        response=outcome.response,
        alias=outcome.alias,
        retry_count=outcome.retry_count,
        fallback_from=outcome.fallback_from,
        degraded=degraded,
        cache_hit=cache_hit,
    )


def _canned(
    call: ChatCall, alias: str, text: str, error: LLMError | None
) -> ChatOutcome:
    """造一个"明说不行"的响应。

    `finish_reason` 用 `UNKNOWN` + `raw_finish_reason="degraded"`，
    **不扩 `FinishReason` 枚举**：那是与 adapter 共享的契约层，
    而"降级"是客户端层的运营概念（与 `retry_count` 同类），
    扩进去会让契约层为客户端实现背锅。
    下游判"正常完成"靠 `finish_reason == stop`，所以这条也不会混进正常率。
    """

    response = LLMResponse(
        message=Message(role=MessageRole.ASSISTANT, content=text),
        finish_reason=FinishReason.UNKNOWN,
        raw_finish_reason="degraded",
        model=alias,
    )
    return ChatOutcome(
        response=response,
        alias=alias,
        degraded=True,
        cache_hit=False,
        # 把上游真实的错误类型带上：报告里要能回答"这次降级是因为超时还是 429"。
        fallback_from=type(error).__name__ if error is not None else None,
    )
