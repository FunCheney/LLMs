"""W10 内容流水线的测试替身。

为什么不用真 `LLMGateway`：四段串行 + 三平台扇出，一个用例要 7 次模型调用，
而且每次都要证明"第几个 stage 拿到了哪段回放"——真 gateway 的 fallback/重试
逻辑会把这个顺序搅乱（这恰恰是它在生产里该做的事）。

为什么不做成"按 prompt 关键字匹配回放"：那样提示词一改，测试全绿但行为全变，
`test_*_stage` 就退化成了 `test_prompt_contains_string`。按**调用序**回放，
任何新增/删除调用都会立刻让用例失败——这才是想要的那种脆弱。

`ScriptedGateway` 只实现 `chat`，不继承 `LLMGateway`：流水线对网关的依赖
只有一个方法签名，替身越窄，"流水线偷偷依赖了网关内部状态"就越不可能发生。
"""

from __future__ import annotations

from agent_framework.llm.client import ChatCall, ChatOutcome
from agent_framework.llm.errors import UpstreamServerError
from agent_framework.llm.types import (
    FinishReason,
    LLMResponse,
    Message,
    MessageRole,
    Usage,
)


def response(
    text: str,
    *,
    alias: str = "scripted",
    input_tokens: int = 100,
    output_tokens: int = 50,
    estimated: bool = False,
) -> LLMResponse:
    return LLMResponse(
        id="resp-scripted",
        model="scripted-model",
        provider="scripted",
        message=Message(role=MessageRole.ASSISTANT, content=text),
        finish_reason=FinishReason.STOP,
        raw_finish_reason="stop",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated=estimated,
        ),
    )


class ScriptedGateway:
    """按调用序吐回放。元素可以是字符串（正文）或异常实例。"""

    def __init__(
        self,
        replies: list[str | Exception],
        *,
        alias: str = "scripted",
        fallback_from: str | None = None,
    ) -> None:
        self._replies = list(replies)
        self._alias = alias
        self._fallback_from = fallback_from
        self.calls: list[ChatCall] = []

    @property
    def count(self) -> int:
        return len(self.calls)

    async def chat(self, call: ChatCall) -> ChatOutcome:
        self.calls.append(call)
        if not self._replies:
            raise AssertionError("ScriptedGateway: 回放用尽")

        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item

        return ChatOutcome(
            response=response(item, alias=self._alias),
            alias=self._alias,
            retry_count=0,
            fallback_from=self._fallback_from,
        )

    def prompts(self) -> list[str]:
        """取每次调用的最后一条 user 消息正文。断言"注入了什么"时用它。"""

        out: list[str] = []
        for call in self.calls:
            for message in reversed(call.messages):
                if message.role == MessageRole.USER:
                    out.append(str(message.content))
                    break
        return out


def upstream_error(message: str = "上游 500") -> Exception:
    return UpstreamServerError(message, provider="scripted", model="scripted-model")


__all__ = ["ScriptedGateway", "response", "upstream_error"]
