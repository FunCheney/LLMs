"""异常层次与错误映射（契约见 docs/architecture.md §3.6）。

两条纪律：

1. **映射只写在这里**。adapter 只负责从响应体里取出 (status_code, code, message)，
   然后调用 map_provider_error()。绝不在 adapter 里写 if status == 429。
2. **对外 HTTP 状态码由这里的类决定**，不由 adapter 决定。上游鉴权失败映射 502
   而不是 401（ADR-005）——我方 401 只表示"调用方没通过我方鉴权"。
"""

from dataclasses import dataclass
from typing import Any

# 各家的"欠费"落在不同状态码上（OpenAI 429 / DeepSeek 402 / Qwen 400-403 / Anthropic 400），
# 按状态码分类必然出错，只能按 code 或错误文案再兜一层。
_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",
        "arrearage",
        "bil.userarrears",
    }
)

_AUTH_CODES = frozenset(
    {
        "invalid_api_key",
        "authentication_error",
        "permission_denied",
        "account_deactivated",
    }
)

_RATE_LIMIT_CODES = frozenset(
    {
        "rate_limit_exceeded",
        "throttling",
        "requests rate limit exceeded",
        "allocated quota exceeded",
        "request rate increased too quickly",
        "overloaded_error",
    }
)

_CONTENT_FILTER_CODES = frozenset(
    {
        "content_filter",
        "content_policy_violation",
        "refusal",
    }
)

# Anthropic 的余额不足没有独立 code，只能靠文案识别。
# W1 先按文案兜底，Day5 真机录制后收敛成固定 code。
_QUOTA_MESSAGE_HINTS = (
    "credit balance",
    "insufficient balance",
    "arrearage",
    "欠费",
    "余额",
)


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """一次 LLM 错误的公共上下文。"""

    provider: str | None = None
    model: str | None = None
    status_code: int | None = None
    retry_after: float | None = None
    request_id: str | None = None


class LLMError(Exception):
    """LLM 客户端基础异常。"""

    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        model: str | None = None,
        retry_after: float | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
        raw_error: Any = None,
    ) -> None:
        super().__init__(message)

        self.context = ErrorContext(
            provider=provider,
            model=model,
            status_code=status_code,
            retry_after=retry_after,
            request_id=request_id,
        )

        # 直接暴露公共字段，方便日志与 HTTP 响应复用。
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retry_after = retry_after
        self.request_id = request_id

        # 未显式指定时，采用异常类的默认策略。
        self.retryable = self.default_retryable if retryable is None else retryable

        # 可选的原始错误信息，供日志与排查使用。
        self.raw_error = raw_error


class UpstreamAuthError(LLMError):
    """上游鉴权失败或欠费（ADR-005：对外 502，不是 401）。"""


class RateLimitError(LLMError):
    """请求频率或配额限流（可重试）。"""

    default_retryable = True


class UpstreamTimeoutError(LLMError):
    """连接或首 token 超时（可重试）。"""

    default_retryable = True


class BadRequestError(LLMError):
    """请求参数无效或不符合 API 要求（不可重试）。"""


class ContentFilteredError(LLMError):
    """内容被上游拦截（不可重试）。"""


class UpstreamServerError(LLMError):
    """上游服务异常、资源不足或网络错误（可重试）。"""

    default_retryable = True


class CapabilityError(LLMError):
    """客户端或模型不支持调用方要求的能力（不可重试）。"""


# 内部异常 → (对外 HTTP, 对外 code)。architecture §3.6 是这张表的唯一来源。
# 放在 errors.py 而不是 server/：客户端层打日志也要用 code，两处各写一份必然漂移。
ERROR_HTTP: dict[type[LLMError], tuple[int, str]] = {
    UpstreamAuthError: (502, "UPSTREAM_AUTH"),
    RateLimitError: (429, "RATE_LIMITED"),
    UpstreamTimeoutError: (504, "UPSTREAM_TIMEOUT"),
    BadRequestError: (400, "BAD_REQUEST"),
    ContentFilteredError: (422, "CONTENT_FILTERED"),
    UpstreamServerError: (502, "UPSTREAM_ERROR"),
    CapabilityError: (400, "CAPABILITY_UNSUPPORTED"),
}

# 未登记的异常一律按上游故障处理：宁可 502 可重试，也不要把未知错误伪装成 400
# 让调用方以为是自己参数写错了。
_HTTP_FALLBACK: tuple[int, str] = (502, "UPSTREAM_ERROR")


def http_status_for(exc: LLMError) -> int:
    """异常 → 对外 HTTP 状态码。"""

    return ERROR_HTTP.get(type(exc), _HTTP_FALLBACK)[0]


def error_code_for(exc: LLMError) -> str:
    """异常 → 对外/日志里的稳定 code。"""

    return ERROR_HTTP.get(type(exc), _HTTP_FALLBACK)[1]


def map_status_code(status_code: int) -> type[LLMError]:
    """将 HTTP 状态码映射到内部异常类（只做分类，不创建实例）。"""

    if status_code in (401, 403):
        return UpstreamAuthError

    # 402 是 DeepSeek 的"余额不足"，语义等价于 401/403 的上游凭证失效。
    if status_code == 402:
        return UpstreamAuthError

    if status_code == 429:
        return RateLimitError

    if status_code in (408, 504):
        return UpstreamTimeoutError

    # Anthropic 用 529 表示过载，与 5xx 同语义且可重试。
    if status_code >= 500:
        return UpstreamServerError

    if 400 <= status_code < 500:
        return BadRequestError

    return UpstreamServerError


def _matches_message_hint(message: str) -> bool:
    """欠费类错误没有统一 code 时，按文案兜底识别。"""

    lowered = message.lower()

    return any(hint in lowered for hint in _QUOTA_MESSAGE_HINTS)


def map_provider_error(
    *,
    provider: str,
    message: str = "LLM request failed",
    status_code: int | None = None,
    provider_code: str | None = None,
    model: str | None = None,
    retry_after: float | None = None,
    request_id: str | None = None,
    raw_error: Any = None,
    extra_code_map: dict[str, type[LLMError]] | None = None,
) -> LLMError:
    """统一将厂商错误映射为内部异常。

    判定顺序（顺序本身是设计的一部分）：

    1. 调用方通过 extra_code_map 注入的厂商特有 code（来自 models.yaml）
    2. 欠费 code / 欠费文案 —— 必须排在 429 之前，否则欠费会被当成限流无限重试
    3. 鉴权 code
    4. 内容拦截 code
    5. 其余按 HTTP 状态码分类
    """

    normalized_code = (provider_code or "").strip().lower()
    error_class: type[LLMError] | None = None

    if extra_code_map and normalized_code:
        error_class = {
            code.lower(): cls for code, cls in extra_code_map.items()
        }.get(normalized_code)

    if error_class is None and normalized_code in _QUOTA_CODES:
        error_class = UpstreamAuthError

    # 欠费没有统一 code 时只能靠文案兜底（Anthropic 就是 400 + invalid_request_error）。
    # 这一步必须排在按 code 判限流之前吗？不需要——限流 code 是明确的，文案是模糊的。
    if error_class is None and _matches_message_hint(message):
        error_class = UpstreamAuthError

    if error_class is None and normalized_code in _AUTH_CODES:
        error_class = UpstreamAuthError

    if error_class is None and normalized_code in _CONTENT_FILTER_CODES:
        error_class = ContentFilteredError

    if error_class is None and normalized_code in _RATE_LIMIT_CODES:
        # 注意：overloaded_error（Anthropic 529）语义更接近上游过载，
        # 但它同样可重试，这里统一进 RateLimitError 会让 429/529 的告警混在一起。
        error_class = (
            UpstreamServerError
            if normalized_code == "overloaded_error"
            else RateLimitError
        )

    if error_class is None:
        # 无状态码 = 传输层失败（连接断开/DNS），可重试。
        error_class = (
            UpstreamServerError if status_code is None else map_status_code(status_code)
        )

    return error_class(
        message,
        provider=provider,
        model=model,
        status_code=status_code,
        retry_after=retry_after,
        request_id=request_id,
        raw_error=raw_error,
    )
