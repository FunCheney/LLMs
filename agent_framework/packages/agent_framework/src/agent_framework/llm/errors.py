from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """
    一次 LLM 错误请求的公共上下文
    """
    provider: str | None = None
    model: str | None = None
    status_code: int | None = None
    retry_after: float | None = None
    request_id: str | None = None



class LLMError(Exception):
    """LLM 客户端基础异常。"""
    default_retryable = False

    def __init__(self,
                 message: str,
                 *,
                 provider: str,
                 model: str,
                 status_code: int,
                 retry_after: float,
                 request_id: str,
                 retryable: bool,
                 raw_error: Any
            ):
        super().__init__(message)
        self.context = ErrorContext(
            provider=provider,
            model=model,
            status_code=status_code,
            retry_after=retry_after,
            request_id=request_id,
        )
        # 直接暴露公共字段，方便调用方访问。
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retry_after = retry_after
        self.request_id = request_id

        # 未显式指定时，采用异常类的默认策略。
        self.retryable = (
            self.default_retryable
            if retryable is None
            else retryable
        )

        # 可选的原始错误信息，供日志和排查使用。
        self.raw_error = raw_error




class AuthenticationError(LLMError):
    """认证失败，例如 API Key 无效。"""


class RateLimitError(LLMError):
    """请求频率或配额限制。"""

    default_retryable = True


class InvalidRequestError(LLMError):
    """请求参数无效或不符合 API 要求。"""


class CapabilityError(LLMError):
    """客户端不支持调用方要求的能力。"""


class LLMTimeoutError(LLMError):
    """请求超时。"""

    default_retryable = True


class UpstreamError(LLMError):
    """上游服务异常、网络错误或其他未分类错误。"""



def map_status_code(
    status_code: int,
) -> type[LLMError]:
    """将通用 HTTP 状态码映射到内部异常类。

    这里只负责分类，不创建异常实例。
    不同供应商的特殊错误码应在 map_provider_error() 中处理。
    """

    if status_code in (401, 403):
        return AuthenticationError

    if status_code == 429:
        return RateLimitError

    if status_code in (400, 404, 413, 422):
        return InvalidRequestError

    if status_code in (408, 504):
        return LLMTimeoutError

    if status_code >= 500:
        return UpstreamError

    return UpstreamError



def map_provider_error(
    *,
    provider: str,
    model: str | None = None,
    status_code: int | None = None,
    provider_code: str | None = None,
    message: str = "LLM request failed",
    retry_after: float | None = None,
    request_id: str | None = None,
    raw_error: Any = None,
) -> LLMError:
    """统一将厂商错误映射为内部异常。

    adapter 负责提取供应商响应中的字段，然后调用本函数。
    这里不依赖供应商 SDK，也不发起 HTTP 请求。
    """

    normalized_provider = provider.lower()

    # 1. 先处理已明确识别的厂商特殊错误。
    #
    # 示例规则应根据实际 API 文档确认。
    if (
        normalized_provider == "deepseek"
        and provider_code == "insufficient_system_resource"
    ):
        return UpstreamError(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            retry_after=retry_after,
            request_id=request_id,
            retryable=True,
            raw_error=raw_error,
        )

    # 2. 没有 HTTP 状态码时，使用通用上游异常。
    if status_code is None:
        return UpstreamError(
            message,
            provider=provider,
            model=model,
            retry_after=retry_after,
            request_id=request_id,
            retryable=False,
            raw_error=raw_error,
        )

    # 3. 按统一 HTTP 状态码分类。
    error_class = map_status_code(status_code)

    # 4. 创建异常，并附带公共上下文。
    return error_class(
        message,
        provider=provider,
        model=model,
        status_code=status_code,
        retry_after=retry_after,
        request_id=request_id,
        raw_error=raw_error,
    )
