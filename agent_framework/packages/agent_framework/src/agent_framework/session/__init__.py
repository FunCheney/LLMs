"""会话层：只存取，不编排（W2 D1）。

三条边界，review 会逐条查：

1. **`SessionStore` 不许调 `LLMGateway`**。会话层负责"攒消息"，调用才是编排方的事；
   一旦这里发起 LLM 调用，重试/成本/超时就会出现第三份定义。
2. **`provider_meta` 必须原样存回来**。DeepSeek 的第二轮要回传 `reasoning_content`，
   丢了它线上表现为"多轮对话第二轮 400"（ADR-007，W1 已经踩过一次）。
3. **幂等靠 `request_id`**。调用方重试（网关重发、前端双击）不能让历史翻倍。
"""

from .base import (
    InvalidSessionIdError,
    SessionError,
    SessionStore,
    Turn,
    decode_turn,
    encode_turn,
    resolve_request_id,
    validate_session_id,
)
from .factory import build_session_store
from .memory import InMemorySessionStore
from .redis import RedisSessionStore

__all__ = [
    "InMemorySessionStore",
    "InvalidSessionIdError",
    "RedisSessionStore",
    "SessionError",
    "SessionStore",
    "Turn",
    "build_session_store",
    "decode_turn",
    "encode_turn",
    "resolve_request_id",
    "validate_session_id",
]
