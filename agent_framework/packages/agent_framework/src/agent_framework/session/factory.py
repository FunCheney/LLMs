"""会话存储工厂（按 `settings.session_backend` 选实现）。

工厂而不是 `if` 散在调用处：`create_app`、W7 的记忆模块、W10 的调研工作流
都要拿同一个 store，各写一次 `if backend == "redis"` 就会出现"某个入口还在用内存"，
表现为"重启后部分会话丢失"——这种问题只在多副本环境才显形。
"""

from __future__ import annotations

from ..config import Settings
from .base import SessionStore
from .memory import InMemorySessionStore
from .redis import RedisSessionStore

__all__ = ["build_session_store"]


def build_session_store(settings: Settings | None = None) -> SessionStore:
    resolved = settings or Settings()

    if resolved.session_backend == "redis":
        return RedisSessionStore(
            resolved.redis_url,
            ttl_s=resolved.session_ttl_s,
            max_turns=resolved.session_max_turns,
        )

    return InMemorySessionStore(max_turns=resolved.session_max_turns)
