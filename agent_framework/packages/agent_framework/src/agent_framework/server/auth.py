"""我方鉴权（W10 D5）。**只认我方的 key，不碰上游。**

## 一条铁律：我方 401 ≠ 上游 502

| 谁错了 | 状态码 | code | 值班该做什么 |
|---|---|---|---|
| 调用方没带/带错我方 key | **401** | `AUTH_REQUIRED` / `AUTH_INVALID` | 让对方去补 key |
| 上游厂商 key 失效/欠费 | **502** | `UPSTREAM_AUTH` | 我们去换厂商 key |

两者绝不能混（ADR-005）。混了会怎样：厂商欠费触发 401，调用方以为是自己
key 的问题，反复换 key 换不出来；反过来我方 key 被拒返回 502，值班跑去查
厂商账单。**状态码是分流的唯一依据**，`UPSTREAM_AUTH` 这个 code 由
`llm/errors.py` 唯一产出，本文件不重复定义它——这里只产出 `AUTH_*`。

## 为什么是中间件而不是 `Depends`

写成 FastAPI 依赖，每个路由都要显式写一次 `Depends(require_key)`。新加一个
路由忘了写，端点就是**公开的**，而且没有任何测试会失败——这正是 ADR-008 说
的那类"横跨所有路径的约束"。中间件只有一个插口，漏不掉；漏掉的那条必须
显式写进 `PUBLIC_PATHS`，而那个集合在 code review 时看得见。

## key 怎么存

环境变量 `API_KEYS`，格式：

```
API_KEYS="sk-live-alice=alice:admin|growth, sk-live-bob=bob:viewer"
```

`key=subject:group1|group2`。`=` 右边可省——只给 key 不给身份时，
subject 取 key 的哈希前 8 位（**不是** key 本身：subject 会进日志和 ACL 判断，
把 key 写进日志等于把凭据写进日志）。

比较用 `hmac.compare_digest`：`==` 的短路比较会泄漏"前几位对了"，
攻击者可以按字节爆破。这不是理论问题——44 字符的 key 逐字节猜，
在能拿到计时差的前提下是可行的。

## 不做什么

- 不做 OAuth / JWT / 用户体系：本周要的是"朋友能用，别人不能白用"。
- 不做限流：那是 W2 的 `RequestQueue` 与 W11 的熔断，鉴权只回答"你是谁"。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ..config import Settings
from ..observability import mask_secret
from ..retrieval import Principal
from .errors import error_body

__all__ = [
    "AUTH_INVALID",
    "AUTH_REQUIRED",
    "APIKeyAuth",
    "AuthError",
    "PUBLIC_PATHS",
    "install_api_key_auth",
    "parse_api_keys",
]

_log = logging.getLogger(__name__)

AUTH_REQUIRED = "AUTH_REQUIRED"
AUTH_INVALID = "AUTH_INVALID"

# 免鉴权路径。**只放探针与文档**：`/health` 要给容器编排与负载均衡用，
# 让它带 key 意味着 key 得写进 compose/K8s 探针配置，泄漏面反而更大。
PUBLIC_PATHS = frozenset({"/health", "/healthz", "/ready", "/docs", "/redoc", "/openapi.json"})

_KEY_HEADERS = ("x-api-key", "authorization")


class AuthError(Exception):
    """鉴权失败。`code` 是稳定的对外 code，`status` 恒为 401。"""

    status = 401

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class _KeyEntry:
    key: str
    principal: Principal


def _subject_of(key: str) -> str:
    """没有显式 subject 时用 key 的哈希前 8 位。

    不用 key 本身：subject 会出现在日志、trace、ACL 判断里，
    把 key 当 subject 等于把凭据写进了每一个观测面。
    """

    return "key-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def parse_api_keys(raw: str) -> tuple[_KeyEntry, ...]:
    """解析 `API_KEYS`。格式错误**直接抛**，不静默跳过。

    静默跳过一条写错的 key 的表现是"某个人突然 401"，而日志里什么都没有。
    启动时炸掉，改配置的人立刻知道。
    """

    entries: list[_KeyEntry] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue

        key, _, identity = item.partition("=")
        key = key.strip()
        if not key:
            # 只有"整条是空白"才允许跳过（上面的 `continue` 已经处理）。
            # 走到这里说明写了 `=alice` 这种缺 key 的条目——报错，不是跳过。
            raise ValueError(f"API_KEYS 条目缺少 key：{mask_secret(item)}")

        subject = _subject_of(key)
        groups: frozenset[str] = frozenset()
        if identity.strip():
            name, _, group_part = identity.strip().partition(":")
            if not name.strip():
                raise ValueError(f"API_KEYS 条目缺少 subject：{mask_secret(key)}=")
            subject = name.strip()
            groups = frozenset(
                part.strip() for part in group_part.split("|") if part.strip()
            )

        entries.append(_KeyEntry(key=key, principal=Principal(subject_id=subject, groups=groups)))

    return tuple(entries)


@dataclass
class APIKeyAuth:
    """我方 API key 校验。`entries` 为空时等于关闭（`AUTH_MODE=off`）。"""

    entries: tuple[_KeyEntry, ...] = ()
    public_paths: frozenset[str] = field(default=PUBLIC_PATHS)

    @classmethod
    def from_settings(cls, settings: Settings) -> APIKeyAuth:
        if settings.auth_mode == "off":
            return cls()
        entries = parse_api_keys(settings.api_keys)
        if not entries:
            # 开了鉴权却没有 key：这是配置错误，不是"允许所有"。
            # 往"开放"方向失败的安全配置是最坏的一种默认值。
            raise ValueError("AUTH_MODE=key 但 API_KEYS 为空；启动被拒绝")
        return cls(entries=entries)

    @property
    def enabled(self) -> bool:
        return bool(self.entries)

    # -- 校验 --------------------------------------------------------------- #

    def extract(self, headers: Mapping[str, str]) -> str:
        """从请求头取 key。`Authorization: Bearer` 优先，其次 `X-API-Key`。"""

        raw = ""
        for name in _KEY_HEADERS:
            value = headers.get(name)
            if value:
                raw = value.strip()
                break

        if not raw:
            raise AuthError(AUTH_REQUIRED, "缺少凭据：请在 X-API-Key 或 Authorization 中提供")

        scheme, _, token = raw.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
        return raw

    def authenticate(self, headers: Mapping[str, str]) -> Principal:
        if not self.enabled:
            return Principal.anonymous()

        presented = self.extract(headers)

        # 遍历全部条目而不是"找到即返回"：比较时间上的差异会泄漏
        # "这个 key 存在但不对"，给爆破一个便宜的筛选器。
        matched: Principal | None = None
        for entry in self.entries:
            if hmac.compare_digest(entry.key, presented):
                matched = entry.principal

        if matched is None:
            _log.warning("我方鉴权失败：presented=%s", mask_secret(presented))
            raise AuthError(AUTH_INVALID, "凭据无效")

        return matched


class _AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, auth: APIKeyAuth) -> None:
        super().__init__(app)
        self._auth = auth

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if (
            not self._auth.enabled
            or path in self._auth.public_paths
            or request.method == "OPTIONS"
        ):
            return await call_next(request)

        try:
            principal = self._auth.authenticate(request.headers)
        except AuthError as exc:
            return JSONResponse(
                error_body(exc.code, exc.message),
                status_code=exc.status,
                # 401 必须带 WWW-Authenticate，否则客户端库不会提示补凭据
                # （curl / fetch 都靠它决定要不要走认证流程）。
                headers={"WWW-Authenticate": 'Bearer realm="growth-agent"'},
            )

        # 身份只放 `request.state`：下游（工具链路要 ACL）从这里取，
        # 而不是各自再解析一次请求头——两份解析必然分叉。
        request.state.principal = principal
        return await call_next(request)


def install_api_key_auth(app: FastAPI, settings: Settings) -> APIKeyAuth:
    """挂上鉴权中间件。`AUTH_MODE=off` 时什么都不挂（开发与测试的默认档）。"""

    auth = APIKeyAuth.from_settings(settings)
    if auth.enabled:
        app.add_middleware(_AuthMiddleware, auth=auth)
        _log.info("我方鉴权已启用：%d 个 key", len(auth.entries))
    return auth
