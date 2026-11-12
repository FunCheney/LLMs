"""脱敏（W8 D2）：录制落盘与工具日志的**唯一**出口。

为什么单独一个模块而不是各写一个 `_mask()`：脱敏有两处消费者——录制夹具
（落盘、会进 git）和工具调用日志（会进 trace 存储）。两处各写一份的必然结果
是"夹具里打码了、日志里漏了"，而漏的那一份恰好是可被 grep 到的那一份。

打码规则只针对**已知形态**，不做"敏感词猜测"：

- 键名命中 `api_key` / `token` / `secret` / `password` / `authorization` /
  `cookie` / `signature` → 值整体替换成 `***`
- 值形如 `sk-xxxx` / `Bearer yyyy` / `ghp_xxxx` → 保留前缀与尾 4 位
  （`observability.mask_secret` 同款口径，便于对账）
- URL 的 query 参数按同名规则打码，**其他参数原样保留**：
  把整个 URL 打成 `***` 会让回放无法验证"当初抓的是哪个页面"

刻意**不**做的两件事：

1. 不删除键（替换成 `***` 而不是 pop）。删掉之后回放方会以为"上游没返回这个字段"，
   于是把"打过码"误判成"本来就没有"，这类差异在排障时极难发现。
2. 不对纯文本做全量正则扫描。工具返回的正文里出现 "token" 这个词很正常，
   全文扫描会把正常内容打花，导致夹具不可读、diff 无法审阅。
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..observability import mask_secret

__all__ = ["MASK", "SENSITIVE_KEYS", "mask_url", "sanitize"]

MASK = "***"

# 键名匹配是**小写子串**匹配：`X-Api-Key`、`apiKey`、`API_KEY` 都要命中。
# 用子串而不是全等，因为各家写法太杂（`access_token` / `accessToken`）。
SENSITIVE_KEYS = (
    "api_key",
    "apikey",
    "access_token",
    "accesstoken",
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "cookie",
    "signature",
    "x-api-key",
)

# `sk-abc123...` / `Bearer eyJ...` / `ghp_xxxx`：只认这几种明确的前缀形态。
_SECRET_VALUE_RE = re.compile(
    r"(?P<prefix>sk-|ghp_|gho_|xoxb-|Bearer\s+)(?P<body>[A-Za-z0-9._\-]{8,})"
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(marker in lowered for marker in SENSITIVE_KEYS)


def _mask_value(value: str) -> str:
    """短值直接 `***`（保留长度会泄漏"密钥多长"这种信息量）。"""
    if len(value) < 8:
        return MASK
    return mask_secret(value)


def _mask_text(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        body = match.group("body")
        tail = body[-4:] if len(body) > 8 else ""
        return f"{prefix}{MASK}{tail}"

    return _SECRET_VALUE_RE.sub(_replace, text)


def mask_url(url: str) -> str:
    """URL 只在 query 参数上打码，path 与域名保留（回放要靠它验证目标）。"""
    try:
        parts = urlsplit(url)
    except ValueError:
        return _mask_text(url)

    if not parts.query:
        return _mask_text(url)

    pairs: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        pairs.append((key, MASK if _is_sensitive_key(key) else _mask_text(value)))

    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(pairs, doseq=True), parts.fragment)
    )


def sanitize(value: Any) -> Any:
    """递归脱敏。返回**新对象**，不改调用方的数据。

    不改原对象：录制链路里同一个 payload 还要原样返回给执行器（真调用档），
    就地打码会让"录制"顺手把真实结果也毁掉。
    """
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _is_sensitive_key(text_key):
                cleaned[text_key] = MASK
            else:
                cleaned[text_key] = sanitize(item)
        return cleaned

    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]

    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return mask_url(value)
        return _mask_text(value)

    # int / bool / None 原样：数字和布尔里没有可打码的结构，
    # 而把所有非字符串都转成 `***` 会让夹具里的分页游标、计数全部丢失。
    return value
