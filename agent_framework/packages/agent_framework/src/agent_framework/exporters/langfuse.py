"""Langfuse exporter（W3 D4）：只实现 W2 定好的 `TraceExporter.export(spans)`。

四条设计约束
------------
1. **缺 key 不抛错，返回 `NoopExporter`**。评测和服务都不该因为"观测后端没配"而挂；
   这条在 CI 里必须成立，否则没有 Langfuse 账号的人连 `make eval` 都跑不了。
2. **上报失败只降级、不影响主链路**。`tracing._safe_export` 已经兜了一层异常，
   这里再加一层"连续失败就自禁用"，避免后端宕机时每个请求都等 3 秒超时。
3. **默认不上报 prompt / 回答全文**（ADR-W3-02）。`attributes` 里任何看起来像内容的
   字段都被替换成 `{len, sha256[:8]}`。要全文必须显式 `capture_content=True`，且要
   在 ADR 里写清合规依据——W10 的安全周会回来查这条。
4. **按 `trace_id` 分组**：一条 trace 一次请求，多个 span 靠 `parent_span_id` 成树。
   分组的键必须来自 span 自身，不能从 contextvar 读——export 发生在 `finally` 里，
   那时 contextvar 已经开始回滚了。

不做 OTel
---------
W3 只需要"一次任务在平台上可回放"。引 OTel 要连带 sampler / batch processor /
跨进程传播，属于为还没出现的问题付复杂度（与 W2 的判断一致）。
"""

from __future__ import annotations

import base64
import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import Settings
from ..tracing import NoopExporter, Span, TraceExporter

__all__ = ["LangfuseExporter", "build_exporter"]

_LOG = logging.getLogger("agent_framework.trace.langfuse")

# 看起来像"内容"的键。命中即替换为长度 + 哈希。
_CONTENT_KEYS = ("content", "prompt", "completion", "text", "messages", "answer")

# 字符串值超过这个长度也做哈希：防止有人把整段 prompt 塞进一个自定义 attribute。
_MAX_ATTR_STRLEN = 200

# 连续失败这么多次后自禁用。3 次足够区分"偶发网络抖动"与"后端挂了"。
_FAILURE_THRESHOLD = 3


class LangfuseExporter:
    """把 `Span` 批量推给 Langfuse Ingestion API。

    只依赖 `httpx`（项目已有），不引 `langfuse` SDK：SDK 自带后台线程与全局状态，
    在 pytest 里很难收干净，而这里需要的只是一个 POST。
    """

    def __init__(
        self,
        *,
        # Langfuse 服务地址（含 scheme）。这里会拼成 `{host}/api/public/ingestion`，
        # 自建实例也能用，所以不从配置里硬编码 SaaS 域名。
        host: str,
        # 项目级 API key。与 secret_key 一起 base64 成 Basic 认证头。
        public_key: str,
        secret_key: str,
        # True = 上报 prompt / 回答**全文**。默认 False 时这些字段只上报
        # `{len, sha256[:8]}`。打开它等于把用户输入送到第三方，属于合规决定，
        # 必须在 ADR 里写清依据（W10 会回来查）。
        capture_content: bool = False,
        # 单次上报超时（秒）。3s 是"比后端快一点点"的取值：再长会让主链路
        # 在后端宕机时跟着变慢，再短会把正常的大批量上报判成失败。
        timeout_s: float = 3.0,
        # 注入用传输层（测试里塞 `httpx.MockTransport`）。生产留 None 走默认连接池。
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._endpoint = host.rstrip("/") + "/api/public/ingestion"
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        }
        self._capture_content = capture_content
        self._timeout_s = timeout_s
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._failures = 0
        self._disabled = False

    # -- 生命周期 -------------------------------------------------------- #

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            )

        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def disabled(self) -> bool:
        """自禁用状态（测试与运维都用得上）。"""

        return self._disabled

    # -- 上报 ------------------------------------------------------------ #

    async def export(self, spans: list[Span]) -> None:
        if not spans or self._disabled:
            return

        try:
            await self._post(self._payload(spans))
        except Exception as exc:  # noqa: BLE001 - 观测后端异常绝不能冒泡到业务
            self._failures += 1

            if self._failures >= _FAILURE_THRESHOLD:
                # 自禁用：后端挂着时，每个请求都等满超时比丢弃 trace 更糟。
                self._disabled = True
                _LOG.warning(
                    "langfuse exporter disabled after %d failures: %s",
                    self._failures,
                    exc,
                )
            else:
                _LOG.warning("langfuse export failed (%d): %s", self._failures, exc)

    async def _post(self, payload: dict[str, Any]) -> None:
        client = self._ensure_client()
        response = await client.post(self._endpoint, json=payload, headers=self._headers)

        if response.status_code >= 400:
            raise RuntimeError(
                f"langfuse ingestion HTTP {response.status_code}: {response.text[:200]}"
            )

    def _payload(self, spans: list[Span]) -> dict[str, Any]:
        """按 trace_id 聚成 batch：一条 trace 一条 trace-create + N 条 span-create。"""

        grouped: dict[str, list[Span]] = {}

        for item in spans:
            grouped.setdefault(item.trace_id, []).append(item)

        batch: list[dict[str, Any]] = []

        for trace_id, items in grouped.items():
            root = min(items, key=lambda span: span.started_at)
            batch.append(
                {
                    "type": "trace-create",
                    "id": f"{trace_id}-trace",
                    "timestamp": _iso(datetime.now(UTC)),
                    "body": {
                        "id": trace_id,
                        "name": root.name,
                        "timestamp": _iso(root.started_at),
                        "metadata": {"spans": len(items)},
                    },
                }
            )

            for item in items:
                batch.append(
                    {
                        "type": "span-create",
                        "id": f"{item.span_id}-create",
                        "timestamp": _iso(datetime.now(UTC)),
                        "body": {
                            "id": item.span_id,
                            "traceId": trace_id,
                            "parentObservationId": item.parent_span_id,
                            "name": item.name,
                            "startTime": _iso(item.started_at),
                            "endTime": _iso(
                                datetime.fromtimestamp(
                                    item.started_at + item.latency_ms / 1000, UTC
                                )
                            ),
                            "level": "ERROR" if item.status == "error" else "DEFAULT",
                            "statusMessage": item.error_code,
                            "metadata": _sanitize(
                                item.attributes, capture=self._capture_content
                            ),
                        },
                    }
                )

        return {"batch": batch}


def _sanitize(attributes: dict[str, Any], *, capture: bool) -> dict[str, Any]:
    """把可能带内容的字段替换成 `{len, sha256}`。

    默认 `capture=False`：观测平台是**第三方**，prompt 里可能有用户数据。
    这不是"以后再说"的事——一旦上过线，数据就已经在对方的库里了。
    """

    if capture:
        return dict(attributes)

    clean: dict[str, Any] = {}

    for key, value in attributes.items():
        lowered = key.lower()

        if isinstance(value, str):
            if any(marker in lowered for marker in _CONTENT_KEYS) or len(value) > _MAX_ATTR_STRLEN:
                clean[key] = _fingerprint(value)
            else:
                clean[key] = value
        elif isinstance(value, list | dict) and any(
            marker in lowered for marker in _CONTENT_KEYS
        ):
            clean[key] = _fingerprint(str(value))
        else:
            clean[key] = value

    return clean


def _fingerprint(value: str) -> dict[str, Any]:
    return {
        "len": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:8],
    }


def _iso(value: Any) -> str:
    """`Span.started_at` 是 epoch 秒；Langfuse 要 ISO8601。"""

    if isinstance(value, datetime):
        return value.isoformat()

    return datetime.fromtimestamp(float(value), UTC).isoformat()


def build_exporter(settings: Settings | None = None, **kwargs: Any) -> TraceExporter:
    """按配置产出 exporter。**缺 key 返回 `NoopExporter`，绝不抛。**

    这条是硬约束：没有 Langfuse 账号的机器必须能跑完整的评测与测试。
    """

    resolved = settings or Settings()

    public_key = (
        resolved.langfuse_public_key.get_secret_value()
        if resolved.langfuse_public_key
        else ""
    )
    secret_key = (
        resolved.langfuse_secret_key.get_secret_value()
        if resolved.langfuse_secret_key
        else ""
    )

    if not public_key or not secret_key:
        _LOG.info("langfuse keys missing → NoopExporter（离线档与 CI 的正常路径）")

        return NoopExporter()

    return LangfuseExporter(
        host=resolved.langfuse_host,
        public_key=public_key,
        secret_key=secret_key,
        **kwargs,
    )
