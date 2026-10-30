"""产品端点的验收测试（离线，`model="fixture"` 走 mock 回放）。

这里测的不是"文案好不好"（那是 W3 评测集的事），而是三条边界：

1. **投影干净**：响应里不能出现 `provider_meta` / `raw_finish_reason` / 任何厂商字段；
2. **错误契约复用**：未知 alias 必须走框架的 400 BAD_REQUEST，而不是产品层自己的 500；
3. **request_id 贯穿**：响应头 `X-Request-ID` 与 body 里的 `request_id` 一致，
   否则线上排障时无法把两边日志串起来。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from agent_framework.bootstrap import build_gateway
from agent_framework.config import Settings
from growth_agent.app import create_app

FORBIDDEN_KEYS = {
    "provider_meta",
    "raw_finish_reason",
    "reasoning_content",
    "arguments_error",
    "raw_arguments",
}


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(gateway=build_gateway(Settings()))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://product.test"
    ) as opened:
        yield opened


async def test_draft_returns_projected_body(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/draft", json={"brief": "给新用户发一条首购优惠推送", "model": "fixture"}
    )

    assert response.status_code == 200

    body = response.json()

    assert body["text"]
    assert body["model"] == "fixture"
    # 夹具里写死的 58/22：能对上说明真的走完了 adapter → usage → 投影
    assert body["input_tokens"] == 58
    assert body["output_tokens"] == 22
    assert body["usage_estimated"] is False
    assert body["latency_ms"] >= 0


async def test_draft_body_has_no_vendor_fields(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/draft", json={"brief": "写一句 slogan", "model": "fixture"})

    leaked = FORBIDDEN_KEYS & set(response.json())

    assert leaked == set()


async def test_request_id_travels_in_header_and_body(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/draft",
        json={"brief": "写一句 slogan", "model": "fixture"},
        headers={"X-Request-ID": "req_fixed_123"},
    )

    assert response.headers["X-Request-ID"] == "req_fixed_123"
    assert response.json()["request_id"] == "req_fixed_123"


async def test_unknown_alias_maps_to_400_not_500(client: httpx.AsyncClient) -> None:
    """产品层没有自己写异常处理，这条能过就说明它复用的是框架的契约。"""

    response = await client.post("/v1/draft", json={"brief": "写一句 slogan", "model": "ghost"})

    assert response.status_code == 400
    assert response.json()["code"] == "BAD_REQUEST"


async def test_empty_brief_is_rejected_by_schema(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/draft", json={"brief": ""})

    assert response.status_code == 400
    assert response.json()["code"] == "BAD_REQUEST"


async def test_health_reports_framework_version(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    body: dict[str, Any] = response.json()

    assert body["service"] == "growth-agent"
    assert body["framework_version"]
