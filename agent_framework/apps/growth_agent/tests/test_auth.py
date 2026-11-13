"""W10 D5：我方鉴权。

这组用例守的是一条**分流规则**，不是"能不能挡"：

- 我方的 401 只表示"调用方没通过我方鉴权"；
- 上游的 502 表示"厂商 key 失效/欠费"（ADR-005）。

两者混了，值班同学会为厂商欠费去查我方的 key 表。所以要断言的
不只是状态码，还有 `code`——状态码可能被中间层改写，code 不会。
"""

from __future__ import annotations

import logging

import httpx
import pytest
from agent_framework.config import Settings
from agent_framework.server.auth import (
    AUTH_INVALID,
    AUTH_REQUIRED,
    APIKeyAuth,
    AuthError,
    parse_api_keys,
)
from growth_agent.app import create_app

KEY = "sk-live-alice-0123456789abcdef"

FAST_SETTINGS = {
    "retrieval_budget_ms": 200.0,
    "max_attempts": 1,
    "first_token_timeout_s": 1.0,
}


def build_client(**overrides: object) -> httpx.AsyncClient:
    settings = Settings(**{**FAST_SETTINGS, **overrides})  # type: ignore[arg-type]
    app = create_app(settings=settings)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://growth.test"
    )


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #


def test_parse_maps_key_to_principal() -> None:
    entries = parse_api_keys(f"{KEY}=alice:admin|growth, sk-bob=bob")

    assert [entry.principal.subject_id for entry in entries] == ["alice", "bob"]
    assert entries[0].principal.groups == frozenset({"admin", "growth"})
    assert entries[1].principal.groups == frozenset()


def test_subject_defaults_to_hash_not_the_key() -> None:
    """没写 subject 时用哈希前 8 位。

    如果用 key 本身当 subject，key 会流进日志、trace、ACL 判断——
    而那三个面都比 key 文件更容易被看到。
    """

    entries = parse_api_keys(KEY)

    assert entries[0].principal.subject_id.startswith("key-")
    assert KEY not in entries[0].principal.subject_id


def test_malformed_entry_fails_loudly() -> None:
    """写错的条目要炸，不能静默跳过。

    静默跳过的表现是"某个人突然 401"，日志里什么都没有。
    两种典型写法错：漏 key（`=alice`）、有 key 没身份名（`sk-x=:admin`）。
    """

    with pytest.raises(ValueError, match="缺少 key"):
        parse_api_keys("=alice")

    with pytest.raises(ValueError, match="缺少 subject"):
        parse_api_keys("sk-x=:admin")


def test_enabling_auth_without_keys_is_refused() -> None:
    """`AUTH_MODE=key` 但没配 key → 启动失败，而不是"放行所有"。

    安全配置往"开放"方向失败是最坏的一种默认值。
    """

    settings = Settings(auth_mode="key", api_keys="")
    with pytest.raises(ValueError, match="API_KEYS 为空"):
        APIKeyAuth.from_settings(settings)


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #


def test_missing_key_is_auth_required() -> None:
    auth = APIKeyAuth(entries=parse_api_keys(KEY))

    with pytest.raises(AuthError) as caught:
        auth.authenticate({})

    assert caught.value.code == AUTH_REQUIRED
    assert caught.value.status == 401


def test_wrong_key_is_auth_invalid() -> None:
    auth = APIKeyAuth(entries=parse_api_keys(KEY))

    with pytest.raises(AuthError) as caught:
        auth.authenticate({"x-api-key": "sk-wrong"})

    assert caught.value.code == AUTH_INVALID


def test_bearer_and_raw_header_both_work() -> None:
    auth = APIKeyAuth(entries=parse_api_keys(f"{KEY}=alice"))

    assert auth.authenticate({"authorization": f"Bearer {KEY}"}).subject_id == "alice"
    assert auth.authenticate({"authorization": KEY}).subject_id == "alice"
    assert auth.authenticate({"x-api-key": KEY}).subject_id == "alice"


def test_disabled_auth_returns_anonymous() -> None:
    assert APIKeyAuth().authenticate({}).subject_id == "anonymous"


# --------------------------------------------------------------------------- #
# 端到端：中间件真的挡住了
# --------------------------------------------------------------------------- #


async def test_endpoint_requires_key_when_enabled() -> None:
    async with build_client(auth_mode="key", api_keys=f"{KEY}=alice") as client:
        response = await client.post("/v1/draft", json={"brief": "写一句文案"})

    body = response.json()
    assert response.status_code == 401
    assert body["code"] == AUTH_REQUIRED
    # 401 必须带 WWW-Authenticate，否则客户端库不会提示补凭据。
    assert response.headers["www-authenticate"].startswith("Bearer")


async def test_health_stays_public() -> None:
    """`/health` 免鉴权：探针要带 key 意味着 key 得写进编排配置，泄漏面更大。"""

    async with build_client(auth_mode="key", api_keys=f"{KEY}=alice") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_key_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    """密钥进日志等于泄漏。失败路径只打码后的形态。"""

    caplog.set_level(logging.DEBUG)
    async with build_client(auth_mode="key", api_keys=f"{KEY}=alice") as client:
        await client.post("/v1/draft", json={"brief": "x"}, headers={"x-api-key": KEY + "x"})

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert KEY not in joined


async def test_our_401_is_not_upstream_502() -> None:
    """**ADR-005 的那一条**：两种鉴权失败必须给出不同的状态码与 code。

    用 `LLMError` 的映射表当参照，而不是自己再写一份期望值——
    参照物写死在这个测试里，就等于给"同一份契约"造了第二份定义。
    """

    from agent_framework.llm.errors import UpstreamAuthError, error_code_for, http_status_for

    upstream = UpstreamAuthError("厂商 key 失效", provider="openai")
    assert http_status_for(upstream) == 502
    assert error_code_for(upstream) == "UPSTREAM_AUTH"

    async with build_client(auth_mode="key", api_keys=f"{KEY}=alice") as client:
        mine = await client.post("/v1/draft", json={"brief": "x"})

    assert mine.status_code == 401
    assert mine.json()["code"] != error_code_for(upstream)
