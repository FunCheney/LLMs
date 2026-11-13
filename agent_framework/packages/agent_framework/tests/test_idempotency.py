"""W11 D3：幂等键——同一个请求只执行一次，且"正在执行"也要挡住。

这组用例守四条：

1. **键必须带主体**。只有内容哈希，A 和 B 提一样的请求，B 会拿到 A 的结果；
2. **规范化必须消掉"顺序不同"**（`sort_keys`），但**不能**合并列表顺序——
   那是语义；
3. **"完成过"与"正在执行"都要挡**。只判前者，并发两次提交会各跑一遍，
   而这两遍在日志里都合法；
4. **失败不占位**。锁死失败，一次上游抖动就让这个请求永远不能重提——
   而重提是用户唯一的恢复手段。
"""

from __future__ import annotations

import pytest
from agent_framework.resilience.idempotency import (
    DuplicateInFlight,
    EntryState,
    IdempotencyKey,
    InMemoryIdempotencyStore,
    canonical_payload,
    digest_of,
    normalize_principal,
)

PAYLOAD = {"topic": "内容营销怎么做", "platform": "xiaohongshu", "tags": ["a", "b"]}


# --------------------------------------------------------------------------- #
# 规范化
# --------------------------------------------------------------------------- #


def test_field_order_does_not_change_the_key() -> None:
    """参数顺序不同要算同一条。这是这个模块存在的第一条理由。"""

    a = {"topic": "x", "platform": "y"}
    b = {"platform": "y", "topic": "x"}

    assert canonical_payload(a) == canonical_payload(b)
    assert digest_of(a) == digest_of(b)


def test_whitespace_and_unicode_escaping_do_not_change_the_key() -> None:
    """`{"a": 1}` 与 `{"a":1}` 是同一个请求；中文被转义后同一请求会有两个哈希。"""

    assert digest_of({"a": 1}) == digest_of({"a": 1})
    # 中文不能被转成 \\uXXXX：同一请求用两种编码写会算出两个哈希
    assert "\\u4e2d\\u6587" not in canonical_payload({"topic": "中文"})
    assert canonical_payload({"topic": "中文"}) == '{"topic":"中文"}'


def test_list_order_is_semantic_and_must_not_be_normalized() -> None:
    """`["a","b"]` 与 `["b","a"]` 可能真是两个请求（顺序就是语义）。"""

    assert digest_of({"tags": ["a", "b"]}) != digest_of({"tags": ["b", "a"]})


def test_different_content_is_a_different_key() -> None:
    assert digest_of({"topic": "x"}) != digest_of({"topic": "y"})


# --------------------------------------------------------------------------- #
# 键 = 主体 + 内容
# --------------------------------------------------------------------------- #


def test_same_payload_from_different_principals_is_not_the_same_key() -> None:
    """只有内容哈希会跨用户串数据。这是第二条理由。"""

    alice = IdempotencyKey.for_request("alice", PAYLOAD)
    bob = IdempotencyKey.for_request("bob", PAYLOAD)

    assert alice.digest == bob.digest
    assert alice != bob
    assert alice.as_str() != bob.as_str()


def test_principal_is_normalized() -> None:
    assert normalize_principal(None) == "anonymous"
    assert normalize_principal("") == "anonymous"
    assert normalize_principal("  Alice ") == "alice"
    # None 与 "" 必须归到同一个主体，否则会算出两条键
    assert IdempotencyKey.for_request(None, PAYLOAD) == IdempotencyKey.for_request(
        "", PAYLOAD
    )


def test_client_supplied_key_still_carries_the_principal() -> None:
    """客户端自带 `Idempotency-Key` 时也要拼主体：UUID 撞车概率极低但不是零。"""

    alice = IdempotencyKey.from_raw("alice", "uuid-1")
    bob = IdempotencyKey.from_raw("bob", "uuid-1")

    assert alice != bob
    assert alice.digest == bob.digest


# --------------------------------------------------------------------------- #
# 存储：三种状态
# --------------------------------------------------------------------------- #


async def test_first_submit_is_new_and_second_is_in_flight() -> None:
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_request("alice", PAYLOAD)

    assert (await store.begin(key)).state is EntryState.NEW
    # 第二次（并发的那次）不是 new：它不能执行
    assert (await store.begin(key)).state is EntryState.IN_FLIGHT


async def test_completed_request_is_replayed_not_reexecuted() -> None:
    """重复提交同一请求 → 只执行一次，返回同一个 task_id。"""

    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_request("alice", PAYLOAD)

    await store.begin(key)
    await store.complete(key, {"task_id": "t-1"})

    replay = await store.begin(key)
    assert replay.state is EntryState.COMPLETED
    assert replay.value == {"task_id": "t-1"}


async def test_failure_does_not_hold_the_slot() -> None:
    """失败必须能重提，否则一次上游抖动就永久锁死这个请求。"""

    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_request("alice", PAYLOAD)

    await store.begin(key)
    await store.fail(key, "UPSTREAM_TIMEOUT")
    assert (await store.begin(key)).state is EntryState.NEW


async def test_begin_or_raise_rejects_the_second_concurrent_caller() -> None:
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_request("alice", PAYLOAD)

    await store.begin_or_raise(key)
    with pytest.raises(DuplicateInFlight):
        await store.begin_or_raise(key)


async def test_begin_or_raise_lets_completed_through_as_replay() -> None:
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_request("alice", PAYLOAD)

    await store.begin(key)
    await store.complete(key, "done")
    entry = await store.begin_or_raise(key)

    assert entry.state is EntryState.COMPLETED


async def test_different_topic_is_a_separate_slot() -> None:
    store = InMemoryIdempotencyStore()
    alice = IdempotencyKey.for_request("alice", {"topic": "A"})
    also_alice = IdempotencyKey.for_request("alice", {"topic": "B"})

    await store.begin(alice)
    assert (await store.begin(also_alice)).state is EntryState.NEW


async def test_store_snapshot_exposes_internal_state_not_the_caller_view() -> None:
    """`begin()` 返回 `new`，存储里是 `in_flight`——两个视角，别混。"""

    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_request("alice", PAYLOAD)

    await store.begin(key)
    assert store.snapshot()[key.as_str()] is EntryState.IN_FLIGHT
