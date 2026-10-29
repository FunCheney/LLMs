"""A1–A8：provider 一致性（conformance）测试。

一条用例 = 同一份内部输入 + 每家一份 wire 夹具 → 断言"解析回内部后三家语义相等"。

**断言里不许出现 `if provider == ...`**：出现即说明断言的是厂商差异，
而不是归一化是否成立。厂商差异必须已被 adapter 吸收，测试层看不到。

单家特有行为（心跳、ping、input_json_delta）放在 test_openai_compat / test_anthropic，
不污染这里。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from agent_framework.llm.errors import (
    LLMError,
    RateLimitError,
    UpstreamAuthError,
    UpstreamServerError,
)
from agent_framework.llm.providers._common import StreamAccumulator
from conftest import (
    FIXTURES_ROOT,
    MAX_TOKENS,
    PROVIDERS,
    SYNTHETIC_ROOT,
    canonical_messages,
    canonical_tools,
    load_case,
    recorded_client,
)

SUCCESS_CASES = ("text_basic", "tool_call_basic", "usage_missing")

ERROR_CASES = ("error_rate_limit", "error_quota", "error_server")

ALL_CASES = SUCCESS_CASES + ERROR_CASES

# 同一错误语义 → 同一个异常类 + 同一个 retryable（四家落点不同：429 / 402 / 400 / 529）
ERROR_EXPECT: dict[str, tuple[type[LLMError], bool]] = {
    "error_rate_limit": (RateLimitError, True),
    "error_quota": (UpstreamAuthError, False),
    "error_server": (UpstreamServerError, True),
}

# 内部字段一旦出现在 wire 上，就说明 adapter 把内部表示泄漏给了厂商
INTERNAL_ONLY_KEYS = frozenset(
    {
        "provider_meta",
        "arguments_error",
        "raw_arguments",
        "estimated",
        "latency_ms",
        "response_id",
    }
)


def _collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()

    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys |= _collect_keys(item)
    elif isinstance(value, list):
        for item in value:
            keys |= _collect_keys(item)

    return keys


async def _chat(case: Any) -> Any:
    client, _ = recorded_client(case)

    return await client.chat(
        canonical_messages(),
        canonical_tools(),
        max_tokens=MAX_TOKENS,
    )


# --------------------------------------------------------------------------- #
# A1 wire 形态：各家不同，但都由同一份内部输入精确产出
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a1_request_matches_fixture(provider: str) -> None:
    case = load_case(provider, "text_basic")
    client, _ = recorded_client(case)

    prepared = client.build_request(
        canonical_messages(),
        canonical_tools(),
        max_tokens=MAX_TOKENS,
    )

    # 精确比对，而不是"包含 name 即算过"——后者测不出层级/字段名错误
    assert prepared.body == case.request


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a1_request_has_no_internal_fields(provider: str) -> None:
    case = load_case(provider, "text_basic")
    client, _ = recorded_client(case)

    prepared = client.build_request(
        canonical_messages(),
        canonical_tools(),
        max_tokens=MAX_TOKENS,
    )

    leaked = _collect_keys(prepared.body) & INTERNAL_ONLY_KEYS

    assert leaked == set()


# --------------------------------------------------------------------------- #
# A2 工具调用语义：四家必须完全一致
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a2_tool_call_semantics(provider: str) -> None:
    case = load_case(provider, "tool_call_basic")
    expected = case.expect["tool_calls"][0]

    response = await _chat(case)

    assert len(response.tool_calls) == 1

    call = response.tool_calls[0]

    assert call.name == expected["name"]
    assert isinstance(call.arguments, dict)
    assert call.arguments == expected["arguments"]
    assert call.arguments_error is None
    assert call.id


async def test_a2_tool_call_is_identical_across_providers() -> None:
    """归一化是否真的成立的**唯一**证据：四家解析出的内部表示完全相等。"""

    signatures: set[tuple[str, str, str]] = set()

    for provider in PROVIDERS:
        response = await _chat(load_case(provider, "tool_call_basic"))
        call = response.tool_calls[0]

        signatures.add(
            (
                call.name,
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True),
                str(response.finish_reason),
            )
        )

    assert len(signatures) == 1, signatures


# --------------------------------------------------------------------------- #
# A3 流式拼接结果 == 非流式结果
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("case_name", ("text_basic", "tool_call_basic"))
async def test_a3_stream_equals_non_stream(provider: str, case_name: str) -> None:
    case = load_case(provider, case_name)

    stream_client, _ = recorded_client(case)
    events = [
        event
        async for event in stream_client.stream(
            canonical_messages(),
            canonical_tools(),
            max_tokens=MAX_TOKENS,
        )
    ]

    accumulator = StreamAccumulator()
    accumulator.feed(events)

    non_stream = await _chat(case)

    assert accumulator.text == (non_stream.text or "")

    assert [(call.name, call.arguments) for call in accumulator.tool_calls] == [
        (call.name, call.arguments) for call in non_stream.tool_calls
    ]

    assert accumulator.finish_reason == non_stream.finish_reason

    # 顺序保证：text_delta 全部先于 tool_call
    kinds = [event.type for event in events]
    last_text = max(
        (index for index, kind in enumerate(kinds) if kind == "text_delta"),
        default=-1,
    )
    first_call = min(
        (index for index, kind in enumerate(kinds) if kind == "tool_call"),
        default=len(kinds),
    )

    assert last_text < first_call


# --------------------------------------------------------------------------- #
# A4 usage：三家都要能映射；缺失时 estimated=True 且 token > 0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a4_usage_present(provider: str) -> None:
    response = await _chat(load_case(provider, "text_basic"))

    assert response.usage.estimated is False
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.usage.total_tokens > 0


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a4_usage_missing_is_estimated(provider: str) -> None:
    response = await _chat(load_case(provider, "usage_missing"))

    assert response.usage.estimated is True
    assert response.usage.total_tokens > 0


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a4_stream_usage_present(provider: str) -> None:
    """DeepSeek 把 usage 挂在内容块上，靠 choices 为空判断会得到 estimated=True。"""

    case = load_case(provider, "tool_call_basic")
    client, _ = recorded_client(case)

    events = [
        event
        async for event in client.stream(
            canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
        )
    ]

    accumulator = StreamAccumulator()
    accumulator.feed(events)

    assert accumulator.usage is not None
    assert accumulator.usage.estimated is False
    assert accumulator.usage.output_tokens > 0


# --------------------------------------------------------------------------- #
# A5 finish_reason 归一
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("case_name", SUCCESS_CASES)
async def test_a5_finish_reason_normalized(provider: str, case_name: str) -> None:
    case = load_case(provider, case_name)
    response = await _chat(case)

    assert str(response.finish_reason) == case.expect["finish_reason"]
    # 原始值必须保留，否则线上排障无从追起
    assert response.raw_finish_reason


# --------------------------------------------------------------------------- #
# A6 错误语义：同一语义 → 同一异常类 + 同一 retryable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("case_name", ERROR_CASES)
async def test_a6_error_mapping(provider: str, case_name: str) -> None:
    case = load_case(provider, case_name)
    expected_class, expected_retryable = ERROR_EXPECT[case_name]
    client, _ = recorded_client(case)

    with pytest.raises(expected_class) as excinfo:
        await client.chat(
            canonical_messages(),
            canonical_tools(),
            max_tokens=MAX_TOKENS,
        )

    error = excinfo.value

    assert isinstance(error, LLMError)
    assert error.retryable is expected_retryable
    assert error.status_code == case.status_code
    assert error.provider == provider


async def test_a6_error_class_is_identical_across_providers() -> None:
    """欠费落在 429 / 402 / 400 三个位置，但内部必须是同一个类且不可重试。"""

    observed: set[tuple[str, bool]] = set()

    for provider in PROVIDERS:
        case = load_case(provider, "error_quota")
        client, _ = recorded_client(case)

        with pytest.raises(LLMError) as excinfo:
            await client.chat(
                canonical_messages(),
                canonical_tools(),
                max_tokens=MAX_TOKENS,
            )

        observed.add((type(excinfo.value).__name__, excinfo.value.retryable))

    assert observed == {("UpstreamAuthError", False)}


# --------------------------------------------------------------------------- #
# A7 无厂商字段泄漏
# --------------------------------------------------------------------------- #

VENDOR_ONLY_TOKENS = ("reasoning_content", "signature", "partial_json")


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a7_no_vendor_field_leak(provider: str) -> None:
    case = load_case(provider, "tool_call_basic")
    response = await _chat(case)

    exposed = json.dumps(
        {
            "text": response.text,
            "tool_calls": [call.model_dump() for call in response.tool_calls],
        },
        ensure_ascii=False,
    )

    for token in VENDOR_ONLY_TOKENS:
        assert token not in exposed


async def test_a7_reasoning_lives_in_provider_meta() -> None:
    """DeepSeek 的 reasoning_content 必须在 provider_meta 里，供下一轮原样回传。"""

    response = await _chat(load_case("deepseek", "tool_call_basic"))

    assert response.message.provider_meta is not None
    assert "reasoning_content" in response.message.provider_meta
    assert response.message.provider_meta["reasoning_content"] not in (response.text or "")


# --------------------------------------------------------------------------- #
# A8 一次调用只发一次 HTTP（ADR-003 的可执行版本）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a8_chat_sends_exactly_one_request(provider: str) -> None:
    case = load_case(provider, "text_basic")
    client, recorder = recorded_client(case)

    await client.chat(canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS)

    assert recorder.count == 1


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a8_stream_sends_exactly_one_request(provider: str) -> None:
    case = load_case(provider, "tool_call_basic")
    client, recorder = recorded_client(case)

    async for _ in client.stream(
        canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
    ):
        pass

    assert recorder.count == 1


# --------------------------------------------------------------------------- #
# A13 录制夹具 ⊇ 手写基线（防"手写幻觉字段"）
# --------------------------------------------------------------------------- #


def key_paths(value: Any, prefix: str = "$") -> set[str]:
    """把一个 JSON 体压成**键路径**集合（`$.choices[].message.content` 这种）。

    为什么不比键名集合：`id` / `type` / `index` 在四家到处都是，键名集合会让
    "位置放错但字段存在"的 bug 蒙混过关。路径能区分 `$.usage.prompt_tokens`
    与 `$.choices[].usage.prompt_tokens`——后者正是 DeepSeek 的真实形态。
    """

    paths: set[str] = set()

    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            paths.add(path)
            paths |= key_paths(item, path)
    elif isinstance(value, list):
        for item in value:
            paths |= key_paths(item, f"{prefix}[]")

    return paths


def _read_json(path: Any) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("case_name", ALL_CASES)
def test_a13_recorded_body_is_superset_of_handwritten(
    provider: str, case_name: str
) -> None:
    """真录制可以**多**字段（厂商加字段是常态），绝不能**少**字段。

    少的含义只有一个：手写基线里存在一个真实协议里不存在的字段——
    也就是说 adapter 里可能为了这个幻觉字段写了永远不会执行的代码。
    Day5 把四家的 response 换成真录制后，这条断言就是"手写猜测"的清算单。
    """

    baseline_dir = SYNTHETIC_ROOT / provider / case_name
    recorded_path = FIXTURES_ROOT / provider / case_name / "response.json"

    if not baseline_dir.is_dir() or not recorded_path.is_file():
        pytest.skip(f"{provider}/{case_name} 缺少手写基线或录制体")

    baseline = _read_json(baseline_dir / "response.json")
    recorded = _read_json(recorded_path)

    missing = key_paths(baseline) - key_paths(recorded)

    assert missing == set(), (
        f"{provider}/{case_name}: 手写夹具里有真实协议不存在的字段路径：{sorted(missing)}"
    )


def test_a13_meta_flags_recording_state_consistently() -> None:
    """`synthetic` 与 `recorded_at` 必须互相印证，否则"哪些是编的"就查不清了。"""

    for provider in PROVIDERS:
        for case_name in ALL_CASES:
            meta = _read_json(FIXTURES_ROOT / provider / case_name / "meta.json")

            if meta.get("synthetic") is False:
                assert meta.get("recorded_at"), f"{provider}/{case_name}: 录制体缺时间戳"
            else:
                assert meta.get("recorded_at") is None, (
                    f"{provider}/{case_name}: 手写体不该有 recorded_at"
                )


# --------------------------------------------------------------------------- #
# A14 工具调用必须落在自己声明的 schema 内
# --------------------------------------------------------------------------- #


def _declared_tool_schemas(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """从 request.json 抽出 {工具名: parameters}，两种协议结构都要认。"""

    schemas: dict[str, dict[str, Any]] = {}

    for raw in request.get("tools") or []:
        if not isinstance(raw, dict):
            continue

        if "function" in raw:  # OpenAI / DeepSeek / Qwen
            function = raw.get("function") or {}
            schemas[str(function.get("name"))] = function.get("parameters") or {}
        else:  # Anthropic
            schemas[str(raw.get("name"))] = raw.get("input_schema") or {}

    return schemas


def _assert_call_matches_schema(call: Any, schemas: dict[str, dict[str, Any]]) -> None:
    assert call.name in schemas, f"调用了未声明的工具 '{call.name}'；已声明：{sorted(schemas)}"

    parameters = schemas[call.name]
    allowed = set((parameters.get("properties") or {}).keys())
    required = set(parameters.get("required") or [])

    assert set(call.arguments) <= allowed, (
        f"'{call.name}' 收到 schema 外的参数 {sorted(set(call.arguments) - allowed)}"
    )
    assert required <= set(call.arguments), (
        f"'{call.name}' 缺少必填参数 {sorted(required - set(call.arguments))}"
    )


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a14_response_tool_call_matches_declared_schema(provider: str) -> None:
    """真实模型可能返回未声明的工具名或凭空加参数。

    这条断言在真录制替换后才有杀伤力：手写夹具永远是"照 schema 编的"，
    而真实模型会跑偏。跑偏必须在这里变红，由我们改 prompt / 改工具描述，
    而不是放宽断言——放宽等于把"模型乱调工具"放进生产。
    """

    case = load_case(provider, "tool_call_basic")
    schemas = _declared_tool_schemas(case.request)

    assert schemas, f"{provider}: request.json 里没有 tools 声明"

    response = await _chat(case)

    for call in response.tool_calls:
        _assert_call_matches_schema(call, schemas)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a14_expected_tool_call_matches_declared_schema(provider: str) -> None:
    """夹具自带的 expect 也必须落在 schema 内——防止手写 expect 编造参数。"""

    case = load_case(provider, "tool_call_basic")
    schemas = _declared_tool_schemas(case.request)

    for expected in case.expect["tool_calls"]:
        class _Stub:
            name = expected["name"]
            arguments = expected["arguments"]

        _assert_call_matches_schema(_Stub(), schemas)
