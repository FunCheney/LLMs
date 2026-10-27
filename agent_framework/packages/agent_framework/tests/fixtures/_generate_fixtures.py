"""生成 W1 Day3 的夹具（四家 x 六个 case）。

**为什么要有生成器**：夹具是手写的协议样本（meta.synthetic = true），
用脚本落盘是为了保证四家的结构差异一眼可见、且能一键重生成。
真实录制在 Day5 用 `-m live` 跑一遍后原地替换，meta 里改 synthetic=false。

**关键纪律**：这里的 wire body 是按各家文档**手写**的，绝不调用 build_request 生成。
否则 A1 就变成"adapter 跟自己对自己答案"，测不出任何东西。

用法：uv run python packages/agent_framework/tests/fixtures/_generate_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES_ROOT = Path(__file__).parent

# --------------------------------------------------------------------------- #
# 公共输入（必须与 tests/conftest.py 的 CANONICAL_* 保持一致）
# --------------------------------------------------------------------------- #

SYSTEM_TEXT = "你是研究助手，只回答可验证的事实。"
USER_TEXT = "帮我查一下 MCP 协议的最新进展"

TOOL_DESCRIPTION = "当需要最新外部信息时使用；不要用于纯计算或改写。"
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "检索关键词"}},
    "required": ["query"],
    "additionalProperties": False,
}

TEXT_BASIC = "MCP 协议最新进展：新增了工具注解与异步任务支持。"
TOOL_TEXT = "我先检索一下最新进展。"
QUERY = "MCP 协议 最新进展"
TOOL_ARGS_JSON = json.dumps({"query": QUERY}, ensure_ascii=False)

MAX_TOKENS = 512

MODEL = {
    "openai": "gpt-test",
    "deepseek": "deepseek-flash",
    "qwen": "qwen3.7-plus",
    "anthropic": "claude-test",
}

THINKING_TEXT = "用户要的是最新进展，我需要调用搜索工具。"


def _tool_call_expect() -> dict[str, Any]:
    return {"name": "search_web", "arguments": {"query": QUERY}}


# --------------------------------------------------------------------------- #
# request.json（按各家文档手写）
# --------------------------------------------------------------------------- #


def _openai_request(provider: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL[provider],
        "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": USER_TEXT},
        ],
        "max_tokens": MAX_TOKENS,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": TOOL_DESCRIPTION,
                    "parameters": TOOL_PARAMETERS,
                },
            }
        ],
    }

    if provider in {"openai", "qwen"}:
        body["parallel_tool_calls"] = True

    return body


def _anthropic_request() -> dict[str, Any]:
    return {
        "model": MODEL["anthropic"],
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_TEXT,
        "messages": [{"role": "user", "content": USER_TEXT}],
        "tools": [
            {
                "name": "search_web",
                "description": TOOL_DESCRIPTION,
                "input_schema": TOOL_PARAMETERS,
            }
        ],
    }


# --------------------------------------------------------------------------- #
# response.json
# --------------------------------------------------------------------------- #


def _openai_text_response(provider: str, *, with_usage: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": f"chatcmpl-{provider}-text",
        "object": "chat.completion",
        "created": 1700000000,
        "model": MODEL[provider],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": TEXT_BASIC},
                "finish_reason": "stop",
            }
        ],
    }

    if with_usage:
        body["usage"] = {
            "prompt_tokens": 58,
            "completion_tokens": 22,
            "total_tokens": 80,
        }

    return body


def _openai_tool_response(provider: str) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_fixture01",
                "type": "function",
                "function": {"name": "search_web", "arguments": TOOL_ARGS_JSON},
            }
        ],
    }

    body: dict[str, Any] = {
        "id": f"chatcmpl-{provider}-tool",
        "object": "chat.completion",
        "created": 1700000000,
        "model": MODEL[provider],
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 58, "completion_tokens": 18, "total_tokens": 76},
    }

    if provider == "deepseek":
        # 思考模式下 reasoning_content 必须随消息返回，且下一轮要原样回传。
        message["reasoning_content"] = THINKING_TEXT
        body["usage"] = {
            "prompt_tokens": 60,
            "completion_tokens": 30,
            "total_tokens": 90,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 60,
        }

    return body


def _openai_tool_response_with_text(provider: str) -> dict[str, Any]:
    body = _openai_tool_response(provider)
    body["choices"][0]["message"]["content"] = TOOL_TEXT

    return body


def _anthropic_response(kind: str, *, with_usage: bool = True) -> dict[str, Any]:
    if kind == "text":
        content: list[dict[str, Any]] = [{"type": "text", "text": TEXT_BASIC}]
        stop_reason = "end_turn"
        response_id = "msg_fixture_text"
    else:
        content = [
            {"type": "text", "text": TOOL_TEXT},
            {
                "type": "tool_use",
                "id": "toolu_fixture01",
                "name": "search_web",
                "input": {"query": QUERY},
            },
        ]
        stop_reason = "tool_use"
        response_id = "msg_fixture_tool"

    body: dict[str, Any] = {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": MODEL["anthropic"],
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
    }

    if with_usage:
        body["usage"] = {"input_tokens": 120, "output_tokens": 42}

    return body


# --------------------------------------------------------------------------- #
# 错误响应体
# --------------------------------------------------------------------------- #

ERRORS: dict[str, dict[str, dict[str, Any]]] = {
    "openai": {
        "error_rate_limit": (
            {
                "error": {
                    "message": "Rate limit reached for gpt-test",
                    "type": "requests",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
            429,
        ),
        "error_quota": (
            {
                "error": {
                    "message": "You exceeded your current quota, check billing.",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_quota",
                }
            },
            429,
        ),
        "error_server": (
            {
                "error": {
                    "message": "The engine is currently overloaded, please try again later",
                    "type": "server_error",
                    "code": "engine_overloaded",
                }
            },
            503,
        ),
    },
    "deepseek": {
        "error_rate_limit": (
            {
                "error": {
                    "message": "Concurrent requests exceeded your limit",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
            429,
        ),
        "error_quota": (
            {
                "error": {
                    "message": "Insufficient Balance",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            },
            402,
        ),
        "error_server": (
            {
                "error": {
                    "message": "Server is too busy",
                    "type": "server_error",
                    "code": "server_overloaded",
                }
            },
            503,
        ),
    },
    "qwen": {
        "error_rate_limit": (
            {
                "code": "Throttling",
                "message": "Requests rate limit exceeded, please try again later.",
                "request_id": "req_fixture_qwen",
            },
            429,
        ),
        "error_quota": (
            {
                "code": "Arrearage",
                "message": "Access denied, please make sure your account is in good standing.",
                "request_id": "req_fixture_qwen",
            },
            400,
        ),
        "error_server": (
            {
                "code": "InternalError",
                "message": "An internal error has occurred.",
                "request_id": "req_fixture_qwen",
            },
            500,
        ),
    },
    "anthropic": {
        "error_rate_limit": (
            {
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Number of requests has exceeded your rate limit",
                },
                "request_id": "req_fixture_anthropic",
            },
            429,
        ),
        "error_quota": (
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Your credit balance is too low to access the Anthropic API.",
                },
                "request_id": "req_fixture_anthropic",
            },
            400,
        ),
        "error_server": (
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
                "request_id": "req_fixture_anthropic",
            },
            529,
        ),
    },
}


# --------------------------------------------------------------------------- #
# stream.txt（SSE 原文，逐字节保真）
# --------------------------------------------------------------------------- #


def _sse(data: dict[str, Any], *, event: str | None = None) -> list[str]:
    lines: list[str] = []

    if event:
        lines.append(f"event: {event}")

    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    lines.append("")

    return lines


def _openai_stream(provider: str, *, tool: bool) -> str:
    lines: list[str] = []

    if provider == "deepseek":
        # DeepSeek 长思考时会插注释行；解析器必须跳过而不是 json.loads。
        lines.extend([": keep-alive", ""])
        lines.extend(
            _sse(
                {
                    "id": "chatcmpl-stream",
                    "model": MODEL[provider],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": THINKING_TEXT},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )

    if tool:
        lines.extend(
            _sse(
                {
                    "id": "chatcmpl-stream",
                    "model": MODEL[provider],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
        for piece in _split_text(TOOL_TEXT):
            lines.extend(
                _sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            )

        lines.extend(
            _sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_fixture01",
                                        "type": "function",
                                        "function": {"name": "search_web", "arguments": ""},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
        )

        first, second = _split_arguments()
        lines.extend(
            _sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": first}}
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
        )
        lines.extend(
            _sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": second}}
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
        )

        finish_reason = "tool_calls"
    else:
        for piece in _split_text(TEXT_BASIC):
            lines.extend(
                _sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            )

        finish_reason = "stop"

    usage = {"prompt_tokens": 58, "completion_tokens": 22, "total_tokens": 80}

    if provider == "deepseek":
        # DeepSeek 把 usage 挂在最后一个内容块上（choices 非空），
        # 靠 "choices 为空" 判断 usage 会永远读不到。
        lines.extend(
            _sse(
                {
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                    "usage": usage,
                }
            )
        )
    else:
        lines.extend(
            _sse(
                {
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ]
                }
            )
        )
        lines.extend(_sse({"choices": [], "usage": usage}))

    lines.append("data: [DONE]")
    lines.append("")

    return "\n".join(lines)


def _anthropic_stream(*, tool: bool) -> str:
    lines: list[str] = []

    lines.extend(
        _sse(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_fixture_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL["anthropic"],
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 120, "output_tokens": 1},
                },
            },
            event="message_start",
        )
    )

    lines.extend(
        _sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            event="content_block_start",
        )
    )

    # 心跳事件：不是业务事件，忽略它才会把解析器写对。
    lines.extend(_sse({"type": "ping"}, event="ping"))

    for piece in _split_text(TOOL_TEXT if tool else TEXT_BASIC):
        lines.extend(
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": piece},
                },
                event="content_block_delta",
            )
        )

    lines.extend(
        _sse({"type": "content_block_stop", "index": 0}, event="content_block_stop")
    )

    if tool:
        lines.extend(
            _sse(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_fixture01",
                        "name": "search_web",
                        "input": {},
                    },
                },
                event="content_block_start",
            )
        )

        first, second = _split_arguments()
        for piece in (first, second):
            lines.extend(
                _sse(
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": piece},
                    },
                    event="content_block_delta",
                )
            )

        lines.extend(
            _sse({"type": "content_block_stop", "index": 1}, event="content_block_stop")
        )

    stop_reason = "tool_use" if tool else "end_turn"

    lines.extend(
        _sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 42},
            },
            event="message_delta",
        )
    )

    lines.extend(_sse({"type": "message_stop"}, event="message_stop"))

    return "\n".join(lines)


def _split_text(text: str) -> tuple[str, str]:
    midpoint = max(1, len(text) // 2)

    return text[:midpoint], text[midpoint:]


def _split_arguments() -> tuple[str, str]:
    midpoint = max(1, len(TOOL_ARGS_JSON) // 2)

    return TOOL_ARGS_JSON[:midpoint], TOOL_ARGS_JSON[midpoint:]


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #


def _write_case(
    provider: str,
    case: str,
    *,
    request: dict[str, Any],
    response: dict[str, Any] | None,
    meta: dict[str, Any],
    stream: str | None = None,
) -> None:
    case_dir = FIXTURES_ROOT / provider / case
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )

    if response is not None:
        (case_dir / "response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )

    if stream is not None:
        (case_dir / "stream.txt").write_text(stream, "utf-8")

    payload = {
        "synthetic": True,
        "recorded_at": None,
        "note": "手写协议样本，Day5 用真机录制替换",
        "status_code": 200,
        **meta,
    }

    (case_dir / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )


def main() -> None:
    for provider in ("openai", "deepseek", "qwen"):
        request = _openai_request(provider)

        _write_case(
            provider,
            "text_basic",
            request=request,
            response=_openai_text_response(provider),
            meta={
                "expect": {
                    "text": TEXT_BASIC,
                    "finish_reason": "stop",
                    "tool_calls": [],
                    "usage_estimated": False,
                }
            },
            stream=_openai_stream(provider, tool=False),
        )

        _write_case(
            provider,
            "tool_call_basic",
            request=request,
            response=_openai_tool_response_with_text(provider),
            meta={
                "expect": {
                    "text": TOOL_TEXT,
                    "finish_reason": "tool_calls",
                    "tool_calls": [_tool_call_expect()],
                    "usage_estimated": False,
                }
            },
            stream=_openai_stream(provider, tool=True),
        )

        _write_case(
            provider,
            "usage_missing",
            request=request,
            response=_openai_text_response(provider, with_usage=False),
            meta={
                "expect": {
                    "text": TEXT_BASIC,
                    "finish_reason": "stop",
                    "tool_calls": [],
                    "usage_estimated": True,
                }
            },
        )

        for case, (body, status_code) in ERRORS[provider].items():
            _write_case(
                provider,
                case,
                request=request,
                response=body,
                meta={"status_code": status_code, "expect_error": True},
            )

    anthropic_request = _anthropic_request()

    _write_case(
        "anthropic",
        "text_basic",
        request=anthropic_request,
        response=_anthropic_response("text"),
        meta={
            "expect": {
                "text": TEXT_BASIC,
                "finish_reason": "stop",
                "tool_calls": [],
                "usage_estimated": False,
            }
        },
        stream=_anthropic_stream(tool=False),
    )

    _write_case(
        "anthropic",
        "tool_call_basic",
        request=anthropic_request,
        response=_anthropic_response("tool"),
        meta={
            "expect": {
                "text": TOOL_TEXT,
                "finish_reason": "tool_calls",
                "tool_calls": [_tool_call_expect()],
                "usage_estimated": False,
            }
        },
        stream=_anthropic_stream(tool=True),
    )

    _write_case(
        "anthropic",
        "usage_missing",
        request=anthropic_request,
        response=_anthropic_response("text", with_usage=False),
        meta={
            "expect": {
                "text": TEXT_BASIC,
                "finish_reason": "stop",
                "tool_calls": [],
                "usage_estimated": True,
            }
        },
    )

    for case, (body, status_code) in ERRORS["anthropic"].items():
        _write_case(
            "anthropic",
            case,
            request=anthropic_request,
            response=body,
            meta={"status_code": status_code, "expect_error": True},
        )


if __name__ == "__main__":
    main()
    print(f"fixtures written to {FIXTURES_ROOT}")
