"""Day5 · 真机录制夹具（把"手写协议样本"换成"真机流量"）。

为什么需要它
------------
手写夹具（`meta.synthetic = true`）证明的是"adapter 与我理解的协议一致"，
不是"adapter 与厂商真实返回一致"。这两件事的差距，只有真流量能暴露：
厂商加了字段、`usage` 换了位置、`finish_reason` 多了个新值、思考模式多了一种块。

怎么录（关键设计）
------------------
**不另写一套 HTTP 客户端**，而是把真 adapter 的流量接进 `RecordingTransport`：
它包住真的 `httpx.AsyncHTTPTransport`，透传的同时把 request / response 原始
bytes 抄一份。这样录下来的 body 就是"生产路径真正会发出的 body"，
不是"录制脚本以为的 body"——后者是假绿的第二大来源。

`request.json` **不覆盖**：它是 A1 的断言锚（`build_request` 必须精确产出它）。
录制时把它与真实发出的 body 做 diff，一致才算录成功；不一致就停下来让人看，
因为那意味着"要发的"和"断言过的"已经不是同一份东西了。

用法
----
    # 1) 只看会发生什么（默认 dry-run，不写盘、不发请求）
    uv run python evals/scripts/record_fixtures.py --providers openai,deepseek

    # 2) 真录（需要 .env.local 里有 key）
    uv run python evals/scripts/record_fixtures.py --providers openai,deepseek --apply

    # 3) 只为验证"上游鉴权失败 → 502 而不是 401"录一份真实错误体
    uv run python evals/scripts/record_fixtures.py --bad-key openai --apply

    # 4) 录完重跑 conformance（离线，零网络）
    uv run pytest --basetemp=.pytest_tmp

成本：每 provider 2 次成功调用（1 次非流式 + 1 次流式），单次 512 max_tokens，
四家合计约 ¥0.01 量级。
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "packages" / "agent_framework" / "tests"
FIXTURES_ROOT = TESTS_DIR / "fixtures"

# 让脚本与 conformance 用**同一份**输入（tests/ 不是包，靠 sys.path 注入即可）。
sys.path.insert(0, str(TESTS_DIR))

from agent_framework.config import KEY_ENV_TO_FIELD, Settings, load_catalog  # noqa: E402
from agent_framework.llm.errors import LLMError  # noqa: E402
from agent_framework.llm.registry import Registry  # noqa: E402
from agent_framework.llm.types import LLMResponse  # noqa: E402
from canonical_input import MAX_TOKENS, canonical_messages, canonical_tools  # noqa: E402

# 能被安全录制的 case。`usage_missing` 是人为制造的"厂商不给 usage"，
# 真实厂商会给，录出来就不再是 usage_missing —— 这类"异常态"只能手写。
RECORDABLE_CASES: tuple[str, ...] = ("text_basic", "tool_call_basic")

BAD_KEY = "sk-invalid-probe-do-not-use-0000"

_REDACTION = "sk-***REDACTED***"


# --------------------------------------------------------------------------- #
# 抓包
# --------------------------------------------------------------------------- #


@dataclass
class Capture:
    """一次真实请求/响应的快照。"""

    url: str
    request_body: dict[str, Any]
    status_code: int
    content_type: str
    sink: bytearray = field(default_factory=bytearray)

    @property
    def body(self) -> bytes:
        return bytes(self.sink)

    @property
    def is_event_stream(self) -> bool:
        return "text/event-stream" in self.content_type

    def json_body(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.body or b"{}")
        except ValueError:
            return {}

        return payload if isinstance(payload, dict) else {}


class _TeeStream(httpx.AsyncByteStream):
    """边透传边抄一份。

    必须包在 transport 返回的 `Response.stream` 上而不是在 adapter 里截流：
    adapter 的薄壳用 `aiter_lines()` 消费，按行截流会丢掉 SSE 的分帧细节
    （`event:` 行、冒号心跳、空行），而夹具要的是**逐字节**原文。
    """

    def __init__(self, inner: httpx.AsyncByteStream, sink: bytearray) -> None:
        self._inner = inner
        self._sink = sink

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        async for chunk in self._inner:
            self._sink.extend(chunk)
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()


class RecordingTransport(httpx.AsyncBaseTransport):
    """透传真请求，顺带把原始 bytes 抄下来。"""

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()
        self.captures: list[Capture] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            request_body = json.loads(request.content or b"{}")
        except ValueError:
            request_body = {}

        response = await self._inner.handle_async_request(request)
        capture = Capture(
            url=str(request.url),
            request_body=request_body if isinstance(request_body, dict) else {},
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
        )
        self.captures.append(capture)

        if response.is_stream_consumed:
            # 替身（MockTransport / 已 aread 的响应）走这里：body 已在内存里，
            # 没有 stream 可 tee。真实网络路径永远走下面的 tee 分支。
            capture.sink.extend(response.content)
        else:
            # 换成 tee：httpx 之后的 aread()/aiter_lines() 都会经过它
            response.stream = _TeeStream(response.stream, capture.sink)

        return response

    async def aclose(self) -> None:
        await self._inner.aclose()

    def clear(self) -> None:
        self.captures.clear()


# --------------------------------------------------------------------------- #
# 脱敏
# --------------------------------------------------------------------------- #


def sanitize(raw: bytes, secrets: list[str]) -> bytes:
    """错误体里厂商经常回显 key 片段，落盘前必须抹掉。

    只抹长度 >= 8 的串：把空串或 "sk" 当秘密替换，会把整个 body 打成马赛克。
    """

    text = raw.decode("utf-8", errors="replace")

    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, _REDACTION)

    return text.encode("utf-8")


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )


def expect_from_response(response: LLMResponse) -> dict[str, Any]:
    """expect 只做**结构投影**，不搬运任何厂商字段。

    它是"回放路径必须复现这次真机解析结果"的锚：跨进程、跨时间都成立。
    """

    return {
        "text": response.text or "",
        "finish_reason": str(response.finish_reason),
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ],
        "usage_estimated": response.usage.estimated,
    }


def write_case(
    provider: str,
    case: str,
    *,
    response_body: dict[str, Any] | None,
    stream_text: str | None,
    meta: dict[str, Any],
    root: Path | None = None,
) -> Path:
    case_dir = (root or FIXTURES_ROOT) / provider / case
    case_dir.mkdir(parents=True, exist_ok=True)

    if response_body is not None:
        _write_json(case_dir / "response.json", response_body)

    if stream_text is not None:
        (case_dir / "stream.txt").write_text(stream_text, "utf-8")

    _write_json(case_dir / "meta.json", meta)

    return case_dir


def recorded_meta(
    *,
    provider_model: str,
    status_code: int,
    expect: dict[str, Any] | None = None,
    expect_error: bool = False,
    note: str = "真机录制（Day5）",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "synthetic": False,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": note,
        "status_code": status_code,
        "provider_model": provider_model,
    }

    if expect is not None:
        payload["expect"] = expect

    if expect_error:
        payload["expect_error"] = True

    return payload


# --------------------------------------------------------------------------- #
# request 漂移检查
# --------------------------------------------------------------------------- #


def diff_request(
    provider: str,
    case: str,
    body: dict[str, Any],
    *,
    root: Path | None = None,
) -> list[str]:
    """比对"真实要发的 body"与 A1 的断言锚。返回 unified diff 行（空=一致）。"""

    path = (root or FIXTURES_ROOT) / provider / case / "request.json"

    if not path.is_file():
        return []

    expected = json.loads(path.read_text("utf-8"))

    if expected == body:
        return []

    return list(
        difflib.unified_diff(
            json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True).splitlines(),
            json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True).splitlines(),
            fromfile=f"{provider}/{case}/request.json（断言锚）",
            tofile="真实发出的 body",
            lineterm="",
        )
    )


# --------------------------------------------------------------------------- #
# 录制主流程
# --------------------------------------------------------------------------- #


def resolve_alias(catalog: Any, provider: str) -> str:
    """provider → 第一个 alias。录哪个模型由 models.yaml 决定，不写死在脚本里。"""

    for alias, model in catalog.models.items():
        if model.provider == provider:
            return str(alias)

    raise SystemExit(f"models.yaml 里没有 provider '{provider}' 的模型")


async def record_provider(
    provider: str,
    cases: tuple[str, ...],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    settings: Settings | None = None,
    apply: bool = False,
    allow_request_drift: bool = False,
    fixtures_root: Path | None = None,
    out: Any = print,
) -> list[str]:
    """录一个 provider 的指定 case。返回"本次写盘/待写盘"的路径列表。

    `fixtures_root` 只为测试而存在：默认写真实夹具目录，测试必须写 tmp_path，
    否则一次跑测试就把真机基线冲掉了。
    """

    root = fixtures_root or FIXTURES_ROOT

    resolved_settings = settings or Settings()
    catalog = load_catalog(resolved_settings.models_yaml)
    alias = resolve_alias(catalog, provider)
    provider_cfg = catalog.get_provider(alias)[1]

    secrets: list[str] = []

    try:
        secrets.append(resolved_settings.api_key(provider_cfg.api_key_env))
    except Exception as exc:  # noqa: BLE001 - 缺 key 时给出可操作提示，而不是栈
        raise SystemExit(f"[{provider}] 取不到密钥：{exc}") from exc

    recorder = RecordingTransport(transport)
    registry = Registry(catalog, resolved_settings, transport=recorder)
    adapter = registry.get(alias)
    model_name = adapter.model

    written: list[str] = []

    try:
        for case in cases:
            recorder.clear()
            out(f"[{provider}] {case}: 录制中（wire model = {model_name}）")

            response = await adapter.chat(
                canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
            )

            if not recorder.captures:
                raise SystemExit(f"[{provider}] {case}: 没有捕获到任何请求")

            chat_capture = recorder.captures[0]

            drift = diff_request(provider, case, chat_capture.request_body, root=root)

            if drift and not allow_request_drift:
                out("\n".join(drift))
                raise SystemExit(
                    f"[{provider}/{case}] 将要发出的 body 与 request.json 不一致。\n"
                    "先确认 A1 的断言锚是否该更新（这通常是改了 canonical 输入），"
                    "确认后再加 --allow-request-drift。"
                )

            response_body = chat_capture.json_body()
            stream_text: str | None = None

            if case == "tool_call_basic":
                recorder.clear()
                async for _event in adapter.stream(
                    canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
                ):
                    pass

                stream_capture = next(
                    (item for item in recorder.captures if item.is_event_stream), None
                )

                if stream_capture is None:
                    raise SystemExit(f"[{provider}] {case}: 流式请求没有被捕获")

                stream_text = sanitize(stream_capture.body, secrets).decode(
                    "utf-8", errors="replace"
                )

            meta = recorded_meta(
                provider_model=model_name,
                status_code=chat_capture.status_code,
                expect=expect_from_response(response),
            )

            target = root / provider / case

            if apply:
                write_case(
                    provider,
                    case,
                    response_body=response_body,
                    stream_text=stream_text,
                    meta=meta,
                    root=root,
                )
                out(f"[{provider}] {case}: 已写入 {target}")
            else:
                out(f"[{provider}] {case}: dry-run，将写入 {target}（加 --apply 生效）")

            written.append(str(target))
    finally:
        await registry.aclose()

    return written


async def probe_bad_key(
    provider: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    settings: Settings | None = None,
    apply: bool = False,
    fixtures_root: Path | None = None,
    out: Any = print,
) -> None:
    """用故意无效的 key 打一次，录真实鉴权失败体（成本 0、无副作用）。

    这是"上游鉴权失败必须映射成 502 而不是 401"唯一可信的证据来源：
    手写的 401 体只能证明我们的映射函数跟自己的想象一致。
    """

    resolved_settings = settings or Settings()
    catalog = load_catalog(resolved_settings.models_yaml)
    alias = resolve_alias(catalog, provider)
    provider_cfg = catalog.get_provider(alias)[1]

    if not provider_cfg.api_key_env:
        raise SystemExit(f"[{provider}] 未声明 api_key_env，无法探测")

    field_name = KEY_ENV_TO_FIELD[provider_cfg.api_key_env]
    real_key = resolved_settings.api_key(provider_cfg.api_key_env)
    broken = resolved_settings.model_copy(update={field_name: SecretStr(BAD_KEY)})

    recorder = RecordingTransport(transport)
    registry = Registry(catalog, broken, transport=recorder)
    adapter = registry.get(alias)

    try:
        try:
            await adapter.chat(
                canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS
            )
        except LLMError as exc:
            capture = recorder.captures[0] if recorder.captures else None

            if capture is None:
                raise SystemExit(f"[{provider}] 鉴权探测没有被捕获到") from exc

            body = json.loads(sanitize(capture.body, [real_key, BAD_KEY]).decode("utf-8"))
            meta = recorded_meta(
                provider_model=adapter.model,
                status_code=capture.status_code,
                expect_error=True,
                note="真机录制：故意使用无效 key，验证鉴权失败映射",
            )

            out(
                f"[{provider}] error_auth: HTTP {capture.status_code} → "
                f"{type(exc).__name__}（retryable={exc.retryable}）"
            )

            if apply:
                target = write_case(
                    provider,
                    "error_auth",
                    response_body=body,
                    stream_text=None,
                    meta=meta,
                    root=fixtures_root,
                )
                out(f"[{provider}] error_auth: 已写入 {target}")
            else:
                out(f"[{provider}] error_auth: dry-run（加 --apply 生效）")
        else:
            raise SystemExit(
                f"[{provider}] 无效 key 居然调用成功了——请确认 .env.local 是否被代理覆盖"
            )
    finally:
        await registry.aclose()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真机录制 LLM 夹具（Day5）")
    parser.add_argument(
        "--providers",
        default="openai,deepseek,qwen,anthropic",
        help="逗号分隔；默认四家全录",
    )
    parser.add_argument(
        "--cases",
        default=",".join(RECORDABLE_CASES),
        help=f"逗号分隔；可录的只有 {RECORDABLE_CASES}",
    )
    parser.add_argument("--apply", action="store_true", help="真发请求并写盘（默认 dry-run）")
    parser.add_argument(
        "--allow-request-drift",
        action="store_true",
        help="允许真实 body 与 request.json 不一致（会破坏 A1，慎用）",
    )
    parser.add_argument(
        "--bad-key",
        default="",
        help="逗号分隔的 provider：用无效 key 录真实鉴权失败体",
    )
    args = parser.parse_args(argv)

    providers = _parse_list(args.providers)
    cases = _parse_list(args.cases)

    unknown = set(cases) - set(RECORDABLE_CASES)

    if unknown:
        raise SystemExit(
            f"不可录制的 case：{sorted(unknown)}；"
            f"异常态（usage_missing / error_*）只能手写，真实厂商不会稳定复现"
        )

    async def runner() -> None:
        if args.bad_key:
            for provider in _parse_list(args.bad_key):
                await probe_bad_key(provider, apply=args.apply)
            return

        for provider in providers:
            await record_provider(
                provider,
                cases,
                apply=args.apply,
                allow_request_drift=args.allow_request_drift,
            )

    asyncio.run(runner())

    if not args.apply:
        print("\n以上是 dry-run。确认无误后加 --apply 真录。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
