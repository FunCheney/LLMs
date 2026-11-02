"""W2 端到端自检：过载 503 / 任务端点 / trace 贯穿（离线档，零网络）。

为什么单独一个脚本而不是写进 pytest
--------------------------------
这三条都要**真起一个 HTTP 服务**并把并发限额压到 1。放进 pytest 会让
`make test` 变慢且和端口绑定；放在这里人工跑一次，专治"单测全绿但线上行为不同"。

为什么过载用例要单独起一个"慢"的服务
----------------------------------
`mock` provider 是**同步读文件**回放：它在自己的协程里不 await 任何 IO，
于是 60 个并发请求在事件循环里被逐个执行，同一时刻只有 1 个在飞——
限额 1 也就不可能超，"压不出来"不是限流没接上，是压测对象本来就没并发。
所以过载用例用一个只多了 `await asyncio.sleep(0.2)` 的真实 gateway 子类：
路由、错误映射、中间件、限流全部走生产代码，唯一差别是"这次调用确实花时间"。

为什么在**进程内**起 uvicorn 而不是 subprocess
--------------------------------------------
限额靠临时 catalog。subprocess + env 一旦没生效，"没触发 503"就无法区分是
**背压没接上**还是**限额没读对**。同进程里直接持有 `limiter`，先把
`limit_for()` 打出来，两类问题立刻分开（第一版就踩过：30 并发全 200）。

用法
----
    uv run python evals/scripts/w2_e2e_check.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import httpx
import uvicorn
import yaml
from agent_framework.bootstrap import build_gateway, build_limiter
from agent_framework.config import Catalog, Settings, load_catalog
from agent_framework.llm.client import ChatCall, ChatOutcome, LLMGateway
from agent_framework.llm.registry import Registry
from agent_framework.llm.router import TaskRouter
from agent_framework.server.app import create_app
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "packages/agent_framework/src/agent_framework/models.yaml"
FAST_PORT = 8011
SLOW_PORT = 8012
LOG_PATH = Path("/tmp/w2_e2e_server.log")


class _SlowGateway(LLMGateway):
    """真实 gateway + 一点真实耗时。

    只覆盖 `chat`，其余（precheck/路由/重试/降级/日志）全部继承：
    这样测到的 503 是**入口层**产生的，而不是某个替身自己造的。
    """

    def __init__(
        self,
        *,
        delay_s: float,
        registry: Registry,
        router: Any,
        catalog: Catalog,
        settings: Settings,
    ) -> None:
        super().__init__(
            registry=registry, router=router, catalog=catalog, settings=settings
        )
        self._delay_s = delay_s

    async def chat(self, call: ChatCall) -> ChatOutcome:
        await asyncio.sleep(self._delay_s)

        return await super().chat(call)


def write_models(global_limit: int, mock_limit: int) -> Path:
    """把限额压到 1 的临时 catalog。

    `per_provider.mock` 也要一起压：只压 global 时 alias 层仍是 32，
    并发根本碰不到总闸——"限额设成 1 还不 503"通常就是这个原因。
    """

    catalog = yaml.safe_load(MODELS.read_text(encoding="utf-8"))
    catalog["limits"]["global"] = global_limit
    catalog["limits"]["per_provider"]["mock"] = mock_limit

    tmp = Path(tempfile.mkstemp(suffix=".yaml")[1])
    tmp.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")

    return tmp


class _RecordCollector(logging.Handler):
    """把框架日志收进内存。

    不用 `FileHandler`：本机沙箱对文件句柄有包装，第一版写出来的文件是空的，
    结果"trace 找不到"看起来像功能坏了。内存列表没有这个不确定性。
    """

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


_COLLECTOR = _RecordCollector()


def parse_records() -> list[dict[str, Any]]:
    records = []

    for line in _COLLECTOR.lines:
        start = line.find('{"')

        if start < 0:
            continue

        try:
            records.append(json.loads(line[start:]))
        except ValueError:
            continue

    return records


async def wait_ready(client: httpx.AsyncClient, port: int) -> None:
    for _ in range(300):
        try:
            response = await client.get(f"http://127.0.0.1:{port}/health")

            if response.status_code == 200:
                return
        except httpx.HTTPError:
            await asyncio.sleep(0.05)

    raise RuntimeError(f"port {port} 上的服务没起来")


async def serve(app: FastAPI, port: int) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    return server, asyncio.create_task(server.serve())


async def stop(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    server.should_exit = True
    await task


async def check_chat_and_tasks(client: httpx.AsyncClient, failures: list[str]) -> None:
    payload = {"messages": [{"role": "user", "content": "跑一下"}], "model": "fixture"}

    response = await client.post("/v1/chat", json=payload)
    alias = response.headers.get("x-model-alias")
    print(f"[A] POST /v1/chat → {response.status_code} alias={alias}")

    if response.status_code != 200:
        failures.append(f"A: /v1/chat 返回 {response.status_code}")

    submit = await client.post(
        "/v1/tasks",
        json={"messages": [{"role": "user", "content": "异步跑一下"}], "model": "fixture"},
    )
    task = submit.json()
    task_id = task.get("task_id", "")
    status = task.get("status")

    for _ in range(400):
        poll = await client.get(f"/v1/tasks/{task_id}")
        status = poll.json().get("status")

        if status in {"succeeded", "failed"}:
            break

        await asyncio.sleep(0.02)

    print(f"[B] POST /v1/tasks → {submit.status_code} task_id={task_id} 终态={status}")

    if submit.status_code not in (200, 202):
        failures.append(f"B: 提交返回 {submit.status_code}")

    if status != "succeeded":
        failures.append(f"B: 终态是 {status}（应为 succeeded）")


async def check_trace(client: httpx.AsyncClient, failures: list[str]) -> None:
    payload = {"messages": [{"role": "user", "content": "追一下 trace"}], "model": "fixture"}

    async with client.stream("POST", "/v1/chat", json={**payload, "stream": True}) as response:
        request_id = response.headers.get("x-request-id", "")
        trace_id = response.headers.get("x-trace-id", "")
        events = [line async for line in response.aiter_lines() if line]

    await asyncio.sleep(0.3)
    mine = [item for item in parse_records() if item.get("request_id") == request_id]
    stream_records = [item for item in mine if item.get("stream")]

    print(f"[C] stream 帧数={len(events)} header_trace={trace_id[:14]} 日志条数={len(mine)}")

    if not request_id or not trace_id:
        failures.append("C: 响应头缺 X-Request-ID / X-Trace-ID")

    if not stream_records:
        failures.append("C: 日志里找不到这条流式请求的 RequestLog")

        return

    record = stream_records[0]

    if record.get("trace_id") != trace_id:
        failures.append(f"C: 日志 trace_id {record.get('trace_id')} != 响应头 {trace_id}")

    if not record.get("span_id"):
        failures.append("C: RequestLog.span_id 为空，span 没包到 llm.call")

    if record.get("first_token_ms") is None:
        failures.append("C: 流式日志 first_token_ms 为空")

    print(
        f"[C] trace 一致={record.get('trace_id') == trace_id} "
        f"span_id={record.get('span_id')} first_token_ms={record.get('first_token_ms')}"
    )


async def check_overload(client: httpx.AsyncClient, failures: list[str]) -> None:
    payload = {"messages": [{"role": "user", "content": "压我不"}], "model": "fixture"}
    responses = await asyncio.gather(
        *[client.post("/v1/chat", json=payload) for _ in range(20)]
    )
    busy = [response for response in responses if response.status_code == 503]
    ok = [response for response in responses if response.status_code == 200]
    other = len(responses) - len(busy) - len(ok)

    print(f"[D] 20 并发（单请求 200ms、限额 1）: 200={len(ok)} 503={len(busy)} 其他={other}")

    if not busy:
        failures.append("D: 限额 1 且每个请求 200ms，仍未触发 503，背压没生效")

        return

    missing = [response for response in busy if not response.headers.get("retry-after")]
    code = busy[0].json().get("code")

    print(f"[D] Retry-After={busy[0].headers.get('retry-after')} code={code}")

    if missing:
        failures.append(f"D: {len(missing)} 个 503 缺 Retry-After")

    if code != "SERVER_BUSY":
        failures.append(f"D: 503 错误码不是 SERVER_BUSY：{code}")


async def check_no_retry_storm(failures: list[str]) -> None:
    records = parse_records()
    retries = [item for item in records if (item.get("retry_count") or 0) > 0]
    error_codes = {item.get("error_code") for item in records if item.get("error_code")}

    print(
        f"[E] 日志 {len(records)} 条，retry_count>0 的 {len(retries)} 条，"
        f"error_code={error_codes}"
    )

    if retries:
        failures.append(f"E: 有 {len(retries)} 条日志 retry_count>0（过载被当成可重试）")


async def main() -> int:
    models_path = write_models(global_limit=1, mock_limit=1)
    settings = Settings(_env_file=None, models_yaml=models_path, overload_wait_s=0.02)
    catalog = load_catalog(models_path)

    handler = _COLLECTOR
    # 框架的结构化日志挂在 `agent_framework` 这棵树上（configure_logging 只动它、
    # 不动 root），抓包必须抓这棵树，否则文件里一条 RequestLog 都没有。
    framework_logger = logging.getLogger("agent_framework")
    framework_logger.addHandler(handler)
    framework_logger.setLevel(logging.INFO)

    limiter = build_limiter(catalog, settings)
    fast_app = create_app(build_gateway(settings, catalog=catalog), limiter=limiter)

    slow_limiter = build_limiter(catalog, settings)
    slow_gateway = _SlowGateway(
        delay_s=0.2,
        registry=Registry(catalog, settings),
        router=TaskRouter(catalog),
        catalog=catalog,
        settings=settings,
    )
    slow_app = create_app(slow_gateway, limiter=slow_limiter)

    fast_server, fast_task = await serve(fast_app, FAST_PORT)
    slow_server, slow_task = await serve(slow_app, SLOW_PORT)

    failures: list[str] = []

    try:
        print(
            f"[0] limit_for('fixture')={limiter.limit_for('fixture')} "
            f"overload_wait_s={settings.overload_wait_s}"
        )

        if limiter.limit_for("fixture") != 1:
            failures.append("0: 限额没读到 1，过载用例没有意义")

        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            await wait_ready(client, FAST_PORT)
            await wait_ready(client, SLOW_PORT)

            fast = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{FAST_PORT}", timeout=30.0, trust_env=False
            )
            slow = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{SLOW_PORT}", timeout=30.0, trust_env=False
            )
            async with fast, slow:
                await check_chat_and_tasks(fast, failures)
                await check_trace(fast, failures)
                await check_overload(slow, failures)

            await check_no_retry_storm(failures)
    finally:
        await stop(fast_server, fast_task)
        await stop(slow_server, slow_task)
        framework_logger.removeHandler(handler)
        handler.close()

    if failures:
        print("\n失败项:")

        for item in failures:
            print(f" - {item}")

        return 1

    print("\n全部通过：聊天/任务/过载 503+Retry-After/trace 一致/无重试风暴")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
