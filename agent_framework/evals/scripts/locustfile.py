"""W2 D4 · 压测脚本（locust）：打 `/v1/chat` 的 stream 与非 stream 两类。

为什么用 locust 而不是自己写 asyncio 循环
----------------------------------------
自己写循环最后一定演化成一个半成品压测框架（并发爬坡、失败重试、分位数统计、
实时曲线），而这些 locust 都有了。我们要的只是"别把定义搞成两套"。

口径（与 W1 Day5 一致，不许另起一套）
------------------------------------
- **延迟/首 token 只从 `RequestLog` 读**（`evals/scripts/bench_baseline.py` 的 LogCapture）。
  这里额外报的 `first_byte_ms` 只是"客户端视角的到达时间"，用于排查代理/网络，
  **不要**拿它当 `first_token_ms` 用：两者定义不同（首字节 vs 首个 text_delta）。
- 报告必须带时间戳：DeepSeek 峰谷计价差 2 倍，"成本降了 40%"可能只是换了时段。

用法
----
    # 0) 先起服务（离线档用 fixture，不花钱）
    uv run agent-server                      # :8000

    # 1) 离线档：确认工具链与端点没被压崩（默认 20 并发）
    uv run --with locust locust -f evals/scripts/locustfile.py \
        --headless -u 20 -r 5 -t 30s --host http://127.0.0.1:8000

    # 2) 在线档：换真实 alias（会花钱），并发按 models.yaml 的 limits 来
    MODEL=qwen3.7-plus uv run --with locust locust -f evals/scripts/locustfile.py \
        --headless -u 100 -r 10 -t 60s --host http://127.0.0.1:8000

判据（Gate「100 并发错误率 < 1%」）
----------------------------------
locust 退出后看 `Failures` 与 `Aggregated` 的 error rate；超过 1% 即不达标。
刻意**不**用 locust 的自动失败判定来做门禁：CI 里跑 100 并发需要真服务 + Redis，
那是 `make load` 的手工步骤，不是每次提交都要跑的单元门禁。
"""

from __future__ import annotations

import json
import os
import time

from locust import HttpUser, between, events, task

# 离线档默认 fixture（mock 回放）；在线档用 MODEL=<alias> 覆盖。
MODEL = os.getenv("MODEL", "fixture")
PROMPT = os.getenv("PROMPT", "用一句话说明什么是增长营销")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "64"))


def _payload(*, stream: bool) -> dict[str, object]:
    return {
        "messages": [{"role": "user", "content": PROMPT}],
        "model": MODEL,
        "stream": stream,
        "max_tokens": MAX_TOKENS,
    }


class ChatUser(HttpUser):
    """3:1 的流式/非流式比例，贴近真实前端（多数请求是非流式批处理）。"""

    wait_time = between(0.2, 0.6)

    @task(3)
    def chat_blocking(self) -> None:
        with self.client.post(
            "/v1/chat",
            json=_payload(stream=False),
            name="POST /v1/chat (blocking)",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code} body={response.text[:200]}")
                return

            try:
                body = response.json()
            except json.JSONDecodeError:
                response.failure("返回体不是 JSON")
                return

            # 200 也可能是坏结果：没有 request_id 说明中间件没生效，
            # finish_reason=error 说明上游挂了但被包成了成功响应。
            if not body.get("request_id"):
                response.failure("缺少 request_id")

    @task(1)
    def chat_stream(self) -> None:
        started = time.perf_counter()

        with self.client.post(
            "/v1/chat",
            json=_payload(stream=True),
            name="POST /v1/chat (stream)",
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code}")
                return

            first_byte_ms: float | None = None
            frames = 0
            saw_end = False

            for line in response.iter_lines():
                if first_byte_ms is None and line:
                    first_byte_ms = (time.perf_counter() - started) * 1000

                if line.startswith(b"event: message_end"):
                    saw_end = True

                if line.startswith(b"event: "):
                    frames += 1

            # 单独记一条自定义指标：客户端视角的首字节到达时间。
            # 它与 RequestLog 的 first_token_ms 定义不同，只用于排查网络/代理。
            if first_byte_ms is not None:
                events.request.fire(
                    request_type="SSE",
                    name="stream first byte",
                    response_time=first_byte_ms,
                    response_length=0,
                    exception=None,
                    context={},
                )

            if not saw_end:
                response.failure("流没有以 message_end 结束（可能半截断开）")

            if frames == 0:
                response.failure("一帧都没有")


@events.test_start.add_listener
def _log_config(environment: object, **_: object) -> None:
    print(f"[load] model={MODEL} prompt={PROMPT!r} max_tokens={MAX_TOKENS}")
    print("[load] 指标口径：延迟/首 token 以 RequestLog 为准，本脚本的 first byte 仅作网络参考")
