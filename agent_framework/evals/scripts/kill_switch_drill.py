#!/usr/bin/env python
"""Kill Switch 演练（W11 D5 / Gate 11 #9）。

    python evals/scripts/kill_switch_drill.py

**判据：从"决定回退"到"行为生效"≤ 5 分钟**，且回退后系统仍能正常应答。

这个脚本真的起一个 uvicorn 进程、真的发 HTTP、真的改 env 重启，
而不是"读一下代码说它会生效"。三件事必须测到：

1. 开着的证据：同一个 brief 发两次，第二次 `input_tokens == 0`（命中缓存）；
2. 关掉的证据：重启后同样发两次，第二次 `input_tokens > 0`（又打上游了）；
3. 代价：重启窗口内的**不可用时长**（这就是"期间错过的请求"的量纲）。

## 为什么用 mock 厂商

演练不能依赖真实密钥与真实账单：一个"要先配 key 才能演练"的回退流程，
在出事那天一定演练不了。`models.yaml` 里的 `mock` provider 走离线回放，
所以这条演练在任何机器上都能跑。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEADLINE_S = 300  # Gate 11 #9：≤5 分钟
BRIEF = "给一款记账 App 写一句开学季 promo 文案"

MODELS_YAML = """\
providers:
  # mock 的 `delegate` 指向一个**已登记**的 provider：它复用那个 provider 的
  # 解析逻辑，但不发网络请求。不写这一段，启动就会报"delegate 不是已知 provider"。
  openai:
    type: openai_compat
    base_url: "https://api.openai.com/v1"
    api_key_env: OPENAI_API_KEY
  mock:
    type: mock
    flags:
      delegate: openai
      case: text_basic
      fixtures_root: packages/agent_framework/tests/fixtures

models:
  fixture:
    provider: mock
    model: ""
    tier: cheap
    capabilities: [stream, tools]
    pricing: { input: 0, output: 0, currency: CNY, per: 1000000 }

routes:
  default: { primary: fixture, fallback: [] }
  cheap:  { primary: fixture, fallback: [] }

tasks:
  chat: cheap

limits:
  global: 8
  per_provider: { mock: 8 }
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(port: int, *, timeout_s: float = 60.0) -> float:
    """等到 /health 能通，返回耗时。起不来就抛——演练脚本不能"默默通过"。"""

    started = time.perf_counter()
    deadline = started + timeout_s

    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                if resp.status == 200:
                    return time.perf_counter() - started
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.05)

    raise RuntimeError(f"server on :{port} did not become healthy in {timeout_s}s")


def post_draft(port: int) -> dict[str, Any]:
    body = json.dumps({"brief": BRIEF}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/draft",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as resp:
        return json.loads(resp.read())


class Server:
    """一个可以被真的杀掉的 uvicorn。

    日志**落文件**而不是留在管道里：管道不读就会写满并阻塞进程，
    而"演练脚本把被测进程卡死"这种事会让人以为回退失败了。
    """

    def __init__(self, models_yaml: Path, features: str, log_path: Path) -> None:
        self.port = free_port()
        self._log_path = log_path
        env = {
            **os.environ,
            "MODELS_YAML": str(models_yaml),
            "FEATURES": features,
            "AUTH_MODE": "off",
            "LOG_PROMPTS": "false",
        }
        self._log = log_path.open("wb")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "growth_agent.app:app",
             "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning"],
            cwd=REPO,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def start(self) -> float:
        return wait_for_health(self.port)

    def cache_hits(self) -> int:
        """数 RequestLog 里的 `cache_hit: true`。

        为什么不看响应体：`/v1/draft` 的 `input_tokens` 描述的是**这个答案当初**
        花了多少（那是它的出处），不是这次请求花了多少。命中与否是**运营口径**，
        它只存在于 RequestLog 里——这也正是给 `RequestLog` 加 `cache_hit` 字段的理由。
        """

        text = self._log_path.read_text("utf-8", errors="ignore")
        return text.count('"cache_hit": true')

    def request_logs(self) -> int:
        text = self._log_path.read_text("utf-8", errors="ignore")
        return text.count('"request_id"')

    def stop(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - 只在挂了才走到
            self._proc.kill()
        self._log.close()


def probe(server: Server) -> dict[str, Any]:
    """同一 brief 连续发两次，看第二次有没有被缓存接住。"""

    first = post_draft(server.port)
    second = post_draft(server.port)

    assert first["text"] and second["text"], "请求本身失败了，命中与否都无从谈起"

    return {
        "first_model": first["model"],
        "hits": server.cache_hits(),
        "logs": server.request_logs(),
        "cached": server.cache_hits() >= 1,
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        models_yaml = Path(tmp) / "models.yaml"
        models_yaml.write_text(MODELS_YAML, "utf-8")

        print("=" * 72)
        print("阶段 1 · 开着（FEATURES=semantic_cache=on）")
        print("-" * 72)
        server = Server(models_yaml, "semantic_cache=on", Path(tmp) / "on.log")
        boot_on = server.start()
        before = probe(server)
        print(f"  启动 {boot_on:.2f}s；同一 brief 两次："
              f"RequestLog {before['logs']} 条，cache_hit {before['hits']} 条")
        assert before["logs"] == 2, "命中也要发 RequestLog：少了它请求量会凭空少一截"
        assert before["cached"], "开着却没命中缓存：Kill Switch 的对象根本不存在"
        server.stop()

        print("=" * 72)
        print("阶段 2 · 决定回退 → 生效（计时开始）")
        print("-" * 72)
        decided_at = time.perf_counter()

        # ── 这就是全部动作：改一个环境变量 + 重启进程 ──
        server = Server(models_yaml, "", Path(tmp) / "off.log")  # 空 = 全部回到旧行为
        boot_off = server.start()
        after = probe(server)
        elapsed = time.perf_counter() - decided_at

        print(f"  启动 {boot_off:.2f}s；同一 brief 两次："
              f"RequestLog {after['logs']} 条，cache_hit {after['hits']} 条")
        print(f"  → 从决定到生效 {elapsed:.1f}s（判据 ≤{DEADLINE_S}s）")
        print(f"  → 期间不可用 ≈ {boot_off:.1f}s（这就是「错过的请求」的量纲）")
        server.stop()

        assert not after["cached"], "回退后仍在命中缓存，开关没生效"
        assert after["logs"] == 2, "回退后每条请求仍应有一条日志"
        assert elapsed <= DEADLINE_S, f"回退耗时 {elapsed:.1f}s 超过判据 {DEADLINE_S}s"

        print("=" * 72)
        print("阶段 3 · 回退后系统仍可用")
        print("-" * 72)
        server = Server(models_yaml, "", Path(tmp) / "after.log")
        server.start()
        health = post_draft(server.port)
        server.stop()
        assert health["text"], "回退后拿不到正文，说明回退把系统弄坏了"
        print(f"  /v1/draft 正常返回 {len(health['text'])} 字，model={health['model']}")
        print("=" * 72)
        print(f"结论：Kill Switch 有效，{elapsed:.1f}s 生效（预算 {DEADLINE_S}s），"
              f"回退后仍可应答。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
