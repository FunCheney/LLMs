"""脚手架 + 打包演练（W12 D2）。

一次跑完三件必须**真做**的事：

1. `uv build` 出 wheel，装进另一个 venv —— 打包不是自嗨；
2. `agent create` 生成项目，装进去 —— 生成的不是空目录；
3. 起服务、发一条 chat、拿到回答 —— 检查点是"1 天内跑通"，不是"生成文件"。

为了让第 3 步不依赖钱包和网络，脚本自己起一个**假的 OpenAI 兼容上游**：
它只做一件事——把问题原样回显。这样"回答能回来"证明的是链路通
（装配 → 路由 → adapter → 解析 → 响应），而不是"上游今天心情好"。

> 口径说明（见 `docs/w12/scaffold-guide.md` §4）：本机无外网，
> 全新 venv 无法解析 fastapi/httpx 这些依赖，所以 venv 用 `--system-site-packages`
> 借用基础依赖，**wheel 本身装进该 venv 的 site-packages**，并断言
> `agent_framework.__file__` 指向 venv 而不是仓库源码。这是这台机器上能做到的
> 最接近"干净环境"的验证；缺的是"依赖从零解析"这一步，如实记在 guide 里。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO / "packages" / "agent_framework"

# 假上游的回答。**故意带一个可断言的特征串**：正文里出现它，
# 才证明这条回答穿过来了，而不是某个兜底话术或缓存。
ECHO_PREFIX = "【假上游】收到："


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ── 假上游：一个只会回显的 OpenAI 兼容服务 ──────────────────────────────


class _Upstream(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的约定
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw or b"{}")
        question = ""
        for message in body.get("messages", []):
            if message.get("role") == "user":
                question = str(message.get("content", ""))

        payload = {
            "id": "chatcmpl-drill",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "gpt-4o-mini"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ECHO_PREFIX + question},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 32, "completion_tokens": 12, "total_tokens": 44},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:  # 静音
        return None


def start_upstream() -> tuple[ThreadingHTTPServer, str]:
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}/v1"


# ── HTTP 辅助 ──────────────────────────────────────────────────────────


def post_json(url: str, payload: dict, *, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get_json(url: str, *, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_health(port: int, proc: subprocess.Popen[bytes], *, deadline_s: float = 45.0) -> float:
    started = time.perf_counter()
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() - started < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError(f"服务在启动阶段就退出了（exit={proc.returncode}）")
        try:
            get_json(url, timeout=1.0)
            return time.perf_counter() - started
        except Exception:  # noqa: BLE001 - 启动期连不上是正常态
            time.sleep(0.15)
    raise TimeoutError(f"/health 在 {deadline_s:.0f}s 内没起来")


# ── 主流程 ─────────────────────────────────────────────────────────────


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"命令失败：{' '.join(cmd)}\n--- stdout ---\n{result.stdout[-2000:]}"
            f"\n--- stderr ---\n{result.stderr[-2000:]}"
        )
    return result.stdout


def main() -> int:
    timings: dict[str, float] = {}
    started_all = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="scaffold-drill-") as tmp:
        tmpdir = Path(tmp)
        dist = tmpdir / "dist"

        print("=" * 74)
        print("阶段 1 · 打包（uv build，从包目录构建）")
        print("-" * 74)
        begin = time.perf_counter()
        run(["uv", "build", "--wheel", "-o", str(dist)], cwd=PACKAGE_DIR)
        wheels = sorted(dist.glob("*.whl"))
        assert wheels, "uv build 没产出 wheel"
        wheel = wheels[0]
        timings["build"] = time.perf_counter() - begin
        size_kb = wheel.stat().st_size / 1024
        print(f"  {wheel.name}  {size_kb:.0f} KB  ({timings['build']:.1f}s)")

        print("=" * 74)
        print("阶段 2 · 生成项目（agent create）")
        print("-" * 74)
        upstream, base_url = start_upstream()
        print(f"  假上游已就绪：{base_url}")

        sys.path.insert(0, str(REPO))
        from agent_framework.cli import ScaffoldOptions, create_project  # noqa: PLC0415

        begin = time.perf_counter()
        project = create_project(
            "demo-agent",
            tmpdir,
            options=ScaffoldOptions(
                base_url=base_url,
                # 让生成出来的 pyproject 直接指向刚打的 wheel：
                # 这一条同时验证"打包产物是可安装的依赖"，不是只能 import 的目录。
                framework_spec=f"agent-framework @ {wheel.as_uri()}",
            ),
        )
        timings["generate"] = time.perf_counter() - begin
        files = sorted(p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file())
        print(f"  {project}  {len(files)} 个文件  ({timings['generate']:.2f}s)")
        for name in files:
            print(f"    {name}")

        print("=" * 74)
        print("阶段 3 · 另一个 venv 里安装（打包产物 + 生成的项目）")
        print("-" * 74)
        venv = tmpdir / "venv"
        begin = time.perf_counter()
        run(["uv", "venv", str(venv), "--quiet"], cwd=tmpdir)
        python = venv / "bin" / "python"
        # 安装是离线 + 无依赖：两个产物（wheel 和生成的项目）都是本地文件，
        # build isolation 用缓存里的 setuptools（本机已验证可用）。
        # 但**运行时依赖**（fastapi/pydantic-settings/httpx/uvicorn）在这台机器上
        # 无外网装不进新 venv——所以只在运行时把仓库 `.venv` 的 site-packages
        # 挂到 PYTHONPATH。注意：`.pth` 在 PYTHONPATH 目录里**不被处理**
        # （只有 site 目录才会），所以仓库的 editable 安装不会泄漏到这个 venv，
        # 待装进本 venv 的 wheel 仍然是唯一的 agent_framework。
        deps_path = str(next((REPO / ".venv" / "lib").glob("python*/site-packages")))
        env_runtime = {**os.environ, "PYTHONPATH": deps_path}

        run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", "--offline",
             "--quiet", str(wheel)],
            cwd=tmpdir,
        )
        run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", "--offline",
             "--quiet", str(project)],
            cwd=tmpdir,
        )
        timings["install"] = time.perf_counter() - begin

        location = run(
            [str(python), "-c", "import agent_framework; print(agent_framework.__file__)"],
            cwd=tmpdir, env=env_runtime,
        ).strip()
        version = run(
            [str(python), "-c", "import agent_framework; print(agent_framework.__version__)"],
            cwd=tmpdir, env=env_runtime,
        ).strip()
        assert str(venv) in location, f"import 到的是仓库源码而不是安装产物：{location}"
        console = venv / "bin" / "agent"
        assert console.exists(), "console script `agent` 没装上（pyproject 的 [project.scripts]）"
        installed = run([str(console), "--help"], cwd=tmpdir, env=env_runtime)
        assert "create" in installed, "`agent` 入口跑不起来"
        print(f"  安装耗时 {timings['install']:.1f}s")
        print(f"  agent_framework {version} ← {location}")
        print("  console script `agent` 可用")

        print("=" * 74)
        print("阶段 4 · 起服务 + 发一条 chat")
        print("-" * 74)
        port = free_port()
        env = {
            **env_runtime,
            "OPENAI_API_KEY": "sk-drill-not-a-real-key",
            "MODELS_YAML": "configs/models.yaml",
            # 日志别把 drill 的输出淹了；也顺便验证 LOG_PROMPTS=off 不炸
            "LOG_PROMPTS": "false",
        }
        log_path = tmpdir / "server.log"
        log_file = log_path.open("wb")
        proc = subprocess.Popen(
            [str(python), "-m", "uvicorn", "demo_agent.app:app",
             "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            cwd=project,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            boot = wait_health(port, proc)
            timings["boot"] = boot
            print(f"  /health 就绪：{boot:.1f}s")

            begin = time.perf_counter()
            answer = post_json(
                f"http://127.0.0.1:{port}/v1/ask",
                {"question": "用一句话解释什么是语义缓存"},
            )
            timings["first_chat"] = time.perf_counter() - begin
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            finally:
                log_file.close()

        print(f"  首条 chat 往返：{timings['first_chat']:.2f}s")
        print(f"  text   = {answer['text']}")
        print(f"  model  = {answer['model']}")
        print(f"  tokens = {answer['input_tokens']} in / {answer['output_tokens']} out")

        # ── 断言：证明它真的走完了链路 ──
        assert answer["text"].startswith(ECHO_PREFIX), (
            f"回答不是假上游给的，链路没通：{answer['text']!r}"
        )
        assert answer["input_tokens"] > 0, "usage 没透传（adapter 解析层可能被跳过）"
        assert answer["model"] == "gpt-4o-mini", f"路由没按 models.yaml 走：{answer['model']}"

        upstream.shutdown()

        timings["total"] = time.perf_counter() - started_all

        print("=" * 74)
        print("阶段 5 · 检查日志里有没有框架该做的事")
        print("-" * 74)
        log_text = log_path.read_text("utf-8", errors="replace")
        checks = {
            "request_id 贯穿": "request_id" in log_text or "rid" in log_text.lower(),
            "有一条 RequestLog": "cost" in log_text.lower() or "provider" in log_text.lower(),
        }
        for name, ok in checks.items():
            print(f"  {'✓' if ok else '·'} {name}")
        print(f"  日志共 {len(log_text.splitlines())} 行：{log_path}")

        print("=" * 74)
        print("结论")
        print("-" * 74)
        print(f"  从空目录到能发 chat：{timings['total']:.1f}s"
              f"（build {timings['build']:.1f}s / install {timings['install']:.1f}s / "
              f"boot {timings['boot']:.1f}s）")
        print("  检查点「第三方 1 天内跑通」：机制上只需一次 uv sync + 一条 curl。")
        print("  未验证：依赖在**全新** venv 里从零解析（本机无外网，见脚本头注释）。")

        payload = {
            "wheel": wheel.name,
            "wheel_kb": round(size_kb, 1),
            "files": len(files),
            "timings_s": {key: round(value, 2) for key, value in timings.items()},
            "answer_model": answer["model"],
            "answer_prefix_ok": True,
        }
        print()
        print("JSON:", json.dumps(payload, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
