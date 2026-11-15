"""`agent create` —— 生成一个**真能跑**的最小项目。

判据不是"生成了文件"，是"装完依赖能起服务、能发一条 chat 拿到回答"
（W12 检查点：第三方场景 1 天内跑通）。所以模板里每一行都必须是被跑通过的，
不是"看起来应该能跑"。

设计上的三个约束：

1. **不用 cookiecutter / jinja2**。模板就是本文件里的字符串常量。
   多一个渲染依赖，就多一处"模板语法错误要等到生成时才发现"；
   而这些模板总共不到 200 行，占位符只需要 `@@NAME@@` 一种。
2. **生成的代码不 import 私有符号**。脚手架是框架公共面的第一个用户，
   它写 `from agent_framework.llm.client import ChatCall` 的话，
   等于教所有第三方穿透实现模块（D1 刚清完这类泄漏）。
3. **`--base-url` 是一等参数，不是事后改 yaml**。自建网关、代理、ollama、
   离线假上游都要改这一行；让用户去编辑生成出来的 yaml，等于假装
   "配置不参与部署"。
"""

from __future__ import annotations

import argparse
import keyword
import sys
from pathlib import Path

__all__ = ["ScaffoldOptions", "create_project", "main", "render_files"]

# ── 模板 ────────────────────────────────────────────────────────────────

_MODELS_YAML = """\
# 一个 provider、一个模型、两条路由——够跑通，不够做容量规划。
# 加第二个模型时注意：`tier` 必须对应一条 route，否则启动即报错（config.py V3）。
providers:
  openai:
    type: openai_compat
    base_url: "@@BASE_URL@@"
    api_key_env: OPENAI_API_KEY

models:
  gpt-4o-mini:
    provider: openai
    model: gpt-4o-mini
    tier: cheap
    capabilities: [stream, tools]
    # per 必须是 1000000（每百万 token）。按千计价会让成本差 1000 倍。
    pricing: { input: 0.15, output: 0.60, currency: USD, per: 1000000 }

routes:
  default: { primary: gpt-4o-mini, fallback: [] }
  cheap:   { primary: gpt-4o-mini, fallback: [] }

tasks:
  chat: cheap

limits:
  global: 8
  per_provider: { openai: 8 }
"""

_APP_PY = '''\
"""@@PKG@@ —— 用 agent-framework 起的最小服务。

三件事：装配 gateway → 加一个业务端点 → 起服务。
重试、路由、熔断、并发闸门、request_id、错误契约都在框架里，这里一行都不写。

跑起来：
    uv run uvicorn @@PKG@@.app:app --reload
    curl -s -X POST localhost:8000/v1/ask \\
      -H 'content-type: application/json' \\
      -d '{"question": "用一句话解释什么是语义缓存"}'
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from agent_framework import ChatCall, LLMGateway, Message, MessageRole, build_gateway
from agent_framework.server import create_app

SYSTEM_PROMPT = "你是 @@PKG@@ 的助手。回答简短、直接，不超过三句话。"


class AskRequest(BaseModel):
    question: str
    task: str = "chat"


class AskResponse(BaseModel):
    text: str
    model: str
    retry_count: int
    input_tokens: int
    output_tokens: int


def build(gateway: LLMGateway | None = None):
    """装配：先造链路（读 configs/models.yaml），再把它**显式注入** app。

    `gateway` 是参数而不是模块全局：测试要换假实现时，注入是一个参数，
    monkeypatch 是一处魔法。
    """
    resolved = gateway if gateway is not None else build_gateway()
    app = create_app(resolved)
    router = APIRouter()

    @router.post("/v1/ask", response_model=AskResponse)
    async def ask(body: AskRequest) -> AskResponse:
        outcome = await resolved.chat(
            ChatCall(
                task=body.task,
                messages=[
                    Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
                    Message(role=MessageRole.USER, content=body.question),
                ],
            )
        )
        usage = outcome.response.usage
        return AskResponse(
            text=outcome.response.text or "",
            model=outcome.alias,
            retry_count=outcome.retry_count,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    app.include_router(router)
    return app


# 服务（不是库）在模块级建实例：uvicorn 的 `module:app` 形态需要它，
# 而这里本来就是进程的入口，没有"别人 import 我"的第二种用法。
app = build()
'''

_TEST_PY = '''\
"""两条冒烟：/health 能通、/v1/ask 走完整条链路。

**用假 gateway 而不是真调上游**：这条测试要证明的是"装配对不对"，
不是"上游今天在不在"。真调上游的测试会在没网/没额度时红，
然后被改成 skip —— 于是它再也不证明任何事。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_framework import (
    ChatOutcome,
    LLMResponse,
    Message,
    MessageRole,
    Settings,
    Usage,
)


class FakeGateway:
    """只实现被调用到的两个方法。这是"注入"的红利：不用 patch httpx。"""

    def __init__(self) -> None:
        self.calls: list[object] = []
        # create_app 会从 gateway 读 settings 来装配 session store / 并发限额
        self.settings = Settings(_env_file=None)

    async def chat(self, call):
        self.calls.append(call)
        return ChatOutcome(
            response=LLMResponse(
                model="fake",
                provider="fake",
                message=Message(role=MessageRole.ASSISTANT, content="假答案"),
                usage=Usage(input_tokens=7, output_tokens=3),
            ),
            alias="fake-alias",
            retry_count=1,
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def client():
    # 延迟 import：模块顶层的 `app = build()` 会读配置，不该在收集阶段就跑
    from @@PKG@@.app import build

    with TestClient(build(FakeGateway())) as test_client:
        yield test_client


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_ask_goes_through_the_gateway(client) -> None:
    response = client.post("/v1/ask", json={"question": "你好"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "假答案"
    assert body["model"] == "fake-alias"
    assert body["input_tokens"] == 7
'''

_README = """\
# @@NAME@@

由 `agent create` 生成的最小项目。装完依赖 → 起服务 → 发一条 chat，三步。

## 1. 装依赖

```bash
export OPENAI_API_KEY=sk-...
uv sync
```

## 2. 起服务

```bash
uv run uvicorn @@PKG@@.app:app --reload
```

## 3. 发一条 chat

```bash
curl -s -X POST localhost:8000/v1/ask \\
  -H 'content-type: application/json' \\
  -d '{"question": "用一句话解释什么是语义缓存"}'
```

另有一个 `GET /health`，返回版本、session 后端与在飞请求数。

## 目录

| 路径 | 作用 |
|---|---|
| `configs/models.yaml` | provider / 模型 / 路由 / 并发限额。**改模型只改这里** |
| `src/@@PKG@@/app.py` | 装配 gateway → 注入 `create_app` → 加业务端点 |
| `tests/test_app.py` | 用假 gateway 验装配，不依赖网络 |

## 想改的三处

- **换模型**：`configs/models.yaml` 的 `routes`，不用改 Python。
- **加端点**：`app.py` 里 `app.include_router(router)` 之前加一个 `@router.post`。
- **换业务 prompt**：`app.py` 顶部的 `SYSTEM_PROMPT`。

## 框架已经替你做的事

重试与退避、按 task 路由到便宜/强档、熔断、并发闸门（超了返 503 + `Retry-After`）、
`request_id` 贯穿日志、错误码统一投影（`ERROR_HTTP`）、每次调用一条 `RequestLog`。
这些在 `app.py` 里都看不见——**这就是要的**：业务代码不该出现重试循环。
"""

_PYPROJECT = """\
[project]
name = "@@NAME@@"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "@@FRAMEWORK_SPEC@@",
]

[dependency-groups]
dev = ["pytest>=8", "httpx>=0.28"]

[tool.uv.sources]
# 框架还没上 PyPI 时指向本地路径/whl；发布后删掉这一段。
@@FRAMEWORK_SOURCE@@

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["src"]
"""

_ENV_EXAMPLE = """\
# 复制成 .env 后填真实值。**不要 commit .env**（.gitignore 已含）。
OPENAI_API_KEY=
# 指向自建网关/代理时改这里；空 = 用 configs/models.yaml 里的 base_url
# MODELS_YAML=configs/models.yaml
"""

_GITIGNORE = """\
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
dist/
"""

_FILES: dict[str, str] = {
    "README.md": _README,
    "pyproject.toml": _PYPROJECT,
    ".env.example": _ENV_EXAMPLE,
    ".gitignore": _GITIGNORE,
    "configs/models.yaml": _MODELS_YAML,
    "src/@@PKG@@/__init__.py": '"""@@NAME@@。"""\n\n__version__ = "0.1.0"\n',
    "src/@@PKG@@/app.py": _APP_PY,
    "tests/test_app.py": _TEST_PY,
}


# ── 渲染 ────────────────────────────────────────────────────────────────


class ScaffoldOptions:
    """生成参数。做成一个对象而不是八个关键字参数——调用方（测试/演练脚本）
    需要"改一个字段再生成一次"，字典式参数容易漏传。"""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        framework_spec: str = "agent-framework>=0.1.0",
        framework_source: str = "",
    ) -> None:
        self.base_url = base_url
        self.framework_spec = framework_spec
        self.framework_source = framework_source


def package_name(project: str) -> str:
    """项目名 → 合法包名。

    `我的助手` → 直接拒绝，而不是音译成 `wo_de_zhu_shou`：
    生成之后 import 不进去，用户要先读一遍我们编的音译规则。
    """

    candidate = project.strip().lower().replace("-", "_").replace(" ", "_")
    if not candidate.isidentifier() or keyword.iskeyword(candidate):
        raise ValueError(
            f"项目名 {project!r} 不能转成合法的 Python 包名；"
            "请用字母/数字/下划线（例：my-agent）"
        )
    return candidate


def render_files(project: str, options: ScaffoldOptions | None = None) -> dict[str, str]:
    """返回「相对路径 → 内容」。纯函数：不碰文件系统，便于单测直接断言内容。"""

    opts = options or ScaffoldOptions()
    pkg = package_name(project)
    source = ""
    if opts.framework_source:
        source = f'agent-framework = {{ path = "{opts.framework_source}" }}'

    values = {
        "@@NAME@@": project,
        "@@PKG@@": pkg,
        "@@BASE_URL@@": opts.base_url,
        "@@FRAMEWORK_SPEC@@": opts.framework_spec,
        "@@FRAMEWORK_SOURCE@@": source,
    }

    rendered: dict[str, str] = {}
    for raw_path, template in _FILES.items():
        path = raw_path
        body = template
        for key, value in values.items():
            path = path.replace(key, value)
            body = body.replace(key, value)
        rendered[path] = body
    return rendered


def create_project(
    project: str,
    target: Path | None = None,
    *,
    options: ScaffoldOptions | None = None,
    force: bool = False,
) -> Path:
    """把项目写到 `target/<project>/`，返回那个目录。"""

    files = render_files(project, options)
    root = (target or Path.cwd()) / project

    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"{root} 已存在且非空；确认要覆盖请加 --force（会覆盖同名文件，不删其他）"
        )

    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, "utf-8")

    return root


# ── CLI ─────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="agent-framework 命令行",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="生成一个最小可跑项目")
    create.add_argument("name", help="项目名（同时决定目录名）")
    create.add_argument("--target", default=".", help="写到哪个目录下（默认当前目录）")
    create.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="OpenAI 兼容入口；自建网关/代理/ollama/离线假上游都改这里",
    )
    create.add_argument(
        "--framework-spec",
        default="agent-framework>=0.1.0",
        help="写进 pyproject 的依赖声明（本地验证打包产物时传 file://...whl）",
    )
    create.add_argument(
        "--framework-path",
        default="",
        help="源码路径（写进 [tool.uv.sources]，开发期用）",
    )
    create.add_argument("--force", action="store_true", help="目录非空也照写")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "create":
        options = ScaffoldOptions(
            base_url=args.base_url,
            framework_spec=args.framework_spec,
            framework_source=args.framework_path,
        )
        try:
            root = create_project(
                args.name, Path(args.target), options=options, force=args.force
            )
        except (ValueError, FileExistsError) as exc:
            print(f"生成失败：{exc}", file=sys.stderr)
            return 2

        pkg = package_name(args.name)
        print(f"已生成 {root}")
        print()
        print("下一步：")
        print(f"  cd {root}")
        print("  export OPENAI_API_KEY=sk-...")
        print("  uv sync")
        print(f"  uv run uvicorn {pkg}.app:app --reload")
        print(
            "  curl -s -X POST localhost:8000/v1/ask -H 'content-type: application/json'"
            ' -d \'{"question": "用一句话解释什么是语义缓存"}\''
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover - 只走 `python -m`
    raise SystemExit(main())
