"""检索运行时的装配与惰性就绪（W6）。

为什么要惰性：`build_runtime` 会加载 bge-m3（本机 18–20 秒）。放进 lifespan 意味着
容器启动要等 20 秒才通过 healthcheck，而 healthcheck 的 start_period 只有 10 秒
（见 ops/docker-compose.yml）——部署会直接失败。所以**启动只装配 LLM 网关，
检索在第一个 `/v1/answer` 请求时加载**，代价明确写在 RUNBOOK 里：
上线后打一次预热请求，别让第一个真实用户吃冷启动。

为什么不做后台预热任务：一个只在 lifespan 里 create_task 的预热，会在
测试、CLI、多 worker 场景下各起一份，而失败信息只留在后台任务的异常里
（没人看得到）。惰性加载 + 显式预热请求的可观测性更好。

失败语义：模型/索引不可用时抛 `ConfigError`（框架的 CONFIG_ERROR → 500），
**不是**返回"没检索到资料"。后者会把环境问题归因成语料缺失——W4 playbook 里
那条最贵的误判。
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_framework.config import ConfigError, Settings
from agent_framework.llm.client import LLMGateway
from fastapi import FastAPI

from growth_agent.answer import KnowledgeAnswerer


def _build_answerer_blocking(settings: Settings, gateway: LLMGateway) -> tuple[Any, Any]:
    """同步部分：加载模型、装配流水线。放在线程里调用，别占事件循环。

    `build_runtime` 内部先 `create_base_retriever`（加载 bge-m3），
    再把 Settings 翻译成 pipeline 参数——两件事都在这里，且只在这里。
    """

    from growth_agent.rag import build_runtime

    runtime = build_runtime(settings)
    return runtime, KnowledgeAnswerer(runtime.pipeline, gateway, settings=settings)


async def ensure_answerer(app: FastAPI) -> KnowledgeAnswerer:
    """拿到（必要时构造）回答器。并发首请求只会加载一次。"""

    existing = getattr(app.state, "answerer", None)
    if isinstance(existing, KnowledgeAnswerer):
        return existing

    lock: asyncio.Lock = app.state.rag_lock
    async with lock:
        # 拿到锁后再查一遍：等锁期间别的请求可能已经建好了。
        existing = getattr(app.state, "answerer", None)
        if isinstance(existing, KnowledgeAnswerer):
            return existing

        settings: Settings = app.state.settings
        gateway: LLMGateway = app.state.gateway
        try:
            runtime, answerer = await asyncio.to_thread(
                _build_answerer_blocking, settings, gateway
            )
        except ImportError as exc:
            raise ConfigError(
                "检索链路加载失败：当前环境没有本地模型（torch / sentence-transformers）。"
                "执行 `uv sync --extra local`，或设 RETRIEVAL_MODE=sparse 只走 BM25。"
                f"原始错误：{exc}"
            ) from exc
        except FileNotFoundError as exc:
            raise ConfigError(
                f"检索链路加载失败：{exc} 设 RAG_ROOT 指向 rag_mvp 仓库根后重试。"
            ) from exc

        app.state.rag = runtime
        app.state.answerer = answerer
        return answerer


async def close_runtime(app: FastAPI) -> None:
    """释放线程池与连接池。**必须在 lifespan 里调用**，否则进程退出会挂住。"""

    runtime = getattr(app.state, "rag", None)
    if runtime is not None:
        await runtime.aclose()
        app.state.rag = None
        app.state.answerer = None
