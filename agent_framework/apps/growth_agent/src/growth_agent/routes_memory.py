"""GET/DELETE /v1/memory —— 记忆的运维与调试面（W7）。

为什么单独开路由而不是塞进 `/v1/answer`：回答链路只该"用"记忆，
"管理"记忆（查看召回了什么、删掉某条被污染的）是另一类操作——
混在一起会让"一次回答"的请求体带上管理语义，两边的演进互相绑死。

这里的操作全部**直接操作记忆库**，不经过检索与生成；失败语义与
`/v1/answer` 不同：记忆库连不上是 500（CONFIG_ERROR），不是降级——
运维主动查记忆时，"静默空列表"比报错更危险（会让人以为没记过）。
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import ConfigError
from agent_framework.observability import current_request_id
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from growth_agent.runtime import ensure_memory

router = APIRouter()


class MemoryEntry(BaseModel):
    key: str
    value: str
    confidence: float
    source_session: str
    expires_at: float | None = None


class RecallResponse(BaseModel):
    request_id: str
    query: str
    entries: list[MemoryEntry]


class ForgetRequest(BaseModel):
    key: str = Field(min_length=1, max_length=64, description="要遗忘的记忆 key")


class ForgetResponse(BaseModel):
    request_id: str
    deleted: int


@router.get("/v1/memory")
async def recall(
    request: Request, query: str = Query(min_length=1, max_length=2000)
) -> JSONResponse:
    runtime = await ensure_memory(request.app)
    try:
        items = await runtime.long_term.recall(query, k=runtime.settings.memory_recall_k)
    except Exception as exc:  # noqa: BLE001 - 运维查询失败必须报出来，不静默
        raise ConfigError(f"记忆库查询失败：{exc}") from exc

    payload = RecallResponse(
        request_id=current_request_id() or "-",
        query=query,
        entries=[
            MemoryEntry(
                key=item.key,
                value=item.value,
                confidence=item.confidence,
                source_session=item.source_session,
                expires_at=item.expires_at,
            )
            for item in items
        ],
    )
    return JSONResponse(payload.model_dump(mode="json"))


@router.delete("/v1/memory")
async def forget(body: ForgetRequest, request: Request) -> JSONResponse:
    runtime = await ensure_memory(request.app)

    def match_key(record: Any, now: float) -> bool:
        return record.key == body.key

    try:
        deleted = await runtime.long_term.forget(match_key)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"记忆库删除失败：{exc}") from exc

    payload = ForgetResponse(request_id=current_request_id() or "-", deleted=deleted)
    return JSONResponse(payload.model_dump(mode="json"))
