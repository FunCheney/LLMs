import asyncio
import uuid
from typing import AsyncIterable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from temporalio.client import Client

from agent_framework.config import TASK_QUEUE, TEMPORAL_ADDRESS
from agent_framework.workflows import SayHelloWorkflow


class chatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="user input message",
    )

    model: str = Field(
        default="mock-model",
        description="模型名称",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="生成随机性参数",
    )

class ChatResponse(BaseModel):
    answer: str
    model: str
app = FastAPI()


async def get_temporal_client(request: Request) -> Client:
    client = getattr(request.app.state, "temporal_client", None)
    if client is None:
        try:
            client = await Client.connect(TEMPORAL_ADDRESS)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Temporal at {TEMPORAL_ADDRESS}",
            ) from exc
        request.app.state.temporal_client = client
    return client


@app.get("/health")
async def health_endpoint():
    return {"status": "ok"}


@app.post("/v1/run")
async def create_run(request: Request):
    workflow_id = f"job-{uuid.uuid4()}"
    client = await get_temporal_client(request)
    handle = await client.start_workflow(
        SayHelloWorkflow.run,
        "some-input",
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    return {"workflow_id": handle.id, "status": "started"}



message = """
Rick: (stumbles in drunkenly, and turns on the lights) Morty! You gotta come on. You got--... you gotta come with me.
Morty: (rubs his eyes) What, Rick? What's going on?
Rick: I got a surprise for you, Morty.
Morty: It's the middle of the night. What are you talking about?
Rick: (spills alcohol on Morty's bed) Come on, I got a surprise for you. (drags Morty by the ankle) Come on, hurry up. (pulls Morty out of his bed and into the hall)
Morty: Ow! Ow! You're tugging me too hard!
Rick: We gotta go, gotta get outta here, come on. Got a surprise for you Morty.
"""


@app.get("/story/stream", response_class=StreamingResponse)
async def stream_story() -> AsyncIterable[str]:
    """
    curl -N -X GET http://127.0.0.1:8000/story/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream"

    这里的 -N 很重要：它会关闭 curl 的输出缓冲，方便观察服务端逐步发送的事件。
    """
    for line in message.splitlines():
        await asyncio.sleep(0.1)
        yield line
