import os
from typing import AsyncIterator, Any
import argparse
import asyncio
import json
from typing import Any

import dotenv
from openai import AsyncOpenAI

dotenv.load_dotenv()




async def stream_deepseek(prompt: str) -> AsyncIterator[dict[str, Any]]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_URL")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    stream = await client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": "你是代码审查助手，只报告有证据的问题。",
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
        max_tokens=1024,
        extra_body={"thinking": {"type": "disabled"}},
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta.content if choice else None

        if delta:
            yield {"type": "text.delta", "delta": delta}

        if choice and choice.finish_reason:
            yield {
                "type": "model.finished",
                "finish_reason": choice.finish_reason,
            }

        if chunk.usage is not None:
            yield {
                "type": "model.usage",
                "usage": {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                },
            }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="直接消费 DeepSeek V4 Flash 的原始流式增量。",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="请审查下面这段 Python 代码，指出其中最可能导致线上故障的问题。",
        help="发送给模型的用户提示词",
    )
    args = parser.parse_args()

    try:
        async for event in stream_deepseek(args.prompt):
            print(json.dumps(event, ensure_ascii=False))
    except RuntimeError as exc:
        print(f"运行失败: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
