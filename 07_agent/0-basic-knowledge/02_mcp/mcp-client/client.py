# mcp_http_client.py
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SERVER_URL = "http://localhost:8000/mcp"

async def main():
    async with streamable_http_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("可用工具:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            result = await session.call_tool(
                "get_alerts",
                arguments={"state": "CA"},
            )

            for block in result.content:
                if block.type == "text":
                    print(block.text)

if __name__ == "__main__":
    asyncio.run(main())