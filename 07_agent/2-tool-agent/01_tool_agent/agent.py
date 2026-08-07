from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name='01_tool_agent',
    model="gemini-2.0-flash",
    description="Tool Agent",
    instruction="""
    You are a helpful assistant that can use the following tools:
    - google_search
    """,
    tools=[google_search]
)

