from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import dotenv
import os
dotenv.load_dotenv()


# ============================================================
# 1. 定义 Tool
# ============================================================

@tool
def search_docs(query: str) -> str:
    """Search product documentation for information relevant to a customer question."""

    documentation = [
        "To reset your password, go to Settings > Security > Reset Password.",
        "Password reset links expire after 30 minutes.",
        "If your account is locked, contact customer support.",
        "You can update your billing information in Settings > Billing.",
    ]

    query_words = query.lower().split()

    results = []

    for doc in documentation:
        if any(word in doc.lower() for word in query_words):
            results.append(doc)

    if not results:
        return "No relevant documentation found."

    return "\n".join(results)


tools = [search_docs]


# ============================================================
# 2. 创建 LLM
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
)

# 给 LLM 绑定 Tools
agent_llm = llm.bind_tools(tools)


# ============================================================
# 3. 定义 Agent State
# ============================================================

class AgentState(dict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 4. Agent Node
# ============================================================

def agent(state: AgentState):
    response = agent_llm.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 5. 判断是否继续
# ============================================================

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if (
        isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "tools"

    return END


# ============================================================
# 6. 创建 Tool Node
# ============================================================

tool_node = ToolNode(tools)

# ============================================================
# 7. 构建 Graph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
)

builder.add_edge("tools", "agent")

agent_graph = builder.compile()


# ============================================================
# 8. 运行
# ============================================================

if __name__ == "__main__":

    result = agent_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="How can I reset my password?"
                )
            ]
        }
    )

    print("\n========== FINAL MESSAGES ==========\n")

    for message in result["messages"]:
        print(type(message).__name__)
        print(message)
        print("-" * 50)