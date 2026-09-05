"""
email_assistant agent 版本
1. 结合 workflow 和 agentLoop
2. 最终实现完成之后
                              START
                                │
                                ▼
                         classify_intent
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
             question          bug           other
                 │              │              │
                 ▼              ▼              ▼
          support_agent   bug_tracking    human_review
                 │              │              │
          ┌──────┴──────┐       │              │
          │             │       │              │
       Tool Call      Final     │              │
          │             │       │              │
          ▼             ▼       ▼              ▼
   support_tools   finish_response            END
          │             │
          │             │
          └──────► support_agent
                        │
                        ▼
                        END
"""
import json
from typing import TypedDict, Literal, Annotated, List
import os
import dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

dotenv.load_dotenv()


class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature_request", "other"]
    urgency: Literal["high", "medium", "low"]
    summary: str


# ============================================================
# 1. 定义state
# ============================================================
# 业务状态 + agent state
class EmailState(TypedDict):
    email_content: str
    classification: EmailClassification | None
    messages: Annotated[List[BaseMessage], add_messages]
    response: str


# ============================================================
# 2. 创建 LLM
# ============================================================
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
)

# ============================================================
# 3. 定义工具
# ============================================================
DOCUMENTATION = [
    "To reset your password, go to Settings > Security > Reset Password.",
    "Password reset links expire after 30 minutes.",
    "If your account is locked, contact customer support.",
    "You can update your billing information in Settings > Billing.",
]


@tool
def search_docs(query: str) -> str:
    """Search product documentation."""

    query_words = query.lower().split()

    results = []

    for doc in DOCUMENTATION:
        if any(
                word in doc.lower()
                for word in query_words
        ):
            results.append(doc)

    if not results:
        return "No relevant documentation found."

    return "\n".join(results)


tools = [search_docs]

# ============================================================
# 创建 agent_llm
# ============================================================
agent_llm = llm.bind_tools(tools)


# ============================================================
# 4. 定义 node
# ============================================================
def classify_intent(state: EmailState, ) -> Command[
    Literal[
        "support_agent",
        "bug_tracking",
        "human_review",
    ]
]:
    email = state["email_content"]

    prompt = f"""
You are a customer support email classifier.

Classify the following customer email.

Return ONLY valid JSON.

The JSON must have exactly these fields:

{{
    "intent": "question",
    "urgency": "low",
    "summary": "short summary"
}}

Allowed intent values:

- question
- bug
- billing
- feature_request
- other

Allowed urgency values:

- low
- medium
- high

Customer Email:

{email}
"""

    result = llm.invoke(prompt)

    classification = json.loads(result.content)

    intent = classification["intent"]

    if intent == "question":
        goto = "support_agent"

    elif intent == "bug":
        goto = "bug_tracking"

    else:
        goto = "human_review"

    return Command(
        update={
            "classification": classification
        },
        goto=goto,
    )


# 定义 finish_response
def finish_response(state: EmailState):
    messages = state["messages"][-1]
    return {
        "response": messages.content,
    }


def bug_tracking(state: EmailState):
    print(">>> Bug tracking")

    return {
        "response": (
            "Thank you for reporting this issue. "
            "Our engineering team will investigate it."
        )
    }


def human_review(state: EmailState):
    print(">>> Human review")

    return {
        "response": (
            "Your request requires additional review. "
            "A support representative will contact you."
        )
    }


# 定义 toolNode
support_tool_node = ToolNode(tools)


# ============================================================
# Agent Node
# ============================================================
def support_agent(state: EmailState):
    messages = state["messages"]

    if not messages:
        messages = [
            HumanMessage(
                content=state["email_content"]
            )
        ]

    response = agent_llm.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# Agent router
# ============================================================

def should_continue(state: EmailState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "support_tools"
    return "finish_response"


# ============================================================
# 5. 构建图
# ============================================================

builder = StateGraph(EmailState)

# Node
builder.add_node("classify_intent", classify_intent)
builder.add_node("finish_response", finish_response)
builder.add_node("bug_tracking", bug_tracking)
builder.add_node("human_review", human_review)
builder.add_node("support_agent", support_agent)
builder.add_node("support_tools", support_tool_node)

# Edge
builder.add_edge(START, "classify_intent")
builder.add_conditional_edges("support_agent", should_continue)
# tool --> agent
builder.add_edge("support_tools", "support_agent")
# final response
builder.add_edge("finish_response", END)
# other workflow branch
builder.add_edge("bug_tracking", END)
builder.add_edge("human_review", END)

graph = builder.compile()

if __name__ == "__main__":

    result = graph.invoke(
        {
            "email_content": """
Hi,

I forgot my password and cannot log into my account.

How can I reset it?

Thanks!
""",

            "classification": None,

            "messages": [],

            "response": "",
        }
    )

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION")
    print("=" * 60)

    print(
        json.dumps(
            result["classification"],
            indent=2,
        )
    )

    print("\n")
    print("=" * 60)
    print("MESSAGES")
    print("=" * 60)

    for message in result["messages"]:
        print(
            f"\n[{type(message).__name__}]"
        )

        print(message)

    print("\n")
    print("=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)

    print(result["response"])
