"""
当前版本项目结构
 email_assistant/
│
├── schemas.py
│   ├── EmailClassification
│   └── EmailState
│
├── prompts.py
│   ├── TRIAGE_PROMPT
│   └── SUPPORT_AGENT_SYSTEM_PROMPT
│
└── email_assistant.py
    ├── LLM
    ├── Tools
    ├── Triage
    ├── Agent
    └── Graph
"""

import json
from typing import Literal
import os
import dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from schemas import EmailState,EmailClassification
from prompts import TRIAGE_PROMPT, SUPPORT_AGENT_SYSTEM_PROMPT

dotenv.load_dotenv()

# ============================================================
# 2. 创建 LLM
# ============================================================
llm = ChatOpenAI(
    model="kimi-k3",
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url=os.getenv("KIMI_BASE_URL"),
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
def classify_intent(state: EmailState, ):
    email = state["email_content"]

    prompt = TRIAGE_PROMPT.format(email=email)

    # 标准化输出格式
    triage_llm = llm.with_structured_output(EmailClassification)
    classification = triage_llm.invoke(prompt)
    return {"classification": classification}


# 定义 finish_response
def finish_response(state: EmailState):
    print(">>> finish_response")
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
    print(">>> support_agent")
    messages = state["messages"]
    # 只是给本次调用增加了 SystemMessage，没有被写入 state["messages"]
    messages = [
        SystemMessage(
            content=SUPPORT_AGENT_SYSTEM_PROMPT
        ),
        *messages,
    ]

    response = agent_llm.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# Agent router
# ============================================================

def should_continue(state: EmailState):
    print(">>> should_continue")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "support_tools"
    return "finish_response"

def route_after_triage(state: EmailState)->Literal[
        "support_agent",
        "bug_tracking",
        "human_review",
    ]:

    intent = state["classification"]["intent"]
    if intent == "question":
        return "support_agent"
    elif intent == "bug":
        return "bug_tracking"
    else:
        return "human_review"



# ============================================================
# 5. 构建图
# ============================================================
"""
             Workflow
                 │
                 ▼
          classify_intent
                 │
                 ▼
              Router
                 │
                 ▼
          ┌──────────────┐
          │    Agent     │
          │              │
          │ LLM decides  │
|————————>│ whether to   │
│         │ call tools   │
│         └──────┬───────┘
│                │
│          tool_call?
│           /       \
│         yes        no
│         ↓          ↓
|─── ToolNode    final answer
"""


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
builder.add_conditional_edges("classify_intent", route_after_triage)
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

    email = """
    Hi,

    I forgot my password and cannot log into my account.

    How can I reset it?

    Thanks!
    """

    result = graph.invoke({
        "email_content": email,
        "classification": None,
        "messages": [
            HumanMessage(content=email)
        ],
        "response": "",
    })

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
