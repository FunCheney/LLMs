# email_assistant.py

import json
from typing import Literal

from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import dotenv
import os

dotenv.load_dotenv()


# ============================================================
# 1. 定义 State
# ============================================================

class EmailClassification(TypedDict):
    """邮件分类结果"""
    intent: Literal[
        "question",
        "bug",
        "billing",
        "feature_request",
        "other",
    ]

    urgency: Literal[
        "low",
        "medium",
        "high",
    ]

    summary: str


class EmailState(TypedDict):
    """Email Assistant 的全局状态"""
    # 原始邮件
    email_content: str
    # 邮件分类结果
    classification: EmailClassification | None
    # 文档搜索结果
    search_results: list[str]
    # 最终回复
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

# 3. 模拟 Documentation Database

# ============================================================

DOCUMENTATION = [
    "To reset your password, go to Settings > Security > Reset Password.",
    "Password reset links expire after 30 minutes.",
    "If your account is locked, contact customer support.",
    "You can update your billing information in Settings > Billing.",
]


# ============================================================

# 4. Tool：搜索文档

# ============================================================

def search_docs(query: str) -> list[str]:
    """
    搜索产品文档。

    目前只是一个简单的 keyword search，
    后面可以替换成：
    - Vector Store
    - RAG
    - Web Search
    """

    results = []

    query_words = query.lower().split()

    for doc in DOCUMENTATION:

        if any(
                word in doc.lower()
                for word in query_words
        ):
            results.append(doc)

    if not results:
        return [
            "No relevant documentation found."
        ]

    return results


# ============================================================
# 5. Node：分类邮件
# ============================================================

def classify_intent(state: EmailState, ) -> Command[
    Literal[
        "search_documentation",
        "bug_tracking",
        "human_review",
    ]
]:
    email = state["email_content"]

    prompt = f"""
    ```
    
    You are a customer support email classifier.
    
    Classify the following customer email.
    
    Return ONLY valid JSON.
    
    The JSON must have exactly these fields:
    
    {{
    "intent": "...",
    "urgency": "...",
    "summary": "..."
    }}
    
    Allowed intent values:
    
    * question
    * bug
    * billing
    * feature_request
    * other
    
    Allowed urgency values:
    
    * low
    * medium
    * high
    
    Customer Email:
    
    {email}
    """

    result = llm.invoke(prompt)

    classification = json.loads(
        result.content
    )

    intent = classification["intent"]

    # --------------------------------------------------------
    # Routing Decision
    # --------------------------------------------------------
    if intent == "question":

        goto = "search_documentation"

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


# ============================================================
# 6. Node：搜索文档
# ============================================================

def search_documentation(
        state: EmailState,
) -> Command[
    Literal["draft_response"]
]:
    email = state["email_content"]

    results = search_docs(email)

    return Command(
        update={
            "search_results": results
        },
        goto="draft_response",
    )


# ============================================================
# 7. Node：生成回复
# ============================================================

def draft_response(
        state: EmailState,
):
    email = state["email_content"]

    classification = state["classification"]

    search_results = state["search_results"]

    context = "\n".join(search_results)

    prompt = f"""
    ```
    
    You are a helpful customer support assistant.
    
    Write a professional response to the customer.
    
    Customer Email:
    
    {email}
    
    Email Classification:
    
    {classification}
    
    Relevant Documentation:
    
    {context}
    
    Requirements:
    
    * Answer based on the relevant documentation.
    * Be helpful and concise.
    * Do not invent information.
    * If the documentation is insufficient,
      clearly tell the customer.
      """

    result = llm.invoke(prompt)

    return {
        "response": result.content
    }


# ============================================================
# 8. Node：Bug Tracking
# ============================================================

def bug_tracking(
        state: EmailState,
):
    print(">>> Bug tracking node")

    return {
        "response": (
            "Thank you for reporting this issue. "
            "Our engineering team will investigate it."
        )
    }


# ============================================================
# 9. Node：Human Review
# ============================================================

def human_review(
        state: EmailState,
):
    print(">>> Human review node")

    return {
        "response": (
            "Your request requires additional review. "
            "A support representative will contact you."
        )
    }


# ============================================================
# 10. 构建 Graph
# ============================================================

builder = StateGraph(EmailState)

# ------------------------------------------------------------
# 注册 Nodes
# ------------------------------------------------------------

builder.add_node("classify_intent", classify_intent)
builder.add_node("search_documentation",search_documentation)
builder.add_node("draft_response",draft_response)
builder.add_node("bug_tracking",bug_tracking)
builder.add_node("human_review",human_review)

# ------------------------------------------------------------
# Static Edges
# ------------------------------------------------------------

builder.add_edge(START,"classify_intent")
builder.add_edge("draft_response",END)
builder.add_edge("bug_tracking",END)
builder.add_edge("human_review",END)

# ============================================================
# 11. Compile Graph
# ============================================================

email_assistant = builder.compile()

# ============================================================
# 12. Run
# ============================================================

if __name__ == "main":

    initial_state = {
        "email_content": """
    ```
    
    Hi,
    
    I forgot my password and cannot log into my account.
    
    How can I reset it?
    
    Thanks!
    """,

        "classification": None,

        "search_results": [],

        "response": "",
    }

    result = email_assistant.invoke(
        initial_state
    )

    print("\n")
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)

    print("\nClassification:")

    print(
        json.dumps(
            result["classification"],
            indent=2,
        )
    )

    print("\nSearch Results:")

    for item in result["search_results"]:
        print(f"- {item}")

    print("\nResponse:")

    print(result["response"])
