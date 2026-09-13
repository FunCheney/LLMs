"""
 copy - vo
"""

from typing import TypedDict, Literal, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


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
    email_content: str
    classification: EmailClassification | None
    messages: Annotated[List[BaseMessage], add_messages]
    response: str