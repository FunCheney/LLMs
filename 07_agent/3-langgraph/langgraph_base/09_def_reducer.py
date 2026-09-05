"""
本例展示  state 中 reducer 的简单定义

在 TypedDict 里，指定 Reducer 的常见写法是 Annotated[字段类型, reducer函数]
"""
import operator
from typing import TypedDict, Annotated

from langgraph.graph import add_messages


class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    tags: Annotated[list, operator.add]
    count: Annotated[int, operator.add]
    latest_answer: str

