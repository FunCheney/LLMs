"""
LangGraph Graph API 的最小构建流程可以概括成这 5 步：

定义 State：说明这张图里要流转哪些字段。
定义 Node：把每一步逻辑写成函数，函数接收当前 state，返回要更新的字段。
定义 Edge：用 add_edge 把节点连起来，并用 START / END 指定入口和出口。
编译 Graph：调用 compile()，把图构建器编译成一个真正可运行的应用对象。
执行 Graph：调用 invoke(initial_state) 传入初始状态，拿到最终状态结果。
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class MyState(TypedDict):
    value: str

def node_a(state: MyState):
    return {"value": state["value"] + " A"}

def node_b(state: MyState):
    return {"value": state["value"] + " B"}

builder = StateGraph(MyState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

app = builder.compile()
result = app.invoke({"value": "start"})
print(result)
