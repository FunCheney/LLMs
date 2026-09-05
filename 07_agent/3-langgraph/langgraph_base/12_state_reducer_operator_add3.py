"""
【案例】operator.add 作为 Reducer（数值）：对数值字段做「累加」，多节点返回的数会与当前值相加，适合计数、积分等场景。

知识点速览：
- Annotated[int, operator.add] 表示该字段用 operator.add 规约：语义为数值加法，即 current + update。
- 初始状态提供起点（如 count: 10），各节点返回 {"count": 增量}，最终 state["count"] 为累加结果。
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class NumberAddState(TypedDict):
    count: Annotated[int, operator.add]


def increment_1(state: NumberAddState) -> dict:
    return {"count": 5}


def increment_2(state: NumberAddState) -> dict:
    return {"count": 3}


def run_demo():
    print("3.3 数值累加 Reducer 演示:")
    builder = StateGraph(NumberAddState)
    builder.add_node("increment_1", increment_1)
    builder.add_node("increment_2", increment_2)
    builder.add_edge(START, "increment_1")
    builder.add_edge("increment_1", "increment_2")
    builder.add_edge("increment_2", END)
    graph = builder.compile()
    result = graph.invoke({"count": 10})
    print(f"初始状态: {{'count': 10}}")
    print(f"执行结果: {result}\n")


if __name__ == "__main__":
    run_demo()

"""
【输出示例】
3.3 数值累加 Reducer 演示:
初始状态: {'count': 10}
执行结果: {'count': 18}
"""
