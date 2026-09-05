"""
【案例】operator.mul 作为 Reducer（数值相乘）的「陷阱」
演示：LangGraph 会用类型默认值（float 的 0.0）先做一次规约，导致 0.0 * 初始值 = 0，后续乘法始终为 0；理解后可用自定义 Reducer 解决。

知识点速览：
- 这个案例的重点不是“operator.mul 不能跑”，而是理解“乘法这类对初始值很敏感的规约逻辑，不能只看 reducer 函数名，还要看首次合并边界”。
- 当字段默认值是 `0.0` 时，乘法规约很容易在第一次合并就变成 `0.0`，后面再乘什么都还是 `0.0`。
- 解决方式通常是改成自定义 Reducer，在函数里显式处理首次合并逻辑。参见 `StateReducer_Custom.py`。
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class MultiplyState(TypedDict):
    factor: Annotated[float, operator.mul]


def multiplier(state: MultiplyState) -> dict:
    return {"factor": 2.0}


# 这里故意保留 operator.mul 的“踩坑版”写法，目的是先让你观察问题，再对照 StateReducer_Custom.py 理解为什么真实项目更适合写自定义 Reducer


def run_demo():
    print("4. operator.mul Reducer（数值相乘）演示:")
    builder = StateGraph(MultiplyState)
    builder.add_node("multiplier", multiplier)
    builder.add_edge(START, "multiplier")
    builder.add_edge("multiplier", END)
    graph = builder.compile()

    result = graph.invoke({"factor": 5.0})
    print(f"初始状态: {{'factor': 5.0}}")
    print(f"执行结果: {result}")
    print(
        "说明: 因 float 默认 0.0 先参与规约，0.0 * 5.0 = 0.0，后续乘 2.0 仍为 0.0；乘法场景请用自定义 Reducer。\n"
    )


if __name__ == "__main__":
    run_demo()

"""
【输出示例】
4. operator.mul Reducer（数值相乘）演示:
初始状态: {'factor': 5.0}
执行结果: {'factor': 0.0}
说明: 因 float 默认 0.0 先参与规约，0.0 * 5.0 = 0.0，后续乘 2.0 仍为 0.0；乘法场景请用自定义 Reducer。
"""
