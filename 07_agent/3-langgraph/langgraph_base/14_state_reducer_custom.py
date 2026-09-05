"""
【案例】自定义 Reducer：用函数签名 (current, update) -> 合并结果，解决 operator.mul 在首次规约边界上不适合直接用于乘法累计的问题。

知识点速览：
- Reducer 可以写成普通函数：接收当前字段值 `current` 与本次更新值 `update`，返回新的合并结果。
- 自定义 Reducer 的价值不在“语法复杂”，而在于你可以按业务语义处理首次合并、空值、重复值、顺序稳定性等边界。
- 节点仍只返回增量（如 `{\"factor\": 2.0}`），真正决定怎么合并的是 Reducer，而不是节点本身。
"""
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph


def MyOperatorMul(current: float, update: float) -> float:
    """自定义乘法 Reducer：首次合并时把 current 的边界情况单独处理，再继续乘法累计。"""
    # 第一次调用时 current 往往是类型默认值 0.0，若直接 current * update 会得到 0，后续无法恢复
    if current == 0.0:
        print(current)
        print(update)
        return 1.0 * update

    return current * update

class MultiplyState(TypedDict):
    factor: Annotated[float, MyOperatorMul]

def multiply(state: MultiplyState) -> dict:
    # 节点返回的 update 会与 state["factor"] 经 MyOperatorMul 合并
    return {"factor": 2.0}

if __name__ == "__main__":
    graph = StateGraph(MultiplyState)
    graph.add_node("multiply", multiply)

    graph.add_edge(START, "multiply")
    graph.add_edge("multiply", END)

    app = graph.compile()

    result = app.invoke({"factor": 5.0})
    print(f"初始状态: {{'factor': 5.0}}")
    print(f"执行结果: {result}")
    print(f"解释: 5.0 * 2.0 = 10.0\n")

