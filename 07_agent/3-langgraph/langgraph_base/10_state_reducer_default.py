"""
默认 Reducer（覆盖更新）：未为状态字段指定 Reducer 时，节点返回的值会直接覆盖该字段，后执行节点的结果覆盖先执行节点的结果。

知识点速览：
- Reducer 决定「节点返回的更新如何合并到当前状态」；不指定时采用默认行为：覆盖。
- 多节点依次更新同一字段时，最终状态中该字段只保留最后一个节点返回的值。
- 适合「单写」场景；若需追加、累加等，需使用 add_messages、operator.add 等 Reducer。
"""
from typing import TypedDict, List

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class DefaultReducerState(TypedDict):
    foo: int
    bar: List[str]



def node_default_1(state: DefaultReducerState) -> dict:
    print(state["foo"])
    print(state["bar"])
    return {"foo": 22}

def node_default_2(state: DefaultReducerState) -> dict:
    print(state["foo"])
    print(state["bar"])
    return {"bar": ["bye1", "bye2", "bye3"]}

def main():
    builder = StateGraph(DefaultReducerState)
    builder.add_node("node1", node_default_1)
    builder.add_node("node2", node_default_2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)

    app = builder.compile()

    result = app.invoke({
        "foo": 1,
        "bar": ["hi"],
    })

    print(result)

if __name__ == "__main__":
    main()