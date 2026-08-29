"""
多节点固定边的完整图：input --> process --> output 三个节点，状态字段 process_data 在节点之间传递，并在默认覆盖规则下，被后续节点更新。

知识点速览：
- StateGraph(GraphState) 指定状态类型后，各节点接收完整 state，返回对 state 的「部分更新」字典。
- 本例的重点是“图怎么搭起来”，不是“状态怎么累积起来”；未为字段指定 Reducer 时，默认覆盖：后一节点返回的 process_data 会覆盖前一节点的值。
- 固定边：add_edge 依次串联 START → input → process → output → END，执行顺序确定。
- 编译后 invoke(initial_state) 传入初始 process_data，结果中最终保留的是最后一次写入该字段的内容。
"""

import dotenv
import os
from typing import Dict, TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

dotenv.load_dotenv()



#1.定义状态 process_data 用于在状态之间传递，本例未指定 Reducer，会按照默认覆盖规则
class GraphState(TypedDict):
    process_data: dict


def input_node(state: GraphState) -> dict:
    """入口节点：写入初始 process_data。"""
    print(f"input_node 节点执行 state.get('process_data'): {state.get('process_data')}")
    return {"process_data": {"input": "input_value"}}


def process_node(state: dict) -> dict:
    """处理节点：更新 process_data。"""
    print(
        f"process_node 节点执行 state.get('process_data'): {state.get('process_data')}"
    )
    return {"process_data": {"process": "process_value9527"}}


def output_node(state: GraphState) -> dict:
    """出口节点：读取并返回当前 process_data。"""
    print(
        f"output_node 节点执行 state.get('process_data'): {state.get('process_data')}"
    )
    return {"process_data": state.get("process_data")}

# 创建图并指定状态
graph = StateGraph(GraphState)
graph.add_node("input", input_node)
graph.add_node("process", process_node)
graph.add_node("output", output_node)

# 固定边 START --> input --> process --> output --> END
graph.add_edge(START, "input")
graph.add_edge("input", "process")
graph.add_edge("process", "output")
graph.add_edge("output", END)

# 编译
app = graph.compile()

result = app.invoke({"process_data": {"name": "测试数据", "value": 12345}})

print(result)

# 可视化
print(app.get_graph().print_ascii())

print()

print(app.get_graph().draw_mermaid())