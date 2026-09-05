from functools import partial
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy
from requests import RequestException


class GraphState(TypedDict):
    process_data: dict


def input_node(state: GraphState) -> dict:
    print(f"input_node 收到初始值: {state}")
    return {"process_data": {"input": "input_value"}}

# 节点可以带额外参数，用 partial 绑定后传给 add_node
def process_node(state: dict, param1: int, param2: int) -> dict:
    print(state, param1, param2)
    return {"process_data": {"process": "process_value"}}

# 重试策略：仅对 RequestException、Timeout 重试，最多 3 次
policy = RetryPolicy(max_attempts=3, initial_interval=1, jitter=True, backoff_factor=2,
                     retry_on=[RequestException, TimeoutError])

graph = StateGraph(GraphState)
graph.add_node("input", input_node)
process_node_with_params = partial(process_node, param1=100, param2="test")
graph.add_node("process", process_node_with_params, retry_policy=policy)

graph.add_edge(START, "input")
graph.add_edge("input", "process")
graph.add_edge("process", END)

app = graph.compile()

print(graph.edges)
print(graph.nodes)
print(app.get_graph().print_ascii())
print()

result = app.invoke({"process_data": 5})
print(f"最后的结果是:{result}")


