"""
【案例】流式传输图状态：对比 stream_mode 为 updates 与 values 时，每一步向调用方推送的内容差异。

知识点速览：
- `stream(..., stream_mode="updates")`：每步只推送“本节点本次改了什么”，更像增量日志。
- `stream(..., stream_mode="values")`：每步推送“当前完整状态长什么样”，更像全量快照。
"""
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class DiliState(TypedDict):
    topic: str
    joke: str

def refine_topic(state: DiliState):
    return {"topic": state["topic"] + "and cats"}

def generate_joke(state: DiliState):
    return {"joke": f"This is a joke about {state['topic']}"}

def main():
    graph = (StateGraph(DiliState).add_node(refine_topic).add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile())

    for chunk in graph.stream({"topic": "ice cream"}, stream_mode="updates"):
        print(chunk)

    print()

    for chunk in graph.stream({"topic": "ice cream"}, stream_mode="values"):
        print(chunk)

if __name__ == "__main__":
    main()


'''
{'refine_topic': {'topic': 'ice creamand cats'}}
{'generate_joke': {'joke': 'This is a joke about ice creamand cats'}}

{'topic': 'ice cream'}
{'topic': 'ice creamand cats'}
{'topic': 'ice creamand cats', 'joke': 'This is a joke about ice creamand cats'}
'''