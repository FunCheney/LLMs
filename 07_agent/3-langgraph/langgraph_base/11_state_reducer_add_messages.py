"""
【案例】add_messages Reducer：消息列表专用，节点只返回「增量消息」，由 add_messages 自动追加到 state["messages"]，适合多轮对话与多节点共同写消息的场景。

知识点速览：
- Annotated[List, add_messages] 表示该字段使用 add_messages 规约：新消息追加到列表末尾，而非覆盖。
- 节点返回格式可为 [("role", content)] 或 [AIMessage/HumanMessage] 等，由 add_messages 统一合并。
- 多节点共同写 messages 时，本例重点是“消息按 add_messages 规则合并”，不要把并行分支下的最终顺序直接当成业务契约。
"""
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph


class AddMessagesState(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node_1(state: AddMessagesState):
    return {"messages": [("assistant", "Hello from node 1")]}

def chat_node_2(state: AddMessagesState) -> dict:
    return {"messages": [("assistant", "Hello from node 2")]}

def main():
    builder = StateGraph(AddMessagesState)
    builder.add_node("node1", chat_node_1)
    builder.add_node("node2", chat_node_2)
    builder.add_edge(START, "node1")
    builder.add_edge(START, "node2")
    builder.add_edge("node1", END)
    builder.add_edge("node2", END)
    app = builder.compile()

    result = app.invoke({"messages": [("user", "Hi there!")]})
    print(f"初始状态: {{'messages': [('user', 'Hi there!')]}}")
    print(f"执行结果: {result}\n")
    print("*" * 60)
    print(app.get_graph().print_ascii())


if __name__ == "__main__":
    main()
