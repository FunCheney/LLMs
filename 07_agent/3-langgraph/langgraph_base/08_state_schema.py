"""
图的输入/输出 Schema：用 input_schema 和 output_schema 限制「调用时只能传 question、返回时只拿 answer」，实现对外接口的契约化，
适合需要明确 I/O 边界的场景。

知识点速览：
- 本例最适合拿来理解“State 不只有一种 Schema”：`OverallState` 是内部完整 State Schema，`InputState` / `OutputState` 是图对外暴露的输入输出契约。
- 构建时 `StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)`，第一个位置参数描述内部完整状态，
    后两个参数负责限制边界输入输出。
- 节点内部仍围绕完整状态空间工作；只有「图的边界」受 input/output 约束，这种分层更贴近真实项目接口封装。
"""

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# 仅包含「输入」字段的 Schema：限制调用方进图时能传什么
class InputState(TypedDict):
    question: str


# 仅包含「输出」字段的 Schema：限制图最终对外返回什么
class OutputState(TypedDict):
    answer: str


# 图内部使用的完整 State Schema（输入 + 输出）
class OverallState(InputState, OutputState):
    pass


def answer_node(state: InputState):
    """处理节点：根据 question 生成 answer。"""
    print(f"执行 answer_node 节点:")
    print(f"  输入: {state}")
    answer = "再见" if "bye" in state["question"].lower() else "你好"
    result = {"answer": answer, "question": state["question"]}
    print(f"  输出: {result}")
    return result


def demo_input_output_schema():
    """演示：调用时只传 question，返回时只得到 answer。"""
    print("=== 演示输入输出模式 ===")

    # 指定 input_schema / output_schema，约束图的对外接口
    builder = StateGraph(
        OverallState, input_schema=InputState, output_schema=OutputState
    )
    builder.add_edge(START, "answer_node")
    builder.add_node("answer_node", answer_node)
    builder.add_edge("answer_node", END)
    graph = builder.compile()

    # invoke 只传 InputState 的字段；返回结果仅包含 OutputState 的字段
    result = graph.invoke({"question": "你好"})
    print(f"图调用结果: {result}")
    print(graph.get_graph().print_ascii())
    print()


def main():
    print("=== LangGraph 图输入输出模式===\n")
    demo_input_output_schema()
    print("=== 演示完成 ===")


if __name__ == "__main__":
    main()

"""
【输出示例】
=== LangGraph 图输入输出模式===

=== 演示输入输出模式 ===
执行 answer_node 节点:
  输入: {'question': '你好'}
  输出: {'answer': '你好', 'question': '你好'}
图调用结果: {'answer': '你好'}
 +-----------+   
 | __start__ |   
 +-----------+   
        *        
        *        
        *        
+-------------+  
| answer_node |  
+-------------+  
        *        
        *        
        *        
  +---------+    
  | __end__ |    
  +---------+    
None

=== 演示完成 ===
"""
