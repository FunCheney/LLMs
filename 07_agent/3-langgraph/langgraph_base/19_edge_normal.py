"""
【案例】普通边（Normal Edges）：用 add_edge 串联节点，形成固定执行顺序 START → node_a → node_b → node_c → END，无条件跳转。


知识点速览：
- 普通边：add_edge(源节点, 目标节点)，表示执行完源节点后必定进入目标节点，无分支。
- START、END 为 LangGraph 内置虚拟节点，分别表示图入口与出口。
- 线性链是最简单的图结构，适合理解“节点负责处理、边负责流转”这条主线。
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 定义状态
class DiliState(TypedDict):
    value: int
    step: str


# 定义节点函数
def node_a(state: DiliState) -> dict:
    """节点A"""
    print("执行节点A")
    return {"value": state["value"] + 1, "step": "A执行完毕"}


def node_b(state: DiliState) -> dict:
    """节点B"""
    print("执行节点B")
    return {"value": state["value"] * 2, "step": "B执行完毕"}


def node_c(state: DiliState) -> dict:
    """节点C"""
    print("执行节点C")
    return {"value": state["value"] - 1, "step": "C执行完毕"}


def main():
    """演示普通边"""
    print("=== 普通边演示 ===")

    # 创建图
    builder = StateGraph(DiliState)

    # 添加节点
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)

    # 添加普通边
    builder.add_edge(START, "node_a")  # 从开始到A
    builder.add_edge("node_a", "node_b")  # 从A到B
    builder.add_edge("node_b", "node_c")  # 从B到C
    builder.add_edge("node_c", END)  # 从C到结束

    # 编译图
    app = builder.compile()

    # 执行图
    result = app.invoke({"value": 1})
    print(f"执行结果: {result}\n")
    # 打印图的边和节点信息
    print(builder.edges)
    # print(builder.nodes)
    # 打印图的ascii可视化结构
    print(app.get_graph().print_ascii())
    print("=================================")
    print()
    # 打印图的可视化结构，生成更加美观的Mermaid 代码，通过processon 编辑器查看
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()

"""
【输出示例】
=== 普通边演示 ===
执行节点A
执行节点B
执行节点C
执行结果: {'value': 3, 'step': 'C执行完毕'}

{('node_b', 'node_c'), ('__start__', 'node_a'), ('node_a', 'node_b'), ('node_c', '__end__')}
+-----------+  
| __start__ |  
+-----------+  
      *        
      *        
      *        
  +--------+   
  | node_a |   
  +--------+   
      *        
      *        
      *        
  +--------+   
  | node_b |   
  +--------+   
      *        
      *        
      *        
  +--------+   
  | node_c |   
  +--------+   
      *        
      *        
      *        
 +---------+   
 | __end__ |   
 +---------+   
None
=================================

---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        node_a(node_a)
        node_b(node_b)
        node_c(node_c)
        __end__([<p>__end__</p>]):::last
        __start__ --> node_a;
        node_a --> node_b;
        node_b --> node_c;
        node_c --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc

"""
