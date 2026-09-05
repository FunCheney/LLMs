"""
【案例】入口点与出口点：用 set_entry_point / set_finish_point 指定图的第一个和最后一个节点，等价于 add_edge(START, node) 与 add_edge(node, END)，写法更简洁。


知识点速览：
- set_entry_point(node_id)：图从该节点开始执行，底层等价于 add_edge(START, node_id)。
- set_finish_point(node_id)：执行到该节点后图结束，底层等价于 add_edge(node_id, END)。
- 适合线性链或单入口单出口的图，减少重复写 START/END 边。
- 本例重点是理解“入口/出口的声明方式”，不是引入新类型的边；它本质上仍然是在配置普通边。
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
    print("state[value]:" + str(state["value"]))
    print("state[step]:" + str(state["step"]))
    return {"value": state["value"] + 1, "step": "A执行完毕"}


def node_b(state: DiliState) -> dict:
    """节点B"""
    print("执行节点B")
    return {"value": state["value"] * 2, "step": "B执行完毕"}


def main():
    """演示入口点"""
    print("=== 入口点演示 ===")

    # 创建图
    builder = StateGraph(DiliState)

    # 添加节点
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)

    # set_entry_point / set_finish_point 是更简洁的入口出口配置方式，本质上仍然是在帮你建立 START/END 的边
    builder.set_entry_point("node_a")
    builder.add_edge("node_a", "node_b")
    builder.set_finish_point("node_b")

    # 编译图
    graph = builder.compile()
    # 执行图
    result = graph.invoke({"value": 0, "step": "hello"})
    print(f"执行结果: {result}\n")

    print()
    # 打印图的ascii可视化结构
    print(graph.get_graph().print_ascii())
    print("=================================")
    print()
    # 打印图的可视化结构，生成更加美观的Mermaid 代码，通过processon 编辑器查看
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()

"""
【输出示例】
=== 入口点演示 ===
执行节点A
state[value]:0
state[step]:hello
执行节点B
执行结果: {'value': 2, 'step': 'B执行完毕'}


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
        __end__([<p>__end__</p>]):::last
        __start__ --> node_a;
        node_a --> node_b;
        node_b --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
"""
