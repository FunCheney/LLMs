"""
【案例】条件入口点：从 START 开始就根据状态分支，
    使用 add_conditional_edges(START, route_fn, mapping)，根据初始输入（如 user_input）决定进入哪个处理节点。

知识点速览：
- add_conditional_edges(START, route_input, {"greeting": "greeting_node", ...})：
    invoke 传入的 state 先交给 route_input，返回值作为 key 在 mapping 中查下一节点，实现「不同输入走不同入口」。
- 与「条件边」区别：条件边是“某节点执行完后”再分支；条件入口点是“图一启动”就分支，常用于做一级路由。
- 本例重点是理解“图从哪里开始”可以由输入动态决定；至于问候、告别、问题这三类文案本身，只是为了帮助观察路由效果。
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义简单的状态
class SimpleState(TypedDict):
    user_input: str
    response: str
    node_visited: str


# 2. 路由函数 - 决定从START去哪
def route_input(state: SimpleState) -> str:
    """根据用户输入决定去哪个节点"""
    text = state["user_input"].lower()

    if "hello" in text or "hi" in text:
        return "greeting"  # 返回路由键
    elif "bye" in text or "exit" in text:
        return "farewell"  # 返回路由键
    else:
        return "question"  # 返回路由键


# 3. 各个处理节点
def handle_greeting(state: SimpleState) -> SimpleState:
    """处理问候"""
    state["response"] = "你好！很高兴见到你！"
    state["node_visited"] = "greeting_node"
    return state


def handle_farewell(state: SimpleState) -> SimpleState:
    """处理告别"""
    state["response"] = "再见！祝你有个美好的一天！"
    state["node_visited"] = "farewell_node"
    return state


def handle_question(state: SimpleState) -> SimpleState:
    """处理问题"""
    state["response"] = "我听到了你的问题，需要更多帮助吗？"
    state["node_visited"] = "question_node"
    return state


# 4. 创建图
def create_simple_graph():
    """创建一个简单的图"""
    stateGraph = StateGraph(SimpleState)

    # 添加节点
    stateGraph.add_node("greeting_node", handle_greeting)
    stateGraph.add_node("farewell_node", handle_farewell)
    stateGraph.add_node("question_node", handle_question)

    # 条件入口点：图从 START 进入后，先调用 route_input，再根据 mapping 决定第一跳去哪个业务节点
    stateGraph.add_conditional_edges(
        START,  # 起点
        route_input,  # 路由函数
        # 路由映射（可选）：路由函数的返回值 -> 节点名
        {
            "greeting": "greeting_node",  # route_input返回"greeting"时，去greeting_node
            "farewell": "farewell_node",  # route_input返回"farewell"时，去farewell_node
            "question": "question_node",  # route_input返回"question"时，去question_node
        },
    )

    # 所有节点都到END
    stateGraph.add_edge("greeting_node", END)
    stateGraph.add_edge("farewell_node", END)
    stateGraph.add_edge("question_node", END)

    return stateGraph.compile()


# 5. 使用示例
def run_example():
    # 创建图
    graph = create_simple_graph()
    # 测试不同的输入
    test_inputs = ["Hello everyone!", "Goodbye now", "What time is it?"]

    for user_input in test_inputs:
        print(f"\n输入: {user_input}")
        print("-" * 30)

        # 创建初始状态
        initial_state = SimpleState(user_input=user_input, response="", node_visited="")

        # 执行图
        result = graph.invoke(initial_state)

        print(f"路由决策: {route_input(initial_state)}")
        print(f"访问的节点: {result['node_visited']}")
        print(f"响应: {result['response']}")

    print()
    # 打印图的ascii可视化结构
    print(graph.get_graph().print_ascii())
    print("=================================")
    print()
    # 打印图的可视化结构，生成更加美观的Mermaid 代码，通过processon 编辑器查看
    print(graph.get_graph().draw_mermaid())


# 运行示例
if __name__ == "__main__":
    print("简单条件入口点示例")
    print("=" * 40)
    run_example()


"""
【输出示例】
简单条件入口点示例
========================================

输入: Hello everyone!
------------------------------
路由决策: greeting
访问的节点: greeting_node
响应: 你好！很高兴见到你！

输入: Goodbye now
------------------------------
路由决策: farewell
访问的节点: farewell_node
响应: 再见！祝你有个美好的一天！

输入: What time is it?
------------------------------
路由决策: question
访问的节点: question_node
响应: 我听到了你的问题，需要更多帮助吗？

                              +-----------+                                
                              | __start__ |.                               
                         .....+-----------+ .....                          
                     ....           .            ....                      
                .....               .                .....                 
             ...                    .                     ...              
+---------------+           +---------------+           +---------------+  
| farewell_node |           | greeting_node |           | question_node |  
+---------------+****       +---------------+        ***+---------------+  
                     ****           *            ****                      
                         *****      *       *****                          
                              ***   *    ***                               
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
        greeting_node(greeting_node)
        farewell_node(farewell_node)
        question_node(question_node)
        __end__([<p>__end__</p>]):::last
        __start__ -. &nbsp;farewell&nbsp; .-> farewell_node;
        __start__ -. &nbsp;greeting&nbsp; .-> greeting_node;
        __start__ -. &nbsp;question&nbsp; .-> question_node;
        farewell_node --> __end__;
        greeting_node --> __end__;
        question_node --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
"""
