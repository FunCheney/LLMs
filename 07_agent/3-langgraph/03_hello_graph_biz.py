"""
不接入大模型的业务图：用自定义加法/减法节点演示“同一个状态字段 x 如何沿着节点逐步更新”，帮助理解 Graph 不一定每个节点都要调用 LLM

知识点速览：
- 用 dict 作为 State 类型时，无需预定义 TypedDict，适合快速试验；但如果状态字段逐渐变多，真实项目里更建议改成 TypedDict 或 Pydantic，避免字段名和类型失控。
- 节点函数接收 state，返回要更新的键值对（如 {"x": state["x"] + 1}），LangGraph 会按默认 Reducer 合并；未显式指定 Reducer 时，通常就是“新值覆盖旧值”。
- add_edge 串联 START → addition → subtraction → END，形成固定执行顺序。
- graph.edges / graph.nodes 可在 compile() 前查看当前图已注册的边与节点，适合排查“节点没加进去”或“边连错了”的问题。
- 本例是一条固定线性业务流；如果后面出现“根据 x 的值决定走不同分支”或“某一步失败后回退重跑”，就更能体现 LangGraph 相比单纯 LCEL Chain 的价值。
"""
from langgraph.constants import START, END
from langgraph.graph import StateGraph


def addition(state):
    """加法节点"""
    print(f'加法节点中收到的初始值：{state}')
    return {"x": state["x"] + 1}

def subtraction(state):
    """ 加法节点"""
    print(f'减法节点中收到的初始值：{state}')
    return {"x": state["x"] - 2}


# 使用 dict 作为状态类型，无需预定义 TypedDict
graph = StateGraph(dict)
graph.add_node("addition", addition)
graph.add_node("subtraction", subtraction)
# 定义执行顺序：START → addition → subtraction → END
graph.add_edge(START, "addition")
graph.add_edge("addition", "subtraction")
graph.add_edge( "subtraction", END)

# 查看图的边和节点
print(graph.nodes)
print(graph.edges)

# 编译图，得到可执行的图应用对象
app = graph.compile()
# invoke() 的核心输入是一整个状态字典，这里给 x 一个初始值 5
init_state = {"x": 5}
# invoke 只接收一个核心参数：初始状态字典
result = app.invoke(init_state)

print(result)

# 打印可视化结构
print(app.get_graph().print_ascii())
print()

# 打印图的可视化结构，生成更加美观的Mermaid 代码，通过processon 编辑器查看
print(app.get_graph().draw_mermaid())