"""
最简 State 定义，与无中间节点图：用 TypedDict 定义状态，构建一条从 START 到 END 的边，验证 invoke(initial_state) 的用法
知识点速览：
- State 由 Schema（模式）和 Reducer（规约函数）两部分组成。
- 本例用 TypedDict（下方 `BasicState`）定义 State Schema（字段名与类型）；图中所有节点读写同一份状态结构。
- 字段未用 `Annotated[..., reducer]` 指定 Reducer 时，使用 LangGraph 默认 Reducer（常见为节点返回的新值覆盖该字段旧值）。
- add_edge(START, END) 表示没有业务节点，图从入口直接到出口；本例重点是理解“State 原样透传”和 `invoke(initial_state)` 的基本调用方式。
- invoke() 只接收一个核心位置参数：状态字典；不要传入多个独立参数。可选第二参数为 config。
- 嵌套类型（如 process_data: dict）在 initial_state 中需传入合法字典。
"""
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class BasicState(TypedDict):
    """本图的 State Schema：字段名 + 类型共同定义这张图允许流转的状态结构。"""

    user_input: str
    response: str
    count: int
    process_data: dict


# 创建图，BasicState 是本图的 state_schema。本例不写 Annotated，因此个字段走默认覆盖的规则
baseState = StateGraph(BasicState)
# 无中间节点直接从 START 到 END
baseState.add_edge(START, END)
# 编译
app = baseState.compile()


# invoke 只接收一个核心参数（状态字典）；process_data 为 dict，需传入嵌套字典
initial_state = {
    "user_input": "a",
    "response": "resp",
    "count": 25,
    "process_data": {"k1": "v1"},
}

result = app.invoke(initial_state)
print("执行结果：", result)