"""
【案例】条件边（Conditional Edges）：根据状态（如 x 的奇偶）在多个后继节点中选一个执行，使用 add_conditional_edges(节点名, 路由函数, 映射)。

知识点速览：
- add_conditional_edges(source, route_fn, mapping)：route_fn(state) 的返回值作为 key，在 mapping 中查到下一节点名；若为 bool，常用 {True: "node_a", False: "node_b"}。
- 条件边的重点是“让边负责分流，而不是把所有 if/else 都塞回节点里”；路由函数在 source 节点执行后被调用，根据当前 state 决定下一跳。
- 本例顺手演示了 State 也可以用 Pydantic BaseModel 定义，这更适合需要默认值和校验的场景。
"""

from typing import Optional
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from loguru import logger
from pydantic import BaseModel


class MyState(BaseModel):
    """
    定义状态模型，用于在图节点之间传递数据
    Attributes:
        x (int): 输入的整数
        result (Optional[str]): 处理结果，可为"even"或"odd"
    """

    x: int
    result: Optional[str] = None


# 检查输入状态的节点函数
def check_x(state: MyState) -> MyState:
    """
    检查输入状态的节点函数
    Args:
        state (MyState): 包含输入数据的状态对象
    Returns:
        MyState: 返回原始状态对象，未做修改
    """
    logger.info(f"[check_x] Received state: {state}")
    return state


# 判断状态中x值是否为偶数的条件函数
def is_even(state: MyState) -> bool:
    """
    判断状态中x值是否为偶数的条件函数
    Args:
        state (MyState): 包含待判断数值的状态对象
    Returns:
        bool: 如果x是偶数返回True，否则返回False
    """
    return state.x % 2 == 0


# 处理偶数情况的节点函数
def handle_even(state: MyState) -> MyState:
    """
    处理偶数情况的节点函数
    Args:
        state (MyState): 包含偶数输入的状态对象
    Returns:
        MyState: 返回更新后的状态对象，result设置为"even"
    """
    logger.info("[handle_even] x 是偶数")
    return MyState(x=state.x, result="even")


# 处理奇数情况的节点函数
def handle_odd(state: MyState) -> MyState:
    """
    处理奇数情况的节点函数
    Args:
        state (MyState): 包含奇数输入的状态对象
    Returns:
        MyState: 返回更新后的状态对象，result设置为"odd"
    """
    logger.info("[handle_odd] x 是奇数")
    return MyState(x=state.x, result="odd")


builder = StateGraph(MyState)
# 添加节点
builder.add_node("check_x", check_x)
builder.add_node("handle_even", handle_even)
builder.add_node("handle_odd", handle_odd)


# 添加条件边，根据is_even函数的返回值决定流向哪个节点
builder.add_conditional_edges(
    "check_x", is_even, {True: "handle_even", False: "handle_odd"}
)

# 添加起始边，从START节点流向check_x节点
builder.add_edge(START, "check_x")

# 添加结束边，从处理节点流向END节点
builder.add_edge("handle_even", END)
builder.add_edge("handle_odd", END)

# 编译图结构
graph = builder.compile()

# 打印图的可视化结构
print(graph.get_graph().print_ascii())

# 测试用例：输入偶数4
logger.info("输入 x=4（偶数）")
graph.invoke(MyState(x=4))

# # 测试用例：输入奇数3
# logger.info("输入 x=3（奇数）")
# graph.invoke(MyState(x=3))

"""
【输出示例】
              +-----------+               
              | __start__ |               
              +-----------+               
                    *                     
                    *                     
                    *                     
               +---------+                
               | check_x |                
               +---------+                
             ...          ..              
            .               ..            
          ..                  ..          
+-------------+           +------------+  
| handle_even |           | handle_odd |  
+-------------+           +------------+  
             ***          **              
                *       **                
                 **   **                  
               +---------+                
               | __end__ |                
               +---------+                
None
2026-03-23 16:38:23.954 | INFO     | __main__:<module>:108 - 输入 x=4（偶数）
2026-03-23 16:38:23.955 | INFO     | __main__:check_x:40 - [check_x] Received state: x=4 result=None
2026-03-23 16:38:23.955 | INFO     | __main__:handle_even:65 - [handle_even] x 是偶数
"""
