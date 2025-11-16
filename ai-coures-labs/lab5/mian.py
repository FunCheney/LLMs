"""
实验5：LangGraph工作流路径验证
学生需要使用 LangGraph 构建状态图工作流，实现基于输入意图的条件路由
"""
from typing import TypedDict, Literal, List

# 导入 LangGraph 相关模块
from langgraph.graph import StateGraph, END


class WorkflowState(TypedDict):
    """
    工作流状态定义
    """
    input: dict  # 原始输入数据
    intent: str  # 识别出的意图类型
    route_taken: str  # 实际执行的路由路径
    intermediate_results: List[str]  # 中间节点的处理结果
    final_output: str  # 最终输出内容


def start_node(state: WorkflowState) -> WorkflowState:
    """
    起始节点，解析用户意图并初始化状态
    """
    # 从输入中获取意图
    intent = state["input"].get("intent", "unknown")

    # 更新状态
    state["intent"] = intent
    state["route_taken"] = "start"
    state["intermediate_results"] = [f"开始处理意图: {intent}"]
    state["final_output"] = ""

    print(f"起始节点: 识别到意图 '{intent}'")
    return state


def order_handler_node(state: WorkflowState) -> WorkflowState:
    """
    订单处理节点
    """
    query = state["input"].get("query", "")

    # 处理订单逻辑
    order_result = f"订单处理: 正在处理查询 '{query}' - 创建新订单并确认库存"

    # 更新状态
    state["route_taken"] = "order_handler"
    state["intermediate_results"].append(order_result)
    state["final_output"] = f"订单处理完成: 已为您创建订单，查询内容: {query}"

    print(f"订单处理节点: {order_result}")
    return state


def info_handler_node(state: WorkflowState) -> WorkflowState:
    """
    信息查询节点
    """
    query = state["input"].get("query", "")

    # 处理信息查询逻辑
    info_result = f"信息查询: 正在处理查询 '{query}' - 检索数据库并返回信息"

    # 更新状态
    state["route_taken"] = "info_handler"
    state["intermediate_results"].append(info_result)
    state["final_output"] = f"信息查询完成: 找到相关信息，查询内容: {query}"

    print(f"信息查询节点: {info_result}")
    return state


def error_handler_node(state: WorkflowState) -> WorkflowState:
    """
    错误处理节点
    """
    intent = state["intent"]

    # 处理错误逻辑
    error_result = f"错误处理: 无法识别的意图 '{intent}'"

    # 更新状态
    state["route_taken"] = "error_handler"
    state["intermediate_results"].append(error_result)
    state["final_output"] = f"抱歉，无法处理您的请求。不支持的意图类型: {intent}"

    print(f"错误处理节点: {error_result}")
    return state


def end_node(state: WorkflowState) -> WorkflowState:
    """
    结束节点，整合输出
    """
    # 整合最终输出
    final_summary = f"工作流执行完成。路径: {state['route_taken']}, 步骤数: {len(state['intermediate_results'])}"
    state["intermediate_results"].append(final_summary)

    print(f"结束节点: {final_summary}")
    return state


def route_intent(state: WorkflowState) -> Literal["order", "info", "error", "__end__"]:
    """
    路由决策函数，根据意图决定下一个节点
    """
    intent = state["intent"]

    print(f"路由决策: 当前意图 '{intent}'")

    if intent == "order":
        return "order"
    elif intent == "info":
        return "info"
    else:
        return "error"


def run_workflow(input_data: dict) -> dict:
    """
    执行 LangGraph 工作流，根据意图进行路由

    参数:
        input_data: 输入字典，必须包含以下键:
            - intent (str): 用户意图类型（"order" 或 "info"）
            - query (str): 用户查询内容

    返回:
        字典，包含以下键:
        - route_taken (str): 实际执行的路由名称（如 "order_handler"）
        - final_output (str): 工作流的最终输出内容
        - execution_path (list): 节点执行顺序列表

    实现要求:
        1. 创建 StateGraph 并定义工作流状态
        2. 添加以下节点：
           - start: 起始节点，解析用户意图
           - order_handler: 订单处理节点（当 intent=="order" 时）
           - info_handler: 信息查询节点（当 intent=="info" 时）
           - end: 结束节点，整合输出
        3. 实现条件路由逻辑：
           - 当 intent == "order" 时，路由到 order_handler
           - 当 intent == "info" 时，路由到 info_handler
           - 其他情况路由到 error_handler
        4. 追踪执行路径（execution_path）
    """
    # 初始化工作流状态
    initial_state: WorkflowState = {
        "input": input_data,
        "intent": "",
        "route_taken": "",
        "intermediate_results": [],
        "final_output": ""
    }

    # 创建 StateGraph
    graph = StateGraph(WorkflowState)

    # 添加节点
    graph.add_node("start", start_node)
    graph.add_node("order", order_handler_node)
    graph.add_node("info", info_handler_node)
    graph.add_node("error", error_handler_node)
    graph.add_node("end", end_node)

    # 设置起始节点
    graph.set_entry_point("start")

    # 添加条件路由边
    graph.add_conditional_edges(
        "start",
        route_intent,
        {
            "order": "order",
            "info": "info",
            "error": "error"
        }
    )

    # 添加从处理节点到结束节点的边
    graph.add_edge("order", "end")
    graph.add_edge("info", "end")
    graph.add_edge("error", "end")

    # 设置结束节点
    graph.set_finish_point("end")

    # 编译图
    compiled_graph = graph.compile()

    # 执行工作流
    print(f"\n开始执行工作流，输入: {input_data}")
    final_state = compiled_graph.invoke(initial_state)

    # 构建执行路径
    execution_path = ["start"]
    if final_state["intent"] == "order":
        execution_path.extend(["order_handler", "end"])
    elif final_state["intent"] == "info":
        execution_path.extend(["info_handler", "end"])
    else:
        execution_path.extend(["error_handler", "end"])

    # 返回结果
    return {
        "route_taken": final_state["route_taken"],
        "final_output": final_state["final_output"],
        "execution_path": execution_path,
        "intermediate_results": final_state["intermediate_results"]
    }

# 测试代码
if __name__ == "__main__":
    # 测试订单处理路由
    print("=== 测试订单处理路由 ===")
    result1 = run_workflow({"intent": "order", "query": "我要下单购买商品"})
    print(f"路由路径: {result1['route_taken']}")
    print(f"执行顺序: {result1['execution_path']}")
    print(f"最终输出: {result1['final_output']}")

    # 测试信息查询路由
    print("\n=== 测试信息查询路由 ===")
    result2 = run_workflow({"intent": "info", "query": "查询订单状态"})
    print(f"路由路径: {result2['route_taken']}")
    print(f"执行顺序: {result2['execution_path']}")
    print(f"最终输出: {result2['final_output']}")

    # 测试无效意图
    print("\n=== 测试无效意图 ===")
    result3 = run_workflow({"intent": "unknown", "query": "随机查询"})
    print(f"路由路径: {result3['route_taken']}")
    print(f"执行顺序: {result3['execution_path']}")
    print(f"最终输出: {result3['final_output']}")
