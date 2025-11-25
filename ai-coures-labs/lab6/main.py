"""
实验6：工具调用的结果验证（最终版，可通过测试）
兼容 LangChain 1.0.3
"""
import json
from typing import List, Dict, Any, Callable
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ========== 模拟工具函数（测评系统提供，学生直接使用） ==========
def get_stock_price(symbol: str) -> float:
    """
    查询股票价格（模拟工具）

    参数:
        symbol: 股票代码（如 "AAPL"）

    返回:
        股票价格（固定返回值以确保测试稳定性）
    """
    print(f"[DEBUG] symbol = {repr(symbol)}")
    print(f"[调试] 符号 = {repr(symbol)}")
    print(f"[DEBUG] type = {type(symbol)}")
    if symbol.upper() == "AAPL":
        return 175.0
    elif symbol.upper() == "GOOGL":
        return 140.5
    else:
        return 100.0

def add_numbers(a: int, b: int) -> int:
    """
    计算两数之和（模拟工具）

    参数:
        a: 第一个数
        b: 第二个数

    返回:
        两数之和
    """
    return a + b


def  get_weather(city: str) -> dict:
    """
    查询天气信息（模拟工具）

    参数:
        city: 城市名称

    返回:
        天气信息字典，包含 temp 和 condition
    """
    weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"},
        "深圳": {"temp": 30, "condition": "雨"},
    }
    return weather_data.get(city, {"temp": 20, "condition": "未知"})


def agent_executor(query: str, available_tools: List[Callable]) -> dict:
    """
    执行 Agent 工具调用

    参数:
        query: 用户查询文本
        available_tools: 可用工具函数列表

    返回:
        字典，包含以下键:
        - tool_used (str): 实际调用的工具名称
        - tool_input (dict): 传递给工具的参数
        - tool_output (any): 工具的返回值
        - final_answer (str): Agent的最终回答
    """
    # TODO: 实现 Agent 工具调用逻辑
    # 提示:
    # 1. 导入 from langchain.agents import AgentExecutor, create_react_agent
    # 2. 将 available_tools 包装为 LangChain Tool 对象
    # 3. 创建 Agent 并执行
    # 4. 解析结果，返回 tool_used, tool_input, tool_output, final_answer
    # 1. 包装工具
    tools = wrap_tools(available_tools)
    # 2. 定义模型
    llm = OllamaLLM(model="qwen3:8b")  # 需要本地运行 Ollama，模型名称可调整为可用的
    # 3. 创建 Agent
    template ="""Answer the following questions as best you can. You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Action Input: the input to the action, **MUST be a valid JSON object like {{"param1": value1, "param2": value2}}**,When you use the tool, you need to treat this as a dict and correctly parse the parameters to pass in
    Final Answer: the final answer to the original input question
    Begin!
    Question: {input}
    Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    # 4. 创建执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True,return_intermediate_steps=True)
    # 5. 执行查询
    result = agent_executor.invoke({"input": query})
    print("result=", result)
    # 6. 解析结果
    return parse_agent_output(result)


# 辅助函数示例
def wrap_tools(tool_functions: List[Callable]) -> List:
    """
    将 Python 函数包装为 LangChain Tool

    参数:
        tool_functions: Python 函数列表

    返回:
        LangChain Tool 对象列表
    """
    # TODO: 实现工具包装逻辑
    # 提示: 使用 Tool.from_function() 或 @tool 装饰器
    def make_wrapped_tool(func):
        def wrapper(json_input: str):
            try:
                params = json.loads(json_input)
                return func(**params)
            except Exception as e:
                return f"Error in tool '{func.__name__}': {e}"
        return wrapper
    tools = []
    for func in tool_functions:
        wrapped_func = make_wrapped_tool(func)
        tool = Tool(
            name=func.__name__,
            func=wrapped_func,
            description=func.__doc__ or f"Call the {func.__name__} function."
        )
        tools.append(tool)
    return tools

def parse_agent_output(agent_result: dict) -> dict:
    """
    解析 Agent 执行结果

    参数:
        agent_result: AgentExecutor 的返回值

    返回:
        标准化的结果字典
    """
    # TODO: 实现结果解析逻辑
    # 提示: 从 agent_result 中提取 tool_used, tool_input, tool_output
    tool_used = None
    tool_input = None
    tool_output = None
    if agent_result.get("intermediate_steps"):
        last_step = agent_result["intermediate_steps"][-1]
        action = last_step[0]
        observation = last_step[1]
        tool_used = action.tool
        tool_input = json.loads(action.tool_input)
        tool_output = observation
    # 构造标准返回格式
    return {
        "tool_used": tool_used,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "final_answer": agent_result.get("output", "")
    }

# 测试代码
if __name__ == "__main__":
    # 准备工具列表
    tools = [get_stock_price, add_numbers, get_weather]

    # 测试1：股票价格查询
    print("=== 测试1：股票价格查询 ===")
    try:
        result1 = agent_executor("苹果公司的股价是多少？", tools)
        print(f"使用工具: {result1['tool_used']}")
        print(f"工具输入: {result1['tool_input']}")
        print(f"工具输出: {result1['tool_output']}")
        print(f"最终回答: {result1['final_answer'][:50]}...")
    except Exception as e:
        print(f"错误: {e}")

    # 测试2：数学计算
    print("\n=== 测试2：数学计算 ===")
    try:
        result2 = agent_executor("计算15加27等于多少", tools)
        print(f"使用工具: {result2['tool_used']}")
        print(f"工具输入: {result2['tool_input']}")
        print(f"工具输出: {result2['tool_output']}")
        print(f"最终回答: {result2['final_answer'][:50]}...")
    except Exception as e:
        print(f"错误: {e}")

    # 测试3：天气查询
    print("\n=== 测试3：天气查询 ===")
    try:
        result3 = agent_executor("北京的天气怎么样？", tools)
        print(f"使用工具: {result3['tool_used']}")
        print(f"工具输入: {result3['tool_input']}")
        print(f"工具输出: {result3['tool_output']}")
        print(f"最终回答: {result3['final_answer'][:50]}...")
    except Exception as e:
        print(f"错误: {e}")
