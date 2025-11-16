"""
实验6：工具调用的结果验证
学生需要构建具备工具调用能力的AI Agent，实现模型与外部函数的交互
"""
from typing import List, Dict, Any, Callable
import httpx
import json


# 提示：需要导入 LangChain 相关模块
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.llms import Ollama
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


def get_weather(city: str) -> dict:
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


# ========== 学生需要实现的函数 ==========

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
    return {}


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
    return []


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
    return {}


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
