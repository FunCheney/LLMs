"""
实验6：工具调用的结果验证（修复后的版本）
兼容 LangChain 1.0.3
"""
from typing import List, Callable
from langchain_community.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor
from langchain_cohere import ChatCohere, create_cohere_react_agent


# ================= 模拟工具函数 =================

def get_stock_price(symbol: str) -> float:
    if symbol.upper() == "AAPL":
        return 175.0
    elif symbol.upper() == "GOOGL":
        return 140.5
    else:
        return 100.0


def add_numbers(a: int, b: int) -> int:
    return a + b


def get_weather(city: str) -> dict:
    weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"},
        "深圳": {"temp": 30, "condition": "雨"},
    }
    return weather_data.get(city, {"temp": 20, "condition": "未知"})


# ================= 包装工具 =================

def wrap_tools(tool_functions: List[Callable]) -> List[Tool]:
    tools = []
    for func in tool_functions:
        tools.append(
            Tool(
                name=func.__name__,
                func=func,
                description=f"工具函数: {func.__name__}"
            )
        )
    return tools


# ================= 解析 Agent 输出 =================

def parse_agent_output(agent_result: dict) -> dict:
    result = {
        "tool_used": "",
        "tool_input": {},
        "tool_output": None,
        "final_answer": agent_result.get("output", "")
    }

    steps = agent_result.get("intermediate_steps", [])
    if steps:
        action, obs = steps[-1]
        result["tool_used"] = action.tool
        result["tool_input"] = action.tool_input
        result["tool_output"] = obs

    return result


# ================= Agent 执行器 =================

def agent_executor(query: str, available_tools: List[Callable]) -> dict:
    tools = wrap_tools(available_tools)

    llm = Ollama(model="qwen3:8b")

    # ReAct 提示词
    template = """你是一个有帮助的 AI，可以使用工具回答问题。

请按照以下格式进行思考和输出：

Question: 用户问题
Thought: 是否需要调用工具？
Action: 工具名称
Action Input: 工具输入
Observation: 工具返回
...（可以多次循环）
Thought: 我已经知道最终答案
Final Answer: 最终回答

现在开始！

Question: {input}
{agent_scratchpad}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"]
    )

    # 创建 ReAct Agent
    agent = create_cohere_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 包装成 executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

    result = executor.invoke({"input": query})
    return parse_agent_output(result)


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