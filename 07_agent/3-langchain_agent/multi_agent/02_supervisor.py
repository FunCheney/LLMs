"""
supervisor 多 Agent 示例
"""

import os
import re
import dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

dotenv.load_dotenv()


# 1.初始化大模型
def init_llm_model() -> ChatOpenAI:

    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.1,
        max_tokens=1024
    )


# 2. Tools (必须要有 docString)
def book_flight(from_airport: str, to_airport: str) -> str:
    """预订航班工具。根据出发机场和到达机场预订一张机票，并返回预订结果。"""
    return f"✅ 成功预订了从 {from_airport} 到 {to_airport} 的航班"


def book_hotel(hotel_name: str) -> str:
    """预订酒店工具。根据酒店名称完成酒店预订，并返回预订结果。"""
    return f"✅ 成功预订了 {hotel_name} 的住宿"



# 3. 子agent
fight_agent = create_agent(
    model=init_llm_model(),
    tools=[book_flight],
    name="fight_assistant",
)

hotel_agent = create_agent(
    model=init_llm_model(),
    tools=[book_hotel],
    name="hotel_assistant",
)

# 4. 创建 supervisor，协调者主管
supervisor_agent = create_supervisor(
    agents=[fight_agent, hotel_agent],
    model=init_llm_model(),
    prompt=(
        "你是旅行预订系统的调度主管，负责协调航班预订和酒店预订。\n"
        "当用户提出航班和酒店预订请求时，你的工作流程是：\n"
        "1. 首先调用flight_assistant来预订航班\n"
        "2. 然后调用hotel_assistant来预订酒店\n"
        "3. 收到两个助手的结果后，汇总并向用户报告\n"
        "4. 完成后结束对话\n"
        "重要规则：\n"
        "- 每个助手只能调用一次\n"
        "- 不要重复任何内容\n"
        "- 不要输出任何英文\n"
        "- 所有通信都使用中文\n"
    ),
).compile()




