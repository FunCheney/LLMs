"""
单智能体示例
"""


import os
import dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()



def get_weather(city: str) -> str:
    """获取指定城市的天气信息。

    Args:
        city: 城市名称
    Returns:
        返回该城市的天气描述（本案例为写死返回值，仅作演示）
    """
    return f"今天{city}是晴天，仅做测试，固定写死"



def main():

    llm = init_chat_model(
        model= os.getenv("DEEPSEEK_MODEL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )

    agent = create_agent(
        model=llm,
        tools=[get_weather]
    )

    human_message = HumanMessage(content="今天北京的天气怎么样")

    response = agent.invoke(human_message)

    print()
    print("模型回答：", response["messages"][-1].content)
    print()
    response["messages"][-1].pretty_print()



if __name__ == "__main__":
    main()