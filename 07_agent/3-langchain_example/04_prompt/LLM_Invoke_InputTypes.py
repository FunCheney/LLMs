"""
    案例：invoke，ainvoke 的多种输入类型 (字符串，message 列表，元组列表)
        invoke 不止接受字符串，也常接受消息对象列表、`(role, content)` 元组列表、`{"role": "...", "content": "..."}` 字典列表。
        这些写法的目标都是表达“这次输入由哪些角色、哪些内容组成”；LangChain 会在内部转成统一的消息表示。
"""
import asyncio
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import ChatPromptValue

load_dotenv()

model = init_chat_model(
    model=os.getenv("DEEPSEEK_MODEL"),
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

def demo_message_objects():
    message = [
        SystemMessage(content="你是一个 AI 助手，回答下面问题"),
        HumanMessage(content="你好，你是谁？")
    ]

    response = model.invoke(message)
    print(response)
    print(type(response), response.content[:80] if response.content else "")

def demo_tuple_list():
    """元组列表：与 ChatPromptTemplate.from_messages 的写法一致。"""
    messages = [
        ("system", "你是一个 AI 助手，回答下面问题"),
        ("human", "你好，你是谁？"),
    ]
    resp = model.invoke(messages)
    print(type(resp), resp.content[:80] if resp.content else "")


def demo_dict_list():
    """字典列表：与 OpenAI Chat Completions 等 API 的请求体形状接近。"""
    messages = [
        {"role": "system", "content": "你是一个 AI 助手，回答下面问题"},
        {"role": "user", "content": "你好，你是谁？"},
    ]
    resp = model.invoke(messages)
    print(type(resp), resp.content[:80] if resp.content else "")

async def demo_ainvoke_tuple():
    resp = await model.invoke([
        ("system", "你是一个 AI 助手，回答下面问题"),
        ("user", "你是谁?")
    ])

    print(type(resp), resp.content[:80] if resp.content else "")


if __name__ == "__main__":
    print("--- Message 对象列表 ---")
    demo_message_objects()
    print("--- 元组列表 ---")
    demo_tuple_list()
    print("--- 字典列表 ---")
    demo_dict_list()
    print("--- ainvoke + 元组 ---")
    asyncio.run(demo_ainvoke_tuple())

    print("---- 字符串 -----")
    resp = model.invoke("用一句话解释什么是 LangChain")
    print(resp)
    print("---- format ---")
    template = PromptTemplate.from_template("用不超过 50 字介绍：{topic} 是什么？")
    resp = model.invoke(template.format(topic="langchain"))
    print(resp)

    print("---- 多角色消息列表 ----")
    # 在实际项目里，这种写法特别常见：
    #
    # 系统提示词放在 SystemMessage
    # 用户问题放在 HumanMessage
    # 历史回复可放回 AIMessage
    # 工具执行结果后续可用 ToolMessage
    messages = [
        SystemMessage(content="你是只回答技术问题的助手，回答要简短。"),
        HumanMessage(content="什么是 LangChain？"),
        # 多轮示例：
        # AIMessage(content="LangChain 是用于编排 LLM 应用的框架……"),
        # HumanMessage(content="它和直接调 API 有什么区别？"),
    ]
    resp = model.invoke(messages)
    print(resp.content)

    print("---- 多角色消息列表 另一种等价的写法 ----")
    prompt_value = ChatPromptValue(
        messages=[
            SystemMessage(content="You are a helpful AI bot. Your name is Bob."),
            HumanMessage(content="Hello, how are you doing?"),
            AIMessage(content="I'm doing well, thanks!"),
            HumanMessage(content="What is your name?"),
        ]

    )
    resp = model.invoke(prompt_value.to_messages())
    print(resp)

