"""
supervisor 多 Agent 示例

用 agent 的动态推理能力做动态路由
知识点速览：
- 这是本章最重要的 Supervisor 案例：用 create_agent 定义子 Agent，再由 create_supervisor 统一调度，形成更贴近当前主流写法的多智能体结构。
- 这里的“主管调子 Agent”本质上对应官方多智能体文档里的 Subagents 模式；主管负责统一入口与路由，子 Agent 负责狭窄领域任务。
- pip install langgraph-supervisor；子 Agent 的工具函数必须具备清晰 docstring，便于模型绑定工具模式。
- create_supervisor(...).compile() 得到可 stream/invoke 的图；主管 prompt 不只是提示词，更是在约束整个调度流程与角色边界。
- filter_messages 只是教学辅助工具，用于弱化移交过程中的英文提示、去重和压缩噪声；重点应放在观察主管—子 Agent 的数据流与控制流。

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


# 5. 消息过滤器：帮助更清楚地观察主管和子 Agent 的有效中文输出
def filter_messages(chunk: dict) -> str:
    """提取并过滤消息，只返回中文内容，去除重复和英文"""
    output = ""

    if isinstance(chunk, dict):
        for role, payload in chunk.items():
            if isinstance(payload, dict) and "messages" in payload:
                for msg in payload["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        content = msg.content.strip()

                        # 过滤英文系统消息
                        if (
                            content
                            and not content.startswith("Successfully")
                            and not content.startswith("Transferring")
                            and "Successfully transferred" not in content
                            and "transferred back to" not in content
                            and not content.startswith("帮我预订从")
                        ):

                            # 只保留中文内容
                            chinese_content = re.sub(
                                r'[^\u4e00-\u9fff，。！？：；""、\s\d✅]', "", content
                            )
                            if chinese_content and len(chinese_content.strip()) > 5:
                                output += f"{role}: {chinese_content.strip()}\n"

    return output


# 6. 主程序
def main():
    print("=" * 60)
    print()

    # 收集用户信息
    print("请按顺序提供以下信息：")
    print("-" * 40)

    # 1. 询问出发机场
    from_airport = input("1. 您的出发机场是哪里？: ").strip()
    while not from_airport:
        print("请输入有效的出发机场名称")
        from_airport = input("1. 您的出发机场是哪里？: ").strip()

    # 2. 询问到达机场
    to_airport = input("\n2. 您的到达机场是哪里？: ").strip()
    while not to_airport:
        print("请输入有效的到达机场名称")
        to_airport = input("2. 您的到达机场是哪里？: ").strip()

    # 3. 询问酒店名称
    hotel_name = input("\n3. 您要预订的酒店名称是什么？: ").strip()
    while not hotel_name:
        print("请输入有效的酒店名称")
        hotel_name = input("3. 您要预订的酒店名称是什么？: ").strip()

    # 构造更明确的用户请求
    user_request = (
        f"请帮我预订以下旅行安排：\n"
        f"1. 航班：从 {from_airport} 飞往 {to_airport}\n"
        f"2. 酒店：{hotel_name}\n"
        f"请完成这两个预订。"
    )

    print("\n" + "=" * 60)
    print("正在处理您的预订请求...")
    print("=" * 60)
    print()

    # 准备输入数据：Supervisor 图和普通 Agent 一样，入口仍然是 messages
    input_data = {"messages": [{"role": "user", "content": user_request}]}

    # 使用流式处理，便于观察主管如何依次调度两个子 Agent
    try:
        # 记录已打印内容，避免在演示时重复刷屏
        seen_contents = set()

        for chunk in supervisor.stream(input_data):
            filtered_output = filter_messages(chunk)
            if filtered_output:
                lines = filtered_output.strip().split("\n")
                for line in lines:
                    if line and line not in seen_contents:
                        print(line)
                        seen_contents.add(line)

        # 如果流式输出过少，就给一个兜底总结，避免读者误以为程序没有完成
        if len(seen_contents) < 2:
            print("\n" + "=" * 60)
            print("预订已完成！")
            print(f"航班：从 {from_airport} 到 {to_airport}")
            print(f"酒店：{hotel_name}")
            print("=" * 60)
    except Exception as e:
        print(f"\n处理过程中出现错误: {e}")
        # 教学兜底：即使多智能体流程异常，也能直接调用工具帮助理解业务目标
        print("\n正在直接执行预订...")
        flight_result = book_flight(from_airport, to_airport)
        hotel_result = book_hotel(hotel_name)
        print(flight_result)
        print(hotel_result)

    print("\n感谢使用智能旅行预订系统！")


# 7. 运行主程序
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        print(f"\n系统出现错误: {e}")





