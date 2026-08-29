"""
supervisor 模式和 handoff 模式容易混淆，但是本质不同
1. supervisor 像是主管一直在调度别人。（中心调度）
2. handoff 像是当前角色把控制权交给下一个角色。（控制权转移）

使用场景介绍：
1. 会话中角色真正切换
2. 下一位 Agent 需要接着当前上下文继续和用户互动
3. 不只是内部调用工具，而是“换一个会说话的角色接手”

举例：
1. 先由客服接待，再把问题交给技术
2. 先由一个 Agent 完成分析，再把控制权交给执行 Agent

易误解点：并不是所有的“角色切换”都必须拆成多个 Agent。很多 handOff 场景使用单 agent + middleware/状态机就能实现；
只有当角色边界，上下文边界，团队维护边界都已经清晰时，拆成多个 Agent 收益才会明显。

知识点速览：
- Handoff 和 Supervisor 的最大区别，不是“也有多个 Agent”，而是“控制权会被正式交给下一位 Agent”，而不是始终由一个中央主管调度。
- Handoff 与“把子 Agent 当工具调”不同：这里显式构造下一跳输入 state，并用 Command(goto=[Send(...)], graph=Command.PARENT) 跳转到兄弟节点。
- InjectedState 把当前 MessagesState 注入工具，便于携带对话历史；task_description 充当“交给下一位的工单说明”，这正是 Handoff 里最值得关注的上下文工程。
- flight_assistant / hotel_assistant 由 create_agent 构建并作为节点加入同一 StateGraph，START 指向默认入口 Agent；这说明 Agent 完全可以作为 LangGraph 图中的节点来组织。
- @tool 装饰的业务工具仍需 docstring；本案例重点不是预订业务本身，而是观察“状态 + 任务说明 + 下一跳目标”如何一起交接。
"""
from typing import Annotated

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.types import Command, Send
import dotenv
import os


dotenv.load_dotenv()


# 1.初始化大模型
def init_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.1,
        max_tokens=1024
    )


model = init_model()


# 2.通用的 Handoff 工具工厂
def create_task_description_handoff_tool(
        *, agent_name: str, description: str | None = None
):

    name = f"transfer_to_{agent_name}"
    description = description or f"移交给 {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
            task_description: Annotated[
                str, "描述下一个 Agent 应该做什么，包括所有必要信息"
            ],
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],  # ← 新增参数
    ) -> Command:

        # task_description_message = {
        #     "role": "user",
        #     "content": task_description
        # }
        #
        # agent_input = {
        #     **state,
        #     "messages": [task_description_message]
        # }
        #
        # return Command(
        #     goto=[Send(agent_name, agent_input)],
        #     graph=Command.PARENT
        # )
        # 1. 构造 tool 消息（响应本次工具调用）
        tool_message = ToolMessage(
            content=f"已移交至 {agent_name}，任务描述：{task_description}",
            tool_call_id=tool_call_id,  # 必须与模型请求的 id 一致
        )

        # 2. 构造用户消息（给下一任 Agent 看的“工单”）
        task_description_message = {
            "role": "user",
            "content": task_description
        }

        # 3. 更新状态：将原始消息 + tool_message + 用户消息 拼接在一起
        agent_input = {
            **state,
            "messages": state["messages"] + [tool_message, task_description_message]
        }

        # 4. 返回 Command 跳转
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT
        )

    return handoff_tool


# 3. 业务工具 必须有（docstring）
@tool("book_flight")
def book_flight(from_airport: str, to_airport: str) -> str:
    """预订航班，根据出发地和目的地完成机票预订"""
    print(f"✅ 成功预订了从 {from_airport} 到 {to_airport} 的航班")
    return f"成功预订了从 {from_airport} 到 {to_airport} 的航班。"


@tool("book_hotel")
def book_hotel(hotel_name: str) -> str:
    """预订酒店，根据酒店名称完成预订"""
    print(f"✅ 成功预订了 {hotel_name} 的住宿")
    return f"成功预订了 {hotel_name} 的住宿。"


# 4.handoff 工具
transfer_to_flight_assistant = create_task_description_handoff_tool(
    agent_name="flight_assistant",
    description="任务移交给航班助手"
)

transfer_to_hotel_assistant = create_task_description_handoff_tool(
    agent_name="hotel_assistant",
    description="任务移交给酒店预定助手"
)

# ===============================
# 5. 定义 Agent（create_agent 新接口）
# 这里不额外写长 prompt，而是更多依赖：
# 1. 工具 schema / 名称 / docstring
# 2. Handoff 工具本身描述的交接语义
# 3. MessagesState 中持续携带的历史消息
# ===============================
flight_assistant = create_agent(
    model=model,
    tools=[book_flight, transfer_to_hotel_assistant],
    name="flight_assistant",
)

hotel_assistant = create_agent(
    model=model,
    tools=[book_hotel, transfer_to_flight_assistant],
    name="hotel_assistant",
)

multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(flight_assistant)
    .add_node(hotel_assistant)
    .add_edge(START, "flight_assistant")
    .compile()
)


# ===============================
# 7. 运行
# ===============================
if __name__ == "__main__":
    result = multi_agent_graph.invoke(
        {
            "messages": [
                HumanMessage(content="帮我预订从北京到上海的航班，并预订如家酒店")
            ]
        }
    )

    print("\n====== 最终对话结果 ======")
    for msg in result["messages"]:
        if msg.type in ("human", "ai"):
            print(msg.content)

