"""
【案例】messages 流模式：从图中调用 LLM 的节点逐 token（或片段）推送输出，便于打字机效果。

知识点速览：
- stream_mode="messages" 时，每次迭代一般为 (message_chunk, metadata)：chunk 为模型输出片段，metadata 标明节点等上下文。
- 这个案例最适合用来建立“LangGraph Streaming 不只流状态，也能流模型输出”这层认知。
- 流式消费侧通常关心 `chunk.content` 和 `metadata`；前者是输出片段，后者帮助你知道这些片段来自哪个节点。
- 需配置环境变量（如 aliQwen-api）与网络；模型、base_url 按你本地教程为准。
"""
from typing import TypedDict
import dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import StateGraph

dotenv.load_dotenv()


class State(TypedDict):
    query: str
    answer: str


def llm_node(state: State):
    print("llm node 节点执行")
    llm = init_chat_model(
        model="kimi-k3",
        model_provider="openai",
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url=os.getenv("KIMI_BASE_URL"),
    )

    llm_result = llm.invoke([("user", state["query"])])
    print("llm invoke 结束", end="\n\n")

    return {"answer": llm_result}

def main():
    graph = (StateGraph(state_schema=State)
             .add_node(llm_node)
             .add_edge(START, "llm_node")
        .compile())


    inputs = {"query": "帮我生成一个200字的小学生作文，主题为我的一天"}

    # messages：从图内触发的大模型调用处流式输出；(chunk, metadata) 见官方文档
    for chunk, _metadata in graph.stream(inputs, stream_mode="messages"):
        # print(f"type of chunk:{type(chunk)}")  # 调试时可打开
        print(chunk.content, end="")
        # print(chunk, end="")


if __name__ == "__main__":
    main()