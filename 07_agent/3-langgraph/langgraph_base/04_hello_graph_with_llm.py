"""
接入大模型的最小对话图：用户消息 → model 节点 → 模型回复写回 messages，演示 LangGraph 如何和 LangChain 模型调用衔接，
以及为什么对话状态通常要配 add_messages

知识点速览：
- add_messages：LangGraph 内置 Reducer，专门适合消息列表字段；它的语义是“把新消息追加进历史消息”，而不是把 messages 整个覆盖掉。
- messages 字段写成 Annotated[List, add_messages] 后，节点只需要 return {"messages": [reply]} 这种增量更新，历史对话不会丢失。
- model_node(state) 直接把 state["messages"] 交给 llm.invoke(...)，说明“节点可以只是普通函数，函数内部再调用 LangChain 模型”。
- 图结构是最小单节点对话流 START → model → END；invoke 时传入初始 messages，执行后从 result["messages"][-1].content 读取最新模型回复。
- 这个案例只是“单节点 LLM 图”的最小雏形；后面学习 add_messages 背后的 Reducer 机制 ，会继续扩展多节点和条件边。
"""
import json
from typing import TypedDict, Annotated, List

import dotenv
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, message_to_dict, BaseMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

dotenv.load_dotenv()


# 1.定义状态 state： messages 使用 add_messages 归约器，节点返回的每条新消息会自动追加到列表
class DiliState(TypedDict):
    # add_messages: 是 langgraph 提供的归约器（Reducer），来自langgraph.graph.message
    # 含义：该字段不是更新。而是追加节点返回的（新增）消息
    # 框架会自动合并到当前消息列表末尾，适合多轮对话、多节点共同往同一列表写消息。
    # 若不用 add_messages，节点返回 {"messages": [reply]} 会直接覆盖掉之前的对话历史。
    messages: Annotated[List, add_messages]

# 2. 初始户大模型
llm = init_chat_model(
        model= "kimi-k3",
        model_provider="openai",
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url=os.getenv("KIMI_BASE_URL"),
)

# 3. 定义节点 Nodes：将当前列表交给模型，返回新消息字典（add_messages 会追加到 state）
def model_node(state: DiliState):
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}

# 4. 构建图，单节点 model, START --> model --> end
graph = StateGraph(DiliState)
graph.add_node("model", model_node)
graph.add_edge(START, "model")
graph.add_edge("model", END)

# 5. 编译并执行

app = graph.compile()

result = app.invoke(
    {"messages": [HumanMessage(content="请用一句话解释什么是 LangGraph。")]},
)
# 或: result = app.invoke({"messages": "请用一句话解释什么是 LangGraph。"})
print(f'模型回答：: {result["messages"][-1].content}')

# 直接格式化输出 result：default 把消息对象转成 dict，其它不可序列化用 str 兜底
print("\n--- result 格式化输出 ---")
print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=lambda o: message_to_dict(o) if isinstance(o, BaseMessage) else str(o),
    )
)

# 可视化
print(app.get_graph().print_ascii())
print("=" * 50)
print(app.get_graph().draw_mermaid())
print("=" * 50)

'''
模型回答：: LangGraph 是由 LangChain 团队开发的框架,它把大模型应用建模为"图"结构(节点代表计算步骤、边代表控制流),从而支持构建带循环、有状态、可持久化的复杂智能体(Agent)工作流。

--- result 格式化输出 ---
{
  "messages": [
    {
      "type": "human",
      "data": {
        "content": "请用一句话解释什么是 LangGraph。",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "human",
        "name": null,
        "id": "45fe1a22-b572-4da3-bdfc-1ae7b7bb4f92"
      }
    },
    {
      "type": "ai",
      "data": {
        "content": "LangGraph 是由 LangChain 团队开发的框架,它把大模型应用建模为\"图\"结构(节点代表计算步骤、边代表控制流),从而支持构建带循环、有状态、可持久化的复杂智能体(Agent)工作流。",
        "additional_kwargs": {
          "refusal": null
        },
        "response_metadata": {
          "token_usage": {
            "completion_tokens": 454,
            "prompt_tokens": 93,
            "total_tokens": 547,
            "completion_tokens_details": {
              "accepted_prediction_tokens": null,
              "audio_tokens": null,
              "reasoning_tokens": 385,
              "rejected_prediction_tokens": null
            },
            "prompt_tokens_details": null
          },
          "model_provider": "openai",
          "model_name": "kimi-k3",
          "system_fingerprint": null,
          "id": "chatcmpl-6a8eb582cf43d877ba1ec0b5",
          "finish_reason": "stop",
          "logprobs": null
        },
        "type": "ai",
        "name": null,
        "id": "lc_run--01a03d75-0106-7ff1-96f7-b1b272196981-0",
        "tool_calls": [],
        "invalid_tool_calls": [],
        "usage_metadata": {
          "input_tokens": 93,
          "output_tokens": 454,
          "total_tokens": 547,
          "input_token_details": {},
          "output_token_details": {
            "reasoning": 385
          }
        }
      }
    }
  ]
}
+-----------+  
| __start__ |  
+-----------+  
      *        
      *        
      *        
  +-------+    
  | model |    
  +-------+    
      *        
      *        
      *        
 +---------+   
 | __end__ |   
 +---------+   
None
==================================================
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	model(model)
	__end__([<p>__end__</p>]):::last
	__start__ --> model;
	model --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

==================================================
'''