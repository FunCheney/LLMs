'''
比较第一版：
目标是跟着官方 agents-from-scratch 学，我们现在应该实现官方 Evaluation notebook 的思路，而不是继续维护我们自己写的 find_tool_calls()
改动点如下：
1. 先实现 Single-step Tool Call Evaluation
2. Dataset 增加 expected_tool
3. 要让 target_email_assistant() 返回 messages
4. 现最简单的 Tool Call Evaluator
5. 删除 debug_evaluator
'''
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client
from dotenv import load_dotenv
import os

load_dotenv()

from email_assistant import email_assistant
from email_dataset import EVAL_DATASET

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# DATASET_NAME = "email-assistant-triage"
DATASET_NAME = "email-assistant-tool-call-v1"
# --------------------------------------------------
# Create Dataset
# --------------------------------------------------
# 创建 langsmith dataset
if not client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Email intent classification dataset",
    )

    client.create_examples(
        dataset_id=dataset.id,
        examples=EVAL_DATASET,
    )
    print(f"Created dataset: {DATASET_NAME}")

else:
    print(f"Dataset already exists: {DATASET_NAME}")


# 创建一个评估 classify_intent 的 Target Function
# Target Function 应该返回 evaluator 真正需要评估的结果，而不是把整个内部 State 都暴露出来。
def target_email_assistant(inputs: dict) -> dict:
    state = {
        "email_content": inputs["email_content"],
        "classification": None,
        "messages": [
            HumanMessage(
                content=inputs["email_content"],
            )
        ],
        "response": [],
    }

    intent = email_assistant.invoke(state)

    return {
        "intent": intent["classification"]["intent"],
        "messages": intent["messages"], # 返回 messages
    }

# --------------------------------------------------
# classification_evaluator
# --------------------------------------------------
def classification_evaluator(
    run,
    example,
) -> bool:
    actual_intent = (run.outputs or {}).get("intent")
    expected_intent = (example.outputs or {}).get("intent")

    return actual_intent == expected_intent


"""
在 email_assistant 还有 
support_agent -> tool call -> search_docs -> ToolMessage -> support_agent -> final response
所以真正完整的 Agent Evaluation 至少应该拆成几个维度：

                    Agent Evaluation
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     Classification    Tool Calling     Response Quality
          │                │                │
       intent          是否调用工具       回答是否正确
       urgency         调了什么工具       是否完整
       summary         参数是否正确       是否符合要求
"""

# --------------------------------------------------
# tool_call_evaluator
# --------------------------------------------------
def tool_call_evaluator(outputs: dict, reference_outputs: dict) -> bool:
    expected_tools = reference_outputs["expected_tool"]

    for message in outputs["messages"]:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call["name"] == expected_tools:

                    return True

    return False


# --------------------------------------------------
# run Evaluator
# --------------------------------------------------
result = client.evaluate(
    target_email_assistant,
    data=DATASET_NAME,
    evaluators=[
        classification_evaluator,
        tool_call_evaluator,
    ],
    experiment_prefix="email-assistant-triage",
    max_concurrency=1,
)

print(result)
