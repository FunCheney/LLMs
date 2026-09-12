"""
这个版本的目标不是评估“回答得好不好”，而是评估 Agent 是否按照预期的方式完成任务。
我们现在有一个 Email Assistant：
    用户邮件
       ↓
    classify_intent
       ↓
    question ?
       ↓
    support_agent
       ↓
    需要查询文档？
       ↓
    search_docs
       ↓
    support_agent
       ↓
    最终回答
对于这封：
    I forgot my password and cannot log into my account.

    How can I reset it?

我们期望 Agent：
    Human
      ↓
    AI → search_docs
      ↓
    Tool → search_docs
      ↓
    AI → final answer

所以 V4 要验证的是：

Agent 有没有走这条正确的“行为路径”。

修改点：
1. 把 Target 的 trajectory 简化成 Assistant Messages
2. Reference 也只描述这两个 Assistant Message
3. 这里有一个容易误解的地方
    Reference 的第二个 AIMessage 为什么 content=""？
    因为 AgentEvals 的 strict trajectory evaluation 允许 message content 不同。
    官方文档明确说，strict 要求：
        相同 message
            +
        相同顺序
            +
        相同 tool calls
    但允许 message content 不同
4. 换一个 Dataset 名称
"""

import json

from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client

from agentevals.trajectory.match import (
    create_trajectory_match_evaluator,
)

from email_assistant import email_assistant


# ============================================================
# 1. Dataset
# ============================================================

DATASET_NAME = "email-assistant-trajectory-v4"


# ============================================================
# 2. Evaluation Dataset
# ============================================================

EVAL_DATASET = [
    {
        "inputs": {
            "email_content": (
                "I forgot my password and cannot log into my account.\n\n"
                "How can I reset it?\n\n"
                "Thanks!"
            )
        },
        "outputs": {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": "{}",
                            }
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                },
            ]
        },
    }
]


# ============================================================
# 3. Target
# ============================================================

def target_email_assistant(inputs: dict) -> dict:
    """
    Execute the Email Assistant.

    Return the agent trajectory as messages.
    """

    result = email_assistant.invoke({
        "email_content": inputs["email_content"],
        "classification": None,
        "messages": [
            HumanMessage(
                content=inputs["email_content"]
            )
        ],
        "response": "",
    })

    assistant_messages = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage)
    ]

    print("\n===== ASSISTANT TRAJECTORY =====")
    for i, message in enumerate(assistant_messages):
        print(f"\n--- {i} ---")
        print(message)

    return {
        "messages": assistant_messages
    }


# ============================================================
# 4. Trajectory Evaluator
# trajectory 的整体结构必须一致，但是 search_docs 到底传什么 query，我不关心。
# ============================================================
trajectory_evaluator = create_trajectory_match_evaluator(
    # 两条 trajectory 的消息结构和顺序都应该一致，同时 tool call 也要匹配
    trajectory_match_mode="strict",

    # Default is "exact".
    #
    # For search_docs we only care whether the tool
    # was selected, not the exact wording of the query.
    tool_args_match_overrides={
        "search_docs": "ignore",
    },
)


# ============================================================
# 5. Prepare Dataset
# ============================================================

def prepare_dataset(client: Client):

    if not client.has_dataset(
        dataset_name=DATASET_NAME
    ):
        print(
            f"Creating dataset: {DATASET_NAME}"
        )

        dataset = client.create_dataset(
            dataset_name=DATASET_NAME
        )

        client.create_examples(
            dataset_id=dataset.id,
            examples=EVAL_DATASET,
        )

        print(
            f"Created {len(EVAL_DATASET)} examples."
        )

        return dataset

    print(
        f"Using existing dataset: {DATASET_NAME}"
    )

    return client.read_dataset(
        dataset_name=DATASET_NAME
    )


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":

    client = Client()

    dataset = prepare_dataset(client)

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    print("name:", dataset.name)
    print("id:", dataset.id)


    # ========================================================
    # Run LangSmith Evaluation
    # ========================================================

    print("\n" + "=" * 70)
    print("START TRAJECTORY EVALUATION")
    print("=" * 70)

    results = client.evaluate(
        target_email_assistant,

        data=DATASET_NAME,

        evaluators=[
            trajectory_evaluator,
        ],

        experiment_prefix=(
            "email-assistant-trajectory-v4"
        ),
        max_concurrency=1,
    )

    print("\nEvaluation started.")

    print(
        "Check the experiment in LangSmith."
    )

"""
们把 Email Assistant Evaluation 做成下面这条路线：
                         Email Assistant
                               │
              ┌────────────────┼────────────────┐
              ↓                ↓                ↓
        Classification      Tool Call       Trajectory
              │                │                │
          intent 对吗？      Tool 对吗？      行为路径对吗？
              │                │                │
              └────────────────┼────────────────┘
                               ↓
                         Final Answer
                               ↓
                       Answer Evaluation
                               ↓
                         Overall Quality
"""
