"""
                    Email
                      │
                      ▼
              Email Assistant
                      │
                      ▼
                Final Response
                      │
                      │
                      ▼
              ┌───────────────┐
              │  LLM Judge    │
              └───────┬───────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Accuracy   Relevance   Helpfulness
          │           │           │
          └───────────┼───────────┘
                      ▼
                    Score
1. 设计 Evaluation Dataset
2. Target 只返回最终 Response
3. 先明确 Judge 的职责。
    第一版只放三个标准
        3.1 Correctness 回答的信息是否正确？
        3.2 Relevance 是否真正回答了用户的问题？
        3.3 Helpfulness 用户看完能不能解决问题？
4. 引入 openevals
    这里我们不自己手写一个复杂的 Judge Graph。
    官方生态现在有 openevals，其中提供了 create_llm_as_judge，用于构建 LLM-as-a-judge evaluator。
5. 设计 Judge Prompt
                ┌─────────────┐
                │ User Email  │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │ LLM Judge   │
                └──────┬──────┘
                       ▲
             ┌─────────┴─────────┐
             │                   │
       Reference             Actual
        Answer              Response
6. 创建 Judge Evaluator
7. Evaluator 函数
8. 执行 Evaluation

"""
from dotenv import load_dotenv
import os
from langsmith import Client
from openai import base_url
from openevals.llm import create_llm_as_judge

from langchain_core.messages import HumanMessage
from posthog import api_key

from email_assistant import email_assistant


load_dotenv()


DATASET_NAME = "email-assistant-response-v5"


EVAL_DATASET = [
    {
        "inputs": {
            "email_content": """
I forgot my password and cannot log into my account.

How can I reset it?

Thanks!
"""
        },
        "outputs": {
            "reference_response": """
You can reset your password by going to
Settings > Security > Reset Password.

The password reset link expires after 30 minutes.
"""
        },
    }
]


def target_email_assistant(inputs: dict) -> dict:
    result = email_assistant.invoke(
        {
            "email_content": inputs["email_content"],
            "classification": None,
            "messages": [
                HumanMessage(
                    content=inputs["email_content"]
                )
            ],
            "response": "",
        }
    )

    return {
        "response": result["response"]
    }


RESPONSE_EVALUATION_PROMPT = """
You are an expert customer support evaluator.

Evaluate the assistant's response to the customer's email.

Evaluate the response based on:

1. Correctness
2. Relevance
3. Helpfulness

A good response should:
- contain correct information
- directly answer the customer's question
- provide useful and actionable information

Customer Email:
{inputs}

Reference Answer:
{reference_outputs}

Assistant Response:
{outputs}

Give a score from 0 to 1.

1.0 = excellent response
0.5 = partially correct or incomplete
0.0 = incorrect or unhelpful

Provide a brief explanation.
"""
judge = create_llm_as_judge(
    prompt=RESPONSE_EVALUATION_PROMPT,
    model="openai:o3-mini",
    feedback_key="response_quality",
)

def response_evaluator_fn(
    run,
    example,
):
    actual_response = run.outputs["response"]

    reference_response = (
        example.outputs["reference_response"]
    )

    result = judge(
        inputs=example.inputs["email_content"],
        outputs=actual_response,
        reference_outputs=reference_response,
    )

    print("\n===== RESPONSE EVALUATION =====")
    print("Score:", result["score"])
    print("Comment:", result.get("comment"))

    return result


def main():
    client = Client()

    if not client.has_dataset(
        dataset_name=DATASET_NAME
    ):
        client.create_dataset(
            dataset_name=DATASET_NAME
        )

        client.create_examples(
            dataset_name=DATASET_NAME,
            examples=EVAL_DATASET,
        )

    result = client.evaluate(
        target_email_assistant,
        data=DATASET_NAME,
        evaluators=[
            response_evaluator_fn
        ],
        experiment_prefix="email-assistant-response-v5",
        max_concurrency=1,
    )

    print(result)


if __name__ == "__main__":
    main()