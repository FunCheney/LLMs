'''
                  LangSmith
                     │
                 Dataset
                     │
                     ▼
              target function
                     │
                     ▼
              classify_intent
                     │
                     ▼
              actual intent
                     │
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
           actual        reference
          "question"     "question"
              │             │
              └──────┬──────┘
                     ▼
          classification_evaluator
                     │
                     ▼
                  True/False
'''
from langsmith import Client
from dotenv import load_dotenv
import os

load_dotenv()

from email_assistant import classify_intent
from email_dataset import EVAL_DATASET

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

DATASET_NAME = "email-assistant-triage"

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
        dataset_name=DATASET_NAME,
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
        "messages": [],
        "response": [],
    }

    intent = classify_intent(state)

    return {
        "intent": intent["classification"]["intent"],
    }

# --------------------------------------------------
# Evaluator
# --------------------------------------------------
def classification_evaluator(input: dict, reference: dict) -> bool:

    return input["intent"].lower() == reference["intent"].lower()

# --------------------------------------------------
# run Evaluator
# --------------------------------------------------
result = client.evaluate(
    target_email_assistant,
    data=DATASET_NAME,
    evaluators=[
        classification_evaluator,
    ],
    experiment_prefix="email-assistant-triage",
    max_concurrency=2,
)

print(result)