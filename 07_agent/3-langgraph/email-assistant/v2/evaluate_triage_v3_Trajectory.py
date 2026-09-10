"""
本例把整个评估链路变成：
LangSmith Dataset
       │
       │
       ├── input
       │     └── email_content
       │
       └── reference_outputs
             └── messages
                    │
                    ▼
              client.evaluate()
                    │
                    ▼
              target_agent
                    │
                    ▼
             actual messages
                    │
                    ▼
          Trajectory Evaluator
                    │
                    ▼
                  PASS
                  FAIL
修改点：
1. Reference 不需要包含最终回答
2. 先建立一个真正的 Reference
3. 让 search_docs 的参数允许语义变化

"""

import json

from langchain_core.messages import HumanMessage
from langsmith import Client

from agentevals.trajectory.match import (
    create_trajectory_match_evaluator,
)

from email_assistant import email_assistant


# ============================================================
# 1. Dataset
# ============================================================

DATASET_NAME = "email-assistant-trajectory-v3"


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
                    "role": "user",
                    "content": (
                        "I forgot my password and cannot log into "
                        "my account.\n\n"
                        "How can I reset it?\n\n"
                        "Thanks!"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": json.dumps({
                                    "query": "password reset"
                                }),
                            }
                        }
                    ],
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

    print("\n===== ACTUAL TRAJECTORY =====")
    for message in result["messages"]:
        print(message)

    return {
        "messages": result["messages"],
    }


# ============================================================
# 4. Trajectory Evaluator
# ============================================================

trajectory_evaluator = create_trajectory_match_evaluator(
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
            "email-assistant-trajectory-v3"
        ),
        max_concurrency=1,
    )

    print("\nEvaluation started.")

    print(
        "Check the experiment in LangSmith."
    )

