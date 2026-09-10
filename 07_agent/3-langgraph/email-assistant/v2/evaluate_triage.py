'''
                    LangSmith
                       │
                  Evaluation
                       │
              ┌────────┴────────┐
              │                 │
           Dataset          Evaluator
              │                 │
              │                 │
              ▼                 │
    target_email_assistant      │
              │                 │
              ▼                 │
       email_assistant          │
              │                 │
              ▼                 │
         完整 Graph              │
              │                 │
              ▼                 │
       actual intent ───────────┘
                         compare
                            │
                      expected intent
'''
from langchain_core.messages import HumanMessage
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


def debug_evaluator(run, example):
    print("\n========== RUN ==========")
    print(run)

    print("\n========== LEVEL1 ==========")

    for child in run.child_runs:
        print(
            child.name,
            child.run_type,
            child.id,
        )

        print("    LEVEL 2:")

        for grandchild in child.child_runs:
            print(
                "    ",
                grandchild.name,
                grandchild.run_type,
                grandchild.id,
            )

    print("\n========== RUN OUTPUTS ==========")
    print(run.outputs)

    print("\n========== EXAMPLE ==========")
    print(example)

    return True

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
def tool_call_evaluator(run, example):
    expected_tool = example.outputs["expected_tool"]

    tool_runs = find_tool_calls(run)

    actual_tools = [
        tool_run.name
        for tool_run in tool_runs
    ]

    print("\n========== TOOL CALL EVALUATION ==========")
    print("Expected:", expected_tool)
    print("Actual:", actual_tools)

    return expected_tool in actual_tools

def find_tool_calls(run):
    tool_calls = []

    def work(current_run):
        if current_run.run_type == "tool_call":
            tool_calls.append(current_run)

            for child in current_run.children:
                work(child)

    work(run)

    return tool_calls


# --------------------------------------------------
# run Evaluator
# --------------------------------------------------
result = client.evaluate(
    target_email_assistant,
    data=DATASET_NAME,
    evaluators=[
        debug_evaluator,
        tool_call_evaluator,
    ],
    experiment_prefix="email-assistant-triage",
    max_concurrency=1,
)

print(result)

"""
本版输出结果
>>> support_agent
>>> should_continue
>>> support_agent
>>> should_continue
>>> support_agent
>>> should_continue
>>> finish_response

========== RUN ==========
id=UUID('01a0890e-8230-7673-b410-60582363964d') name='Target' start_time=datetime.datetime(2026, 9, 10, 2, 3, 44, 560874, tzinfo=datetime.timezone.utc) run_type='chain' end_time=datetime.datetime(2026, 9, 10, 2, 4, 21, 610972, tzinfo=datetime.timezone.utc) extra={'metadata': {'revision_id': '5ff5593-dirty', '__ls_runner': 'py_sdk_evaluate', 'num_repetitions': 1, 'example_version': '2026-09-10T01:58:54.891810+00:00', 'ls_method': 'traceable'}, 'runtime': {'sdk': 'langsmith-py', 'sdk_version': '0.10.18', 'library': 'langsmith', 'platform': 'macOS-15.6.1-arm64-arm-64bit', 'runtime': 'python', 'py_implementation': 'CPython', 'runtime_version': '3.12.7', 'langchain_version': '1.3.15', 'langchain_core_version': '1.5.4'}} error=None serialized=None events=[] inputs={'email_content': '\nHi,\n\nI forgot my password and cannot log into my account.\n\nHow can I reset it?\n\nThanks!\n'} outputs={'intent': 'question'} reference_example_id=UUID('74bb1175-687b-4bd5-bd6c-a187baf654c8') parent_run_id=None tags=[] attachments={} parent_run=None parent_dotted_order=None child_runs=[RunTree(id=01a0890e-8252-7163-b5c8-964839c44e45, name='LangGraph', run_type='chain', dotted_order='20260910T020344560874Z01a0890e-8230-7673-b410-60582363964d.20260910T020344594448Z01a0890e-8252-7163-b5c8-964839c44e45')] session_name='email-assistant-triage-58820cf7' session_id=None ls_client=Client (API URL: https://api.smith.langchain.com) dotted_order='20260910T020344560874Z01a0890e-8230-7673-b410-60582363964d' trace_id=UUID('01a0890e-8230-7673-b410-60582363964d') dangerously_allow_filesystem=False replicas=[]

========== LEVEL1 ==========
LangGraph chain 01a0890e-8252-7163-b5c8-964839c44e45
    LEVEL 2:
     classify_intent chain 01a0890e-8254-7e03-9caf-a7d3be095bd6
     support_agent chain 01a0890e-a82f-79a0-8209-1a8ffa61f849
     support_tools chain 01a0890e-c284-76c3-98a0-c483fbd82ca4
     support_agent chain 01a0890e-c287-71c0-9ef8-ef73cc801eb3
     support_tools chain 01a0890e-ed76-7eb2-881b-65249494bc92
     support_agent chain 01a0890e-ed78-7d42-b691-d68476c79dcc
     finish_response chain 01a0890f-12e9-72c0-be76-afc1abeb5df2

========== RUN OUTPUTS ==========
{'intent': 'question'}

========== EXAMPLE ==========
dataset_id=UUID('b4d88496-0356-4032-b475-29241e8be3ea') inputs={'email_content': '\nHi,\n\nI forgot my password and cannot log into my account.\n\nHow can I reset it?\n\nThanks!\n'} outputs={'intent': 'question', 'expected_tool': 'search_docs'} metadata={'dataset_split': ['base']} id=UUID('74bb1175-687b-4bd5-bd6c-a187baf654c8') created_at=datetime.datetime(2026, 9, 10, 1, 58, 54, 891810, tzinfo=TzInfo(0)) modified_at=datetime.datetime(2026, 9, 10, 1, 58, 54, 891810, tzinfo=TzInfo(0)) source_run_id=None attachments={}

========== TOOL CALL EVALUATION ==========
Expected: search_docs
Actual: []
<ExperimentResults email-assistant-triage-58820cf7>
"""