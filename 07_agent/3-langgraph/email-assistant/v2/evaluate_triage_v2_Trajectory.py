"""
本实验我们要评价的是：Agent 处理一个问题时大致的执行流程 和 预期的执行流程
如 输入：I forgot my password. How can I reset it?
执行里程大致是：
            HumanMessage
                │
                ▼
            AIMessage
              tool_calls = search_docs
                │
                ▼
            ToolMessage
              search_docs result
                │
                ▼
            AIMessage
              final answer
这个就是：Trajectory。
1. 安装 AgentEvals： pip install agentevals
2. 增加 REFERENCE_TRAJECTORY
3. 创建 evaluator
4. 让 Target 输出 trajectory
5. 让 Evaluator 拿到 trajectory

"""
from agentevals.trajectory import create_trajectory_match_evaluator
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langsmith import Client
from dotenv import load_dotenv
from email_assistant import email_assistant
import os

load_dotenv()



REFERENCE_TRAJECTORY = [
    HumanMessage( content=""" I forgot my password and cannot log into my account. How can I reset it? Thanks! """.strip() ),
    # 注意：
    # 这里暂时只描述我们期望 Agent 做出的第一步行为：
    # Human # ↓ # AI -> search_docs
    # 实际的 tool args 可能因为模型不同而有所不同。
    # 因此第一次测试建议先打印实际 trajectory，
    # 再根据实际结果调整 Reference。
    ]

# AgentEvals 当前提供四种模式：
# strict
# unordered
# subset
# superset
trajectory_evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict", # "strict" 意味着 Reference 中规定的 trajectory 顺序必须匹配。
)

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
        "messages": intent["messages"], # 返回 messages, 其中是包含 trajectory
    }


if __name__ == "__main__":
    inputs = {
    "email_content": """ I forgot my password and cannot log into my account. How can I reset it? Thanks! """.strip()
    }
    print("=" * 70)
    print("1. INPUT")
    print("=" * 70)
    print(inputs["email_content"])
    # -------------------------------------------------------- # Execute Agent # ---------------------------------------
    result = target_email_assistant(inputs)
    actual_trajectory = result["messages"]
    # -------------------------------------------------------- # Print actual trajectory # -----------------------------
    print("\n" + "=" * 70)
    print("2. ACTUAL TRAJECTORY")
    print("=" * 70)
    for i, message in enumerate(actual_trajectory):
        print(f"\n--- Message {i} ---")
        print("type:", type(message).__name__)
        print("content:", message.content)
        if hasattr(message, "tool_calls"):
            print("tool_calls:", message.tool_calls)
        # -------------------------------------------------------- # Build Reference # ---------------------------------、
        # 第一阶段不要直接假设完整 trajectory。
        # 我们先根据实际输出构造一个可比较的 Reference。
        # 对于当前 password reset 场景， # 我们真正关心的是：
        # Human
        # ↓
        # AI(tool_call: search_docs)
        # 后续 ToolMessage / AI final response
        # 可以根据实际 Agent 行为继续补充。
        # --------------------------------------------------------------------------------------------------------------
        reference_trajectory = actual_trajectory
        print("\n" + "=" * 70)
        print("3. REFERENCE TRAJECTORY")
        print("=" * 70)
        for i, message in enumerate(reference_trajectory):
            print(f"\n--- Message {i} ---")
            print("type:", type(message).__name__)
            print("content:", message.content)
            if hasattr(message, "tool_calls"):
                print("tool_calls:", message.tool_calls)

        # -------------------------------------------------------- # Evaluate # ----------------------------------------
        print("\n" + "=" * 70)
        print("4. TRAJECTORY EVALUATION")
        print("=" * 70)
        evaluation = trajectory_evaluator( outputs=actual_trajectory, reference_outputs=reference_trajectory, )
        print(evaluation)
        # -------------------------------------------------------- # Result # ------------------------------------------
        print("\n" + "=" * 70)
        print("5. RESULT")
        print("=" * 70)
        if evaluation.get("key") == "trajectory_match":
            print("PASS")
        else:
            print("FAIL")



"""
======================================================================
1. INPUT
======================================================================
I forgot my password and cannot log into my account. How can I reset it? Thanks!
>>> support_agent
>>> should_continue
>>> support_agent
>>> should_continue
>>> finish_response

======================================================================
2. ACTUAL TRAJECTORY
======================================================================

--- Message 0 ---
type: HumanMessage
content: I forgot my password and cannot log into my account. How can I reset it? Thanks!

======================================================================
3. REFERENCE TRAJECTORY
======================================================================

--- Message 0 ---
type: HumanMessage
content: I forgot my password and cannot log into my account. How can I reset it? Thanks!

--- Message 1 ---
type: AIMessage
content: I can help you with that! Let me check the product documentation for the password reset process.
tool_calls: [{'name': 'search_docs', 'args': {'query': 'reset forgotten password account login'}, 'id': 'search_docs_0_1352929e', 'type': 'tool_call'}]

--- Message 2 ---
type: ToolMessage
content: To reset your password, go to Settings > Security > Reset Password.
Password reset links expire after 30 minutes.
If your account is locked, contact customer support.

--- Message 3 ---
type: AIMessage
content: No problem! Here's how to reset your password, based on our product documentation:

**To reset your password:**
1. Go to **Settings** > **Security** > **Reset Password**
2. Follow the prompts to receive a password reset link

**A few important things to keep in mind:**
- ⏱️ Password reset links **expire after 30 minutes**, so be sure to use the link promptly once you receive it.
- 🔒 If your account is **locked** (for example, after too many failed login attempts), you'll need to contact customer support directly for help getting back in.

Since you mentioned you can't log in at all — if you're unable to access the Settings menu or you're seeing a message that your account is locked, just let me know and I can help you with the next steps for reaching our support team.
tool_calls: []

======================================================================
4. TRAJECTORY EVALUATION
======================================================================
{'key': 'trajectory_strict_match', 'score': True, 'comment': None, 'metadata': None}

======================================================================
5. RESULT
======================================================================
FAIL

--- Message 1 ---
type: AIMessage
content: I can help you with that! Let me check the product documentation for the password reset process.
tool_calls: [{'name': 'search_docs', 'args': {'query': 'reset forgotten password account login'}, 'id': 'search_docs_0_1352929e', 'type': 'tool_call'}]

======================================================================
3. REFERENCE TRAJECTORY
======================================================================

--- Message 0 ---
type: HumanMessage
content: I forgot my password and cannot log into my account. How can I reset it? Thanks!

--- Message 1 ---
type: AIMessage
content: I can help you with that! Let me check the product documentation for the password reset process.
tool_calls: [{'name': 'search_docs', 'args': {'query': 'reset forgotten password account login'}, 'id': 'search_docs_0_1352929e', 'type': 'tool_call'}]

--- Message 2 ---
type: ToolMessage
content: To reset your password, go to Settings > Security > Reset Password.
Password reset links expire after 30 minutes.
If your account is locked, contact customer support.

--- Message 3 ---
type: AIMessage
content: No problem! Here's how to reset your password, based on our product documentation:

**To reset your password:**
1. Go to **Settings** > **Security** > **Reset Password**
2. Follow the prompts to receive a password reset link

**A few important things to keep in mind:**
- ⏱️ Password reset links **expire after 30 minutes**, so be sure to use the link promptly once you receive it.
- 🔒 If your account is **locked** (for example, after too many failed login attempts), you'll need to contact customer support directly for help getting back in.

Since you mentioned you can't log in at all — if you're unable to access the Settings menu or you're seeing a message that your account is locked, just let me know and I can help you with the next steps for reaching our support team.
tool_calls: []

======================================================================
4. TRAJECTORY EVALUATION
======================================================================
{'key': 'trajectory_strict_match', 'score': True, 'comment': None, 'metadata': None}

======================================================================
5. RESULT
======================================================================
FAIL

--- Message 2 ---
type: ToolMessage
content: To reset your password, go to Settings > Security > Reset Password.
Password reset links expire after 30 minutes.
If your account is locked, contact customer support.

======================================================================
3. REFERENCE TRAJECTORY
======================================================================

--- Message 0 ---
type: HumanMessage
content: I forgot my password and cannot log into my account. How can I reset it? Thanks!

--- Message 1 ---
type: AIMessage
content: I can help you with that! Let me check the product documentation for the password reset process.
tool_calls: [{'name': 'search_docs', 'args': {'query': 'reset forgotten password account login'}, 'id': 'search_docs_0_1352929e', 'type': 'tool_call'}]

--- Message 2 ---
type: ToolMessage
content: To reset your password, go to Settings > Security > Reset Password.
Password reset links expire after 30 minutes.
If your account is locked, contact customer support.

--- Message 3 ---
type: AIMessage
content: No problem! Here's how to reset your password, based on our product documentation:

**To reset your password:**
1. Go to **Settings** > **Security** > **Reset Password**
2. Follow the prompts to receive a password reset link

**A few important things to keep in mind:**
- ⏱️ Password reset links **expire after 30 minutes**, so be sure to use the link promptly once you receive it.
- 🔒 If your account is **locked** (for example, after too many failed login attempts), you'll need to contact customer support directly for help getting back in.

Since you mentioned you can't log in at all — if you're unable to access the Settings menu or you're seeing a message that your account is locked, just let me know and I can help you with the next steps for reaching our support team.
tool_calls: []

======================================================================
4. TRAJECTORY EVALUATION
======================================================================
{'key': 'trajectory_strict_match', 'score': True, 'comment': None, 'metadata': None}

======================================================================
5. RESULT
======================================================================
FAIL

--- Message 3 ---
type: AIMessage
content: No problem! Here's how to reset your password, based on our product documentation:

**To reset your password:**
1. Go to **Settings** > **Security** > **Reset Password**
2. Follow the prompts to receive a password reset link

**A few important things to keep in mind:**
- ⏱️ Password reset links **expire after 30 minutes**, so be sure to use the link promptly once you receive it.
- 🔒 If your account is **locked** (for example, after too many failed login attempts), you'll need to contact customer support directly for help getting back in.

Since you mentioned you can't log in at all — if you're unable to access the Settings menu or you're seeing a message that your account is locked, just let me know and I can help you with the next steps for reaching our support team.
tool_calls: []

======================================================================
3. REFERENCE TRAJECTORY
======================================================================

--- Message 0 ---
type: HumanMessage
content: I forgot my password and cannot log into my account. How can I reset it? Thanks!

--- Message 1 ---
type: AIMessage
content: I can help you with that! Let me check the product documentation for the password reset process.
tool_calls: [{'name': 'search_docs', 'args': {'query': 'reset forgotten password account login'}, 'id': 'search_docs_0_1352929e', 'type': 'tool_call'}]

--- Message 2 ---
type: ToolMessage
content: To reset your password, go to Settings > Security > Reset Password.
Password reset links expire after 30 minutes.
If your account is locked, contact customer support.

--- Message 3 ---
type: AIMessage
content: No problem! Here's how to reset your password, based on our product documentation:

**To reset your password:**
1. Go to **Settings** > **Security** > **Reset Password**
2. Follow the prompts to receive a password reset link

**A few important things to keep in mind:**
- ⏱️ Password reset links **expire after 30 minutes**, so be sure to use the link promptly once you receive it.
- 🔒 If your account is **locked** (for example, after too many failed login attempts), you'll need to contact customer support directly for help getting back in.

Since you mentioned you can't log in at all — if you're unable to access the Settings menu or you're seeing a message that your account is locked, just let me know and I can help you with the next steps for reaching our support team.
tool_calls: []

======================================================================
4. TRAJECTORY EVALUATION
======================================================================
{'key': 'trajectory_strict_match', 'score': True, 'comment': None, 'metadata': None}

======================================================================
5. RESULT
======================================================================
FAIL
"""



