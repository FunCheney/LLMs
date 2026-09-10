# 版本迭代记录
1. 本版本添加的 Evaluation 

### Evaluation 设计

Evaluation
    │
    ├── 1. Outcome
    │
    ├── 2. Single-step
    │
    ├── 3. Trajectory
    │
    └── 4. LLM-as-Judge

#### Outcome
例如：

```text
    用户邮件
       ↓
    Agent
       ↓
    intent = bug
```
评估：

```text
    Expected: bug
    Actual:   bug
    
            ↓
    
           PASS
```

#### Single-step
关注一个决策：

```text
    User
     ↓
    Agent
     ↓
    Tool?
     ↓
    search_docs
```

评估：

```text
    Agent 有没有选择正确的工具？
```
#### Trajectory

关注：
```text
    Human
      ↓
    AI
      ↓
    Tool
      ↓
    AI
      ↓
    Tool
      ↓
    AI
```

评估：

整个 Agent 执行路径是否符合预期？

##### 把本地 Trajectory Match 跑通 

 evaluate_triage_v2_Trajectory.py

##### 把 Reference Trajectory 正确设计出来，然后接入 LangSmith Dataset

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

也就是说：

```text
Example
{
    inputs: {
        email_content: ...
    },

    outputs: {
        messages: [
            ...
        ]
    }
}

```
这也是 AgentEvals/LangGraph 当前推荐的数据结构：Dataset 的输入可以是消息，输出保存预期的消息 trajectory


#### LLM-as-Judge

```text
                  Agent
                    │
                    ▼
                  Answer
                    │
                    ▼
              ┌───────────┐
              │ Judge LLM │
              └─────┬─────┘
                    │
             ┌──────┼──────┐
             ▼      ▼      ▼
           factual helpful complete
```