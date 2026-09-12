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

## 总结 四种 Evaluation
先把本版本中的四种 Evaluation 串起来

```
                Email Assistant Evaluation

                         Email
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Triage Evaluation           Tool Call Evaluation
       「做什么？」                 「调用什么？」
             │                           │
             └─────────────┬─────────────┘
                           ▼
                   Trajectory Evaluation
                      「怎么做？」
                           │
                           ▼
                   Response Evaluation
                      「做得好吗？」
                           │
                           ▼
                    Overall Evaluation
```

Agent Evaluation 不是只评价“最后回答得好不好”，而是分别评价 Agent 的「决策、动作、过程、结果」。

* Triage：评价 决策 是否做对？
* Tool call：评价 动作，工具是否选对？
* Trajectory：评价 过程，Agent 按正确路径执行了吗？
* Response：评价结果，Agent 最终回答得好吗？

上述四个层次分别对应 Agent 的四个维度

1. Decision
2. Action
3. Process
4. Outcome

### 第一层：Triage Evaluation

#### 1.评价什么
我们的 classify_intent：

```python
def classify_intent(state: EmailState):
    ...
    classification = triage_llm.invoke(prompt)
    return {"classification": classification}
```
它负责：

```
用户邮件
   ↓
Intent Classification
   ↓
question / bug / billing / feature_request / other

```

所以 Triage Evaluation 问的是：

```
Agent 对用户问题的理解是否正确？
```

#### 2.例如
用户：

```
I forgot my password and cannot log into my account.

How can I reset it?
```


正确分类：

```json
{
    "intent": "question",
    "urgency": "low"
}
```


如果 Agent 输出：

```json
{
    "intent": "bug",
    "urgency": "low"
}
```

那么

```
Triage
❌
```
即使它最后碰巧给出了一个正确的密码重置答案，我们仍然认为：Agent 的内部决策是错误的。


#### 3. Triage 为什么通常可以用确定性 Evaluation？
因为它是一个分类问题。



### 第二层：Triage Evaluation


### 第三层：Triage Evaluation


### 第四层：Triage Evaluation


