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
因为它是一个分类问题。因此它很适合：Deterministic Evaluation


### 第二层：Tool Call Evaluation
在 Triage 之后，Agent 需要决定：我应该使用哪个工具。因此 Tool Call Evaluation 就是用来评估 Agent 选择的工具是否正确。


#### Tool Call 和 Triage 有什么区别？
Triage: Agent 理解的用户的问题。
Tool Call: Agent 选择的工具。

如果 Agent 在 Triage 层做出的了正确的判断，说明它正确的理解了用户的问题。但是如果选择错了工具，说明没有把正确的理解转化成正确的动作。


### 第三层：Trajectory Evaluation
这一步，Evaluation 开始从 ”做了什么“ 升级到 ”整个过程怎么做的？“。Trajectory Evaluation 用来评价：Agent 是否按照预期的行为路径完成任务。

#### 为什么 Tool Call Evaluation 还不够？

假设我们只检查：
```text
Agent called search_docs
```

结果：
```text
search_docs ✅
```

但 Agent 实际执行：
```text
Human
 ↓
AI(search_docs)
 ↓
Tool(search_docs)
 ↓
AI(search_docs)
 ↓
Tool(search_docs)
 ↓
AI(search_docs)
 ↓
Tool(search_docs)
 ↓
AI(final)
```

它虽然调用了正确工具，但可能：

* 无限循环 
* 重复搜索 
* 没有正确结束 
* 调用了不必要的工具

所以：
```text
Tool Call Evaluation
```


只能告诉我们： 某个动作对不对。

而： Trajectory Evaluation 告诉我们： 动作组合起来形成的行为过程对不对。


### 第四层：Response Evaluation

Agent 最终给用户的答案是怎么样的？


