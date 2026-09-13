### HITL 到底解决什么问题？

### 最应该理解的核心概念：interrupt

### 最简单的 HITL 模型

### HITL 和我们之前学的 Graph 是怎么连接起来的？

### 为什么必须有 Checkpointer？



## 学习路线

```
                 Email Assistant
                       │
        ┌──────────────┴──────────────┐
        │                             │
       V1                            V2
     Agent                     Agent + Evaluation
        │                             │
        │                   ┌─────────┼─────────┐
        │                   │         │         │
        │                Triage    ToolCall  Response
        │                             │
        │                         Trajectory
        │
        ▼
       V3
   Agent + HITL
        │
        ├── interrupt()
        ├── Checkpointer
        ├── thread_id
        └── Human Decision
              ├── approve
              ├── edit
              └── reject
```