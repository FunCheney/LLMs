### 阶段 1
Basic Agent
│
│ EmailState
│
├── email_content
├── classification
├── messages
└── response
│
▼
### 阶段 2
Evaluation
│
│ Schema 基本不变
│
├── evaluate triage
├── evaluate tool calls
└── evaluate responses
│
▼
### 阶段 3
Human-in-the-loop
│
│ State 增加 Human Review
│
├── pending action
├── human decision
└── ...
│
▼
### 阶段 4
Memory
│
│ 不只是修改 State
│
├── State
│    └── 当前执行
│
└── Store
     └── 长期记忆
│
▼
Gmail / Deployment


#### 整个项目的演进路线

                    Email Assistant
                          │
                          ▼
┌─────────────────────────────────────────────┐
│ Phase 1  LangGraph 基础                      │
│ State / Node / Edge / Reducer / Pregel      │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Phase 2  Workflow                           │
│ classify → route → handler                  │
│                                             │
│ 学会：Graph 如何控制流程                       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Phase 3  Agent                              │
│ LLM → Tool Call → Tool → LLM                │
│                                             │
│ 学会：Agent 如何自主决定下一步                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Phase 4  Evaluation                         │
│                                             │
│ ├─ Classification Evaluation                │
│ ├─ Single-step Tool Evaluation              │
│ └─ Trajectory Evaluation                    │
│     <-  evaluate_triage_v2_Trajectory.py    │
│                                             │
│ 学会：如何证明 Agent 做得好不好                 │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Phase 5  Human-in-the-loop                  │
│                                             │
│ interrupt → human decision → resume         │
│                                             │
│ 学会：Agent 如何与人协作                       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Phase 6  Memory                             │
│                                             │
│ Thread State + Store                        │
│                                             │
│ 学会：Agent 如何跨执行保存信息                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Phase 7  Production                         │
│                                             │
│ Gmail / Persistence / Deployment / Monitor  │
│                                             │
│ 学会：如何把 Agent 变成真正的应用               │
└─────────────────────────────────────────────┘