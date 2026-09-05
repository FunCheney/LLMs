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