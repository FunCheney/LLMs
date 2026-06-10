### 设计问答助手

#### 如何设计系统流程

### 关键挑战
1. 如何判断是否需要调用工具
2. 如何放置无限递归或死循环
3. 如何记录会话状态

#### 用什么技术栈实现


### 设计 Agent 客服系统

#### 悬在那种 Agent 模式

#### 划分为几个 Agent


### angent
1. 对用户输入的问题意图识别
2. tool calling

### Agent 的本质

#### 状态管理

#### 上下文保持
1. memory 设计

   1. working memory conte 
   2. session memory session Id + 用户画像存储 
   3. 长期记忆

#### 对话核心技术

1. intent Recognition + Slot Filling（槽位）: 理解用户意图并提取关键信息
2. Failure Handling：失败重试，降级策略，错误兜底
3. Conversation 生命周期：Session Id 管理，上下文清理


#### 消息协议设计方法论

1. 标准化字段
2. 扩展字段
3. 错误码规范
4. 支持异步响应与超时机制


### 多 Agent 协作
多 Agent 系统（Multi-Agent System）是一种由多个自主，交互的智能体（Agent）组成的分布式系统，每个Agent具有一定的自主性
感知能力，决策能力。
