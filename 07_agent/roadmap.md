
### 如何演进成一个线上可用的 agent


| 优先级 | 知识方向 | 需要重点掌握 | 对本项目的意义 |
|---|---|---|---|
| P0-1 | 业务领域建模 | 业务目标、流程/SOP、角色权限、业务实体、状态机、规则、异常流程、责任边界、成功指标 | 决定 Agent 到底应该做什么、哪些事情不能自主做 |
| P0-2 | 企业系统集成与工具工程 | REST/OpenAPI、数据库、OAuth/SSO、Webhook、消息队列、MCP、参数校验、幂等、事务、超时、错误分类 | 把目前的模拟 `search_docs` 替换成 CRM、ERP、工单、知识库等真实工具 |
| P0-3 | Agent 安全与权限治理 | Prompt Injection、数据泄露、最小权限、RBAC/ABAC、多租户隔离、PII 脱敏、密钥管理、审计日志、工具白名单 | 企业 Agent 最大风险通常来自“错误执行”，而不只是“回答错误” |
| P0-4 | 生产级后端工程 | FastAPI、异步编程、PostgreSQL/Redis、任务队列、持久化 Checkpointer、重试、超时、熔断、限流、部署 | 把 Notebook/脚本升级为可调用、可恢复、可扩展的服务 |
| P0-5 | 自动化测试 | 单元测试、Tool Contract Test、Graph 路径测试、异常测试、Mock 外部系统、CI | 当前已有 Evaluation，但传统软件测试覆盖明显不足 |
| P1-1 | 企业级 RAG | 混合检索、重排、Metadata、ACL-aware retrieval、引用溯源、增量更新、文档版本、删除同步、检索评估 | 当前 RAG 主要覆盖解析、切分、向量和检索概念，见 [RAG 路线](/Users/mico/PycharmProjects/LLMs2/06_rag/README.md:2) |
| P1-2 | Agent Evaluation 与可观测性 | Golden Dataset、回归测试、Trace、成功率、工具错误率、人工接管率、延迟、Token/成本、线上反馈 | 在现有四层 Evaluation 上继续补齐线上指标和回归门禁 |
| P1-3 | 状态、记忆与并发 | Thread State、长期记忆、用户画像、记忆提取/遗忘、并发控制、状态迁移、断点恢复 | 公司业务通常是跨会话、跨天、跨人员的，不是一次性问答 |
| P1-4 | HITL 与审批设计 | 风险分级、approve/edit/reject、超时升级、审批人选择、恢复执行、操作凭证 | 项目已有 HITL 概念，但需要落到真实业务权限和审批流程 |
| P1-5 | 数据治理 | 数据来源、数据血缘、质量、有效期、访问权限、保留期限、合规删除 | 避免 Agent 使用过期、越权或来源不明的数据 |
| P2-1 | 模型工程与成本治理 | 模型路由、Fallback、上下文压缩、Token 预算、语义缓存、批处理、质量/成本权衡 | 项目已有 KV Cache 实验，可以继续提升为系统级成本管理 |
| P2-2 | Agent 产品与交互设计 | 不确定性表达、引用展示、确认步骤、进度反馈、失败恢复、人工接管入口 | 决定业务人员是否敢用、会用、愿意持续使用 |
| P2-3 | 多 Agent 架构决策 | Supervisor、Handoff、共享状态、隔离上下文、冲突处理、终止条件 | 仓库已有多 Agent；应在单 Agent 无法清晰解决问题时再引入 |
| P3 | 微调和底层模型训练 | SFT、LoRA、偏好优化、模型部署 | 对多数公司业务 Agent 不是当前瓶颈，优先级低于业务、数据、安全和工程化 |
[text.md](text.md)
---

## 三个月学习计划（2026-10-20 起，对照「Agent 平台/框架开发」JD）

```text
职位描述
1、负责 Agent 应用的整体架构设计与核心功能开发，包括大模型适配、记忆库、知识库、Skills 等核心模块；
2、基于业务场景需求，设计并实现 Agent 的任务规划、逻辑推理、决策执行等核心能力，提升智能体自主解决复杂问题的能力；
3、负责 Agent 应用的性能优化与稳定性保障，包括响应延迟优化、并发能力提升、错误容灾机制设计等；
4、集成记忆库、知识库、Skills 等核心技术，沉淀通用 Agent 开发框架、组件库与技术解决方案，支撑业务快速落地；
5、跟踪 Agent 领域前沿技术动态，调研最新的智能体架构与技术，落地到业务场景中提升产品技术竞争力。
```

> 节奏：每周 5 天，每天 2–3 小时，共 12 周。每周末用「检查点」自测，不通过则顺延对应任务。
> 主线：**大模型适配层（W1）→ 服务化底座（W2）→ 核心模块：知识库/记忆库/Skills（W3–W6）→ 规划与决策（W7）→ 性能与容灾（W8）→ 评测体系（W9）→ 框架与组件库沉淀（W10）→ 复杂任务整合 Demo（W11）→ 前沿与面试沉淀（W12）**。
> 说明：W9 评测框架会回填到 W3/W7/W8 做回归验证；W11 用沉淀出的框架快速搭建一个"自主解决复杂问题"的演示 Agent。

### Week 1：大模型适配层（P0，职责 1）

- Day1：调研主流 LLM API 差异（OpenAI / Claude / Qwen / DeepSeek）与统一抽象设计
- Day2：实现 `LLMClient` 抽象接口：chat / stream / 重试
- Day3：实现多 provider 适配器（openai-compatible 通用协议 + 各厂商差异处理）
- Day4：统一 function calling / tool calling 归一化（各厂商格式对齐）
- Day5：模型配置化 + 简单路由（按任务选模型）
- **产出物**：`agent_framework/llm/` 适配层，一处切换多模型
- **检查点**：同一段代码可无感切换 ≥ 3 个模型，tool calling 结果一致

### Week 2：服务化底座（P0，职责 3）

- Day1：FastAPI 基础 + Pydantic 校验 + 依赖注入
- Day2：异步并发（`async/await`、Semaphore）+ SSE 流式输出
- Day3：把适配层 + 基础 Agent 包装为 HTTP 服务（`/chat` 接口）
- Day4：Docker + docker-compose 一键起
- Day5：结构化日志 + 统一异常分类
- **产出物**：`agent_server/` 可调用服务，支持 SSE
- **检查点**：curl 流式返回；compose 一键拉起；异常返回规范错误码
 
### Week 3：知识库（RAG）深化（P0，职责 1）

- Day1：RAG 全景回顾 + 混合检索（BM25 + 稠密）+ RRF 融合
- Day2：重排（bge-reranker）接入
- Day3：引用溯源 + chunk 元数据 + 增量更新
- Day4：Query 改写（multi-query / HyDE）
- Day5：检索评估（Recall@K / MRR），量化改造前后对比
- **产出物**：`agent_framework/knowledge/` 知识库模块，可插拔检索策略
- **检查点**：检索指标量化提升；回答可溯源；新增文档免全量重建

### Week 4：记忆库 I —— 工作记忆与上下文治理（P0，职责 1）

- Day1：记忆分层理论（工作 / 短期 / 长期 / 用户画像）
- Day2：工作记忆：会话状态、上下文预算、消息裁剪
- Day3：上下文压缩：对话摘要、tool result 截断与摘要
- Day4：Token 预算与成本埋点
- Day5：接入 `agent_server`，长对话压测
- **产出物**：`agent_framework/memory/` 工作记忆 + 上下文治理
- **检查点**：20+ 轮不爆上下文，token 成本下降可量化

### Week 5：记忆库 II —— 长期记忆与用户画像（P0，职责 1）

- Day1：长期记忆设计：提取、写入、检索、遗忘
- Day2：记忆存储（向量库 + 结构化元数据）
- Day3：用户画像记忆的最小实现
- Day4：记忆与知识库 / 会话的边界与协同
- Day5：整合演练 + 记忆命中效果验证
- **产出物**：完整的记忆库模块 + 记忆检索 demo
- **检查点**：跨会话能回忆关键信息；遗忘策略可清理过期记忆

### Week 6：Skills 系统（P0，职责 1、4）

- Day1：调研 Skill/工具生态：Anthropic Skills、MCP、开源 skill 包格式
- Day2：定义 Skill 规范（元数据 + 指令 + 脚本/资源 + 参数契约）
- Day3：实现 Skill 加载器：注册、发现、动态加载、沙箱执行
- Day4：把已有工具重构为 2–3 个 Skill（搜索、知识库查询、记忆读写）
- Day5：Skill 与 tool calling 的调度集成
- **产出物**：`agent_framework/skills/` Skill 系统 + 2–3 个可复用 Skill
- **检查点**：新增 Skill 无需改主流程即可被 Agent 调用

### Week 7：任务规划、推理与决策（P0，职责 2）

- Day1：规划范式对比：ReAct / Plan-and-Execute / 反思（Reflection）
- Day2：实现 Plan-and-Execute：任务分解 → 子计划 → 逐步执行
- Day3：决策执行循环：工具选择、重规划、终止条件、预算控制
- Day4：反思/自纠错：结果校验 + 失败重试 + 策略调整
- Day5：评测：任务成功率、步数、工具调用准确性
- **产出物**：`agent_framework/planner/` 规划与决策模块
- **检查点**：复杂任务（≥5 步）完成率显著高于朴素 ReAct 基线

### Week 8：性能优化与容灾（P0，职责 3）

- Day1：延迟优化：流式、语义缓存、并发请求批处理
- Day2：并发能力：连接池、信号量限流、背压
- Day3：容灾：重试 / 超时 / 熔断 / 降级 / 兜底模型切换
- Day4：错误分类与自恢复、幂等设计
- Day5：压测 + 混沌演练（模拟模型超时 / 工具故障）
- **产出物**：`agent_framework/resilience/` 容灾中间件 + 压测报告
- **检查点**：注入故障后服务不中断，延迟/吞吐有量化报告

### Week 9：评测体系（P0，支撑职责 3、4）

- Day1：评测方法论：Golden Dataset、指标分层（成功率 / 延迟 / 成本 / 质量）
- Day2：构建 Golden Set（≥ 50 条）+ 批量评测脚本
- Day3：LLM-as-Judge + RAGAS
- Day4：回归门禁接入 CI + Trace 采集（LangSmith / Langfuse）
- Day5：根因归因：写 3 个「现象 → 根因 → 量化改进」案例
- **产出物**：`eval/` 框架 + `cases/` 归因案例 3 篇
- **检查点**：一条命令跑全量评测出报告；回填验证 W3/W7/W8

### Week 10：框架与组件库沉淀（P0，职责 4）

- Day1：架构梳理：把 llm/memory/knowledge/skills/planner 抽象为统一框架
- Day2：定义公共接口与依赖注入，解耦业务与框架
- Day3：组件库：可复用 Agent 组件（tool、retriever、memory 等）
- Day4：文档 + 示例 + 一键脚手架（`agent create` 快速起项目）
- Day5：单元测试补全 + 打包发布（本地 pip 安装）
- **产出物**：`agent_framework` 通用框架 + 组件库 + 脚手架
- **检查点**：第三方业务场景用脚手架 1 天内接入框架跑通

### Week 11：复杂任务整合 Demo（P0，职责 2）

- Day1：选一个复杂业务场景（如深度调研 / 多源数据分析）做需求拆解
- Day2：用沉淀框架搭 Agent：规划 + 知识库 + Skills + 记忆协同
- Day3：端到端联调 + 用 W9 评测框架评估
- Day4：加 HITL 与权限控制，处理不确定性
- Day5：demo 打磨（README + 截图 + 结果报告）
- **产出物**：一个"自主解决复杂问题"的完整 demo Agent
- **检查点**：给定主题自动完成多步任务并产出可交付结果

### Week 12：前沿跟踪与面试沉淀（P0，职责 5）

- Day1：前沿调研：MCP / Skills / 多智能体 / Computer Use / 上下文工程最新进展
- Day2：写前沿技术调研笔记（含可落地评估）
- Day3：复盘框架设计，输出架构图与技术决策记录（ADR）
- Day4：打磨作品集 README（架构图 + 指标 + 评测 + demo）
- Day5：模拟面试：按 JD 五条职责逐条准备 10 分钟讲述
- **产出物**：调研笔记 + ADR + 面试 case 集
- **检查点**：能对着 JD 五条职责逐条讲清架构与落地经验