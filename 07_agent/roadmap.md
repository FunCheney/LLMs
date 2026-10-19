
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

---

## 三个月学习计划（2026-10-20 起，对照「增长营销 Agent 开发」JD）
```text
职位描述
1、设计并实现端到端的增长营销Agent系统，涵盖深度调研、SEO优化、社交媒体营销等多个场景；
2、优化Agent应用性能效率和产出效果，深度理解Agent的实现机制和业务流程，针对评测过程中发现的问题，结合具体案例进行根源分析，量化输出可落地的优化方向与改进建议；
3、持续推进Agent Context Engineering，优化包括RAG、工具系统在内的生态体系，提高Agent的运行效率和稳定性，为高并发、多场景的业务环境提供有力的技术支撑；
4、构建端到端的Agent评测体系，协同模型算法团队，深入探索系统的评测方案，从用户满意度、业务指标等多个维度对Agent进行全面评估，为后续优化提供数据支持和决策依据；
5、不断探索和应用前沿的技术和方法，为团队带来创新思路和解决方案。
职位要求
1、本科及以上学历，计算机软件开发及相关专业；
2、掌握至少一种主流编程语言，如Python、Java、Go等，熟悉Web开发框架，能够高效地进行前后端开发工作，具备良好的数据结构和算法基础；
3、熟悉AI Agent相关技术，了解工具开发，Agent流程设计，评估系统，上下文工程等体系；
4、具有较强的学习能力和求知欲，能够快速掌握新知识、新技术，关注AI领域的发展动态，推动Agent应用的持续优化。

具备以下条件者优先：
1、了解主流AI模型及其应用场景，有从0-1搭建AI Agent应用经验者优先，能够独立完成Agent应用的设计、开发和部署工作；
2、有参与过增长营销，包括SEO优化、社交媒体营销在内的相关项目经验者优先；
```

> 节奏：每周 5 天，每天 2–3 小时，共 12 周。每周末用「检查点」自测，不通过则顺延对应任务。
> 主线：**工程化底座（W1–W2）→ 评测体系（W3–W4）→ RAG 与上下文工程（W5–W8）→ 安全与真实工具（W9）→ 增长营销 Agent 作品（W10–W11）→ 全栈收尾与面试沉淀（W12）**。

### Week 1：生产级后端 I —— FastAPI 与流式服务（P0，JD 要求 2）

- Day1：FastAPI 基础：路由、依赖注入、Pydantic 校验，跑通官方教程
- Day2：异步编程：`async/await`、`asyncio` 并发、Semaphore 限并发
- Day3：SSE 流式输出：`StreamingResponse`，把 LLM token 流推到 HTTP 响应
- Day4：把 `07_agent/2-tool-agent/01_tool_agent` 包装成 FastAPI 服务（`/chat` 接口）
- Day5：错误处理与日志：统一异常、错误分类、结构化日志
- **产出物**：`agent_server/`——可调用的 Agent HTTP 服务，支持 SSE 流式
- **检查点**：curl 可调用并流式返回；异常请求返回规范错误码

### Week 2：生产级后端 II —— 高并发与部署（P0，JD 要求 2 / 职责 3）

- Day1：Redis 基础 + 会话状态缓存
- Day2：任务队列（arq 或 Celery）：长任务异步化
- Day3：重试 / 超时 / 熔断 / 限流的实现模式
- Day4：Docker 化：Dockerfile + docker-compose（app + redis）
- Day5：压测与调优：locust 压测，输出 QPS / 延迟报告
- **产出物**：`rag_mvp` 服务化升级 + 一键启动的 compose 文件 + 压测报告
- **检查点**：compose 一键拉起；100 并发下错误率 < 1%，报告中有量化数据

### Week 3：评测体系 I —— Golden Dataset 与评测脚本（P0，JD 职责 4）

- Day1：Agent 评测方法论：离线/在线评测、指标分层（任务成功率/工具错误率/延迟/成本）
- Day2：为 `rag_mvp` 构建 Golden Dataset（≥ 50 条，含答案与引用）
- Day3：实现评测脚本：批量执行 + 指标统计 + 结果落盘
- Day4：接入 LangSmith 或 Langfuse，跑通 Trace 采集
- Day5：设计评测报告模板（指标 + 失败 case 列表）
- **产出物**：`eval/` 目录（golden set + 评测脚本 + 报告模板）
- **检查点**：一条命令跑完全量评测并生成报告

### Week 4：评测体系 II —— LLM-as-Judge、RAGAS 与根因归因（P0，JD 职责 2、4）

- Day1：LLM-as-Judge：judge prompt 设计、评分一致性验证
- Day2：RAGAS 上手：faithfulness / answer relevancy / context precision
- Day3：回归门禁：评测接入 CI，指标劣化即报警
- Day4：根因归因方法：从 trace 定位问题属于检索/提示词/工具/模型哪一层
- Day5：写 3 个「现象 → 根因 → 量化改进」完整案例
- **产出物**：CI 回归评测 + `cases/` 归因案例集（3 篇）
- **检查点**：案例可 10 分钟讲清，含改进前后指标对比

### Week 5：高级 RAG I —— 混合检索与重排（P1，JD 职责 3）

- Day1：稀疏检索（BM25）与稠密检索的原理与局限
- Day2：混合检索 + RRF 融合，接入 `rag_mvp`
- Day3：重排：bge-reranker 接入与延迟控制
- Day4：Query 改写：multi-query、HyDE
- Day5：用 W3 的评测集做改造前后对比实验
- **产出物**：`rag_mvp` 检索链路升级 + 对比实验报告
- **检查点**：检索相关指标有量化提升（目标 +10% 以上）

### Week 6：高级 RAG II —— 溯源、增量更新与检索评估（P1，JD 职责 3）

- Day1：引用溯源：chunk 元数据设计与答案引用
- Day2：增量索引与文档版本/删除同步
- Day3：检索侧评估：Recall@K、MRR，离线检索 benchmark
- Day4：Metadata 过滤与权限感知检索（ACL-aware）概念与最小实现
- Day5：整理 RAG 调优清单（参数/策略/适用场景）
- **产出物**：支持引用溯源与增量更新的 `rag_mvp` + 调优清单
- **检查点**：demo 回答可展示出处；新增文档无需全量重建索引

### Week 7：Context Engineering I —— 上下文治理（P1，JD 职责 3）

- Day1：Context Engineering 全景：system prompt 组织、上下文预算分配
- Day2：上下文压缩：对话摘要、tool result 截断与摘要
- Day3：Token 预算与成本统计埋点
- Day4：长上下文退化问题（lost in the middle）与缓解策略
- Day5：在 `agent_server` 中实现上下文管理模块
- **产出物**：上下文管理模块 + 长对话成本对比实验
- **检查点**：20 轮以上对话不爆上下文，token 成本下降可量化

### Week 8：Context Engineering II —— 记忆与模型路由（P1，JD 职责 3）

- Day1：长期记忆：记忆提取、写入、检索、遗忘策略
- Day2：用户画像记忆的最小实现（基于 Redis/向量库）
- Day3：模型路由：按任务复杂度分流不同模型
- Day4：Fallback 与语义缓存
- Day5：整合演练：记忆 + 路由 + 缓存接入 `agent_server`
- **产出物**：记忆模块 + 模型路由层
- **检查点**：模拟主模型故障可自动切换；重复请求命中缓存

### Week 9：Agent 安全与真实工具接入（P1，roadmap P0-2/P0-3）

- Day1：Prompt Injection 攻击手法与防御（输入隔离、指令分层）
- Day2：最小权限与工具白名单、敏感操作二次确认
- Day3：实现安全中间件：注入检测 + 工具调用审计日志
- Day4：MCP 进阶：把网页搜索/抓取封装为 MCP 工具（供 W10 复用）
- Day5：写安全 checklist，做红队自测
- **产出物**：安全中间件 + 搜索/抓取 MCP 工具 + checklist
- **检查点**：10 条注入测试用例全部拦截；工具调用有审计日志

### Week 10：增长营销 Agent I —— 深度调研（P0，JD 职责 1 / 优先条件 1）

- Day1：Deep Research 模式拆解：规划 → 搜索 → 抓取 → 综合 → 引用报告
- Day2：调研规划器：问题拆解与子任务生成
- Day3：搜索/抓取循环：接入 W9 的 MCP 工具，去重与质量过滤
- Day4：综合生成：带引用的结构化调研报告（Markdown）
- Day5：端到端联调 + 用 W4 的评测框架评估报告质量
- **产出物**：`deep_research_agent` 模块 + 一份完整调研报告样例
- **检查点**：给定主题 10 分钟内产出带 ≥ 10 条引用的调研报告

### Week 11：增长营销 Agent II —— SEO 与社媒内容（P0，JD 职责 1 / 优先条件 2）

- Day1：SEO 基础：关键词调研、搜索意图、TDK、结构化数据
- Day2：SEO 内容 pipeline：关键词 → 大纲 → 成文 → meta 生成
- Day3：社媒文案：多平台（小红书/TikTok/Instagram/X）风格化改写
- Day4：多语言内容生成与本地化注意事项
- Day5：与 W10 调研模块串联：调研 → SEO 文章 → 社媒分发文案，端到端 demo
- **产出物**：`growth_marketing_agent`——调研 + SEO + 社媒三场景端到端
- **检查点**：一条命令从主题到 SEO 文章 + 3 平台文案，产出可演示

### Week 12：全栈收尾与面试沉淀（P2，JD 优先条件 3）

- Day1：React/Next.js 快速上手（只学到能改模板）
- Day2：给 Agent 做极简前端：对话框 + SSE 流式渲染
- Day3：增长指标常识：AARRR、CAC/LTV、A/B 测试，能聊业务
- Day4：打磨两个作品的 README（架构图、指标、评测结果、demo 截图）
- Day5：模拟面试：按 JD 五条职责逐条准备 10 分钟讲述
- **产出物**：可演示的产品界面 + 面试 case 集（评测归因 3 篇 + 项目讲稿）
- **检查点**：两个作品可现场 demo；能对着 JD 逐条讲清对应经验