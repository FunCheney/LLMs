# 课程大纲 × 12 周计划 · 对齐评审

> 输入：
>
> 1. 《AI Agent 全栈工程师训练营》课程大纲（7 章，28 页）
> 2. 极客时间《企业级 Agent 工程架构实战营》课程大纲（10 篇，29 页）
>
> 基准：`ROADEMAP.md` v2（12 周单项目贯穿版）+ `docs/wN/wN-tasks.md`
>
> 评审日期：2026-11-04（W4 已合流，W5 未开工）

---

## 0. 结论先行

| 判断                  | 内容                                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| **总覆盖**             | 102 个可执行条目：**49 已覆盖（48%）/ 15 部分（15%）/ 30 真缺口（29%）/ 4 明确不做 / 4 只记笔记**；另有 1 条章节级判断（软件工厂/Codebase Agent，见 §5）      |
| **真缺口值得进计划的只有 8 条** | 毛增 19.5h，但 15.5h 属于"把已有 Day 的实现方式换掉"，**净增 ≈ 4h**                                                                |
| **最大结构性风险**         | 不是漏内容，是 **Trajectory（过程）评测完全空缺**——W9 的单/多 Agent 对比实验目前只有结果指标，"多 Agent 好在哪"讲不深                                   |
| **不要被大纲带偏**         | doc1 的作品集是「Tool Runtime + Codebase Agent + Eval Platform」三件套，正是 ROADEMAP §1 问题 #2 要修掉的"双资产/多半成品"。**搬工程对象，不搬项目** |
| **我们已经超纲的部分**       | judge 一致性 kappa、RAGAS 口径、门禁相对劣化+绝对下限、四层归因证据规则——两份大纲都只到"知道要做"，没到"怎么证明它可信"                                        |
| **P3 决策不变**         | 吸收后余量约 11h < 15h 阈值，**仍不解锁深入档**                                                                                 |

---

## 1. 两份大纲的定位差异（决定哪些能抄）

|      | doc1 全栈训练营           | doc2 架构实战营                      | 本计划 v2                 |
| ---- | -------------------- | ------------------------------- | ---------------------- |
| 目标人群 | 传统后端 → Agent 开发      | 已有 Agent 经验 → 架构师               | 同 doc1，但要求可验证证据        |
| 组织方式 | 7 章 / 能力广度优先         | 10 篇 / 架构深度优先                   | 单产品 12 周增量             |
| 作品   | 平台 + 3 个作品集项目        | 1 个 DataAgent（多租户 BI）           | 1 个 growth-agent（内容生产） |
| 评测占比 | 1 章（约 14%）           | 1 篇（约 10%）                      | **4 周（33%）**           |
| 适合抄的 | 平台能力清单、工具治理、Skill 六层 | Runtime 对象模型、Context 治理、EvalOps | —                      |

**共同点**：三者的场景（代码库 / 零售 BI / 内容生产）完全不同，但**工程对象完全一致**——Runtime、Context、Tool、Memory、Eval 五件套。所以对齐时按"工程对象"比，不按"业务场景"比。

---

## 2. 逐条对齐（判定口径：✅已覆盖 / ◐部分 / ✗真缺口 / ○不做 / 📝只记笔记）

### A. LLM 底座与 Prompt（doc1 ch1、doc2 一篇 5）

| 条目                                   | 来源       | 我们做在哪                     | 判定 |
| ------------------------------------ | -------- | ------------------------- | -- |
| 多厂商 API 差异与适配                        | doc1 1.2 | W1 `llm/providers/`       | ✅  |
| Streaming / SSE / 中断取消               | doc1 1.3 | W1 `POST /v1/chat` stream | ✅  |
| Structured Output + Schema 校验 + 纠错重试 | doc1 1.5 | W1 结构化输出                  | ✅  |
| Prompt 模板                            | doc1 1.4 | W1 gateway 层模板            | ✅  |
| Prompt 版本治理（回滚点 / 适用场景 / 评测结果）       | doc2 六 5 | —                         | ✗  |
| Prompt 注入防护                          | doc1 1.4 | W10 D3 红队 10 条            | ✅  |
| LLM Gateway（统一入口 / 成本埋点）             | doc1 1.6 | W1                        | ✅  |
| 逻辑模型 vs 物理模型分层 + 能力契约路由              | doc2 四 2 | W1 tier 路由（cheap/strong）  | ◐  |
| Logical Call / Attempt 执行模型          | doc2 四 3 | W1 只有重试，无统一执行模型           | ✗  |
| 采样机制 / prefill-decode / kvcache 认知   | doc2 一 1 | —                         | 📝 |
| PE/CE/HE/LE/GE 五层技术体系                | doc2 一 5 | —                         | 📝 |

### B. 工具、MCP、Skill（doc1 ch2、doc2 七篇）

| 条目                                       | 来源         | 我们做在哪                         | 判定 |
| ---------------------------------------- | ---------- | ----------------------------- | -- |
| Function Calling 双向归一化                   | doc1 2.1   | W1 D4                         | ✅  |
| 参数校验（JSON Schema / Pydantic）             | doc1 2.3   | W1                            | ✅  |
| 工具调用审计 + Trace                           | doc1 2.3   | W2 trace + W8 guard           | ✅  |
| 超时 / 重试 / 降级 / 失败恢复                      | doc1 2.3   | W2 resilience                 | ✅  |
| Tool Registry（元数据 / 分类 / 版本 / 依赖 / 启停）   | doc1 2.2   | W8 有 `ToolSpec`，无 registry 治理 | ◐  |
| 工具权限分级（只读 / 写 / 高风险 / 需确认）               | doc1 2.3   | W8 D5 白名单                     | ◐  |
| Tool Contract 三层（模型理解 / Runtime 执行 / 治理） | doc2 七 2   | —                             | ✗  |
| 最小模型可见面 / 动态最小工具集                        | doc2 七 1,2 | —                             | ✗  |
| Action Runtime（提议→校验→授权→审批→执行→验真）        | doc2 七 3   | W10 D4 只有 HITL 审批门            | ✗  |
| 幂等 / 最小权限 / 短期凭据                         | doc2 七 3   | W11 D3 幂等                     | ◐  |
| MCP Client / Server 协议                   | doc1 2.4   | W8 D1–D2                      | ✅  |
| MCP 生产化（鉴权 / 租户隔离 / 限流）                  | doc2 七 5   | —                             | ○  |
| Skill 规范 + 加载器                           | doc1 5.3   | W8 D3–D4                      | ✅  |
| Skill 检索 / 匹配 / 渐进式加载 / Validator        | doc2 七 4   | —                             | ✗  |
| Shell / SQL 风险控制 + Sandbox 边界            | doc1 3.4   | —                             | ✗  |

### C. Loop、Runtime、状态（doc1 ch3、doc2 五篇）

| 条目                                                                                       | 来源       | 我们做在哪       | 判定 |
| ---------------------------------------------------------------------------------------- | -------- | ----------- | -- |
| ReAct / Plan-and-Execute / Reflection                                                    | doc1 3.1 | W9 D1       | ✅  |
| 死循环识别 / 最大循环 / 终止条件                                                                      | doc1 3.1 | W9 D1       | ✅  |
| LangGraph 对照 + 框架选型                                                                      | doc1 3.6 | W9 D4       | ✅  |
| Workflow 六模式（Chaining/Routing/Parallel/Orchestrator-Workers/Evaluator-optimizer/Agentic） | doc2 一 2 | —           | ✗  |
| 何时不该用 Agent（Agent/Workflow/RAG/Chatbot/脚本边界）                                             | doc1 7.4 | —           | ✗  |
| 控制权矩阵 + Autonomy Budget                                                                  | doc2 五 1 | —           | ✗  |
| 显式任务状态机（9 态含 waiting_approval / tool_failed）                                             | doc1 3.3 | —           | ✗  |
| Run / Step / Attempt / Event 统一执行模型                                                      | doc2 五 3 | —           | ✗  |
| Checkpoint / Resume                                                                      | doc1 3.3 | W11 D3 断点续跑 | ◐  |
| Run Timeline（解释一次故障）                                                                     | doc2 五 3 | —           | ✗  |
| Durable Workflow + Human Approval                                                        | doc2 五 4 | W10 D4 部分   | ◐  |
| Agent Harness 抽象（模型+工具+状态+环境的统一组织）                                                       | doc1 3.5 | —           | ✗  |
| 分布式 Runtime（Scheduler/Worker/Lease/Heartbeat/多租户）                                        | doc2 五 5 | —           | ○  |

### D. Context、Memory、RAG（doc1 ch4、doc2 六/八篇）

| 条目                                         | 来源       | 我们做在哪                     | 判定 |
| ------------------------------------------ | -------- | ------------------------- | -- |
| Context 预算分配 / 压缩 / 优先序                    | doc1 4.1 | W7 D1 `context/budget.py` | ✅  |
| 工作记忆 / Scratchpad                          | doc1 4.2 | W7 D2                     | ✅  |
| 长期记忆四阶段（提取/写入/检索/遗忘）                       | doc1 4.3 | W7 D3                     | ✅  |
| Context Schema（来源/版本/权限/可信度/生命周期）          | doc2 六 1 | —                         | ✗  |
| Context Debug Report（为什么这条没进上下文）           | doc1 4.1 | —                         | ✗  |
| Context 故障分类 + Context Eval 五指标            | doc2 六 6 | —                         | ✗  |
| Context Drift 检测（目标偏移/事实失真/重复）             | doc2 六 5 | —                         | ✗  |
| 记忆写入幂等 / 冲突 / 版本失效                         | doc2 八 2 | —                         | ✗  |
| Memory Poisoning 防护 + 删除传播                 | doc2 八 4 | —                         | ✗  |
| 记忆三档对照实验（Off / Blind Top-K / Governed）     | doc2 八 4 | —                         | ✗  |
| Chunking（固定/段落/标题/语义/AST/overlap/metadata） | doc1 4.4 | W5 D1                     | ✅  |
| 混合检索 BM25 + 向量 + RRF                       | doc1 4.4 | W5 D2                     | ✅  |
| Rerank（cross-encoder / LLM）                | doc1 4.4 | W5 D3                     | ✅  |
| Query 改写 / multi-query / HyDE              | doc1 4.4 | W5 D4                     | ✅  |
| 引用溯源 + chunk 元数据                           | doc1 4.6 | W6 D1                     | ✅  |
| 增量索引 + 版本/删除同步                             | doc1 4.4 | W6 D2                     | ✅  |
| ACL-aware 检索                               | doc2 六 2 | W6 D3                     | ✅  |
| 知识资产治理（质量/过期/冲突/来源追踪）                      | doc1 4.5 | W6 只有版本与删除                | ✗  |
| Agentic Search（围绕信息缺口多轮检索 + 预算）            | doc2 六 4 | —                         | ✗  |
| Claim-Evidence 证据包                         | doc2 六 4 | W6 D1 引用链                 | ◐  |
| RAG 评估（召回/命中/正确/幻觉率）                       | doc1 4.4 | W5 D5 + W4 RAGAS          | ✅  |
| 语义层 Metric Registry（指标口径统一）                | doc2 六 2 | —                         | ○  |

### E. 多 Agent（doc1 ch5、doc2 五篇）

| 条目                                           | 来源              | 我们做在哪    | 判定 |
| -------------------------------------------- | --------------- | -------- | -- |
| Supervisor / Planner-Executor / Reviewer 三范式 | doc1 5.1        | W9 D3    | ✅  |
| Subagent 委托契约（目标/上下文/工具/权限/超时/评价）            | doc1 5.2        | W9 D3 部分 | ◐  |
| 并行串行 + 隔离 + 预算                               | doc1 5.1        | W9 D3    | ✅  |
| 结果聚合 + 冲突仲裁                                  | doc1 5.2        | W9 D3    | ✅  |
| 分层复核（validator）                              | doc2 六          | W9 D3    | ✅  |
| A2A 协议                                       | doc1 5.1 / doc2 | W9 D4    | ✅  |

### F. 评测（我们的强项，也是最大缺口所在）

| 条目                                             | 来源           | 我们做在哪                                  | 判定    |
| ---------------------------------------------- | ------------ | -------------------------------------- | ----- |
| Golden Dataset                                 | doc1 5.4     | W3 D2（54 条）                            | ✅     |
| Outcome 层指标                                    | doc2 九 1     | W3 metrics                             | ✅     |
| System 层指标                                     | doc2 九 1     | W3 延迟/成本/token                         | ✅     |
| **Trajectory 层指标（工具选择/参数/步数/路径）**              | **doc2 九 1** | **—**                                  | **✗** |
| Multi-Grader（Code / Environment / Trace / LLM） | doc2 九 3     | 只有 rule + LLM judge                    | ✗     |
| LLM-as-Judge + 一致性校验                           | doc1 5.4     | W4（超出两份大纲）                             | ✅     |
| 人工标注规范                                         | doc1 5.4     | W4 24 条，labeler 待复核                    | ◐     |
| Grader 校准与版本治理                                 | doc2 九 3     | —                                      | ✗     |
| Multi-Trial 稳定性（同 case 跑 N 次看方差）               | doc2 九 3     | —                                      | ✗     |
| Fixture / 环境快照 / 版本元组                          | doc2 九 2     | W4 `model_snapshot` + `catalog_sha256` | ◐     |
| 评测集分层治理（防调优污染测试集）                              | doc2 九 2     | W3 有难度分层                               | ◐     |
| Failure Taxonomy                               | doc1 5.5     | W4 四层归因                                | ✅     |
| 根因：Replay / Fork / Ablation                    | doc2 九 4     | W4 只有规则归因，无对照实验                        | ✗     |
| 回归门禁 + 基线                                      | doc2 九 5     | W4                                     | ✅     |
| Baseline vs Candidate 公平比较                     | doc2 九 5     | W4                                     | ✅     |
| Shadow / Canary / Kill Switch                  | doc2 九 5     | —                                      | ✗     |
| 失败样本聚类                                         | doc1 5.6     | W3 规则分类                                | ◐     |
| 软评分与硬门槛分离                                      | doc2 九 1     | W4 required / optional                 | ✅     |

### G. 生产化（doc1 ch6、doc2）

| 条目                                       | 来源           | 我们做在哪         | 判定 |
| ---------------------------------------- | ------------ | ------------- | -- |
| Docker / compose 一键部署                    | doc1 6.7     | W2 D4         | ✅  |
| 压测基线                                     | doc1 6.2     | W2 D4 locust  | ✅  |
| SLO + 告警                                 | doc1 6.4     | W11 D1        | ✅  |
| 混沌 / 故障注入                                | doc1 6.4     | W11 D2        | ✅  |
| 幂等                                       | doc2 五 4     | W11 D3        | ✅  |
| 灰度发布 + 回滚                                | doc1 6.5     | W11 D3 演练级    | ◐  |
| 资产版本治理（Prompt/Model/Tool/RAG/Skill 统一版本） | doc2 六 5     | —             | ✗  |
| Replay 回放调试（存失败任务的输入+上下文+状态）             | doc1 6.3     | —             | ✗  |
| 成本治理（per-task 预算 + 归因看板）                 | doc1 6.2     | W11 D5 部分     | ◐  |
| Trace 全链路                                | doc1 6.3     | W2 D3 + W3 D4 | ✅  |
| Runbook                                  | doc1 6.7     | W10 D5        | ✅  |
| SDK / 服务化封装                              | doc1 6.7     | W12 D1        | ✅  |
| 私有化部署 / vLLM / TGI / 多租户 SaaS            | doc1 6.1,6.6 | —             | ○  |

### H. 方法论与表达（doc1 ch7、doc2 一/十篇）

| 条目                                | 来源             | 我们做在哪             | 判定 |
| --------------------------------- | -------------- | ----------------- | -- |
| 能力模型 L1–L4 / JD 十条职责              | doc2 一 3       | W12 D5            | ✅  |
| 作品集表达（怎么写进简历）                     | doc1 7.5       | W12 D5            | ✅  |
| Decision Matrix / M0 基线 + 里程碑 ADR | doc2 三         | W4 baseline + ADR | ◐  |
| AI Native 思维 / 人机责任边界             | doc2 一 4 / 十 2 | —                 | 📝 |
| DSH vs Claude Code 架构对照           | doc2 十 4       | —                 | 📝 |

**统计（脚本核对，按表格最后列数出来的）**：✅ 49 · ◐ 15 · ✗ 30 · ○ 4 · 📝 4 = 102 行。

按块看缺口密度，能看出问题集中在哪里：

| 块                    | 行数 | ✅  | ✗ | 缺口率     |
| -------------------- | -- | -- | - | ------- |
| E 多 Agent            | 6  | 5  | 0 | 0%      |
| D Context/Memory/RAG | 22 | 11 | 9 | 41%     |
| B 工具/MCP/Skill       | 15 | 6  | 5 | 33%     |
| F 评测                 | 17 | 8  | 5 | 29%     |
| C Loop/Runtime/状态    | 13 | 3  | 7 | **54%** |
| G 生产化                | 13 | 8  | 2 | 15%     |
| A LLM 底座             | 11 | 6  | 2 | 18%     |
| H 方法论                | 5  | 2  | 0 | 0%      |

**两个结论**：① **C 块（Runtime/状态）缺口率最高**——我们做了队列和重试，但没有"统一的执行对象模型"，这是 W11 要补的；② **F 块那 5 条缺口里有 3 条是同一件事**（Trajectory / Multi-Grader / Multi-Trial 都在回答"过程怎么测"），说明评测的缺口不是一个面，而是一个点。

---

## 3. 值得进计划的 8 条（按 ROI 排序）

> 关键判断：**其中 6 条是"替换已有 Day 的实现方式"，不是新增任务**。  
> 因为这些 Day 本来就还没开工，验收标准抬高的边际成本远小于新增一个 Day。

| # | 缺口                                                                  | 挂载点                  | 毛成本       | 净成本     | 验收判据                                                                            |
| - | ------------------------------------------------------------------- | -------------------- | --------- | ------- | ------------------------------------------------------------------------------- |
| 1 | **Trajectory 层评测 + Trace Grader + Multi-Trial**                     | W9 D5（扩写）+ W4 回填一个脚本 | 3h        | **+2h** | 单/多 Agent 对比同时给出 outcome/trajectory/system 三层数字；工具选择准确率、无效调用率、同 case 跑 3 次的方差   |
| 2 | 统一执行模型 Run/Step/Attempt + 9 态状态机 + Checkpoint/Resume + Run Timeline | W11 D3（换实现）          | 3h        | 0       | `kill -9` 后 resume 成功率 100%；一次故障能用 Run Timeline 讲到"第几步、哪个工具、为什么重试"              |
| 3 | 编排范式选型（Workflow 六模式 + 何时不该用 Agent + 控制权矩阵 + Autonomy Budget）        | W9 D1（换实现）           | 2h        | 0       | 一张选型决策表（任务特征 → 范式）+ W9 D5 的数据支撑其中两格；有 `max_steps / max_cost / max_seconds` 三道上限 |
| 4 | Context 治理四件套（Schema + Debug Report + 故障分类/Eval + Drift）            | W7 D1 + D4（换实现）      | 2.5h      | 0       | 能对任意一次评测回答"这条为什么没进上下文"（6 类原因之一）；Context Eval 五指标有读数                             |
| 5 | 工具治理（Tool Contract 三层 + Action Runtime + 最小可见面 + Skill Validator）   | W8 D3 + D5（换实现）      | 3h        | 0       | 三个 badcase 全拦住：重复请求、参数篡改、接口假成功；每个 role 可见工具数 < 全集的 60%                          |
| 6 | Sandbox 边界（路径/命令/网络白名单 + 超时，本地级不做容器）                                | W8 D4（+0.5h）         | 1.5h      | +0.5h   | 越界路径写入被拒；`rm -rf /` 类命令被拦；超时进程被回收，且三类拒绝都进审计日志                                   |
| 7 | 记忆治理轻量版（写入幂等 + Poisoning 校验 + 删除传播 + 两档对照实验）                        | W7 D3 + D5           | 2h        | +1.5h   | Off vs Governed 两档跑同一批 case，给出成功率/成本/延迟三列；恶意文档注入的假事实不进长期记忆                      |
| 8 | 受控进化（Replay/Fork/Ablation + 资产版本清单 + Feature Flag 分流 + Kill Switch） | W11 D4 + D5（换实现）     | 3h        | 0       | 用 Fork 证明"改 A 不是改 B"的对照实验至少 1 次；`FEATURE_X=off` 能一键回到旧行为                        |
|   |                                                                     | **合计**               | **19.5h** | **+4h** |                                                                                 |

**净增 4h 的算法**：19.5h − 15.5h（原本就要花在这些 Day 上的时间）= 4h。真正新增工作量的只有 #1（多一个评测脚本）和 #6/#7（多跑两轮 eval）。

---

## 4. 只写笔记、不进计划的 4 条（合计 2.5h，全在 W12 D4）

| 条目                              | 为什么不做成工程                                     | 笔记落点                       |
| ------------------------------- | -------------------------------------------- | -------------------------- |
| 采样机制 / prefill-decode / kvcache | 属于推理侧，Agent 工程岗问"为什么长上下文贵"就够了                | W12 D4 一节 + W7 成本实验的脚注     |
| PE/CE/HE/LE/GE 五层体系             | 是给技术命名的框架，不是新工程对象；但**面试时能把我们的东西挂到这五层上，是加分项** | W12 D4 一张对照表（我们的模块 → 五层归属） |
| AI Native / 人机责任边界              | 组织话题，12 周内无可验证产出                             | W12 D4 一节                  |
| DSH vs Claude Code 架构对照         | 值得读，但不值得 12 周内复刻                             | W12 D4 一节                  |

---

## 5. 明确不吸收的 5 条（4 条条目级 + 1 条章节级，写清理由避免下次再纠结）

| 条目 | 不吸收的理由 |
| -- | ------ |


| 分布式 Runtime（Scheduler/Worker/Lease/Heartbeat/多租户资源治理） | doc2 用 1 整篇讲，至少 2 周量。我们 W2 的 Semaphore + 队列已满足 DoD"100 并发错误率 <1%"。塞进去只会挤掉 W5 的 RAG 主线 |  
| 多租户 SaaS 隔离（RBAC / 行列级权限 / 租户成本归集） | ROADEMAP 适用范围已划界："目标是生产级工程质量，不是生产级规模"。加多租户会让每个模块多一层维度，收益在面试里几乎问不到 |  
| 私有化部署 / vLLM / TGI / Triton | 与 Agent 工程岗核心关联弱；问到时用"云 API + 混合部署的取舍"回答即可（W12 D4 笔记覆盖） |  
| 软件工厂 / Codebase Agent | 与 capstone（内容生产）是两个作品方向。做了就是第二个半成品——**这正是 v2 要修的问题 #2** |  
| 语义层 Metric Registry（GMV/退款率指标口径） | 是 BI 域的核心，不是 Agent 工程的核心。我们 capstone 是内容生产，没有"指标口径"问题 |

---

## 6. 三处结构性修正（不只是加内容）

### 6.1 W9 的主题从"多智能体"扩成"编排选型 + 多智能体"

现在的 W9 是"实现 Plan-and-Execute → 做多 Agent → 对比"。缺了前置判断：**什么时候根本不该上 Agent**。  
doc1 ch7 和 doc2 一 2 都把"Agent/Workflow/RAG/Chatbot/脚本的边界"当独立知识点，因为**过度 Agent 化是面试官最常追问的坑**。  
改法：W9 D1 前置一张选型决策表，D5 的对比实验从"单 vs 多"扩成"单 Loop / 固定 Workflow / 多 Agent"三格。

### 6.2 W11 的主题从"生产硬化"扩成"生产硬化 + 受控进化"

现在 W11 有 SLO、混沌、幂等、灰度演练，是"抗故障"。但 doc2 九 5 讲的是另一半：**改坏了怎么证明、怎么只影响 1% 用户、怎么一键退回**。  
改法：D3 用统一执行模型重写（顺带拿到 Run Timeline），D4 加 Fork/Ablation 对照实验，D5 加 Feature Flag 分流 + Kill Switch。

### 6.3 W12 D4 的前沿笔记要有明确结构，别变成"读了几篇文章"

现在只写"前沿调研笔记"。建议固定成 4 节 + 1 张表：

- 4 节 = 采样/推理、五层技术体系（PE/CE/HE/LE/GE）、人机关系、DSH vs Claude Code
- 1 张表 = **我们的模块 → 五层归属**（能在面试里 30 秒说清"我在哪一层做了什么"）

---

## 7. 改后的 12 周表（★ = 本次变更）

| 周     | 主题                           | 变更                                                                                      |
| ----- | ---------------------------- | --------------------------------------------------------------------------------------- |
| W1–W4 | LLM 底座 / 运行底座 / 评测 I / 评测 II | 无变更（已完成）                                                                                |
| W5    | 生产 RAG I                     | 无变更（正在做，不动）                                                                             |
| W6    | 生产 RAG II                    | D5 的"调优清单"补 3 条：知识质量评分、过期检测、冲突文档处理（+0.5h）                                               |
| W7    | 上下文工程与记忆                     | ★ D1 加 Context Schema + Debug Report；D3 加写入幂等与 Poisoning 校验；D4 加 Drift 检测；D5 改成两档记忆对照实验 |
| W8    | Skills 与 MCP                 | ★ D3 加 Tool Contract 三层 + Skill Validator；D4 加 Sandbox 边界；D5 升级为 Action Runtime         |
| W9    | 编排与多智能体                      | ★ D1 前置范式选型表 + 控制权矩阵；D2 标注为 Agentic Search（加检索预算）；D5 扩成三层评测对比                           |
| W10   | 内容生成 + 安全 + 上线               | 无变更                                                                                     |
| W11   | 生产硬化与受控进化                    | ★ D3 用统一执行模型重写；D4 加 Fork/Ablation；D5 加 Feature Flag + Kill Switch                       |
| W12   | 框架提取 / 文档 / 面试               | ★ D4 固定为 4 节 + 1 张五层对照表；D3 补一张 Decision Matrix                                          |

---

## 8. 预算核算

| 项            | 数值                        |
| ------------ | ------------------------- |
| 当前余量（W4 末评估） | < 15h                     |
| 本次净增         | +4h                       |
| 吸收后余量        | ≈ 11h                     |
| P3 解锁阈值      | 15h                       |
| **决策**       | **仍不解锁任何深入档**；W8 末决策点再看一次 |

---

## 9. 下一步

1. 确认上面 8 条是否全要（也可只取 #1 #2 #5，这三条对面试叙事影响最大）
2. 确认后我按顺序改 5 份文件：`w6`、`w7`、`w8`、`w9`、`w11`（每份只动 2–4 个 Day 的"你要做的事 / 自检 / 验收判据"）
3. `ROADEMAP.md` 第五节同步加 ★ 标注
4. W12 D4 的 4 节笔记结构写进 `w12-tasks.md`
