# 前沿笔记（W12 D4）

> 固定四节 + 五层归属表。不是资料堆砌，是能被追问的判断。

## 1. 采样与推理

**prefill 逐 decode 的成本差异**：prefill 是一次性的（把 prompt 变成 KV cache），decode 是逐 token 展开。为什么长上下文贵——prefill 成本与序列长度成正比，decode 成本与序列长度也成正比（每步要读全部 KV cache），但 prefill 能并行、decode 必须串行。所以"多 10K token 输入"在 prefill 层只是多 10K 次矩阵乘（并行），在 decode 层却是每生成一个 token 都多读 10K 行 KV cache（串行）。这就是为什么长 context + 长输出的成本是超线性增长。

**kvcache 与本项目的关系**：我们的 `Usage.input_tokens` 是厂商回报的，当厂商支持 `cache_read`（如 Claude prompt caching），预算里的"输入成本"会降 5-10x。W11 的 `compute_cost` 已经读 `pricing.cache_read`，但成本看板还没把它单独列出来——这是未落地的缺口。

**temperature / top_p 我们怎么用的**：draft 生成用 `temperature=0.7`（多样性要），SEO 分析用 `temperature=0.3`（确定性要），记忆提取用 `temperature=0`（可复现）。`top_p` 不与 `temperature` 同时调（缺少消融实验不知道叠加效果），跟 OpenAI 官方建议一致。

## 2. 五层技术体系

PE / CE / HE / LE / GE 是递进关系（每一层在前一层之上加约束），不是取代——PE 是基础，CE 在 PE 之上做预算/trust，HE 在 CE 之上做工具治理，LE 在 HE 之上做循环控制，GE 在 LE 之上做多 Agent 图。

不是所有项目都需要五层全做。我们的分层见下方归属表。

### 五层归属表

| 层 | 全称 | 我们的对应物 | 证据 |
|---|---|---|---|
| **PE** | Prompt Engineering | W1 prompt 模板、W10 品牌调性约束 | `prompts/`、`docs/w10/` |
| **CE** | Context Engineering | W7 ContextItem/预算分配/Debug Report、W5-W6 检索 | `context/`、`retrieval/`、`docs/w7/` |
| **HE** | Harness Engineering | W8 Skill/工具治理、W9 planner、W11 执行模型 | `skills/`、`planner/`、`runtime/` |
| **LE** | Loop Engineering | W9 重规划 + 三道上限、W11 断点续跑 | `planner/controls.py`、`runtime/resume.py` |
| **GE** | Graph Engineering | W9 多 Agent 图 + handoff、W11 Run Timeline | `multi_agent/` |

## 3. 人与 AI 的关系

我们做了四种关系中的前三种：

- **工具**（用户写 brief，Agent 生成文案，用户审核后发布）——W8/W10 的默认形态
- **依附**（Agent 做调研，用户看结论下判断）——W9 调研模式
- **导学**（Agent 提出 hypothesis，用户修正方向）——W11 planner 的 plan-review 循环
- **内观**——未做。风险等级最高，我们不说 Agent 自己决定做什么。

**什么风险等级留人在环**：publish（发布到平台）必须有人工确认；记忆写入必须有人确认（W7 confirmed_by_user）；工具执行（W8 sandbox + visibility decision）在 research 阶段不需要确认但在 publish 阶段需要。

## 4. Harness 架构对照

| 维度 | 自研 (本项目) | LangGraph 对照 (W9) | DSH / Claude Code |
|---|---|---|---|
| 编排 | Plan-and-Execute / Workflow | Supervisor + State Graph | Workflow + Loop |
| 预算 | per-hop reset + handoff | 无原生预算控制 | 全局 context window |
| 工具治理 | sandbox + visibility + audit | 节点级 tool permission | permission system |
| 续跑 | ResumableRunner + Run Timeline | checkpoint (state-based) | 会话历史 |
| 追溯 | RequestLog + TraceExporter + RunTimeline | LangSmith (外部) | 内置 trace |

关键差异：**budget handoff**。自研的多 Agent 图在每次 handoff 时重置预算上限——LangGraph 的 supervisor 没有这个概念，预算是全局共享的。对于"调研 Agent 可以多花钱，写文案 Agent 要省钱"这种场景，全局预算会把钱花在错误的地方。

## 各方向落地评估

### A2A (Agent-to-Agent)
- 现状：协议草案阶段（Google 2024），无成熟参考实现
- 与本项目关系：W9 多 Agent 图是内部 A2A 的一个特化；跨厂商 A2A 暂无需求
- 是否值得落地：**否**。目标岗位不涉及跨厂商协作
- 成本：2 人天 + 兼容性风险

### AG-UI (Agent-UI Protocol)
- 现状：提案阶段，未标准化
- 与本项目关系：前端单页已用 SSE，部分覆盖
- 是否值得落地：**否**（默认档已够）。产品前端不在目标 JD 核心
- 成本：1 人天

### Computer Use
- 现状：OpenAI/Anthropic 有 API，但通用岗性价比低
- 是否值得落地：**否**。ROI 排序在最后（ROADEMAP §7）
- 成本：学习 0.5 人天 + 实现不计入
