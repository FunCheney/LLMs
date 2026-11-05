# W9 执行清单（规划与多智能体 · P0，稀缺能力 ③）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> 本周是三大稀缺能力之一「多智能体协调」的落地周，**直接长在 capstone 的调研链路上**（v2 计划的核心变化，避免"只能讲概念"）。
> ROADEMAP §1 问题 4：多智能体仅 5 天且与场景脱节 → v2 对策：**产出单/多 Agent 对比数据**。
> ⚠️ 与 W8 的 A2A 轻量深入档（P3-3）联动：若 W8 末已解锁，本周顺势做跨进程 handoff demo。

## 承接面：W1–W8 已交付

| 已有 | 位置 | W9 的接缝 |
|---|---|---|
| `ToolCall` 单次发出（ADR-002）+ `SkillExecutor` 执行闭环 | W1/W8 | 规划器把 `ToolCall` 排进步骤，执行交给 Skill 层 |
| `ContextBuilder` + token 预算 | `context/`（W7） | 多 Agent 共享状态要受预算约束，别爆上下文 |
| `SessionStore` + `trace_id` 贯穿 | W2/W3 | handoff 消息串在同一 trace 下 |
| `RetrievalPipeline`（RAG）、`LongTermMemory` | W5–W7 | researcher 用检索、validator 查引用 |
| `make eval` + golden `difficulty=3` 多步案例 | `evals/` | 单/多 Agent 对比的样本来源 |
| `PermissionGuard` 的 `dangerous` 钩子 | W8 | 规划器高风险步骤可触发审批占位 |
| `PerTask` 计量 + `ConcurrencyLimiter`/背压 | W2、W11 前置 | 多 Agent 必须有**预算上限 + 防死循环** |

> W9 检查点（ROADEMAP）：**≥5 步任务完成率显著高于朴素 ReAct**；能讲清"何时才值得拆多 Agent"并用数据支撑。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接符号 | 推进的 DoD |
|---|---|---|---|
| D1 | `planner/{base,react,plan_execute,controls}.py`（范式选型 + 三道上限） | `ToolCall`、`ContextBuilder` | 可靠性（多步、可中止） |
| D2 | `planner/research_planner.py`（**Agentic Search**：缺口驱动 + 停止条件） | RAG、Skill | 质量（调研链路）、成本 |
| D3 | `multi_agent/`（supervisor/researcher/validator + handoff + 共享状态 + 防死循环/预算） | trace、预算 | 稀缺能力③ |
| D4 | `multi_agent/langgraph_variant.py`（对照实现）+ A2A 认知 | 同样的图 | 面试取舍素材 |
| D5 | **三层评测对比**（`evals/scripts/trajectory_eval.py`）+ 稳定性 + 合流 | make eval、golden、trace | 证据包第 6 件 |

**唯一主线**：先实现"能跑多步"的规划器，再叠加多智能体，**每一步都有预算与死循环守卫**，最后用对比数据回答"何时值得拆多 Agent"。**不要为了多 Agent 而多 Agent**——这是本周最想传达的判断力。

> 本周 3 处按《课程大纲对齐评审》§3 抬高验收标准：D1 前置范式选型 + 控制权矩阵、D2 明确为 Agentic Search、
> D5 从"结果对比"扩成"三层对比 + 稳定性"。真正新增工时只有 D5 的 `trajectory_eval.py`（约 2h）。
> 见 `docs/plan-review-course-outline-gap.md`。

---

## Gate 0 · 开工前（30 min）

- [ ] `make eval` 全绿、门禁生效；`difficulty=3` 案例当前成功率记录（对比基线）
- [ ] 复核 W7 的 token 预算分配器（多 Agent 共享状态要复用它）
- [ ] 复核 W8 SkillExecutor 的执行闭环（规划器只编排，不重写执行）
- [ ] 明确 capstone 的"深度调研"链路：问题拆解 → 多源检索 → 交叉验证 → 成文
- [ ] 若 W8 解锁了 A2A 深入档，准备一个跨进程 demo 目标

---

## Day 1（3h）· ReAct vs Plan-and-Execute vs Reflection

**目标**：理解三种范式的取舍，实现 **Plan-and-Execute** 作为主范式（ROADEMAP D1）。

### 你要做的事
- [ ] `planner/base.py`：`Plan`/`PlanStep` 数据结构 + `Planner` 协议
- [ ] `planner/plan_execute.py`：Plan-and-Execute 实现
- [ ] `planner/react.py`：朴素 ReAct 基线（**对齐 W9 检查点的对照组**，必须实现）
- [ ] `planner/controls.py`：**控制权矩阵 + Autonomy Budget**（`max_steps` / `max_cost` / `max_seconds`）
- [ ] `docs/w9/adr-w9-planner-choice.md`：**编排范式选型**（含 Workflow 六模式 + 何时不该用 Agent）
- [ ] 单测：计划生成、步骤状态机、失败重规划、三道上限各触发一次

### 引导框架

```python
# planner/base.py
@dataclass(frozen=True)
class PlanStep:
    id: str
    goal: str
    tool: str | None            # 需要的 Skill/MCP 工具
    depends_on: tuple[str, ...]
    status: str = "pending"     # pending|running|done|failed|skipped

@dataclass
class Plan:
    task: str
    steps: list[PlanStep]
    budget_tokens: int
    def ready_steps(self) -> list[PlanStep]: ...   # 依赖已满足且未执行

class Planner(Protocol):
    async def plan(self, task: str) -> Plan: ...
    async def replan(self, plan: Plan, failed: PlanStep) -> Plan: ...
    async def execute(self, plan: Plan) -> "TaskResult": ...
```

**先做选型，再写代码：六种编排形态与"何时不该用 Agent"**

这一节放在最前面，是因为**写错编排范式的返工成本最高**——一旦按 Agent Loop 写完，再退回 Workflow 等于重写。
doc1 ch7 与 doc2 都把"Agent/Workflow/RAG/Chatbot/脚本的边界"当独立知识点，因为过度 Agent 化是最常见的工程事故：
本来 20 行确定性代码能保证的事，交给模型后变成"通常能行"。

| 形态 | 适合什么 | 我们 capstone 里的落点 | 判据（出现就选它） |
|---|---|---|---|
| 脚本 | 规则明确、输入结构固定 | 关键词清洗、字数裁剪 | 步骤可枚举且无判断 |
| Chatbot | 单轮问答、无需外部事实 | 品牌客服问答 | 不需要工具、不需要记忆 |
| RAG | 一次检索就够 | "这个竞品的主打卖点" | 答案在文档里且一次召回可得 |
| **Workflow（固定图）** | 步骤稳定、有依赖 | SEO 流水线（关键词→大纲→成文→meta） | 步骤顺序几乎不变 |
| Workflow（Routing） | 任务类型可先分类 | 按选题类型选不同写作模板 | 先分派再执行更省 |
| Workflow（Evaluator-Optimizer） | 有明确评分标准、可迭代 | 文案按品牌调性打分重写 1–2 轮 | 能写出评分函数 |
| Workflow（Orchestrator-Workers） | 子任务数量不定 | 多平台改写（平台数可变） | 子任务同构且可并行 |
| **Autonomous（ReAct / Plan-and-Execute）** | 目标明确但路径不定 | 深度调研（搜什么取决于搜到啥） | 下一步依赖上一步结果 |
| 多 Agent Graph | 需要多视角复核/冲突仲裁 | 调研链的 supervisor+validator | 单 Agent 上下文放不下或需要角色隔离 |

**结论（写进 ADR）**：capstone 只有"深度调研"一格真的需要 Autonomous，其余四格都是 Workflow。
所以本周做 Plan-and-Execute，但**不把整条链路都改成 Agent**——W10 的 SEO 流水线仍用固定 Workflow。

```python
# planner/controls.py —— 把"自主性"变成上限，而不是口号
@dataclass(frozen=True)
class AutonomyBudget:
    max_steps: int          # 最多执行多少步（不是轮次）
    max_cost: Decimal       # 单任务成本上限，超了中止并返回部分结果
    max_seconds: float      # 墙钟上限
    max_replans: int = 2

# 控制权矩阵：哪些决定必须由代码/人做，不能交给模型
CONTROL_MATRIX: dict[str, Literal["model", "code", "human"]] = {
    "下一步做什么":       "model",
    "调用哪个工具":       "model",
    "工具参数":           "model",
    "工具是否可见":       "code",    # W8 visibility
    "是否能继续":         "code",    # 预算/上限
    "是否发布":           "human",   # W10 HITL
    "是否越过权限":       "code",    # W8 guard
}
```

**这张矩阵是给面试官看的**：被问"你怎么保证 Agent 不乱来"时，答案不是"我加了 prompt 约束"，
而是"我把 7 个决定分了权，模型只拿到 2 个"。

**算法步骤（今天最容易写错的地方）**
1. **先写 ReAct 基线再写 Plan-and-Execute**——没有对照组，D5 的"显著高于"无从谈起
2. `Plan` 带 `budget_tokens`；执行到超预算要能中止并给出部分结果（对齐 W11 预算熔断）
3. 步骤状态机：`depends_on` 满足才能进 running；单步失败触发 `replan` 而非整体崩
4. **防死循环**：设 `max_replans` 上限与"重复计划指纹"检测（同一计划被重提≥2 次就中止），今天就要埋
5. Planner 只编排，`ToolCall` 执行仍走 W8 `SkillExecutor`，别重写执行逻辑
6. **三道上限必须能真正中断**：不是"记一条警告然后继续跑"。构造超限场景，断言返回 `partial` 结果而非完整假结果
7. 控制权矩阵写进代码常量而不是文档——**它在运行时被 `controls.py` 引用**，才不是一句口号

**自检**
- [ ] `pytest -k planner` 覆盖：依赖阻塞、失败重规划、超预算中止、重复计划中止
- [ ] 一个 `difficulty=3` 案例，Plan-and-Execute 与 ReAct 各跑一次，记录步数与结果（D5 基线）
- [ ] 注入"永远失败"的步骤 → 在 `max_replans` 内中止，不死循环（反向验证）
- [ ] `max_steps` / `max_cost` / `max_seconds` 各构造一次超限，都返回 `partial` 且不抛未捕获异常
- [ ] ADR 里的选型表对 capstone 五条链路逐条给出形态判定（不是只写"我们选 Plan-and-Execute"）

### 🔍 交给我的锚点
`Plan`/`Planner`/`controls`、ADR、死循环与三道上限测试。我会看：**ReAct 对照组是否实现**、预算与死循环守卫是否今天埋、
Planner 有没有越界重写执行、选型表是否对五条链路都给了判定。

---

## Day 2（3h）· 调研规划器：拆解 → 子任务 → 执行 → 重规划

**目标**：把 capstone 的"深度调研"链路用规划器串起来（ROADEMAP D2）。

> 这条链路的正式名字是 **Agentic Search**：不是"检索一次取 Top-K"，而是"围绕信息缺口反复检索"。
> 换句话说，W5 是单轮召回，今天开始是多轮收敛。名字写清楚很重要——两者的评测指标完全不同。

### 你要做的事
- [ ] `planner/research_planner.py`：`ResearchPlanner`（继承 Plan-and-Execute，注入领域拆解逻辑）
- [ ] 拆解策略：主题 → 子问题 → 检索/验证子任务
- [ ] **信息缺口驱动的检索**：记录"还缺什么"，据此决定下一轮搜什么
- [ ] **检索预算与停止条件**：`max_searches` / 边际收益阈值 / 证据足够判定
- [ ] 与 W5 `RetrievalPipeline`、W8 Skill 集成
- [ ] 长任务走 W2 的 **Temporal workflow**（architecture §7 明令：长任务才走工作流）
- [ ] 单测：拆解合理性、子任务并行、部分失败降级、预算耗尽时停止

### 引导框架

```python
# planner/research_planner.py
class ResearchPlanner(PlanExecutePlanner):
    async def plan(self, task: str) -> Plan:
        # 主题 -> 3~5 个子问题 -> 每问检索(W5) + 验证(validator 雏形)
        ...

@dataclass
class Gap:
    """信息缺口：决定下一轮搜什么，而不是重新搜一遍。"""
    subquestion: str
    missing: str            # 缺数字 / 缺反例 / 缺时效 / 缺来源
    searched: list[str]     # 已经搜过的 query（防重复搜同一句）

@dataclass(frozen=True)
class SearchBudget:
    max_searches: int = 12
    min_new_evidence_chars: int = 200   # 一轮新增证据少于这个数 -> 边际收益为负，停
```

**Agentic Search 的三个停止条件（任一满足即停）**

| 条件 | 判据 | 为什么必须有 |
|---|---|---|
| 证据足够 | 每个子问题都至少有 1 条引用支撑 | 否则会无限"再查一下" |
| 预算耗尽 | `searches >= max_searches` | 成本失控的唯一硬闸 |
| 边际收益低 | 本轮新增有效证据 < 阈值 | 模型会重复搜同义 query，看起来在干活 |

**最贵的失败模式不是"搜不到"，是"搜了很多遍同一件事"**：模型换几个同义词反复查，成本翻三倍，
证据却一条没增加。所以 `Gap.searched` 必须记录已搜 query，新 query 要做近似去重（字符 n-gram 相似度 > 0.8 视为重复）。

**算法步骤**
1. 拆解出的每个子任务对应一个可检索的子问题；**子问题要正交**（避免重复检索浪费预算）
2. 独立子任务**并发执行**，但受 W2 `ConcurrencyLimiter` 约束（别打爆自己）
3. 部分子任务失败 → 重规划或标注"该子问题证据不足"，**不能让整份报告留空洞结论**
4. **长任务走 Temporal**：`POST /v1/tasks`（W2）提交 → workflow 编排步骤 → 每步 activity 调 planner；分钟级任务才算长，**别把单轮 chat 塞进去**
5. 每个子任务的检索引用要带 `Citation`（W6），供 validator 交叉验证
6. **区分假设 / 证据 / 结论**：模型推测出来的内容（"可能是季节性因素"）必须标 `hypothesis`，
   只有带 `Citation` 的才能进结论段——否则报告会把猜测写成事实，这是 W4 归因里 `HALLUCINATION` 的主要来源
7. 停止条件要**可断言**：构造一个"永远有缺口"的假检索器，断言在 `max_searches` 处停下

**自检**
- [ ] 一个调研主题跑通：拆解 → 并发检索 → 汇总，报告有引用
- [ ] 注入一个子任务失败 → 报告标注"证据不足"而非编造（反向验证）
- [ ] 长任务通过 `/v1/tasks` 走 workflow，短 `chat` 仍直连（边界验证）
- [ ] 构造"永远搜不满"的假检索器 → 在 `max_searches` 处停止，不无限循环
- [ ] 同义重复 query 被去重（相似度 > 0.8 不再发起）
- [ ] 报告里 `hypothesis` 与 `evidence` 分区可辨，结论段无未引用断言

### 🔍 交给我的锚点
ResearchPlanner、`Gap`/`SearchBudget`、一次调研产物、workflow 接线。我会看：子问题是否正交、部分失败是否诚实标注、
**有没有把 chat 塞进 Temporal**（违反 §7）、三个停止条件是否都能断言。

---

## Day 3（3h）· 多智能体：supervisor / researcher / validator

**目标**：实现 handoff 协议、共享状态、**防死循环与预算控制**（ROADEMAP D3 明确点名这三件）。这是稀缺能力③的核心。

### 你要做的事
- [ ] `multi_agent/roles.py`：三种角色 + `AgentRole` 协议
- [ ] `multi_agent/handoff.py`：handoff 消息结构 + 协议
- [ ] `multi_agent/shared_state.py`：共享状态（带版本/预算）
- [ ] `multi_agent/supervisor.py`：调度 + 防死循环/预算守卫
- [ ] `docs/w9/adr-w9-multi-agent.md`：何时拆多 Agent、handoff 语义、预算模型
- [ ] 单测：handoff 环路检测、预算超限、共享状态一致性

### 引导框架

```python
# multi_agent/handoff.py
@dataclass(frozen=True)
class Handoff:
    from_role: str
    to_role: str
    task: str
    artifacts: tuple[str, ...]   # 引用/中间产物 id
    budget_left_tokens: int      # 预算随 handoff 传递，不能每跳重置
    step_no: int                 # 防死循环：总步数上限

# multi_agent/shared_state.py
@dataclass
class SharedState:
    task: str
    findings: dict[str, list[Citation]]
    version: int
    budget_used_tokens: int
    def commit(self, role: str, delta: dict) -> None: ...  # 带版本号，防并发覆盖
```

```python
# multi_agent/supervisor.py —— 三道守卫是本周技术核心
class Supervisor:
    MAX_STEPS = 20
    MAX_BUDGET_TOKENS = ...       # 从 settings 读
    async def run(self, task: str) -> TaskResult:
        # 每轮：选角色 -> handoff(带剩余预算) -> 执行 -> 更新共享状态 -> 检查终止
        # 守卫1 步数上限 守卫2 预算上限 守卫3 handoff 环路指纹
        ...
```

**算法步骤（今天最容易写错的地方）**
1. **预算随 handoff 传递、不重置**：`budget_left_tokens` 每次递减；否则 3 个角色各花一份预算，总成本翻倍
2. **handoff 环路指纹**：`(from,to,task)` 的序列若出现重复模式 → 立即中止（防死循环，ROADEMAP 点名）
3. **共享状态带版本号**：并发角色 commit 用版本校验，防止覆盖（对齐 W6/W11 幂等主线）
4. 三个角色职责单一：supervisor 只调度、researcher 只检索、validator 只查引用充分性；**别让 supervisor 自己又检索又写**
5. 终止条件明确：达成目标 / 步数超限 / 预算超限 / 环路 → 都要有确定结果和原因（可归因）

**自检**
- [ ] 注入"两个角色互相 handoff"的死循环 → 在 MAX_STEPS/指纹检测内中止（反向验证）
- [ ] 设一个小预算 → 超限中止且 `budget_used` 累计正确（不是每跳清零）
- [ ] 并发 commit 共享状态，版本号能检出冲突

### 🔍 交给我的锚点
Supervisor 三道守卫、ADR、死循环/预算测试。我会看：**预算是否跨 handoff 累计**、环路检测是否真生效、角色职责是否单一。

---

## Day 4（2.5h）· LangGraph 对照实现 + A2A 认知

**目标**：同样的图用 LangGraph 实现一遍，比较自研 vs 框架的取舍（ROADEMAP D4）；了解 A2A 协议。

### 你要做的事
- [ ] `multi_agent/langgraph_variant.py`：用 LangGraph 表达同一张图（若引入依赖，记进 ADR）
- [ ] `docs/w9/langgraph-vs-inhouse.md`：取舍对比（可控性/可观测/成本/迭代速度）
- [ ] A2A 认知笔记：`docs/w9/a2a-notes.md`（协议分层、与 MCP 的区别）
- [ ] 单测：两种实现在 golden 子集上结果可比

### 引导框架

**对比表模板（写进文档）**
| 维度 | 自研（本周） | LangGraph 对照 |
|---|---|---|
| 可控性（预算/死循环守卫） | 完全自控 | 需适配 |
| 可观测（trace 对接） | 直接复用 W3 | 需桥接 |
| 迭代速度 | 慢（自己写） | 快 |
| 依赖/黑盒风险 | 无 | 引入第三方 |
| 面试怎么讲 | "我懂底层" | "我会用生态" |

**算法步骤**
1. LangGraph variant 只做**对照**，不替换主实现；默认档"认知+轻量对照"即可（ROADEMAP §7 A2A 默认档 = 读规范 + 一次 demo）
2. 若引入 langgraph 依赖，**评估它与现有 ADR-008（编排方法）的冲突**并记进 ADR
3. A2A 笔记写清：A2A 是 Agent 间协作协议，MCP（W8）是工具接入协议，**分层不同**（这是 27% JD 提及的加分点）
4. 若 W8 解锁了 A2A 深入档，本周做一次跨进程 handoff demo；否则只写笔记

**自检**
- [ ] 两种实现在同一 golden 子集上跑通，结果可对比
- [ ] 对比表每格有结论（不是只列维度）
- [ ] `pytest` 不因 langgraph 依赖引入而破坏离线档（若引入，要能 skip）

### 🔍 交给我的锚点
langgraph-vs-inhouse.md、a2a-notes.md、对照测试。我会看：对比有没有真实结论、A2A 与 MCP 分层是否讲清、依赖引入是否有 ADR。

---

## Day 5（2.5h）· 三层评测对比：Outcome / Trajectory / System + 合流

**目标**：产出**证据包第 6 件**——单/多 Agent 对比数据，并回答"何时值得拆多 Agent"。
关键变化：**对比不能只有结果指标**，必须同时给出过程指标，否则"多 Agent 更好"无法归因到机制。

> 这是《课程大纲对齐评审》§3-1 的落地：W3–W4 的评测只覆盖 Outcome 与 System 两层，
> Trajectory（过程）层是空的。没有它，本周实验只能得出"多 Agent 成功率 +8%"，然后被追问
> "为什么高？"——答不上来，因为不知道是**分工对了**、**工具选对了**，还是**只是多花了 3 倍 token 多试了几次**。

### 你要做的事
- [ ] `evals/scripts/trajectory_eval.py`：从 trace 算过程指标（**新脚本，本周唯一的新增工时**）
- [ ] `evals/metrics.py` 回填：`TaskMetrics` 增加 `tool_selection_acc` / `invalid_call_rate` / `steps_used` / `path_repeat_rate`
- [ ] `docs/w9/single-vs-multi-ab.md`：**三层对比报告**（不是只有成功率）
- [ ] 用 golden `difficulty` 分层跑：1/2 档与 3 档分别对比
- [ ] **Multi-Trial**：复杂档每个配置跑 3 次，报均值与极差（单次偶然成功没有说服力）
- [ ] 更新 `evals/history/YYYY-Www.json`、`docs/log.md`；周五合流仪式

### 引导框架

**三层指标各自回答什么问题**

| 层 | 指标 | 回答 | 数据来源 |
|---|---|---|---|
| **Outcome**（结果） | 成功率、引用准确率、judge_score | 做对了吗 | W3 双判据 + W4 judge |
| **Trajectory**（过程） | 工具选择准确率、参数准确率、无效调用率、步数、路径重复率 | 为什么对/为什么错 | trace（W2 D3 已有，只是没算过） |
| **System**（代价） | token 成本、p95 延迟、工具调用次数 | 值不值 | `RequestLog` |

**Trajectory 四个指标怎么算（不引外部库，从我们自己的 trace 算）**

| 指标 | 定义 | 为什么它能解释归因 |
|---|---|---|
| `tool_selection_acc` | 期望工具序列与实到序列的最长公共子序列 / 期望长度 | 低 = 选错工具（归因到 prompt 层） |
| `invalid_call_rate` | schema 校验失败或 `tool_not_visible` 的次数 / 总调用 | 高 = 工具描述不清（归因到工具层） |
| `steps_used` | 实际步数（与期望步数一起报） | 远超期望 = 绕路或死循环前兆 |
| `path_repeat_rate` | 重复 `(tool, args_hash)` 占比 | 高 = 规划器在打转（归因到 planner） |

**路径重复率是最好用的一个**：它能在成功率还没掉的时候先报警。多 Agent 的"两个角色互踢皮球"，
在 Outcome 层表现为"成功率略降"，在 `path_repeat_rate` 上则是**从 0.05 跳到 0.6**——后者更像根因。

**三层对比表模板（写进报告）**
| 任务难度 | 范式 | 成功率 | judge | 工具选择acc | 无效调用率 | 步数(实/期望) | 路径重复率 | cost/task | p95 |
|---|---|---|---|---|---|---|---|---|---|
| 简单(1) | ReAct |  |  |  |  |  |  |  |  |
| 简单(1) | 多 Agent |  |  |  |  |  |  |  |  |
| 复杂(3) | ReAct |  |  |  |  |  |  |  |  |
| 复杂(3) | 多 Agent |  |  |  |  |  |  |  |  |

**Multi-Trial 表（复杂档，各跑 3 次）**
| 范式 | 成功率 | 极差 | 平均步数 | 极差 | 结论 |
|---|---|---|---|---|---|
| ReAct |  |  |  |  | 稳不稳 |
| 多 Agent |  |  |  |  | 稳不稳 |

**算法步骤（今天的分析核心）**
1. **分层看**：预期简单任务多 Agent 反而更慢更贵（协调开销），复杂任务才体现优势——**这个反差本身就是最好的结论**
2. 目标：≥5 步（复杂）任务，多 Agent 完成率显著高于 ReAct（ROADEMAP 检查点）
3. **归因要落到层**：多 Agent 赢在哪一层？如果是 `tool_selection_acc` 高，说明"角色化让工具选择更专一"；
   如果只是 `steps_used` 多（多试几次），那这条证据就**不能**用来证明架构优势——**必须如实写出来**
4. 诚实记录：若多 Agent 在成本/延迟上更差，明确写"值得拆的边界条件"，**别只报喜**
5. Multi-Trial 只为复杂档做 3 次（离线档成本近 0）；**别对全部 54 条都跑 3 次**，那是 3 倍时间换不到更多信息
6. 结论要能回答两个面试问题："何时才值得拆多 Agent？"、"你怎么知道是架构而不是运气？"——用上面两张表支撑
7. 合流：`make eval` 过门禁 → 写 history json → `docs/log.md`

### 🔍 Gate 9 · 周末验收（全绿才算过 W9）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 对照组齐全 | ReAct 与 Plan-and-Execute 都实现可跑 |
| 2 | 规划可用 | ≥5 步任务完成率显著高于朴素 ReAct |
| 3 | 拆解合理 | 子问题正交，部分失败诚实标注 |
| 4 | 三道守卫 | 死循环/步数/预算守卫均生效且反向验证过 |
| 5 | 预算累计 | token 跨 handoff 累计，不每跳重置 |
| 6 | 对照实现 | LangGraph variant 可比，取舍有结论 |
| 7 | 选型有依据 | ADR 对 capstone 五条链路逐条给出形态判定 |
| 8 | Agentic Search | 三个停止条件都能断言，同义 query 被去重 |
| 9 | 三层对比 | Outcome/Trajectory/System 三层数字齐全，含"何时值得"结论 |
| 10 | 归因到层 | 多 Agent 的优势能指到具体某一层指标，说不出就写"无法归因" |
| 11 | 稳定性 | 复杂档 3 次跑完，报均值与极差 |
| 12 | 合流 | history 接上，端到端仍可跑 |

---

## 本周最容易返工的 7 处

1. **没有 ReAct 对照组** → "显著高于"无从证明，检查点直接挂
2. **预算每跳重置** → 多 Agent 成本翻倍，W11 成本看板失真
3. **没有死循环/环路检测** → 两个角色互踢皮球，跑到天荒地老（ROADEMAP 点名）
4. **supervisor 又检索又写** → 职责不单一，共享状态被并发覆盖
5. **把 `/v1/chat` 塞进 Temporal** → 违反 architecture §7（W2 已强调）
6. **对比实验只报喜** → 简单任务多 Agent 更差的事实不写，结论不可信，面试一问就露
7. **对比只有成功率** → 赢了也不知道为什么赢，被追问"是架构还是多花了 3 倍 token"就答不上；必须有三层指标
8. **三道上限只记警告不中断** → "有预算控制"变成文档里的一句话，构造超限仍跑完全程

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 规划器 | `planner/{base,plan_execute,react,research_planner,controls}.py` |
| 多智能体 | `multi_agent/{roles,handoff,shared_state,supervisor,langgraph_variant}.py` |
| ADR/笔记 | `docs/w9/adr-w9-{planner-choice,multi-agent}.md`、`docs/w9/langgraph-vs-inhouse.md`、`docs/w9/a2a-notes.md` |
| 过程评测 | `evals/scripts/trajectory_eval.py`、`evals/metrics.py`（trajectory 四指标） |
| 对比实验 | `docs/w9/single-vs-multi-ab.md`（三层表 + Multi-Trial 表） |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W9 条目） |

---

## P3 挂载点（见 ROADEMAP §7）

- **A2A 深入档**（P3-3）：若 W8 末已解锁，本周做跨进程 handoff demo / Agent Card / task lifecycle；默认档只读规范 + 一次轻量 demo
- **Computer Use**（P3-4）：只写进 W12 前沿笔记，本周不碰
- 本周先把"值得拆多 Agent 的数据"做出来，这是比"用了多 Agent"更有说服力的证据
