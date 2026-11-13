# W11 执行清单（生产硬化周 · P0，稀缺能力 ①）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> 本周是 ROADEMAP 的转折：**capstone 从"能跑"变成"敢给别人用"**。W10 已上线并有真实用户，本周用真实反馈修 bug、做混沌与容灾、冲刺成本，最后 **DoD 九条逐条自评**。
> 本周也是 **W11 末 P3 决策点**（ROADEMAP §7）——前端/多语言深入档的最佳挂载点就在这里。

## 承接面：W1–W10 已交付

| 已有 | 位置 | W11 的接缝 |
|---|---|---|
| 重试 + committed 状态机 + fallback（ADR-003/008） | `llm/client.py` | 本周加熔断器/降级兜底，不改重试归属 |
| `ConcurrencyLimiter` + 背压（W2） | `resilience/` | 混沌演练验证它在故障下生效 |
| `/v1/tasks` + Temporal workflow（W2/W9） | `queue/`、`workflow/` | 幂等与断点续跑基于它 |
| `PermissionGuard`/`ApprovalGate`（W8/W10） | `skills/guard.py`、`guardrails/hitl.py` | 灰度/回滚要带审批 |
| `RequestLog` + Langfuse trace + `first_token_ms`/cost | W2/W3 | SLO 三张图、混沌归因、成本看板的数据源 |
| `make eval` + `regression_gate` + baseline | W3/W4 | 每次硬化改动都要过门禁 |
| `models.yaml` 的 pricing（峰谷/阶梯口径，W1 Day5） | `config.py` | 成本优化冲刺要按真实口径取价 |
| 真实上线 + 反馈渠道 | W10 | 本周修 ≥5 条线上问题 |
| `Router` 协议（`TaskRouter`/`StaticRouter`） | `llm/router.py` | architecture §8：按成本/延迟的新路由策略在 W11 生效 |

> W11 检查点（ROADEMAP）：**DoD 九条逐项达标或明确标注缺口**；有压测 + 混沌 + 安全三份报告；≥5 条「线上问题 → 根因 → 修复」记录。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接符号 | 推进的 DoD |
|---|---|---|---|
| D1 | `ops/slo.py` + 告警规则（成功率/延迟/成本三张图） | RequestLog/trace | 可观测、可用性 |
| D2 | `resilience/{circuit_breaker,degradation}.py` + 混沌演练 | client 重试、fallback | 可靠性 |
| D3 | `runtime/{model,timeline}.py`（**统一执行模型**）+ 幂等 + 断点续跑 | /v1/tasks、workflow、W9 Plan | 可靠性、可解释 |
| D4 | 真实反馈修 bug（≥5 条）+ **Fork/Ablation 对照实验** | 反馈渠道、trace 归因 | 可用性、根因可证明 |
| D5 | **受控进化**（Feature Flag / 资产版本 / Kill Switch）+ 成本冲刺 + DoD 自评 | pricing、history 曲线 | 成本、可回退、证据包 |

**唯一主线**：用"故障注入 + 真实反馈 + 量化指标"证明 DoD 九条。本周的产出大多是**报告与证据**，不是新功能——这正是"生产硬化"区别于"堆功能"的地方。

> 本周从"抗故障"扩成"**抗故障 + 受控进化**"：D3 收敛执行状态（Run/Step/Attempt/Event + 9 态），
> D4 用对照实验证明根因，D5 让行为可灰度可一键回退。灰度从 D3 挪到 D5——D3 是运行时状态，D5 是资产版本。
> 见 `docs/plan-review-course-outline-gap.md` §6.2。

---

## Gate 0 · 开工前（30 min）

- [ ] 导出 W10 上线后的 `evals/history` 曲线（成功率/p95/成本/judge 四个维度已有数据点）
- [ ] 汇总真实用户反馈（W10 收集渠道），列出 ≥5 个待修问题候选
- [ ] 复核 ADR-003：**重试只在 client**——本周的熔断器要放在 client 之外/之上，别造第二次重试
- [ ] `make eval` 全绿、门禁生效
- [ ] 准备混沌注入手段（mock transport 故障、限流、超时），断网可复现

---

## Day 1（2.5h）· SLO 定义 + 告警规则

**目标**：把 DoD 的可用性/延迟/成本指标变成**可告警的 SLO**（成功率/延迟/成本三张图，ROADEMAP D1）。

### 你要做的事
- [ ] `ops/slo.py`：SLO 定义（SLI + 目标 + 窗口）+ 误差预算计算
- [ ] `ops/alerts.yml`：告警规则（成功率跌破、p95 超标、成本超预算）
- [ ] 三张图的数据查询（从 `RequestLog`/history 取，别另造埋点）
- [ ] `docs/w11/slo-spec.md`：每个 SLO 的 SLI/目标/窗口/告警阈值
- [ ] 单测：误差预算计算、告警触发边界

### 引导框架

```python
# ops/slo.py
@dataclass(frozen=True)
class SLO:
    name: str                 # availability | latency_p95 | cost_per_task
    sli: str                  # 对应 RequestLog 字段/查询
    target: float             # 如 0.95 / 1500ms / 2.0 CNY
    window: str               # 如 "30d" / "7d"
    comparison: str           # "gte" | "lte"

def error_budget(slo: SLO, observed: float, window_total: int) -> float:
    """剩余误差预算；耗尽即应停止发布新功能、转入硬化。"""
    ...
```

**算法步骤（今天最容易写错的地方）**
1. SLO 的 SLI 必须直接来自 `RequestLog`/history（W1 Day5 已建口径），**不许自己 time.monotonic 另造**
2. 延迟 SLO 用 **p95 而非均值**（DoD 原话："首 token p95 ≤ 1.5s"），口径与 W3 `TaskMetrics.latency_p95` 一致
3. 成本 SLO 要区分 `usage_estimated`（W1 Day5 教训：估算占比高时结论不可信）；峰谷价按时间戳取（W1 Day5）
4. 告警规则要有**持续时间**（连续 N 分钟才告警），避免单点抖动天天呼值班
5. 误差预算概念写进 spec：预算耗尽时**冻结新功能、只做硬化**（这是 SRE 的常见话术，面试加分）

**自检**
- [ ] 三张图能从 history 数据画出（W1–W10 数据点连续）
- [ ] `pytest -k slo` 覆盖误差预算、p95 口径、告警边界
- [ ] 注入一个"成功率跌破"的历史窗口 → 告警规则触发（反向验证）

### 🔍 交给我的锚点
slo-spec、alerts.yml、三张图。我会看：SLI 是否复用 RequestLog、延迟是否用 p95、成本是否区分 estimated、告警是否有持续时间。

---

## Day 2（3h）· 混沌演练 + 降级与兜底模型

**目标**：模型超时 / 工具 500 / 限流三类故障注入可自愈（DoD 可靠性项），产出**混沌演练报告**。

### 你要做的事
- [x] `resilience/circuit_breaker.py`：熔断器（连续失败→打开→半开→闭合）
- [x] `resilience/degradation.py`：降级策略（缓存/兜底话术/明确降级说明）
- [x] `evals/scripts/chaos.py`：三类故障注入 + 自愈断言
- [x] `docs/w11/chaos-report.md`：注入项 / 预期 / 实际 / 自愈证据
- [x] 单测：熔断状态机、降级路径、恢复（22 + 9 条）

> **偏差 1：降级里没有"兜底模型（强→便宜）"。** 它就是第二次重试，而候选链已经试过一遍
> （ADR-003）。落地顺序：语义缓存 → 兜底话术（默认关）→ 原样抛出。见 ADR-W11-002。
> **偏差 2：熔断器挂在 `LLMGateway(breaker=...)` 上，不是包在 4 个业务调用点。**
> 包在调用点意味着新增一处调用就漏一处保护，而漏掉的那处不会有测试变红。见 ADR-W11-001。

### 引导框架

```python
# resilience/circuit_breaker.py —— 放在 client 之外的编排层，别造第二次重试
class CircuitBreaker:
    def __init__(self, *, failure_threshold: int, reset_after_s: float) -> None: ...
    async def call(self, fn, *args):
        # CLOSED -> 失败计数；达阈值 -> OPEN(快速失败) -> 超时后半开试探 -> 闭合
        ...

# resilience/degradation.py
class DegradationPolicy:
    async def fallback(self, call: ChatCall) -> ChatOutcome:
        # 顺序：兜底模型(强->便宜) -> 语义缓存(W11 D5) -> 精简输出/明确降级说明
        ...
```

**算法步骤（今天最容易写错的地方）**
1. **熔断器不重复实现重试**：重试仍是 `LLMGateway` 的职责（ADR-003），熔断器只做"连续失败后快速失败 + 半开探测"。两者叠加会放大流量
2. 三类故障分别注入：① 模型超时（`first_token_timeout`）② 工具 500（Skill 执行）③ 限流（429 + Retry-After）；**用 mock transport 断网可复现**
3. 每类断言"自愈"：故障期间成功率 ≥90%、不崩、不静默失败；故障恢复后自动闭合
4. 降级要有**明确标记**（`fallback_from`/`degraded` 进日志与响应），别让用户以为拿到了正常结果
5. 兜底模型走 `models.yaml` 的 fallback 链（W1 已定），别硬编码

**自检**
- [x] `chaos.py` 三类注入全部可复现，输出自愈证据（F1/F2/F3）
- [x] 注入持续超时 → 熔断打开后快速失败，不再打上游（`calls_after_open == 0`，反向验证没被重试放大）
- [x] 故障恢复 → 半开试探成功后闭合（F5：探测飞行中 4 个请求被拒、0 次上游调用）
- [x] `make eval` 在演练后仍过门禁：全量 `1296 passed / 30 skipped`，`test_regression_gate.py` 全绿；
      完整 54 条 eval 在 Gate 11 统一跑一次并落 `evals/history`（避免演练当天落一份半成品基线）
- [ ] **故障期间"成功率 ≥90%"未在演练里量化**——注入的是 100% 失败，报出来是 0 没有信息量。
      这条只能从真实 `RequestLog` 算，见 `chaos-report.md` §7 缺口 1

### 🔍 交给我的锚点
chaos-report、CircuitBreaker、自愈断言输出。我会看：**熔断器有没有造第二次重试**（ADR-003）、三类是否都可复现、降级是否有明确标记。

---

## Day 3（3h）· 统一执行模型 + 幂等 + 断点续跑 + Run Timeline

**目标**：长任务可断点续跑、请求幂等（DoD 可靠性项），并且**任意一次执行都能被完整解释**。
今天不是"加一个断点续跑"，而是把散落在 W2/W9/W10 的执行状态收敛成**一套对象模型**。

### 你要做的事
- [x] `runtime/model.py`：**`Run` / `Step` / `Attempt` / `Event`** 四个对象 + 9 态状态机
- [x] `runtime/timeline.py`：`RunTimeline`（一次执行的时间线，能回答"第几步、哪个工具、为什么重试"）
- [x] `resilience/idempotency.py`：幂等键（请求/任务级）
- [x] 断点续跑：`runtime/resume.py`（事件流落盘 + 重启跳过已完成步 + 中断步重跑）
- [x] `docs/w11/reliability-report.md`：执行模型 + 幂等 + 续跑 + Timeline 四项证据
- [x] 单测：状态机非法迁移被拒、重复请求幂等、续跑跳过已完成步、Timeline 可重建（26 + 14 + 12）
- [x] 附：码表从 `evals/scripts/classify_failures.py` 搬进 `agent_framework/error_codes.py`，
      分类器改为 re-export——`Attempt.error_code` 要判"是不是厂商的锅"，而 runtime 不能 import 评测脚本

> **偏差：断点续跑没走 Temporal。** 本项目没有真正的 Temporal 集群，
> `ResumableRunner` 用"事件流落盘 + 从磁盘重建"实现同一性质，重启是真的重启（内存态全丢）。
> **缺口：新模型尚未替换 `/v1/tasks` 的四态记录**，见 `reliability-report.md` §9 缺口 1。

### 引导框架

**为什么要统一执行模型：现在有四个"状态"互相对不上**

W2 的 `/v1/tasks` 有自己的 task 状态，W9 的 `PlanStep` 有 `pending|running|done|failed|skipped`，
W10 的审批有 `pending|approved`，Temporal 自己还有一套。**四套状态机都不可能覆盖对方**，
结果就是"任务显示 running，但某一步其实卡在等审批"这类问题只能靠人肉看日志。

```python
# runtime/model.py
class RunState(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"          # 正在等工具返回
    WAITING_APPROVAL = "waiting_approval"  # 等人（W10 HITL）—— 最容易被漏的一个状态
    TOOL_FAILED = "tool_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Run:
    run_id: str
    state: RunState
    steps: list[Step]
    events: list[Event]      # 追加式，不改写历史

@dataclass
class Step:
    step_id: str
    goal: str
    attempts: list[Attempt]  # 一步可以试多次
    state: RunState

@dataclass(frozen=True)
class Attempt:
    n: int
    started_at: datetime
    error_code: str | None   # 复用 W4 RELIABILITY_CODES，不新造码表
```

**三层对象的分工（面试高频追问）**
| 层 | 回答什么 | 粒度 |
|---|---|---|
| `Run` | 这次任务整体怎么样 | 一次用户请求 |
| `Step` | 任务分几步、每步干什么 | 一个子目标 |
| `Attempt` | 为什么这一步试了 3 次 | 一次工具/模型调用 |
| `Event` | 过程中发生了什么（只追加） | 状态迁移、错误、审批 |

**关键**：`Attempt.error_code` 直接复用 W4 的 `RELIABILITY_CODES` / `is_reliability_code()`。
W4 刚刚把两份漂移的码表收敛成一份（10 个码里 6 个判定相反那次），**这里再抄一份必然重演**。

**Run Timeline：让"解释一次失败"变成查表**

```python
# runtime/timeline.py
class RunTimeline:
    def render(self, run: Run) -> str:
        """形如：
        00:00 step-1  attempt 1  search_web   ok     412ms
        00:02 step-2  attempt 1  fetch_page   timeout 30000ms  [UPSTREAM_TIMEOUT]
        00:32 step-2  attempt 2  fetch_page   ok     890ms   (重试，参数未变)
        00:33 step-3  --         publish      waiting_approval
        """
```

**判据**：拿任何一条失败 trace，**不打开日志文件**、只看 Timeline 就能说出"哪一步、什么错、重试了几次、现在卡在哪"。
做不到就说明状态没收敛，还是四套并存。

**算法步骤（今天最容易写错的地方）**
1. 幂等键用 `(principal, 请求内容规范化哈希)`；**参数顺序不同要算同一条**（对齐 W8 回放哈希思路）
2. 断点续跑靠 Temporal：每个 activity 完成时持久化状态，重启后**跳过已完成步**——别让已花钱的检索重跑（成本）
3. **状态机必须拒绝非法迁移**（`COMPLETED` 不能再回 `RUNNING`），并且非法迁移要抛错而不是静默忽略；
   否则"状态对不上"会变成最难查的一类 bug
4. `WAITING_APPROVAL` 是独立状态，不是 `RUNNING` 的子状态——**否则监控会把"等人"算成"卡住"**，
   D1 的 SLO 图直接失真
5. `Event` 只追加不改写：续跑时靠重放 Event 恢复内存态，不依赖某张表的当前值
6. 幂等与 HITL（W10）联动：审批过的危险动作重复请求不再执行第二次
7. 续跑/Timeline 演练都记进 `reliability-report`，带时间戳与证据

**自检**
- [ ] 重复提交同一 `POST /v1/tasks` → 只执行一次，返回同一 task_id
      **（限定：`InMemoryIdempotencyStore` 的语义已实测；接口层还没接，见缺口 1）**
- [x] 长任务跑到第 3 步时"重启" → 从第 4 步续跑，前 3 步不重跑（反向验证）
      实测：`skipped=('step-1','step-2')`、`executor 实际调用=['step-3','step-4']`
- [x] 构造非法迁移（`COMPLETED` → `RUNNING`）→ 抛错，不是静默
- [x] 一条失败 trace 只凭 Timeline 能讲清全过程（不打开原始日志）
- [x] `WAITING_APPROVAL` 不计入"卡住"：`stuck_runs()` 排除它、`RunTimeline.summary()`
      单独报"卡在等人审批"；**但 `ops/slo_report.py` 还没把 run 状态纳入周报**（缺口 5）

### 🔍 交给我的锚点
`runtime/{model,timeline}.py`、idempotency、续跑与非法迁移测试、reliability-report。
我会看：状态机是否真收敛（不是又加第五套）、码表是否复用 W4、`WAITING_APPROVAL` 是否独立、Timeline 能否独立解释失败。

---

## Day 4（2.5h）· 真实反馈修 bug（≥5 条）

**目标**：用 W10 收集的真实反馈，修 ≥5 条 bug，产出「线上问题 → 根因 → 修复」记录（证据包第 7 件）。

### 你要做的事
- [x] 从反馈渠道整理 ≥5 个真实问题 → **限定：无外部用户，5 条来自 W10 上线打通时的实跑 + W11 演练**
      （`docs/w11/incidents/README.md` §0 有口径声明）
- [x] 每个用 W4 `root-cause-playbook` 定位根因 —— **结论是 5/5 都不在四层里**：
      四层树的前提是"系统在跑、模型在答"，而这几条是"没跑起来 / 没设防"。见 README §2
- [x] **至少 1 条用 Fork/Ablation 证明根因**：做了 2 条（INC-1 消融、INC-5 分叉），
      `evals/scripts/w11_fork_ablation.py` 可复跑，结论已进门禁
- [x] 逐个修复 + 补回归用例（`test_injection.py` / `test_article_route.py` / `test_auth.py` / `test_circuit_breaker.py`）
- [x] `docs/w11/incidents/incident-{1..5}.md`：五篇记录 + `README.md` 索引
- [ ] 更新 `docs/log.md` —— 留给 W11 合流那一刀统一写（避免一周写两次、两次不一致）

### 引导框架

**incident 模板（每篇固定）**
```
## 现象
真实用户怎么触发的、复现命令、当时的指标/trace 链接
## 根因
W4 playbook 定位到哪层，证据字段（span/日志值）
## 对照实验（至少一篇必须有）
Fork 的是哪一步、只改了什么、另一条路径指标有无变化
## 修复
改了哪个文件、引用哪个 ADR
## 验证
修复前后指标 + 新增的回归用例（make eval 通过）
```

**Replay / Fork / Ablation：三个词必须分清**

W4 做的是**规则归因**（看证据字段判层），它能给出"怀疑对象"，但不能证明。
证明需要对照实验，而这三个词在面试里最容易被连在一起说糊：

| 手段 | 是什么 | 回答什么问题 | 我们靠什么做 |
|---|---|---|---|
| **Replay** | 拿失败任务的原始输入+上下文+工具结果，重跑一遍 | "还能复现吗" | W11 D3 的 `Event` 重放 |
| **Fork** | 从某一步分叉，只改一个变量继续跑 | "是不是这一步的错" | 复用 W9 `Plan` 的步骤 + 改单点 |
| **Ablation** | 关掉某个能力整体对比 | "这个能力有没有贡献" | W7 的 `MEMORY=off`、W8 的 `FEATURE_X=off` |

**判据（这就是"根因被证明"的样子）**：
Fork 出两条路径，**只改 A**，如果 B 路径指标不变、A 路径变好 → 根因是 A；
如果两边一起变 → 说明改的是共同上游，第一次归因就错了，回去重判层。

**算法步骤**
1. **只记真实问题**，别把 W1–W10 的内部调试凑数——ROADEMAP §3.3 要的是"上线后遇到的最难的三个问题"
2. 每条修复必须补回归用例并过门禁，否则等于没修
3. 优先修"影响成功率/成本/安全"的，按 SLO 误差预算（D1）排序
4. 至少挑 3 条整理成"最难的三个问题"叙述（W12 模拟面试素材）
5. **对照实验挑最贵的那一条做**（不是最容易的）：成本高的 bug 才值得花 40 分钟做 Fork，
   便宜的 bug 用规则归因就够——这个取舍本身也是面试可讲的判断力

**自检**
- [x] ≥5 篇 incident 各有现象、根因、修复、验证四段
- [x] 至少 1 篇有 Fork/Ablation 对照实验，且写清了"只改了什么、另一条没变"（做了 2 篇）
- [x] 每篇对应的回归用例在全量门禁中通过（1350 passed / 30 skipped）
- [ ] 修复后 history 曲线的成功率/成本有可见改善 —— **做不到**：这 5 条里 4 条离线档走不到，
      第 5 条修的是证据链不是成功率。能报的数（`/v1/article` 500→200、deploy_check 3/5→5/5）
      记在 README §4，不拿它冒充"曲线改善"

### 🔍 交给我的锚点
5 篇 incident、对照实验、回归用例、修复前后指标。我会看：是否真实、根因是否用 playbook 归因、
对照实验有没有做到"只改一个变量"、有没有补回归用例。

---

## Day 5（2.5h）· 受控进化：Feature Flag / 灰度 / Kill Switch + 成本冲刺 + DoD 自评

**目标**：让"改坏了"能**只影响 1% 用户、并且一键退回**；成本优化可量化；**DoD 九条逐条自评**；
周五合流 + W11 末 P3 决策。

> 灰度与回滚从 D3 挪到这里，是因为它们和 Feature Flag 是同一件事的三个面：
> **D3 解决"能不能恢复执行"，D5 解决"能不能退回行为"**。前者是运行时状态，后者是资产版本。

### 你要做的事
- [x] `runtime/flags.py`：Feature Flag（配置驱动，不改代码不发版）
- [x] **资产版本清单**：六类资产各自有版本号且可回滚
- [x] 灰度：按 principal/比例切新老逻辑（复用 W6 `Principal`）
- [x] **Kill Switch**：一条命令让某能力回到旧行为（演练并计时）
- [x] 回滚演练：镜像 tag 回退 + 配置回退（复用 W10 runbook）
- [x] `resilience/semantic_cache.py`：语义缓存（相似 query 复用结果）
- [x] 模型分流：按难度把简单任务路由到 cheap（实现 architecture §8 的 Router 策略）
- [x] 批处理：可并行的改写/检索合并
- [x] `docs/w11/dod-self-assessment.md`：DoD 九条逐条自评（达标/缺口/证据链接）
- [x] 更新 `evals/history/YYYY-Www.json`、`docs/log.md`；周五合流仪式
- [x] **W11 末 P3 决策**：余量 ≥15h 才解锁一个深入档（前端/多语言最佳挂载点）

> **偏差 1：回滚演练只做了一半。** 配置回退（`FEATURES` 改值 + 重启）实跑到 **0.5s**；
> **镜像 tag 回退没实跑**——本机没有既有镜像，`ops/Dockerfile` 的构建需要仓库外的
> `06_rag/rag_mvp` 作为 `additional_contexts`，构建不起来。没有产物就没有"回退到上一版"，
> 所以如实记为缺口（`asset-versions.md` §4 给了可执行步骤）。
>
> **偏差 2：批处理只取"并发"部分。** `StyleRewriter.rewrite_all` 的并发是 W8 就有的；
> "合并成一次调用"W8 已实测否决（LinkedIn 段落结构污染 X，空行被 `validate()` 拦下），
> 本周不重测——重测需要真实上游输出，离线造一个"串味文本"再断言它被拦是自证。
> 于是 `FLAGS` 里**没有** `batched_rewrite`：给被否决的方案配开关，等于给没人敢开的代码办长期居留。
>
> **偏差 3：`circuit_breaker` 也没做成 Flag。** 已经有 `BREAKER_ENABLED`（D2）。
> 同一行为两个开关必然出现"A 关了它还在跑"，而排查时两个人各看一个开关。
>
> **缺口：`skill` 这一类资产没进快照**（`missing_kinds=('skill',)`）。
> W8 的 Skill 在 `tools_skills_dir` 里按需加载，`capture()` 只在显式传 `skills=` 时才有这一类。
>
> **发现（成本这条最贵的一处）**：语义缓存最初把"`opts` 非空"判成不可缓存，
> 而产品端**每一个**调用都带 `temperature`——它在真实路径上从未生效，而测试全绿。
> 改成"opts 进缓存键"之后才真正命中（`kill_switch_drill.py` 里 2 条请求 1 条命中）。

### 引导框架

**资产版本清单：只回滚代码是不够的**

我们的行为由六样东西共同决定，**任何一样变了，系统行为就变了**。只给代码打 tag，等于只能回滚 1/6。

| 资产 | 版本载体 | 回滚方式 | 已经有的 |
|---|---|---|---|
| 代码 | 镜像 tag / commit | 回退 tag | ✅ W10 runbook |
| 模型 | `catalog` 里的 alias 指向 | 改 alias → cheap/strong 换回 | ✅ W1（`model_snapshot` 已进 baseline） |
| Prompt | prompt 模板版本号 | 切版本 | ❌ 现在模板没版本号 |
| 工具/Skill | `SkillSpec` 的 version | 切 version | ❌ 今天只在 spec 里有 name |
| RAG 策略 | chunk 参数 + 检索参数 + rerank 模型 | 切配置组 | ❌ 参数散在配置里 |
| 记忆策略 | `MEMORY=off\|governed`（W7 D5 建的） | 切 env | ✅ W7 |

**结论**：本周只需补 **Prompt / Skill / RAG 策略**三样版本号 + 一张"当前生效版本快照"，
写进 `evals/history` 的 snapshot（W4 已有 `model_snapshot`，扩成 `asset_snapshot`）。
**不建版本管理平台**——六个字符串 + 一个环境变量就够，多了就是过度设计。

**Kill Switch 的判据（必须可计时）**
1. 触发方式：`FEATURE_X=off` 重启 或 配置中心热更新（本周做 env + 重启，够用）
2. **判据：从决定回退到行为生效 ≤ 5 分钟**，且 `make eval` 仍绿
3. 演练要记录：开始时间、生效时间、期间错过的请求数、回退后第一次成功的 trace

**成本优化三手段（都要可量化、可关闭）**
| 手段 | 做法 | 度量 | 风险 |
|---|---|---|---|
| 语义缓存 | 相似 query 命中即复用 | 命中率 / 节省成本 | 结果过期 → 带 TTL + 版本 |
| 模型分流 | 简单任务走 cheap | cheap 占比 / 质量 | 误分流 → 过门禁验证 |
| 批处理 | 并发改写合并 | p95 / 成本 | 别打爆背压（W2 限流） |

```python
# resilience/semantic_cache.py
class SemanticCache:
    async def get(self, query_embedding, *, ttl_s: int) -> ChatOutcome | None: ...
    async def put(self, query_embedding, outcome: ChatOutcome) -> None: ...
    # 命中要标记 cache_hit 进日志，成本记 0 但与 usage_estimated 区分（对齐 W8 replay）
```

**算法步骤**
1. 每个优化**单变量 A/B**，用 `make eval` 证明成本下降且质量过门禁（W4/W5/W7 一贯方法）
2. 语义缓存命中/回放松记 `cache_hit`/`replayed`，**成本记 0 但要在报告里区分**（W11 成本看板，W8 已强调）
3. 模型分流实现 `Router` 协议（architecture §8 预留），按任务难度/预算选 alias，不改业务代码
4. **每个 Flag 必须有"关掉它"的默认值 = 旧行为**——默认值是新行为，等于回滚时要改代码，就没意义了
5. **DoD 自评表**：九条逐项写「达标 / 部分达标 / 缺口 + 证据链接」，**缺口如实标注**（ROADEMAP 检查点原话："达标或明确标注缺口"）
6. 合流：合 main → `make eval` 过门禁 → 写 history json → `docs/log.md` → P3 决策

### 🔍 Gate 11 · 周末验收（全绿才算过 W11）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | SLO | 三张图可告警，延迟用 p95，成本区分 estimated |
| 2 | 混沌 | 三类故障注入可复现且自愈，≥90% 不崩 |
| 3 | 不重复重试 | 熔断器未与 client 重试叠加（ADR-003） |
| 4 | 执行模型 | Run/Step/Attempt/Event 统一，非法迁移抛错，码表复用 W4 |
| 5 | 幂等续跑 | 重复请求幂等，续跑跳过已完成步 |
| 6 | 可解释 | 任一失败 trace 只凭 Run Timeline 能讲清全过程 |
| 7 | 真实修 bug | ≥5 篇 incident，各带根因与回归用例 |
| 8 | 根因被证明 | ≥1 条有 Fork/Ablation 对照实验，只改一个变量 |
| 9 | 可回退 | 六类资产版本齐全，Kill Switch 生效 ≤5 分钟且回退后 eval 仍绿 |
| 10 | 成本优化 | 三手段单变量 A/B，成本下降且过门禁 |
| 11 | DoD 自评 | 九条逐条达标/缺口 + 证据链接 |
| 12 | P3 已判定 | 按余量明确解锁/不解锁，写进 log |
| 13 | 合流 | history 接上，端到端仍可跑 |

---

## 本周最容易返工的 8 处

1. **熔断器又造了一次重试** → 与 ADR-003 叠加，故障时流量放大（本周头号坑）
2. **延迟 SLO 用均值不用 p95** → 与 DoD 原话不符，长尾问题被掩盖
3. **成本不区分 estimated / cache_hit / replay** → 成本看板失真（W1/W8 已两次强调）
4. **断点续跑重跑已完成步** → 已花钱的检索重复执行，成本失控
5. **incident 拿内部调试凑数** → 不是"上线后真实问题"，证据包第 7 件不成立
6. **DoD 自评报喜不报缺口** → 检查点要求"达标或明确标注缺口"，注水会被一眼看穿
7. **统一执行模型变成"第五套状态机"** → 不删旧的、只加新的，问题更严重；必须替换而非并存
8. **Flag 默认值设成新行为** → 回滚时还得改代码发版，Kill Switch 等于没有

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| SLO/告警 | `ops/{slo.py,alerts.yml}`、`docs/w11/slo-spec.md` |
| 混沌/容灾 | `resilience/{circuit_breaker,degradation}.py`、`evals/scripts/chaos.py`、`docs/w11/chaos-report.md` |
| 执行模型 | `runtime/{model,timeline}.py`、`docs/w11/reliability-report.md`（含 Run Timeline 样例） |
| 可靠性 | `resilience/idempotency.py` |
| 真实反馈 | `docs/w11/incidents/incident-{1..5}.md`（≥1 篇含 Fork/Ablation） |
| 受控进化 | `runtime/flags.py`、`docs/w11/asset-versions.md`（六类资产版本快照 + Kill Switch 演练记录） |
| 成本优化 | `resilience/semantic_cache.py`、Router 新策略、`docs/w11/dod-self-assessment.md` |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W11 条目 + P3 决策） |

---

## P3 挂载点（W11 末 = 第三个决策点，见 ROADEMAP §7）

- **前端极简**（P3-1）：最佳挂载点就在 W11 末 / W12——单页 HTML + SSE 渲染（默认档 0.5 天），让 demo 观感最直观
- **多语言**（P3-2）：若目标 JD 有"出海/跨境"，W10 内容周已铺好，本周补评测集维度
- 余量 ≥15h 才解锁**一个**深入档，一次只开一个；**默认档一律完成**
- 若余量不足，明确写"不解锁"，把时间留给 W12 的框架提取与文档（主线）
