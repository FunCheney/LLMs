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
| D3 | 幂等键 + 断点续跑 + 灰度/回滚演练 | /v1/tasks、workflow | 可靠性 |
| D4 | 真实反馈修 bug（≥5 条） | 反馈渠道、trace 归因 | 可用性、真实使用 |
| D5 | 成本优化冲刺 + DoD 逐条自评 + 合流 | pricing、history 曲线 | 成本、证据包 |

**唯一主线**：用"故障注入 + 真实反馈 + 量化指标"证明 DoD 九条。本周的产出大多是**报告与证据**，不是新功能——这正是"生产硬化"区别于"堆功能"的地方。

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
- [ ] `resilience/circuit_breaker.py`：熔断器（连续失败→打开→半开→闭合）
- [ ] `resilience/degradation.py`：降级策略（兜底模型/缓存/精简输出）
- [ ] `evals/scripts/chaos.py`：三类故障注入 + 自愈断言
- [ ] `docs/w11/chaos-report.md`：注入项 / 预期 / 实际 / 自愈证据
- [ ] 单测：熔断状态机、降级路径、恢复

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
- [ ] `chaos.py` 三类注入全部可复现，输出自愈证据
- [ ] 注入持续超时 → 熔断打开后快速失败，不再打上游（反向验证没被重试放大）
- [ ] 故障恢复 → 半开试探成功后闭合
- [ ] `make eval` 在演练后仍过门禁

### 🔍 交给我的锚点
chaos-report、CircuitBreaker、自愈断言输出。我会看：**熔断器有没有造第二次重试**（ADR-003）、三类是否都可复现、降级是否有明确标记。

---

## Day 3（3h）· 幂等、断点续跑、灰度与回滚

**目标**：长任务可断点续跑、请求幂等（DoD 可靠性项），并演练灰度与回滚。

### 你要做的事
- [ ] `resilience/idempotency.py`：幂等键（请求/任务级）
- [ ] workflow 断点续跑：Temporal activity 完成态持久化，中途重启不重跑已完成步
- [ ] 灰度：按 principal/比例切新老逻辑（复用 W6 `Principal`）
- [ ] 回滚演练：镜像 tag 回退 + 配置回退（复用 W10 runbook）
- [ ] `docs/w11/reliability-report.md`：幂等/续跑/灰度/回滚四项证据
- [ ] 单测：重复请求幂等、续跑跳过已完成步、回滚后可用

### 引导框架

```python
# resilience/idempotency.py
class IdempotencyStore:
    async def check_and_reserve(self, key: str, ttl_s: int) -> IdempotencyRecord:
        # 同 key 首次 -> 新记录；重复 -> 返回既有结果，不重复执行（对齐 W2/W6 幂等主线）
        ...
```

**算法步骤（今天最容易写错的地方）**
1. 幂等键用 `(principal, 请求内容规范化哈希)`；**参数顺序不同要算同一条**（对齐 W8 回放哈希思路）
2. 断点续跑靠 Temporal：每个 activity 完成时持久化状态，重启后**跳过已完成步**——别让已花钱的检索重跑（成本）
3. 灰度开关用配置驱动（不是改代码重新发版）；回滚要验证"回退后 `make eval` 仍绿"
4. 幂等与 HITL（W10）联动：审批过的危险动作重复请求不再执行第二次
5. 续跑/回滚演练都记进 `reliability-report`，带时间戳与证据

**自检**
- [ ] 重复提交同一 `POST /v1/tasks` → 只执行一次，返回同一 task_id
- [ ] 长任务跑到第 3 步时"重启" → 从第 4 步续跑，前 3 步不重跑（反向验证）
- [ ] 回滚到上一镜像 tag → `/health` 与端到端仍可用

### 🔍 交给我的锚点
idempotency、续跑测试、reliability-report。我会看：幂等键规范化是否正确、续跑是否真跳过已完成步、回滚是否验证过可用。

---

## Day 4（2.5h）· 真实反馈修 bug（≥5 条）

**目标**：用 W10 收集的真实反馈，修 ≥5 条 bug，产出「线上问题 → 根因 → 修复」记录（证据包第 7 件）。

### 你要做的事
- [ ] 从反馈渠道整理 ≥5 个真实问题
- [ ] 每个用 W4 `root-cause-playbook` 定位根因（检索/提示词/工具/模型四层）
- [ ] 逐个修复 + `make eval` 验证；补一条回归用例防止复发
- [ ] `docs/w11/incidents/incident-{1..5}.md`：五篇记录
- [ ] 更新 `docs/log.md`

### 引导框架

**incident 模板（每篇固定）**
```
## 现象
真实用户怎么触发的、复现命令、当时的指标/trace 链接
## 根因
W4 playbook 定位到哪层，证据字段（span/日志值）
## 修复
改了哪个文件、引用哪个 ADR
## 验证
修复前后指标 + 新增的回归用例（make eval 通过）
```

**算法步骤**
1. **只记真实问题**，别把 W1–W10 的内部调试凑数——ROADEMAP §3.3 要的是"上线后遇到的最难的三个问题"
2. 每条修复必须补回归用例并过门禁，否则等于没修
3. 优先修"影响成功率/成本/安全"的，按 SLO 误差预算（D1）排序
4. 至少挑 3 条整理成"最难的三个问题"叙述（W12 模拟面试素材）

**自检**
- [ ] ≥5 篇 incident 各有复现、根因、修复、验证四段
- [ ] 每篇对应的回归用例在 `make eval` 中通过
- [ ] 修复后 history 曲线的成功率/成本有可见改善

### 🔍 交给我的锚点
5 篇 incident、回归用例、修复前后指标。我会看：是否真实、根因是否用 playbook 归因、有没有补回归用例。

---

## Day 5（2.5h）· 成本优化冲刺 + DoD 逐条自评 + 合流

**目标**：成本优化（语义缓存/模型分流/批处理）+ **DoD 九条逐条自评**，周五合流 + W11 末 P3 决策。

### 你要做的事
- [ ] `resilience/semantic_cache.py`：语义缓存（相似 query 复用结果）
- [ ] 模型分流：按难度把简单任务路由到 cheap（实现 architecture §8 的 Router 策略）
- [ ] 批处理：可并行的改写/检索合并
- [ ] `docs/w11/dod-self-assessment.md`：DoD 九条逐条自评（达标/缺口/证据链接）
- [ ] 更新 `evals/history/YYYY-Www.json`、`docs/log.md`；周五合流仪式
- [ ] **W11 末 P3 决策**：余量 ≥15h 才解锁一个深入档（前端/多语言最佳挂载点）

### 引导框架

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
4. **DoD 自评表**：九条逐项写「达标 / 部分达标 / 缺口 + 证据链接」，**缺口如实标注**（ROADEMAP 检查点原话："达标或明确标注缺口"）
5. 合流：合 main → `make eval` 过门禁 → 写 history json → `docs/log.md` → P3 决策

### 🔍 Gate 11 · 周末验收（全绿才算过 W11）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | SLO | 三张图可告警，延迟用 p95，成本区分 estimated |
| 2 | 混沌 | 三类故障注入可复现且自愈，≥90% 不崩 |
| 3 | 不重复重试 | 熔断器未与 client 重试叠加（ADR-003） |
| 4 | 幂等续跑 | 重复请求幂等，续跑跳过已完成步 |
| 5 | 回滚演练 | 回退后端到端仍可用 |
| 6 | 真实修 bug | ≥5 篇 incident，各带根因与回归用例 |
| 7 | 成本优化 | 三手段单变量 A/B，成本下降且过门禁 |
| 8 | DoD 自评 | 九条逐条达标/缺口 + 证据链接 |
| 9 | P3 已判定 | 按余量明确解锁/不解锁，写进 log |
| 10 | 合流 | history 接上，端到端仍可跑 |

---

## 本周最容易返工的 6 处

1. **熔断器又造了一次重试** → 与 ADR-003 叠加，故障时流量放大（本周头号坑）
2. **延迟 SLO 用均值不用 p95** → 与 DoD 原话不符，长尾问题被掩盖
3. **成本不区分 estimated / cache_hit / replay** → 成本看板失真（W1/W8 已两次强调）
4. **断点续跑重跑已完成步** → 已花钱的检索重复执行，成本失控
5. **incident 拿内部调试凑数** → 不是"上线后真实问题"，证据包第 7 件不成立
6. **DoD 自评报喜不报缺口** → 检查点要求"达标或明确标注缺口"，注水会被一眼看穿

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| SLO/告警 | `ops/{slo.py,alerts.yml}`、`docs/w11/slo-spec.md` |
| 混沌/容灾 | `resilience/{circuit_breaker,degradation}.py`、`evals/scripts/chaos.py`、`docs/w11/chaos-report.md` |
| 可靠性 | `resilience/idempotency.py`、`docs/w11/reliability-report.md` |
| 真实反馈 | `docs/w11/incidents/incident-{1..5}.md` |
| 成本优化 | `resilience/semantic_cache.py`、Router 新策略、`docs/w11/dod-self-assessment.md` |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W11 条目 + P3 决策） |

---

## P3 挂载点（W11 末 = 第三个决策点，见 ROADEMAP §7）

- **前端极简**（P3-1）：最佳挂载点就在 W11 末 / W12——单页 HTML + SSE 渲染（默认档 0.5 天），让 demo 观感最直观
- **多语言**（P3-2）：若目标 JD 有"出海/跨境"，W10 内容周已铺好，本周补评测集维度
- 余量 ≥15h 才解锁**一个**深入档，一次只开一个；**默认档一律完成**
- 若余量不足，明确写"不解锁"，把时间留给 W12 的框架提取与文档（主线）
