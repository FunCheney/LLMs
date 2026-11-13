# W11 · DoD 九条逐条自评

> 检查点原话是「**达标或明确标注缺口**」。所以这份表里"部分达标"占了 5 条——
> 它们不是"没做"，是**机制做了、数字没在真实流量上标定**。
> 把这两件事写成同一档（都叫"达标"），这份自评就失去了意义。

| # | 维度 | 判定 | 一句话 |
|---|---|---|---|
| 1 | 可用性 | 🟡 部分达标 | 不崩、不静默失败已验证；"连续 7 天"与"≥90%"没在真实流量上量化 |
| 2 | 延迟 | 🟡 部分达标 | p95 口径 + 告警规则已落地；数字未在真实上游标定 |
| 3 | 成本 | 🟡 部分达标 | 归因看板与三手段 A/B 达标；**per-task 预算上限与超支熔断没做** |
| 4 | 质量 | 🟡 部分达标 | 54 条 golden、门禁绿；judge ≥4.0 未在真实模型上跑 |
| 5 | 可靠性 | ✅ 达标 | 三类故障可自愈 + 断点续跑 + 请求幂等，全部实跑 |
| 6 | 安全 | ✅ 达标 | 注入 10/10 拦截、误报 6/6 放行、工具审计、HITL |
| 7 | 工程 | 🟡 部分达标 | CI 门禁/一键部署/配置回滚达标；**覆盖率未测**、镜像回退未实跑 |
| 8 | 可观测 | 🟡 部分达标 | 本地可回放（span + Run Timeline）；远端 trace 后端未接真实 key |
| 9 | 文档 | 🟡 部分达标 | README/架构/13 篇 ADR/RUNBOOK 齐；**12 周趋势图缺（W12 D3）** |

统计：**2 条达标 / 7 条部分达标 / 0 条未做**。

---

## 1 · 可用性 🟡

| 硬指标 | 状态 | 证据 |
|---|---|---|
| 连续 7 天可用 | ❌ 未验证 | 本机进程，无 7×24 部署（`evals/history/2027-W01.json` 的 `availability.note`） |
| 正常成功率 ≥95% | 🟡 离线档达标 | 全量 1446 passed / 0 failed；golden 54 条 success_rate 1.0（离线档只代表执行管道）；smoke 10/10 |
| 注入工具故障后 ≥90% 且不崩、不静默失败 | ✅ 机制达标 | `evals/scripts/chaos.py` 7 场景；`docs/w11/chaos-report.md` |

**缺口**：混沌演练证明了"受控"（熔断不叠加、探测只放一个、失败有明确码），
但**没有给出一个成功率百分比**——演练里每个场景都是"注入即全挂"，
报出来是 0，没有信息量。那个数只能从真实 `RequestLog` 算，本周没有真实流量。

**为什么这不算达标**：DoD 写的是"≥90%"，是数字；我只有机制。

---

## 2 · 延迟 🟡

| 硬指标 | 状态 | 证据 |
|---|---|---|
| `/v1/chat` 首 token p95 ≤1.5s | 🟡 口径达标、数字缺 | `ops/slo.py`（`first_token_ms` 而非 `latency_ms`）、`ops/alerts.yml`、`docs/w11/slo-spec.md` |
| 完整任务 p95 ≤120s | 🟡 同上 | 同上（`TASK_COMPLETION` 目标） |
| 有分段耗时拆解 | ✅ | `StageTrace`（retrieval / rerank / rewrite / style 分段）+ span per HTTP |

**缺口**：离线档的延迟是 `httpx.MockTransport` 的 0.1–2ms，**拿它标定 SLO 是自欺**。
`cost_ab.py` 的报告里因此**删掉了 p95 列**，改在 JSON 里写 `note`——
一列看起来精确的假数字，比没有这一列更危险。

---

## 3 · 成本 🟡

| 硬指标 | 状态 | 证据 |
|---|---|---|
| 单次完整任务 ≤¥2.0 | ❌ 未测 | 离线 0 元；真实未跑（无凭证） |
| per-task 预算上限 + 超支熔断 | ❌ **没做** | `retrieval_budget_ms` 是**延迟**预算，不是钱的预算 |
| 成本归因看板 | ✅ | `RequestLog` 的 `cost/currency/usage_estimated/cache_hit`；`ops/slo_report.py` 成本图 |

**缺口（最重要的一条）**：**没有任何"钱花超了就停"的机制**。
现在的成本控制全是"事前的比例"（便宜档 + 缓存），没有"事中的闸门"。
一个反复重试 + 长上下文的请求可以安静地花掉一整天预算，而看板要到第二天才显示。
这条本该在 W11 做，没做，记在这里而不是模糊成"成本优化达标"。

**成本优化三手段**（Gate 11 #10，`evals/scripts/cost_ab.py` 实测）：

| 手段 | 结论 | 数字 |
|---|---|---|
| 语义缓存 | ✅ 采用（默认关，可灰度） | 上游调用 16→10，成本 −37.6%，命中 6/16 |
| 难度分流 | ✅ 采用（默认关，可灰度） | 质量优先基线 −47.4%、零误分流；**当前 yaml 下成本 ×13** |
| 批处理 | ⚠️ 只取"并发"部分 | 墙钟 ×2.99 加速，调用数不变；"合并成一次调用"W8 已否决 |

---

## 4 · 质量 🟡

| 硬指标 | 状态 | 证据 |
|---|---|---|
| golden ≥50 条 | ✅ 54 条 | `evals/golden/golden_v1.jsonl`（sha256 `1c31832c2bbf`） |
| 任务成功率 ≥80% | 🟡 离线 1.0 | `evals/history/baseline.json`：50 passed / 0 failed / 4 skipped |
| LLM-as-Judge ≥4.0/5 | ❌ 未测真实模型 | `evals/scripts/judge_agreement.py` 只有规则基线 |

**缺口**：`success_rate 1.0` 是**离线档**，只说明"执行管道没断"，
不说明"答案好"。把它写成"质量达标"是这类项目最常见的一次自我欺骗。

---

## 5 · 可靠性 ✅

| 硬指标 | 状态 | 证据 |
|---|---|---|
| 模型超时 / 工具 500 / 限流 三类故障可自愈 | ✅ | chaos F1–F7，7 场景全绿（`docs/w11/chaos-report.md`） |
| 长任务可断点续跑 | ✅ | `runtime/resume.py` + `evals/scripts/resume_drill.py`：`skipped=('step-1','step-2')`、只跑 `step-3/4`、终态 completed |
| 请求幂等 | ✅ | `runtime/model.py` 稳定幂等键 + `tests/test_idempotency.py` 14 条 |

这一条是本周唯一"可以拿数字说话"的：`skipped` 与 `executed` 是两个不同的集合，
续跑演练里它们是**分开打印**的（`skipped=('step-1','step-2')` / `executor 实际调用=['step-3','step-4']`）。
"续跑跑了 4 步"和"续跑跑了 2 步"在日志里长得一样，只有分开数才看得出省了钱。

---

## 6 · 安全 ✅

| 硬指标 | 状态 | 证据 |
|---|---|---|
| 10 条 prompt injection 全拦截 | ✅ 10/10 | `evals/golden/redteam_v1.jsonl` + `evals/scripts/run_redteam.py` |
| 误报可控 | ✅ 6/6 放行 | 同上（`normal_allowed`） |
| 工具调用全审计 | ✅ | W8 `actions` 审计 + 幂等键 |
| 高风险动作 HITL | ✅ | W10 `WAITING_APPROVAL` 状态 + `tests/.../test_hitl_flow.py` 8 条 |

补充证据（本周新增）：`w11_fork_ablation.py` 实验 A 证明"解封皮是唯一修复"——
不做这件事时，隔离后的正常请求放行率是 **0%**（`fake_delimiter` 误报 6 次）。

---

## 7 · 工程 🟡

| 硬指标 | 状态 | 证据 |
|---|---|---|
| 核心模块单测覆盖 ≥70% | ❌ **未测** | 没有覆盖率工具链（W12 D2 补） |
| CI 跑 lint + test + eval 回归门禁 | ✅ | `make eval` = check + golden + run-eval-judge + gate；本轮 ruff / mypy(134 files) / 1446 passed 全绿 |
| 一键部署 | ✅ | `ops/docker-compose.yml`、`ops/deploy_check.sh` |
| 可回滚 | 🟡 配置回退 ✅ / 镜像回退 ❌ | Kill Switch 实测 **0.5s**（`docs/w11/asset-versions.md` §3）；镜像回退未实跑 |

**缺口**：覆盖率是"工程"这条里唯一必须靠数字判断的项，而它恰恰没测。
用"1446 条测试"代替"覆盖率 ≥70%"是偷换——测试条数多但集中在一个模块，
覆盖率可以很低。

---

## 8 · 可观测 🟡

| 硬指标 | 状态 | 证据 |
|---|---|---|
| 任意一次任务可回放完整 trace | 🟡 本地 ✅ / 远端 ❌ | `tracing.py` 的 span + `runtime/timeline.py` 的 Run Timeline（样例见 `docs/w11/reliability-report.md`） |
| 含每次 LLM 调用、工具调用、token 与耗时 | ✅ | `RequestLog` 20 字段 + per-attempt span |

**缺口**：`Langfuse` exporter 缺 key 时**强制降级为 Noop**（这是对的——观测组件
挂掉不能影响主链路），但结果是"远端 trace 后端从未在真实 key 下跑过"。

「只凭 Run Timeline 能讲清全过程」这一条是实打实过掉的：13 行 Timeline
能看出"step-1 重试后成功、参数未变"、"step-3 首次超时"、"step-4 在审批处停住"。

---

## 9 · 文档 🟡

| 硬指标 | 状态 | 证据 |
|---|---|---|
| README | ✅ | `README.md` |
| 架构图 | ✅ | `docs/architecture.md` |
| ADR ≥5 篇 | ✅ **13 篇** | `docs/w{2,3,4,5,6,8,9}/adr-*.md` + `docs/w11/chaos-report.md` 内 ADR-W11-001/002 |
| runbook | ✅ | `ops/RUNBOOK.md`（含 §10 镜像回退、§11 模型指向、§12 检索参数） |
| **12 周指标趋势图** | ❌ **缺** | 计划在 W12 D3 |
| 检查点三份报告 | ✅ | 压测（W2 基线）、混沌 `chaos-report.md`、安全（W10 redteam + `docs/w11/incidents/`） |

13 篇 ADR 全部是"候选 / 约束 / 证据 / 结论"四段式，不是"我们选了 X"的一句话。

---

## 附：这份自评里最该被记住的两句

1. **"机制达标"和"数字达标"必须分开写。** 7 条"部分达标"里，5 条是机制有了、
   数字没有——把它们合并成"达标"，就是在用一个能跑的系统冒充一个被验证过的系统。
2. **缺口里最贵的不是"没做"，是"做了但没人知道它没生效"。** 本周就踩到两次：
   `skill_drift` 把正确做法也报成漂移（一份整天误报的清单最后没人看）、
   语义缓存把每个带 `temperature` 的请求都判成不可缓存（于是它在真实路径上从未生效）。
   两处都是对方先红、我才发现。
