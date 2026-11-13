# SLO 规格（W11 D1）

> 一句话：**三个数（成功率 / 首 token p95 / 单任务成本）各自有唯一的口径、目标、窗口，
> 并且都能算"还剩多少余量"。** 口径写在 `ops/slo.py`，用法写在 `ops/alerts.yml`，
> 取值写在 `ops/slo_report.py`。三处都不重复定义目标值。

## 1. 三条 SLO

| # | 名称 | SLI（读哪个字段） | 目标 | 窗口 | 判定 |
|---|---|---|---|---|---|
| S1 | `availability` | `RequestLog.error_code` 为空的请求占比 | 0.99 | 28d | `gte` |
| S2 | `latency_p95` | `first_token_ms` ≤ 1500ms 的请求占比 | 0.95 | 7d | `lte` → 折算成占比 |
| S3 | `cost_per_task` | 单任务 `cost` ≤ 2.0 CNY 的任务占比 | 0.95 | 7d | `lte` → 折算成占比 |

**为什么 S2/S3 写成"占比"而不是"毫秒/元"**：写成绝对值的话，三条 SLO 会得到两套
预算语义——S1 说"还能坏 1% 的请求"，S2 说"还能慢 50ms"，这两个数不能相加，
也回答不了"这个窗口总共还剩多少余量"。折算成占比之后只有一套公式。

代价是：报"p95 是多少毫秒"时要从样本重算（`percentile()`），不能从预算反推。
报数用样本、判预算用占比，这个分工写在 `ops/slo.py` 的模块头。

## 2. 每条 SLO 的边界条件（这一节比上一节重要）

没有边界条件的 SLO 会在半年后变成一场关于"算不算"的争论，而两种算法都能自圆其说。

### S1 可用性：分母不含"调用方的错"

| error_code | 计入分母？ | 理由 |
|---|---|---|
| `AUTH_REQUIRED` / `AUTH_INVALID` | **否** | 401 是调用方没带/带错 key |
| `CONTENT_BLOCKED` | **否** | 400/422 是输入或产物不合格 |
| `BAD_REQUEST` | **否** | 参数校验失败 |
| `UPSTREAM_*` / `RATE_LIMITED` / `CONFIG_ERROR` | 是 | 这是我们的问题 |
| `null`（成功） | 是，算达标 | — |

含进去的后果很具体：**被扫的时候可用性看起来下降**，真故障被攻击噪声淹没。
实现见 `ops/slo_report.py::is_our_fault()`。

### S2 延迟：只用 `first_token_ms`，缺了就不参与

DoD 原话是「首 token p95 ≤ 1.5s」，所以取 `first_token_ms`，不取整体 `latency_ms`。
非流式请求没有这个字段——**此时不参与**这条 SLO，而不是拿 `latency_ms` 顶替：
两者量级差 10 倍（W53 离线档：整体 p95 1.6ms vs 首 token 无样本），混算等于把长尾抹平。

> 一个必须说清的落差：当前离线评测档（fixture 回放）几乎不产生 `first_token_ms`，
> 所以这一列的覆盖率很低。`ops/slo_report.py` 会把覆盖率打出来（`样本 0/2，非流式不参与`），
> 而**不是**用 `latency_p95_ms` 填一个好看的数字。这条 SLO 的真实覆盖率
> 要等在线档（真实厂商流式响应）才有意义。

### S3 成本：三个来源必须分开记

| 来源 | cost | 是否进 `cost_per_task` 分母 | 标记 |
|---|---|---|---|
| 真实调用 | 实价 | 是 | — |
| 语义缓存命中（W11 D5） | 0 | **否**（单列命中率） | `cache_hit` |
| 回放（W8） | 0 | **否** | `replayed` |

把命中/回放算进分母，会让"开了缓存 → 成本下降"变成一句自证——
因为分母也是被稀释的。**单列命中率、只对真实调用算单任务成本**，成本下降才是真的。

另有一条：`usage_estimated=True` 的记录要单独给占比。
>20% 时这条 SLO 的结论不可信（W1 Day5 的教训：厂商没给 usage 时我们按字符估，
>估算基准一改，成本曲线就跳一次，而那次跳动会被误读成"优化生效"）。

## 3. 误差预算与"冻结"策略

```
允许坏掉的量   allowed_bad = total × (1 - target)
已经坏掉的量   bad         = total × (1 - observed)
剩余比例       remaining   = max(0, allowed_bad - bad) / allowed_bad
燃烧速率       burn_rate   = (bad / allowed_bad) / elapsed
```

`elapsed` 是窗口已过比例。它**只**用于 burn rate：同样的错误数，在窗口第 1 天发生
比最后 1 天紧迫得多。预算余量本身不按 elapsed 折算——已消耗的额度不会因为时间过去而变多。

**预算耗尽时做什么**（写进流程，不是写在心里）：

| 剩余预算 | 状态 | 允许的动作 |
|---|---|---|
| > 50% | 正常 | 新功能、重构、依赖升级 |
| 10%–50% | 收紧 | 只做修复与小改动；每个 PR 要说明"为什么必须现在做" |
| < 10% | **冻结** | 只做硬化与修复；新功能一律延后 |
| 0（耗尽） | 事故 | 停发布，先写复盘 |

这条表的价值不在"我们会不会遵守"，而在**冲突时有一个第三方依据**：
"这周能不能上这个功能"不该由想做的人判断。

## 4. 告警：三条规则 + 一条流程规则

详见 `ops/alerts.yml`（照 Prometheus 格式写，但我们**不跑** Prometheus——
现在没有秒级写入压力，`python ops/slo_report.py --alert` 够用；
照它的格式写是为了将来接时序库时不用重写规则）。

| 规则 | 阈值 | 持续时间 | 级别 |
|---|---|---|---|
| `AvailabilityBudgetBurningFast` | burn > 5x | 5m | page |
| `FirstTokenLatencyBudgetBurning` | burn > 3x | 10m | warn |
| `CostPerTaskOverBudget` | burn > 2x | 15m | warn |
| `ApprovalBacklog`（流程） | 待审 > 3 | 10m | warn |

三个设计点：

1. **阈值用 burn rate 不用裸成功率**。"成功率 97%"是好是坏取决于目标：
   目标 99.9% 时是灾难，目标 95% 时很健康。裸阈值必然与某个目标绑死。
2. **每条都有持续时间**。没有它，30 秒抖动会叫醒值班，而被误报叫醒三次之后
   这个告警就会被静音——那时真故障也没人看。
3. **审批积压单列**。它和"卡住"在监控上长得一样（都是"没进展"），
   但处置动作完全相反：一个去催人，一个去重启。混在一条规则里，
   值班会先重启，而重启不会让任何人去批。

## 5. 三张图的数据来源

| 图 | 数据源 | 命令 |
|---|---|---|
| 可用性 | `RequestLog` JSONL 的 `error_code` | `python ops/slo_report.py --log app.jsonl` |
| 延迟 | 同上，`first_token_ms` 的 p95（nearest-rank） | 同上 |
| 成本 | 同上，`cost` 求和 ÷ 非缓存任务数；另给 `usage_estimated` 占比 | 同上 |
| 周级趋势 | `evals/history/*-eval.json` 的 `by_difficulty` | `python ops/slo_report.py` |

**不另造埋点**。自己再埋一套"业务指标"的后果是两套数对不上，
而对不上的时候没人知道该信哪个。W11 的成本看板与 W3 的评测报告必须是同一个数。

## 6. 实跑记录 `[实测]`

```
$ python ops/slo_report.py
== 周级趋势（by_difficulty 聚合，直接读落盘，不重算）==
week      档        成功率   首token p95    整体 p95     成本/任务     估算占比
2026-W46  1      1.000            -     2.0ms    0.0000     0.0%
...
2026-W53  3      1.000            -     0.2ms    0.0000     0.0%

== 告警判定（基于 2026-W53）==
  [ ok  ] availability_burn_fast     page
  [ ok  ] latency_p95_burn           warn
  [ ok  ] cost_per_task_burn         warn
```

```
$ python ops/slo_report.py --log /tmp/ga.log      # 真实进程的两条 RequestLog
== RequestLog：2 条 ==
可用性          1.0000  (分母 2，其中失败 0)
首 token      p95=- 达标率=- (样本 0/2，非流式不参与)
成本           合计 0.0000 CNY 估算占比 0.0%
availability   SLI=1.000 target=0.99 bad=0/0.0 剩余=100.0% burn=0.00x
```

## 7. 已知缺口

1. **没有线上连续采样**：告警判定是基于"周级分档数据"离线算的，
   它能回答"如果线上是这个数，规则会不会响"，**不能**回答"响得及时不及时"——
   后者需要 `(timestamp, value)` 序列，本仓库还没有那个数据源（要接时序库或定时抓取）。
2. **首 token 覆盖率低**：离线档不产生该字段，S2 目前在离线数据上基本无样本。
3. **没有多窗口燃烧率**：单窗口 + 持续时间已经能挡住抖动；多窗口（1h/6h/3d）
   是给 7×24 值班团队用的，我们只有一个人。
4. **成本的口径依赖 `models.yaml` 的 pricing**：pricing 改了，历史成本曲线会出现
   台阶，而台阶会被误读成"优化/退化"。
