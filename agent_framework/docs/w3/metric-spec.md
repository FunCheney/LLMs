# 指标字典（W3 D1）

> 这份文档的作用是**在写 runner 之前把"怎么算分"钉死**。
> 顺序反了就是灾难：先有脚本后补定义，指标会变成"代码碰巧算出来的东西"，
> 而 W4 的回归门禁、W11 的成本看板、W12 的趋势图全都建在这几个数上。

数据来源唯一性
--------------

所有指标的数据**只允许来自两处**：

1. `RequestLog`（`observability.py`）——延迟、token、成本、错误码；
2. `expect` 断言结果（`evals/golden/*.jsonl`）——通过 / 失败。

**任何指标都不许自己 `time.monotonic()`。** W1 已经把 `first_token_ms` 的定义固定在
gateway 里（首个 `text_delta`/`reasoning_delta` 到请求开始的毫秒差），再计一次时就是
第二套口径，W11 的曲线会在某一天突然断层，而没人知道是哪次改动断的。

---

## 1. 四态状态机（先定义"什么不算失败"）

一条 case 跑完只可能落在四个状态之一。**分母怎么取，全看这一步。**

| 状态 | 触发条件 | 进 `success_rate` 分母 | 含义 |
|---|---|---|---|
| `passed` | `expect` 全部成立 | 是（分子） | 系统答对了 |
| `failed` | `expect` 有断言不成立，且链路本身正常 | 是 | 系统答错了——**这才是质量问题** |
| `interrupted` | 上游 5xx / 限流 / 鉴权 / 能力不支持 / 超时 | **否** | 可靠性问题，不是模型笨 |
| `skipped` | 该 case 在当前档位没有可执行判据（如离线档遇到语义类 case） | **否** | 没测，不是通过 |

```
success_rate = passed / (passed + failed)
```

**`interrupted` 必须单列。** 把上游 502 算成"答错"会让一次厂商故障把质量分打掉
20 个点，然后团队去调 prompt——查三天,发现是对方在发版。

**`skipped` 不许算进 `passed`。** 这是评测系统最常见的自欺：离线档跑不了语义题，
就把它们都记成通过，报告上 100% 一片绿。本仓库的规矩是——没测就说没测。

---

## 2. 五层指标

### 2.1 任务层

| 指标 | 定义 | 公式 | 数据来源 | 边界 |
|---|---|---|---|---|
| `success_rate` | 断言通过率 | `passed / (passed + failed)` | `expect` 断言 | 分母不含 `interrupted` / `skipped` |
| `interrupted_rate` | 上游中断占比 | `interrupted / total` | `RequestLog.error_code` / 异常类型 | 单列，不并入 `success_rate` |
| `skip_rate` | 未测占比 | `skipped / total` | runner 判定 | 报告里必须显式打印，不许隐藏 |

按 `difficulty`、`tags` 分组的成功率同样按上式算；某组样本数为 0 时返回 `None`
而**不是 0.0**——"没有样本"和"一个都没过"是两件事，`0.0` 会让趋势图画出根本不存在的塌陷。

### 2.2 工具层

| 指标 | 定义 | 公式 | 数据来源 | 边界 |
|---|---|---|---|---|
| `tool_errors` | 工具调用错误数 | 计数 | `ToolCall.arguments_error` 非空 / 调用了未声明的工具 | 判据沿用 conformance A14，**不另立一套** |
| `tool_error_rate` | 工具错误占比 | `tool_errors / tool_calls`（`tool_calls` = **尝试次数，含失败的**） | 同上 | 没有工具调用时返回 `None`，不返回 0 |

分母写成 `tool_calls + tool_errors` 是常见的重复计数错误：失败的调用本身也是一次调用尝试，
加第二次会让错误率凭空减半。

A14 的判据是"工具调用落在声明 schema 内"。golden 里只要 case 声明了 `tools`，
工具名不在声明列表里就算错——不因为"模型调了个差不多的工具"就放过。

### 2.3 延迟层

| 指标 | 定义 | 数据来源 | 边界 |
|---|---|---|---|
| `latency_p50_ms` / `latency_p95_ms` | 请求总耗时分位 | `RequestLog.latency_ms` | 含重试的总时长（这才是用户感受到的） |
| `first_token_p50_ms` / `first_token_p95_ms` | 首 token 分位 | `RequestLog.first_token_ms` | 只统计**非空**值；阻塞式调用不参与（它没有首 token 概念） |

分位算法：线性插值（`p = (n-1) * q`），与 W1 `bench_baseline.py` 一致。
样本数 `< 5` 时只打 p50 与 max，**不打 p95**——5 个点算 p95 是把最大值改个名字。

`first_token` 只对 `stream=True` 的调用有值。把阻塞式请求的 `latency_ms` 混进首 token
样本池，会让首 token 看起来"慢了 10 倍"，而实际是两回事。

### 2.4 成本层

| 指标 | 定义 | 公式 | 数据来源 | 边界 |
|---|---|---|---|---|
| `cost_total` | 总成本 | `Σ RequestLog.cost` | `RequestLog.cost` | **只累加 `usage_estimated=False` 的样本** |
| `cost_per_task` | 单任务成本 | `Σ cost(精确样本) / 精确样本数` | 同上 | 精确样本为 0 时返回 `None` |
| `usage_estimated_ratio` | 估算占比 | `estimated_count / total` | `RequestLog.usage_estimated` | `> 0` 时报告必须显著标注 |

**为什么估算样本不能混进均值**：`usage_estimated=True` 意味着厂商没给 usage，
框架按字符数猜的 token 数。猜出来的数和实测数放在一起求平均，得到的数字既不是
实测也不是上界，是一个没有意义的数。宁可 `cost_per_task=None` 并标注"全部为估算"。

币种：`RequestLog.currency` 恒为 CNY（`models.yaml` 里定价的人民币口径）。
混币种时报告直接报错，不做汇率换算——汇率换算是 W11 成本看板的事，评测层不承担。

### 2.5 质量层

| 指标 | 定义 | 本周状态 |
|---|---|---|
| `judge_score` | LLM-as-judge 打分（对齐/忠实/完整性） | **`null`，标 pending** |
| `faithfulness` | 答案是否有 `references` 支撑 | **`null`，W5 有检索后才有输入** |

本周**不填假数**。`judge_score=0.0` 和 `judge_score=null` 在趋势图上是两条完全不同的线，
前者是"评判结果为零分"，后者是"还没开始评"。

---

## 3. 离线段与在线段的判据分离

这是本仓库对 W3 任务书的一处**刻意扩展**，理由如下。

golden case 有两套判据：

| 字段 | 档位 | 锚在哪 | 能证明什么 |
|---|---|---|---|
| `expect` | 在线档（真实模型） | 语义：答案该包含什么、不该出现什么 | 模型质量 |
| `offline.expect` | 离线档（夹具回放） | 夹具实际输出（`MCP`、`我先检索` 等） | 断言管道、指标埋点、错误映射**能工作** |

只有一套判据会怎样：离线档用语义判据，夹具输出对不上，于是要么全红（没人看门禁），
要么把判据放宽到"只要不报错就算过"（门禁形同虚设）。分离之后，离线档诚实地只声明
"管道通",并在报告页脚写明 `mode=offline → 质量分不可用于对外结论`。

**离线档的数字永远不进 `evals/history/*.json` 的 `quality` 块。** 它只填
`pipeline` 块（证明跑得通），`quality.success_rate` 保持 `null` 直到在线档跑过。

---

## 4. 复现性约定

| 项 | 约定 | 违反的后果 |
|---|---|---|
| 模型标识 | 钉 `alias`（`fixture` / `qwen3.7-plus`），**不写 wire 名** | 厂商改 wire 名 → 昨天的数字对不上 |
| 温度 | `temperature=0` 强制注入，case 里写别的值直接报错 | 两次跑数字不同，"可复现"检查点直接挂 |
| 输入 | 夹具 JSON 固定；在线档记 `model_snapshot`（厂商版本号或日期） | 厂商静默升级模型 → 曲线出现无法解释的跳变 |
| 时间 | 在线档必须记 `period`（时段） | DeepSeek 峰谷价差 2 倍，"成本降 40%"可能只是换了时段 |
| 随机性 | 离线档零随机；在线档 `seed` 仅在厂商支持时传 | — |

---

## 5. 常见误读（报告里要主动防的）

1. **"离线档 100% 通过"≠"系统很好"** —— 夹具回放只证明链路通。
2. **"p95 比上周低了"可能是样本变了** —— 分母是 `passed+failed`，`interrupted` 一多，
   分母变小，幸存样本偏向简单题。看 `success_rate` 必须同时看 `total`。
3. **"成本降了"可能只是切到空闲时段**（DeepSeek 峰谷差 2 倍）。
4. **`usage_estimated_ratio > 0` 时不要引用 `cost_per_task`** —— 那是个混合数。
5. **`skipped` 变多不是"变稳了"** —— 是测试覆盖退化了，`skip_rate` 本身要进趋势图。
