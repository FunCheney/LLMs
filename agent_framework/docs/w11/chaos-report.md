# W11 D2 · 混沌演练与降级报告

> 交付物：`resilience/circuit_breaker.py`、`resilience/degradation.py`、`evals/scripts/chaos.py`、
> 单测 `test_circuit_breaker.py`（22）/ `test_degradation.py`（9）、本报告。
> 所有数字都来自本机实跑（`uv run python evals/scripts/chaos.py`，耗时 62ms，零网络）。

## 0. 一句话结论

熔断器**不重试**、**打开了就真的不打上游**、**只统计上游不健康**；
降级默认**不兜底**，开了兜底也必须能被结构化识别。
三类故障注入 + 四项边界性质全部断言通过，但**"成功率 ≥90%"这条 DoD 没有在演练里量化**——
它是 SLO 口径，由 `ops/slo_report.py` 从真实 `RequestLog` 算，见 §7 缺口 1。

## 1. 为什么注入传输层，而不是连真厂商

| 做法 | 问题 |
|---|---|
| 连真厂商制造故障 | 429/500 什么时候来不由我们决定。跑三次两次没触发，那不是演练，是抽奖 |
| 把整条链路换成假实现 | 变成"测我的替身写得对不对"，而不是测链路 |
| **换最底层的 `httpx.MockTransport`** | 用真的 `LLMGateway`（真重试、真计量、真错误映射），只换传输 |

代价必须说清：注入的错误是**协议层错误**（超时 / 500 / 429 / 400），
不是"模型输出了胡话"或"工具执行抛异常"。后者在 W8 的 `tools/` 单测里，本报告不覆盖（§7 缺口 2）。

## 2. 七个场景与实测

| # | 场景 | 注入 | 预期 code | 上游调用 | 熔断拒绝 | 拒绝后上游 | 末态 |
|---|---|---|---|---|---|---|---|
| F1 | 模型超时 | `timeout` | `UPSTREAM_TIMEOUT` | 18 | 9 | **0** | open |
| F2 | 上游 500 | `http500` | `UPSTREAM_ERROR` | 12 | 6 | **0** | open |
| F3 | 限流 429 | `ratelimit` | `RATE_LIMITED` | 12 | 4 | **0** | open |
| F4 | 我方 400 | `http400` | `BAD_REQUEST` | 24 | 0 | 0 | closed |
| F5 | 半开单探测 | `http500` | `UPSTREAM_ERROR` | 1 | 4 | 0 | closed |
| F6 | 降级标记 | `http500` | `UPSTREAM_ERROR` | 3 | 0 | 0 | closed |
| F7 | **接线后熔断** | `timeout` | `UPSTREAM_TIMEOUT` | 12 | 4 | **0** | open |

合计 82 次上游调用，全部落在注入传输层里，一次真实请求都没有。

逐条读法：

- **F1**：阈值 3、`max_attempts=2`、候选链 3 条 → 每次 `chat()` 打 6 次上游；
  放行 3 次即 18 次，之后 9 次请求**一次都没打上游**。
  "不放大"在这里是可数的：如果熔断器自己也重试，这个数会变成 18 的倍数。
- **F3**：429 计失败。上游说"我扛不住"时继续打是最坏的反应，它只会更扛不住。
- **F4**：8 次全部失败、熔断器**始终 closed**。400 是我方把参数写错了，
  把它算成上游故障，值班会去查厂商状态页，什么也查不到。
- **F5**：探测飞行中来了 4 个请求，4 个被快速拒绝，**期间 0 次上游调用**；
  探测成功后 1 次调用、状态回到 closed。
- **F7**：用线上形状（`breaker=` 传进 `LLMGateway`）再跑一遍 F1 的性质。
  它的存在意义是：**哪天有人把接线拆了，F1 依然绿，F7 会红。**

## 3. 三个"假绿"，都是这一轮自己抓到的

这一节比 §2 更重要——三个错都不是代码错，是**断言错**，而断言错的测试永远绿。

### 3.1 `threshold × max_attempts` 这个公式少了一个乘数

第一版：`expected_calls = threshold * max_attempts`，跑出来 18 不是 6。

差的是**候选链**：`chat()` 会依次试 `models.yaml` 里该 task 的全部候选
（`qwen3.7-plus` → `deepseek-flash` → `gpt-4o-mini`），每个候选再重试 `max_attempts` 次。
真实倍数是 `放行次数 × 候选链长度 × max_attempts`。

这个乘积属于 `LLMGateway`（ADR-003 归它），**不属于熔断器**。
所以修法不是改数字，而是把断言拆成两条：
一条比对总数（证明链路没做计划外的事），一条单独记 `calls_after_open == 0`
（证明熔断器自己的贡献恰好是 0）。**混在一起，就分不清是谁放大的。**

### 3.2 半开"1 个探测"写成 `gather` 5 个并发，实测 5 个全放行

原因不是熔断器漏判：`httpx.MockTransport` 的 handler 是**同步**的，
探测一次到底、中途不让出事件循环，于是第一个调用返回时状态已 CLOSED，
后 4 个被正常放行——**这是对的**。

这条性质的真实表述是"**探测还在飞行时**，其余请求被快速拒绝"。
用一对 `asyncio.Event` 把探测停在半空，它才可观测。改完后：飞行期间 0 次上游调用、4 个拒绝。

顺带记一条：这类"测试替身比真实世界快太多"的坑，只有在你写**否定式断言**
（"不能发生"）时才会暴露；写"应该成功"的断言永远发现不了。

### 3.3 一个 `error` 字段被覆写成最后一条错误

第一版 `Scenario` 只有一个 `error`。`drive()` 每轮无条件赋值，
于是 F1 报告里显示 `CIRCUIT_OPEN`——被拒绝的是最后几轮，真正要看的
`UPSTREAM_TIMEOUT` 被冲掉了。

拆成 `upstream_error`（最后一次**真的打到上游**的码）与 `last_refusal`
（最后一次被拒的码）。这两个数回答的是两个不同问题：上游怎么了 / 我们拒绝了什么。

## 4. 熔断器的边界（ADR-003 的落地形态）

| 职责 | 归谁 | 为什么不能挪 |
|---|---|---|
| 重试、退避、换候选 | `LLMGateway` | `retryable` 的判定只在那里有 |
| 并发准入、过载 503 | `ConcurrencyLimiter`（W2） | 入口过载是"谁收请求谁拒绝" |
| **连续失败后停止调用上游** | `CircuitBreaker` | 唯一决定"不再尝试"的地方 |

失败计数单位是**一次 `gateway.chat()`**，而那次内部可能已经重试 3 次、换过 2 个候选。
所以看板上"熔断器失败 5 次"对应的真实上游调用可能是 30 次——**这个换算必须在报告里写清**，
否则复盘时会以为"只失败了 5 次就熔断，阈值太敏感"。

### ADR-W11-001：熔断器挂在 `LLMGateway` 里，不挂在 4 个业务调用点

产品层有 4 处 `await gateway.chat(...)`（`answer.py`、`routes_draft.py`、
`pipeline/seo.py`、`pipeline/style.py`）。把熔断器包在每处意味着**新增一处调用就漏一处保护**，
而漏掉的那处不会有任何测试变红。

所以做成 `LLMGateway(breaker=...)` 的可选注入：
- `breaker=None` → 行为与 W10 完全一致（既有全量用例一条都没改）；
- 传了 → `chat()` 把**整条候选链**包进 `breaker.call()`。
- 打开期间抛 `CircuitOpenError` → HTTP **503 `CIRCUIT_OPEN`** + `Retry-After: 5`。

不复用 `ServerBusyError`（同为 503）：**两者处置动作相反**。
`SERVER_BUSY` 是"我们自己并发满了，等一会儿"，`CIRCUIT_OPEN` 是"上游在连续失败，已停止调用它"——
后者值班要做的是去看上游，而不是回头扩我们自己的并发。

配置项（`config.py`）：`BREAKER_ENABLED=true`、`BREAKER_THRESHOLD=5`、`BREAKER_RESET_S=30`。
**默认开着**：一个默认关闭的保护等于没有保护，"它其实一直没生效"只在出事那天被发现。

### ADR-W11-002：降级顺序与任务书不一致，我改了

`w11-tasks.md` 的引导框架写的是"兜底模型（强→便宜）→ 语义缓存 → 精简输出"。
实现里去掉了第一项，理由是它就是**第二次重试**：`gateway` 的候选链已经按
`primary → fallback` 试过一遍，降级层再来一次"换个便宜档"，等于把 ADR-003 拆了。

落地顺序是：**语义缓存（W11 D5 注入）→ 兜底话术（默认关）→ 原样抛出**，每个手段最多一次。

兜底话术默认关，因为一个"请稍后再试"的假回答同时毁三件事：
用户以为拿到了答案不会重试 / SLO 的失败计数少一条 / 复盘时 trace 里只有一次成功。
开了之后返回的 outcome 必带 `degraded=True` + `finish_reason=UNKNOWN` +
`raw_finish_reason="degraded"`，且**不扩 `FinishReason` 枚举**——那是与 adapter 共享的契约层，
"降级"是客户端层的运营概念（与 `retry_count` 同类）。

## 5. 单测补了什么（31 条）

`test_circuit_breaker.py`（22）——守"连续"、"不重试"、"只统计上游"、"半开一个名额"、
"接线真的接上了"：

| 用例 | 它挡的事故 |
|---|---|
| `test_opens_only_after_threshold_consecutive_failures` | 阈值差一次就打开（会造成大量误熔断） |
| `test_success_resets_the_consecutive_counter` | 阈值被写成"累计 N 次" |
| `test_open_breaker_never_calls_the_upstream` | 熔断只是个说法，实际还在打 |
| `test_breaker_never_retries_by_itself` | **熔断器自己重试**（ADR-003 明令禁止） |
| `test_cooling_down_is_visible_on_state_read` | 冷却结束了看板还挂着 open |
| `test_half_open_admits_exactly_one_probe_in_flight` | 放行 N 个探测 → 永远探测不成功 |
| `test_failed_probe_reopens_immediately_without_waiting_threshold` | 探测失败后卡在半开 |
| `test_bad_request_neither_counts_nor_clears` | 我方 400 被算成上游故障（故障域跑偏） |
| `test_gateway_without_breaker_behaves_exactly_as_before` | 加保护顺手改了既有行为 |
| `test_gateway_with_breaker_stops_reaching_adapters_once_open` | **接线漏了**（没接上去的熔断器在单测里永远是对的） |
| `test_circuit_open_surfaces_as_503_with_retry_after` | 错误码/状态码/Retry-After 对不上 |

`test_degradation.py`（9）——守"默认不兜底"、"兜底不能像成功"、"缓存命中要复制"、
"每个手段只试一次"。

门禁：`1296 passed, 30 skipped`（收集 1327）、`ruff` All checks passed、`mypy` 124 files 0 issues。

## 6. 复现

```bash
uv run python evals/scripts/chaos.py            # 人读表
uv run python evals/scripts/chaos.py --json     # 机器可读，进 history
uv run pytest packages/agent_framework/tests/test_circuit_breaker.py \
              packages/agent_framework/tests/test_degradation.py -q
```

## 7. 已知缺口（不修，但要写下来）

1. **"故障期间成功率 ≥90%"没有在演练里量化。** 本演练是"注入 100% 失败"的极端形态，
   报出来的成功率是 0，没有信息量。这条 DoD 要求的数只能从真实 `RequestLog` 算，
   在 `ops/slo_report.py` 的可用性曲线里；把演练口径当线上可用性报出去，是另一种假绿。
2. **工具 500 这一类的落点与任务书不同。** 注入的是"上游 HTTP 500"，不是"Skill 执行抛异常"。
   工具执行异常走 `tools/` 执行器的错误分支，W8 有单测，本周没有把它接进混沌脚本。
3. **熔断器是"全链路一个"，不是 per-alias。** 真实故障常是单厂商挂；
   全链路熔断会把"另一家还好"的候选一起拒掉。取舍理由是阈值与候选链同源（§3.1），
   要做到 per-alias 得把熔断下沉到候选循环内——那会把"重试"和"熔断"两个职责搅在一起，
   留到有"单厂商挂"真实案例时再动。
4. **半开名额是进程内的。** 多副本部署时每进程各一个探测名额，
   上游恢复瞬间最多会有 N 个探测打过去。真正的"全局一个"要 Redis 共享状态，本周没做。
5. **流式未接熔断。** 产品层目前没有流式路由，`gateway.stream()` 保持 W10 原样。
   给它接熔断不是"包一层"能解决的：async generator 的失败发生在迭代期，
   而那时 header 已经发出去了。这件事要么在 `precheck` 前做准入，要么等 D3 的统一执行模型。

## 8. 与 Gate 11 的关系

D2 支撑 DoD 的"可靠性"一项，但它单独**不足以**判定达标：
- 熔断解决"上游挂了别继续打"，不解决"单次请求失败后能不能续跑"（D3）；
- 降级解决"给用户什么"，不解决"这次降级该不该计入可用性"（D1 的 SLO 口径）。

Gate 11 的自评里，本报告只能证明"故障注入可复现 + 熔断真实生效 + 降级有结构化标记"三件事。
