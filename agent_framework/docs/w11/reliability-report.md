# W11 D3 · 统一执行模型 / 幂等 / 断点续跑 / Run Timeline

> 交付物：`runtime/{model,timeline,resume}.py`、`resilience/idempotency.py`、
> `error_codes.py`（码表迁入框架包）、`evals/scripts/resume_drill.py`、
> 单测 52 条（model 26 / idempotency 14 / resume+timeline 12）。
> 数字全部来自本机实跑。

## 0. 一句话结论

四套分散的执行状态收敛成**一张九态状态机 + 四张映射表**；
断点续跑实测"前 2 步一次没重跑"，Timeline 能只看表就说清"哪一步、什么错、试了几次、卡在哪"。

但有一个必须写在最前面的缺口：**队列/`/v1/tasks` 目前还在用 W2 的四态记录**，
新模型有映射表、有 runner、有演练，但**没有替换掉线上那条路径**（§8 缺口 1）。

## 1. 为什么要统一：四套状态互相对不上

| 来源 | 状态 |
|---|---|
| W2 `/v1/tasks` | `queued / running / succeeded / failed` |
| W9 `PlanStep` | `pending / running / done / failed / skipped` |
| W10 审批 | `pending / approved / rejected / expired` |
| 时间线 | 只有日志行，没有对象 |

四套状态机都不可能覆盖对方。结果是"任务显示 running，但某一步其实卡在等审批"
这类问题只能靠人翻日志——而它恰恰是最该被人看见的那一种。

**这次不是"再加一套"**：九态是唯一状态，四张映射表把旧值翻译进来（`LEGACY_MAPPINGS`），
`test_every_legacy_state_has_a_mapping` 穷举三套旧枚举，**缺一个就红**。
这是"真的收敛"与"又加了一套"的唯一可执行分界。

## 2. 四层对象的分工

| 层 | 回答什么 | 粒度 |
|---|---|---|
| `Run` | 这次任务整体怎么样 | 一次用户请求 |
| `Step` | 任务分几步、每步干什么 | 一个子目标 |
| `Attempt` | 为什么这一步试了 3 次 | 一次工具/模型调用 |
| `Event` | 过程中发生了什么（只追加） | 状态迁移、尝试、跳过、等审批 |

九态（`RunState`）：`created / planning / running / waiting_tool / waiting_approval /
tool_failed / completed / failed / cancelled`。

**一个枚举、两张边表**。`Run` 与 `Step` 共用九态，但允许的边不同：

- `Step` 永远不进入 `PLANNING`（计划是 run 级的产物）；
- `Step` 的 `FAILED` 可以回 `RUNNING`（续跑就是重试那一步），`Run` 的 `FAILED` 也可以
  （续跑整个 run），但 `COMPLETED` 都不能复活——"完成过又重跑"会让幂等与成本统计同时失真。

共用枚举的理由很朴素：如果 step 用 `done`、run 用 `completed`，
报告里就会出现两种写法指同一个终态，而看报告的人会以为是两件事。

## 3. 三个结构性决定

### 3.1 `run.state` 是**算出来的**，不是人手写的（`derive_state` / `reconcile`）

`Run.derive_state()` 从 steps 推导 run 该处于什么状态，优先级是：

```
waiting_approval > waiting_tool > tool_failed > running > 终态
```

顺序本身是设计的一部分：**"有人在等审批"必须压过"有人在跑"**。
否则一个 8 步任务永远显示 running，而唯一需要人动手的那一步被淹没在正常进度里。

`reconcile()` 在状态分叉时纠正并留一条 `reconciled` 事件（带 `from`）。
这是"任务显示 running 但卡在等审批"的结构性解法：状态不是人写的，是算出来的。

推导出**不可达**状态时抛 `IllegalTransition`，不硬写进去。

### 3.2 非法迁移抛错，且不允许半截状态

`mark_step()` 先校验 step 的边，再校验"对齐后的 run 状态是否可达"，两个都过才落盘；
否则回滚 step 并抛错。`test_step_cannot_start_before_run_starts` 断言"被拒的迁移不留痕迹"。

静默忽略是最坏的选择：代码在跑、状态在漂、日志里全是成功，
这种 bug 平均排查时间以"天"计。

### 3.3 `WAITING_APPROVAL` 独立，且**续跑不越过审批**

- 它是独立状态，不是 running 的子状态。混进 running 会让"等人"被算进"卡住"，
  D1 的可用性/延迟图直接失真；
- 看板口径落在 `stuck_runs()`：等人审批**不算卡住**（`_NOT_STUCK`），
  真正卡住 = 既没结束、也不在等人；
- 续跑遇到等审批的步骤 → 记 `blocked` 并**停下**，不执行、也不跳过去跑后面的步
  （依赖关系上它是前置）。实测见 `test_waiting_approval_is_never_auto_resumed`：
  `executor.calls == []`。

## 4. 码表：搬进框架包，不抄第二份

`Attempt.error_code` 要回答"这次失败是不是厂商的锅"，而 `runtime/` 是生产代码，
不能 import `evals/scripts/classify_failures.py`。所以把码表本体搬到
`agent_framework/error_codes.py`，分类器改为 re-export（历史导入路径不变）。

W4 D5 的教训写在那个模块的文档里：之前两份码表，10 个已知码里 6 个判定相反
（`SERVER_BUSY` 一边算"质量失败"、一边算"上游问题"），一批被限流的请求把
success_rate 拉低，而报告说"模型答错了"。

顺带补了两件小事：

- `CIRCUIT_OPEN`（W11 D2 引入）正式归入 `RELIABILITY_CODES`——它不是"我们传错了参数"；
- 新测试 `test_error_code_table_is_not_duplicated` 断言 **`ERROR_HTTP` 里的每个码都在
  `KNOWN_CODES` 里**。以后新增一个异常类却忘了归类，这条会红。

## 5. 幂等：键 = 主体 + 规范化哈希

```
键 = (normalize_principal(principal), sha256(canonical_json(payload)))
```

三个必须成立的点：

1. **必须带主体**。只有内容哈希，A 和 B 提一样的请求，B 会拿到 A 的结果（跨用户串数据）；
2. **规范化要消掉"顺序不同"**（`sort_keys` + 紧凑分隔符 + `ensure_ascii=False`），
   但**不能**合并列表顺序——`["a","b"]` 与 `["b","a"]` 在"按这个顺序写三章"里就是语义；
3. **"完成过"与"正在执行"都要挡**。只判前者，并发两次提交会各跑一遍，
   而这两遍在日志里都合法。

三种状态与动作：

| `begin()` 返回 | 含义 | 调用方该做什么 |
|---|---|---|
| `new` | 归你执行 | 执行，完成后 `complete(value)` |
| `in_flight` | 有人正在执行 | **409**（`begin_or_raise` 抛 `DuplicateInFlight`），不排队 |
| `completed` | 已做完 | 直接返回 `value`（同一 task_id），不重跑 |

`failed` 不占位：一次上游抖动锁死幂等键，这个请求就永远不能重提，
而"重提"是用户唯一的恢复手段。

## 6. 断点续跑：实测

```bash
uv run python evals/scripts/resume_drill.py
```

第一趟（第 3 步注入 `UPSTREAM_TIMEOUT`）：

```
executed = ('step-1', 'step-2', 'step-3')
failed   = ('step-3',)
run.state= failed
executor 实际调用: ['step-1', 'step-2', 'step-3']
落盘文件: .../drill-run.json（7637 字节）
```

"进程重启"（只从磁盘读回，内存态全丢）后续跑：

```
读回后 run.state = failed；事件 28 条
skipped  = ('step-1', 'step-2')   ← 已完成步，一次都没重跑
executed = ('step-3', 'step-4')
executor 实际调用: ['step-3', 'step-4']
续跑后 run.state = completed
```

**"已完成步不重跑"不是为了省时间，是为了省钱**：一个 20 步的调研任务重跑前 15 步，
账单翻倍，而报告里只会写"重试了一次"。

### 重启时还在 RUNNING 的步骤必须重跑

进程在 `RUNNING` 死掉时，那次调用**结果未知**——可能写完了文件，也可能一个字节都没发。
三种处理里只有一种对：

| 处理 | 后果 |
|---|---|
| 当成功 | 静默丢一次副作用，数据缺一块且没人知道 |
| 当失败且不再重试 | 任务永远卡住 |
| **当"要重跑，副作用由幂等键兜"** | 唯一正确 |

所以恢复时把 `RUNNING` / `WAITING_TOOL` 标成 `TOOL_FAILED`（可重试），
事件 reason 记"进程重启：上一次调用结果未知，需重跑（副作用由幂等键兜）"。
每一步的幂等键 `(run_id, step_id, goal, tool)` 在重启前后**算出来完全一样**
（`test_step_idempotency_key_is_stable_across_restarts`）——否则"重跑不产生第二次副作用"是空话。

## 7. Timeline：判据是"不打开日志就能讲清"

实测输出（续跑后的完整历史）：

```
t           step        try   action                    status          latency
-------------------------------------------------------------------------------
00:01.0     step-1      -     search_web                waiting_tool    -
00:03.0     step-1      #1    search_web                ok              100ms
00:04.0     step-2      -     fetch_page                waiting_tool    -
00:06.0     step-2      #1    fetch_page                ok              100ms
00:07.0     step-3      -     extract_facts             waiting_tool    -
00:09.0     step-3      #1    extract_facts             failed          1000ms     [UPSTREAM_TIMEOUT]
00:09.0     step-3      -     extract_facts             failed          -          (注入超时)
00:11.0     step-1      -     search_web                skipped         -          (已完成，未重跑)
00:12.0     step-2      -     fetch_page                skipped         -          (已完成，未重跑)
00:13.0     step-3      -     extract_facts             waiting_tool    -
00:15.0     step-3      #2    extract_facts             ok              100ms      (重试后成功，参数未变)
00:16.0     step-4      -     compose                   waiting_tool    -
00:18.0     step-4      #1    compose                   ok              100ms

completed：4 步 / 5 次尝试 / 1 步重试
```

这张表能独立回答四个问题：哪一步（`step-3`）、什么错（`UPSTREAM_TIMEOUT`）、
试了几次（`#1` 失败 / `#2` 成功）、钱花在哪（`skipped` 的两行说明前两步没重跑）。

几条实现上的选择：

- **从 `run.events` 渲染，不从 `run.steps`**。steps 是当前值，events 是历史；
  "第一次超时、第二次成功"的痕迹只在 history 里。
- **时间起点取最早事件，而不是第 1 条**。事件按追加顺序排，但 `at` 可能因续跑时钟
  出现回退；取最小值只会让偏移偏大，取第一条会让整张表的偏移全错。
- **`args_digest` 相同就写"重试，参数未变"**。这两个结论在复盘时指向完全不同的修法：
  参数没变却通了 → 上游抖动；改了参数才通 → 第一次的参数错了。

## 8. 单测（52 条）

| 文件 | 条数 | 守什么 |
|---|---|---|
| `test_runtime_model.py` | 26 | 九态穷举、非法迁移抛错、`PLANNING` 不进 step、推导与 reconcile、四套旧枚举全映射、码表不重复、事件重放、等人≠卡住 |
| `test_idempotency.py` | 14 | 顺序无关、列表顺序算语义、跨主体不串、in-flight 挡住、completed 重放、failed 不占位 |
| `test_resume_timeline.py` | 12 | 已完成的步不重跑、中断步重跑、审批不被越过、事件流只增不减、幂等键跨重启稳定、Timeline 能从事件重建 |

全量门禁：`1349 passed / 30 skipped`、`ruff` All checks passed、`mypy` 130 files 0 issues。

## 9. 已知缺口

1. **新模型没有替换线上那条路径。** `/v1/tasks` 仍在用 W2 的四态 `TaskRecord`，
   `ResumableRunner` 只有演练脚本与测试在跑。映射表（`state_from_legacy`）与
   runner 都就绪，缺的是"把 `/v1/tasks` 的提交/查询接到 `Run` 上"这一刀——
   它要动 `queue/base.py` 的公开契约与三处调用点，属于 W12 框架提取时的活。
   **所以 D3 的自检里"重复提交同一 `POST /v1/tasks` 只执行一次"这条，实测的是
   `InMemoryIdempotencyStore` 的语义，不是接口层的行为。**
2. **`InMemoryIdempotencyStore` 只保证单进程幂等。** 多副本部署时同一请求打到两个副本
   就是两次执行——这件事在单机测试里永远看不到。换 Redis 是 D5 及之后的活。
3. **`JsonRunStore` 是整份覆写，不是真的事件追加存储。** 它挡住了"事件变少"
   （`save()` 越界抛错），但两个进程同时续跑同一个 run 仍会互相盖掉。
4. **`Attempt.args_digest` 由执行器提供。** 框架层还没有统一算它；
   W8 的回放哈希（`planner.base.args_digest`）可以直接复用，本周没接。
5. **看板只到模型层。** `state_breakdown` / `stuck_runs` 有测试，但 `ops/slo_report.py`
   还没有把 run 状态纳入周报——缺这条数据源，§3.3 的"单列"在图上还看不到。
6. **D1 的 `ApprovalBacklog` 与这里是两件事。** 前者数"待批的动作"，后者看"卡在等审批的
   run"；两者都该有线，目前只有前者接了告警规则。

## 10. 与 Gate 11 的关系

D3 支撑"可靠性"与"可解释"两项，但它单独判不了达标：

- 续跑解决"重跑不要钱两次"，不解决"某类失败该不该自动重试"（D4 的 incident 才给判据）；
- Timeline 解决"能解释"，不解决"解释得对不对"——那要靠 D4 的 Fork/Ablation 对照实验。
