# ADR-W8-003：Action Runtime——`ToolCall` 是意图，不是执行

- 状态：已采纳（W8 D5）
- 日期：2026-11-11
- 相关：`skills/{action,guard}.py`、`skills/executor.py`、`tests/test_action_runtime.py`

---

## 背景

D4 的 `SkillExecutor` 已经有「校验 → 权限 → 沙箱 → 执行 → 回注」五道门。
它挡住的是**越权**与**越界**，挡不住另外三类问题：

| 问题 | 真实发生方式 | 五道门为什么挡不住 |
|---|---|---|
| 重复请求 | 模型在同一任务里连发三次同样的写操作 | 每次请求单独看都合法 |
| 参数篡改 | 上下文里有恶意文本（抓回来的网页），诱导模型把预算改成 100 万 | 改后的参数仍然过 schema |
| 接口假成功 | 接口返回 200，响应体是 `{"ok": false}` | 执行确实「成功」了，是业务没生效 |

根因是同一个：**`ToolCall` 一产生副作用，就没有拦截点了**。它表达的是模型的意图，
而意图不等于事实。

---

## 决策

在 `ToolCall` 与副作用之间插一层 Action，六个阶段，每个阶段有**唯一**的失败码：

```
propose(模型) → validate(代码) → authorize(代码) → approve(人,仅高风险)
              → execute(代码,幂等) → verify(代码,回读)
```

| 阶段 | 归谁 | 做什么 | 失败码 |
|---|---|---|---|
| propose | 模型 | 生成 `ToolCall`（只是意图） | — |
| validate | 代码 | A14 schema + **业务规则**（数值上限、字段互斥） | `invalid_action` |
| authorize | 代码 | 可见面 + 权限白名单 + 风险分级 | `tool_not_visible` / `forbidden` |
| approve | **人** | 二次确认，展示「将要发生什么」 | `pending` / `rejected` |
| execute | 代码 | 带幂等键执行，超时受控 | `failed` |
| verify | 代码 | 回读真实状态确认生效 | `unverified` |

### 1. 幂等键 = `hash(tool + canonical_args + intent_id)[:24]`，TTL 24h

键里带**意图 id** 而不是 `ToolCall.id`：

`ToolCall.id` 每次重试都会变（模型每次生成新 id）。用它做键等于
「每次重试都是新动作」，幂等直接失效——而重试恰恰是最需要幂等的场景。

`intent_id` 由调用方按「这一轮要干的事」给，默认空串 = 「同一份参数只该执行一次」。
这是最安全也最有用的默认。

### 2. 并发重复靠锁，不靠时序

「先查缓存再执行」在并发下是错的：三个并发请求同时查都 miss，然后三个都执行了。
所以是 **per-key `asyncio.Lock` + 双重检查**（拿到锁后再查一次）。

### 3. 缓存只存「真的执行过」的结果

`verified` / `unverified` 进缓存；`pending` / `failed` / `forbidden` **不进**。

缓存一个 `pending` 会让审批通过后的重试直接拿到 `pending`——审批做完了，
动作永远不会执行，而日志显示「已经处理过」。

### 4. 业务规则与 schema 的 `maximum` 分开

`NumericCapRule(field="budget", maximum=200_000)` 与 schema 里
`budget.maximum = 1_000_000` 是**两件不同的事**：

- schema 的 `maximum` 是**给模型看的契约**（它会照着改）；
- 业务上限是**我们的规矩**（超了必须拒，不管模型怎么想）。

两者可以不同：schema 允许 100 万是历史遗留，业务只批到 20 万。
混在一起，改业务上限就得动 schema——等于把「给模型看的说明书」当成了授权系统。

### 5. 验真只做「读回状态」一种

`verify` 用 `skill.verify.entrypoint` 回读（如 `workspace.get_post`），
比对 `status_field` 是否落在 `ok_status` 里。

不做复杂对账（金额核对、多表一致性），那要真实业务系统（W11 才有）。
「响应体 `ok=false`」+「回读状态不对」这两类覆盖了线上绝大多数假成功。

**`ok` 的三态**：`verified` / `unverified` / **`skipped`**。
`skipped`（Skill 没声明 `verify`）**不算失败**——只读动作没有「生效」可验。
早先的实现把 `skipped` 算成失败，结果是每次搜索都变成 `unverified`，
而「处处出现的信号等于没有信号」。

### 6. 死循环预警只报警不阻断

最近 3 步里同一 `(tool, args)` 出现 ≥2 次 → 置 `loop_warning`（复用 W7 `drift.check_drift`
的 `repeat_action`）。**不阻断**：终止策略是 W9 多 Agent 的职责，
一次误报被升级成任务失败，比多花一点 token 贵得多。

预警必须在幂等检查**之前**算。放后面的话，「同一动作连发 3 次」会先被幂等
拦成 `duplicate`，预警永远不出现——那这条预警就等于不存在。

### 7. 审批门：没有 approver 就是 `pending`，不是放行

`PermissionGuard(approver=None)` 时，`irreversible` 动作返回 `pending`。
approver 自己抛异常 → `rejected`（**fail-closed**）。

默认拒绝而不是默认放行：W10 接真 HITL 之前，没人审批的高风险动作
应该卡住等人，而不是先执行完再看日志。

---

## 三个 badcase 的实测

`evals/scripts/tool_boundary_check.py` 的输出（默认离线档）：

| badcase | 构造 | 实测结果 | 拦在哪一层 |
|---|---|---|---|
| 重复请求 | 同一 `(action, args)` 连发 3 次 | `['verified', 'duplicate', 'duplicate']`，副作用 **1 次** | 幂等键 |
| 参数篡改 | `budget=1000000`（schema 允许，业务只批 20 万） | `invalid_action`，`业务规则 numeric_cap 拒绝：budget=1000000 超过业务上限 200000（schema 允许不等于业务批准）` | validate |
| 接口假成功 | mock 返回 `{"ok": false}` 的 200 | `unverified`，`posts()==0`（没有写成「已发布」） | verify |

第三条的日志原文：`工具返回 ok=false（审核未通过）：HTTP 成功不等于生效`。

---

## 后果

### 好的

- 三类问题各有唯一拦截点，失败码能直接进值班队列。
- 高风险动作在 W10 之前就已经「卡住而不是放行」，接 HITL 时只换 approver 实现。
- 幂等 + 缓存让「重试」变成安全操作，模型可以放心重试。

### 代价

- 多一层 `Action`，`submit()` 比 `executor.execute()` 慢一点（一次哈希 + 一次锁）。
  这是想要的：受控执行的成本应该可见。
- 幂等缓存是进程内的（24h TTL、`_clock()` 可注入）。多副本部署时
  要换成 Redis/DB——`_cache` 的接口已经按「可替换存储」设计，但本周没做。
- `intent_id` 需要调用方正确提供。给错（比如每次都给新 id）会让幂等退化成
  「永不重复」，这是使用者的责任，不是机制能兜的。

### 被否决的方案

| 方案 | 否决理由 |
|---|---|
| `ToolCall` 直接执行 | 重复/篡改/假成功三类全无拦截点（本 ADR 的起因） |
| 用 `ToolCall.id` 做幂等键 | 重试时 id 会变，幂等失效 |
| 「先查后写」判重 | 并发下三个请求同时 miss，然后都执行 |
| 业务上限写进 schema `maximum` | 把给模型看的说明书当成授权系统；改业务上限要动 schema |
| 验真交给模型自评 | W4 已证明模型自评与真实质量相关性很低 |
| 死循环预警改成阻断 | 一次误报 = 一次任务失败，比多花 token 贵 |
| 无 approver 时默认放行 | 高风险动作会在没人看的情况下执行完 |
| 缓存 `pending` 结果 | 审批通过后的重试直接拿到 `pending`，动作永不执行 |
