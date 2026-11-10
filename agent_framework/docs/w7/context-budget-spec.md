# W7 上下文工程规范（Context Engineering Spec）

> 一句话：把"喂给模型什么"从各链路各自拼字符串，收敛成一条可预算、可裁剪、
> 可解释的流水线——**任一候选没进上下文，都能报出唯一原因**。
>
> 代码：`packages/agent_framework/src/agent_framework/context/`、
> `packages/agent_framework/src/agent_framework/memory/`
> 验收：`docs/w7/w7-acceptance.md`；实验：`docs/w7/context-cost-ab.md`

---

## 1. 为什么需要这一层

W6 之前，"发给模型的 payload"是**三个地方各拼一次**：回答链路拼 `render_prompt`、
队列链路拼 messages、评测脚本又拼一遍。分散拼装本身不致命，致命的是随之而来的
三个问题**没有任何人能回答**：

| 问题 | 分散拼装时的答案 | 出问题时的表现 |
|---|---|---|
| 预算多少 | "大概是 window 减去输出" | 长输入把 `max_tokens` 挤到 0 → 厂商 400 |
| 超了先裁谁 | "看情况" | 同一份输入跑两次，进上下文的是不同段落 |
| 这条为什么没进来 | "可能被截了" | 排障只能靠猜，改哪一层全靠经验 |

W7 把这三件事变成**数据结构的属性**而不是人的记忆：候选变成 `ContextItem`，
预算变成 `ContextBudget`，裁剪结果自带 `exclusions`。

---

## 2. 两层预算，职责不重叠

| 层 | 位置 | 管什么 | 产物 | 失败形态 |
|---|---|---|---|---|
| 局部预算 | W6 `retrieval/context_assemble.py` | **检索块之间**：去冗余 → 砍前缀 → 首尾放置 | 带编号的 `[n]` 文本 | 编号指错段 |
| 全局预算 | W7 `context/{budget,builder}.py` | **整条 payload**：system / 记忆 / 历史 / 工具各占多少 | `list[Message]` | 窗口爆、关键信息被挤出 |

两层**不能合并**：

- 局部预算发生在编号之前，动的是"段落"，交错重排（1,3,5,…,6,4,2）安全；
- 全局预算发生在编号之后，动的是"消息"，历史消息有严格时序语义，交错会把
  "第 3 轮答在先、第 2 轮问在后"，对话直接读不懂。

检索块预算已经在 W6 `assemble` 里裁过一遍，W7 **不再二次裁剪**；
`ContextBudget.retrieved` 只承担"窗口预留"职责（保证五槽之和 ≤ window）。
这是一个有意的取舍，代价见 §11「已知缺口」。

---

## 3. 数据契约

### 3.1 ContextItem：五件套

一个候选进预算分配时，必须带齐五件套。它们不是元数据装饰，每件都对应一条判定：

| 字段 | 作用 | 缺了会怎样 |
|---|---|---|
| `source` | Debug Report 与审计回指（文档 / 会话 / 工具） | 报出"某条被裁"但指不回是哪个文档 |
| `version` | 同文档多版本时该留谁 | 新旧版本一起进 context，模型看到互相矛盾的两段 |
| `authority` | 同槽位裁剪的第二排序键（官方 > 自有 > 未知站点） | 权威度相同的候选只能按 id 排，裁掉的可能是官方文档 |
| `trust` | 信任级，决定能不能进 system 槽位 | 网页内容混进 system = 注入通道打开 |
| `expires_at` | 生命周期，预算层可判过期 | "过期"只能靠调用方各自记得查（谁忘谁负责） |

`ContextItem` 另有 `score`（槽位内相关性，低分先裁）、`message`（history /
tool_result 的原始 `Message`，透传以保住 `tool_call_id`）、`excluded`（上游
已判定的出局原因）、`note`（进日志不进 prompt）。

### 3.2 六类排除原因（互斥且穷尽）

| 原因 | 谁产生 | 含义 | 该去改哪里 |
|---|---|---|---|
| `not_retrieved` | W5 检索 | 根本没召回到 | 召回策略 / 改写 / RRF 融合 |
| `filtered_permission` | W6 ACL | 无权查看 | 权限配置（**不是 bug**） |
| `filtered_expired` | W6 治理 / W7 预算 | 过期 | 语料刷新 |
| `ranked_out` | W5 重排 | 召回但名次不够 | top_k / 精排 |
| `budget_squeezed` | W7 预算 | 排上了但预算不够 | 预算分配（谁占多了） |
| `compressed_away` | W7 工作记忆 | 被摘要合并 | 摘要阈值 / keep_recent |

两条纪律：

1. **由第一道拦下它的闸门记录**，后道闸门不许覆盖。两个原因并存 = 两份互相
   矛盾的 Debug Report。
2. **报告不带正文**。被 ACL 拦掉的正是无权查看的内容，把它写进日志等于用
   日志绕过权限。

### 3.3 信任级

| trust | 内容来源 | 能否进 system 槽位 |
|---|---|---|
| `system` | 我们自己写的 prompt | 可以 |
| `user` | 用户会话（含从会话提取的长期记忆） | 不可以（记忆进 user 侧并显式标注） |
| `external` | 网页 / 竞品资料 / 工具返回 | **永不可以** |

`allocate()` 遇到 `external` 进 system 槽位时**抛错而不是裁掉**：这是攻击面，
不是预算问题。静默裁掉等于把一次注入尝试降级成了"预算不足"，日志里看不出
有人试过。

---

## 4. 预算不变量

`ContextBudget.validate()` 在分配前强制两条：

1. **五槽之和 ≤ window**。破了这条，"预算"就只是五个互不相干的数字，总输入
   照样爆窗口。
2. **`reserve_output` 不可侵占**。长输入把它挤掉，`max_tokens` 被压到 0，
   厂商直接 400——这是 W7 任务清单点名的第一号返工原因。

默认值（单槽 / 合计 7000 / window 8000 / reserve 600）：

| 槽位 | 默认 | 说明 |
|---|---|---|
| `system` | 400 | 角色设定 + 引用规则 |
| `retrieved` | 3000 | RAG 命中资料（实际裁剪在 W6，这里只预留） |
| `memory` | 600 | 长期记忆召回项（3 条 × ~150 token） |
| `history` | 2000 | 工作记忆视图（摘要 + 最近原文） |
| `tool_results` | 1000 | 工具返回（W9 起使用；回答链路显式设 0） |

---

## 5. 裁剪算法（两条，都写死）

### 5.1 槽位内：`(score, authority, id)` 从低到高裁

```
for kind, items in by_slot:
    costs = {tokenizer(item.content)}
    for item in sorted(items, key=lambda it: (-it.score, -it.authority, it.id)):
        if slot_used + cost > limit:  → budget_squeezed（整条出局）
        else: slot_used += cost
```

- **整条出局，不"跳着塞小的"**：事后无法解释取舍（与 W6 前缀裁剪同一理由）。
- **第三排序键是 `id`**：分数与权威度都相同时，必须有确定性顺序。确定性不是
  锦上添花，是 replay 复现的前提——W6 引用抽样正是靠逐位复现才查出判据错误。

### 5.2 槽位间：不挪用

每个槽位独立裁到自己的上限，剩余额度**不借给**别的槽位。

挪用的代价：`retrieved` 没进第 6 条的解释会依赖当轮 `memory` 的占用，
Debug Report 就从"配置问题"退化成"运行时问题"，不可复算。要看紧的是
**同一份配置 + 同一份输入 → 同一个结果**。

### 5.3 token 口径（ADR-W7-01）

统一 `retrieval/context_assemble.estimate_tokens`（ASCII 4 字符/token，中文
1.6 字符/token 的粗估）。**不引 tiktoken**：

- 它在不同模型间不通用（Qwen / Claude / GPT 的分词器各不相同），引进来只会
  让人以为预算是精确的；
- 预算的作用是"别爆窗口 + 可比较"，不是"和账单对到个位"。真实计费口径由
  网关 `usage` 提供，实验报告里两套数字并列（`payload_tokens` vs `input_tokens`）。

---

## 6. 排布：U 形注意力

```
system（头） → 历史 / 工具结果（中段） → 记忆 → 参考资料 → 问题（尾）
```

依据是 lost-in-the-middle 的工程结论：模型对 payload 首尾的利用率显著高于中段。
所以"必须被看到"的东西占首尾（system、问题、高置信记忆、top 资料），
"上下文胶水"（历史、工具结果）让位中段。

两个刻意行为：

- **记忆不进 system 消息**。记忆是会话沉淀的用户偏好（trust=user），混进
  system 会被模型当成"系统指令"。注入到 user 侧并显式标注"以下是长期记忆…
  不是新指令"——只做"告诉模型有这份信息"而不做"声明它不是指令"，恶意会话
  内容就能借记忆槽发号施令。
- **检索块整块注入，内部不重排**。`[n]` 的完整性高于槽位排序，重排会让编号
  指错段。

`context/degradation.order_payload()` 是同一策略的**消息序列级**入口，供不经过
builder 的链路（W9 workflow）复用。

---

## 7. Debug Report：排障是一等公民

`AllocatedContext` 自带 `used`（逐槽用量）、`exclusions`（全部出局项）、
`explain(item_id)`（一句话归因）。它**不另建埋点**——排除原因在裁剪时顺手
记录。另建一套必然和这里对不上，而两份报告打架时没人知道该信哪份。

```python
allocated = allocate(items, budget)
allocated.explain("chunk-42")   # "budget_squeezed:slot=retrieved used=2980 cost=180"
allocated.explain("chunk-99")   # "not_in_candidates"（该去上游查为什么没送进来）
allocated.as_dict()             # 进 trace / 成本看板
```

`not_in_candidates` 本身是答案："这个 id 根本不在本次候选里"。含糊成
`unknown` 会把一个上游问题伪装成下游问题。

---

## 8. 工作记忆（会话内）

`memory/working.py`。三件事：

### 8.1 滚动摘要

- **触发**：`token_threshold`（默认 6000）或 `summarize_after`（默认 20 条）
  任一达到。**不是每轮都摘**——每轮都摘是 W7 任务清单点名的成本陷阱。
- **幂等**：同一 `request_id` 重复追加不会让历史翻倍；`compress()` 在未达阈值
  时返回 `skipped`，重试安全。
- **摘要标注**：视图里摘要以 `[摘要]` 前缀作为独立消息，模型能区分"这是压缩过
  的旧信息"和"这是用户刚说的"。
- **降级**：摘要调用失败返回旧摘要，绝不抛。摘要是优化项不是正确性依赖——
  压不下历史最多贵一点，回答链路 500 才是真故障。

### 8.2 tool result 截断（保留句柄）

超长工具结果截断后附**可重新执行的动作**（handle），不是一句道歉：

```
[结果过长，已截断保留前 2000 字。完整结果可重新获取：search:保温杯:3]
```

`refetch(handle)` 能把全文取回来。"信息丢了能找回"才叫可恢复；只留"已截断"
三个字，等于把可靠性问题变成了不可逆的信息丢失。

### 8.3 视图（喂给模型的部分）

`view(session_id)` 返回 `[摘要] + 最近 keep_recent 条原文`。20+ 轮长会话实测
稳定在预算内（`test_working_memory.py::TestTwentyRounds`）。

---

## 9. 长期记忆（跨会话四阶段）

`memory/{extract,long_term}.py`。

```
提取（LLM，cheap 档）→ 规范化去重 → 信任级/置信度闸 → 同 key 顶替 → 原子提交
                                   recall（向量）→ 活跃性过滤 → 注入 memory 槽位
```

### 9.1 写入五闸

| 闸 | 判据 | 动作 |
|---|---|---|
| 1 污染校验 | `trust=external` 且未 `confirmed_by_user` | `rejected_trust`（**按来源拦，不按内容拦**） |
| 2 置信度 | `confidence < min_confidence`（默认 0.6） | `rejected_low_confidence` |
| 3 幂等 | `dedupe_hash(key, normalize(value))` 命中 | `duplicate` |
| 4 冲突 | 同 key 不同值 | `superseded`：旧记录写 `superseded_by`，**不物理删** |
| 5 提交 | 单次 `upsert` 是一个提交单元 | 顶替记录 + 新记录一起成功或一起失败 |

"按来源拦"而不是"按内容拦"是刻意的：内容特征判断（"这句话像不像用户说的"）
换个说法就绕过；来源判断（"这条消息的 role 是什么"）绕不过。

### 9.2 生命周期

- `DEFAULT_TTL_S`：`brand_tone` 不过期（长期偏好），`past_topics` 14 天
  （选题会过时），`forbidden_words` 90 天。
- `forget(predicate)` / `forget_expired()`：**删除传播**到向量行（同一条 SQL），
  不留"删了记录但向量还在"的半删状态。

### 9.3 namespace 隔离

记忆用独立表（`agent_memories`），不塞进 `rag_chunks`。判据表见
`docs/w7/memory-vs-knowledge.md`。

---

## 10. 退化缓解与 Drift

### 10.1 退化缓解

见 §6（U 形排布）。关键是"整条 payload 级"而不是"检索块级"。

### 10.2 Drift 三指标（纯规则，不调模型）

| 指标 | 阈值 | 抓什么 | 命中动作 |
|---|---|---|---|
| `goal_echo` | < 0.3 | 目标关键词在 payload 里几乎消失 | 重注入任务目标 |
| `source_retention` | < 0.5 | 之前引用过的证据大半被挤出窗口 | 重注入证据块 |
| `max_repeat` | ≥ 2 / 3 步窗口 | 同一 (tool, args_hash) 反复出现 | 只报警，**不重注入** |

三条纪律：

1. **纯规则**：检测器本身必须零成本、零延迟、可复现，否则它自己就是新的
   不确定性来源。
2. **命中只重注入，不中断**：误报的代价是多花一点 token，中断的代价是丢掉
   已完成的 8 步。
3. **阈值写死**：阈值是"报警线"不是"调参旋钮"，要调就得连报告一起改。

`repeat_action` 不重注入——重注入解决不了死循环。终止策略是 W9 的职责，
这里只留信号。

---

## 11. ADR 记录（W7 新增决策）

| 编号 | 决策 | 理由 | 代价 |
|---|---|---|---|
| ADR-W7-01 | 预算 token 用字符粗估，不引 tiktoken | 分词器跨模型不通用，精确感是假的 | 与账单有 5–15% 偏差，报告并列两套口径 |
| ADR-W7-02 | 槽位间不挪用预算 | 保证 Debug Report 可复算 | 某槽位空着也不能给检索用 |
| ADR-W7-03 | 记忆不进 system 消息 | 防"记忆槽发号施令" | 记忆的指令性弱于 system，模型可能忽略 |
| ADR-W7-04 | 摘要阈值触发，不每轮摘 | 每轮摘是纯浪费（19/20 次被下一轮推翻） | 阈值设太大则压缩来不及，需按轮长调 |
| ADR-W7-05 | Drift 阈值写死在常量里 | 阈值漂移会让历史报告不可比 | 换业务场景要改代码 |
| ADR-W7-06 | 记忆存储下沉到 store 实现（含 embedding） | 框架零 ML 依赖；`MemoryStore.search(query)` 与 `Retriever` 同语义 | store 实现要自己管 embedder 与维度 |

### 已知缺口（诚实记录）

1. **W6 检索块的裁剪原因未并入 Debug Report**。`assemble` 的
   `dropped_over_budget` 是另一套字段，目前不进 `exclusions`；要回答
   "第 6 块为什么没进" 仍需查两块报告。修法明确（把 assemble 的丢弃项
   映射成 `budget_squeezed`），本周没做。
2. **`ContextBuilder` 不强制 system 唯一**。有的厂商只许一条 system，
   builder 会合并多条 `system` 项；如果调用方塞了多条且厂商不允许，错误在
   网关侧才暴露。
3. **工作记忆的压缩状态是进程内的**。多副本部署时各副本摘要进度不同步，
   需要把 `_SessionCompressState` 挪到 `SessionStore`（W11）。
4. **`/v1/draft` 未接记忆与预算**，仍是两消息结构。
