# ADR-W6-01 · 知识资产：溯源口径、ACL 默认策略、增量事务、治理降权

- 状态：已采纳（2026-W49）
- 影响范围：`knowledge/{citation,incremental,governance}.py`、`retrieval/{acl,context_assemble,pipeline}.py`、`apps/growth_agent/src/growth_agent/{answer,runtime,routes_answer,app,rag}.py`、`Settings` 的 `retrieval_token_budget` / `retrieval_*dedupe*` / `retrieval_acl_*` / `answer_*`、`06_rag/rag_mvp/src/rag_mvp/{vector_store/pgvector.py,contrib/agent_framework.py}`
- 数据来源：`evals/history/2026-W49-citation.json`（30 条抽样）、`evals/history/2026-W49-citation-run.json`（可 replay）、`evals/golden/citation_sample_v1.jsonl`

## 1. 问题

W5 结束时"能检索"，但检索结果和最终答案之间**没有任何契约**：答案可以不带出处、
可以引一个不存在的编号、可以把过期文档和现行规定并列排第一。W6 要补的是四个
必须现在定、以后改代价很大的选择：

1. 引用"算不算对"的**判定口径**——它决定了 Gate 6 #2 是不是一个真门禁；
2. 没有 `acl` 字段的块，默认放行还是拒绝；
3. 增量索引在"更新"时如何保证不留半新半旧；
4. 过期 / 低质量 / 互相矛盾的资料，是过滤、降权，还是裁决。

第 1 条是本 ADR 的核心。抽样第一版跑出来 **0/2**，而两条答案其实都引到了标注段——
这不是模型的问题，是判据的问题。把它写进 ADR 而不是留在脚本注释里，是因为
"指标不合格所以去改 prompt"是这条链路上最贵的一种返工。

## 2. 决策

### D1 编号与映射同源：只在一个循环里产出

`build_context()` 是**唯一**同时决定"文本顺序"和"`[n]` 指向"的地方：循环变量
`index` 既写进 `[{index}]` 也写进 `Citation(index=index, ...)`。

分两次遍历（一遍写文本、一遍建映射）时，只要有一次 `enumerate` 起点写成 0，
答案 `[1]` 就指向第二段，而两边代码看起来都对。这类错不打日志、不抛异常，
只表现为"抽样准确率莫名偏低"。

代价：调用方不能自己拼 context。想改顺序（去冗余 / 首尾放置 / 裁剪）必须**在
`build_context` 之前**做完——这正是 `assemble()` 的定位。

### D2 通过口径 = "至少一条命中标注段"，不是"条条命中"

实测依据（`2026-W49-citation.json`，首版判定）：

| case | 引用条数 | 其中命中标注段 | 首版判定 | 问题 |
|---|---|---|---|---|
| `golden-001` | 5 | 2（c330、c348） | `off_target` ✗ | 另外 3 条是与标注段**同页的相邻块**（c329 与 c330 同为 p152） |
| `golden-002` | 5 | 3 | `off_target` ✗ | 同上 |

根因是**标注本身是部分标注**：425 个块只标了 69 个。按"条条命中"判，等于用
"标注是否穷尽"去衡量"引用是否正确"——模型引对了没标的地方反而扣分。

所以拆成两个数：

- `verdict` 回答"有没有引对"（至少一条命中 → `ok`）；
- `precision = 命中标注段的引用 / 全部引用` 回答"引的东西有多干净"。

**精确率不设门槛。** 它的上限由标注完整性决定（69/425），拿它当门禁等于让
标注工作量决定验收结果。但它必须被报出来：准确率靠"至少一条命中"通过时，
precision 掉下来就是答案在灌水。

代价：这个口径比"条条命中"宽松，必须配 D3 的反制，否则"把整个 context 都引一遍"
可以稳过。

### D3 "全量引用"不算通过

context ≥ 4 块时，若引用条数等于 context 块数，判 `blanket_citation`，不计入通过。
context 只有 2–3 块时全引是合理的（就那么几段），不判。

理由：D2 的宽松口径下，全引必然命中。这不是假想——把 `CITATION_RULES` 改成
"每段都要引用"就能复现。指标的每个放宽点都要配一条反制，否则它只是看起来严格。

### D4 未标注 ACL 的块：默认拒绝

`ACLPolicy.allow_unlabeled = False`。没有 `acl` 字段的块视为"未标注 → 拒绝"。

这条不能反过来：反过来之后，"忘了打标签"与"公开"在审计上**无法区分**，
而漏标是多发事故、误标是罕见事故。

同时 `RETRIEVAL_ACL_ENABLED` 默认 **关**——当前语料没有任何 `acl` 标注，开了等于全拒。
但"D 关掉"与"策略是放行"是两件事，不要把开关关掉当成把默认改成放行。

`Settings.retrieval_acl_enabled=false` 时路由**不解析主体**；开启后没有
`X-Principal-Id` 的请求按 `Principal.anonymous()` 处理（仍走默认拒绝），
而不是"没头就当没开"——否则删掉一个 header 就绕过了权限。

**过滤位置：精排之前。** 两个理由，第二个比第一个重要：

1. 算力：20 条 cross-encoder 在本机 4.7s，其中一半无权限就是纯浪费；
2. 侧信道：精排耗时与降级行为会随"候选里无权限块的数量"变化。放在精排之后，
   一个没有任何权限的调用方也能从响应时间推断"库里有多少他看不到的文档"。

代价是精排池可能因过滤而变小，由 `recall_multiplier` 的 over-fetch 补偿，
而不是靠"先精排再过滤"来凑条数（`test_acl_runs_before_rerank` 断言的是
**reranker 收到的条数**，不只是最终结果）。

### D5 增量更新：同事务替换 + hash 跳过 + model_snapshot 双判据

`IncrementalIndexer.sync()` 的顺序是硬的：

1. 算 `content_hash`（只归一化换行，不动全半角——全角差异是真实内容变化）；
2. 与库中 `DocVersion` 比对，**未变直接 skip**（否则每次 sync 都重新 embed，成本杀手）；
3. 变了 → 调 `store.replace_document()`，**一条 SQL 事务内删旧写新**；
4. 记 `index_version` 与 `model_snapshot`。

`model_snapshot` 单独判：换 embedding 模型后即使内容没变也必须重建，否则新 query
向量与旧库向量不在同一空间，相似度全是噪声且不会报错。

代价：`DocumentStore` 协议要求 `replace_document` 返回 `(写入, 删除)` 两个数，
比"返回写入条数"啰嗦；但只返回一个数时，报告里分不清"更新"和"追加"。

`PgVectorStore` 是同步的，异步门面（`rag_mvp/contrib.AsyncDocumentStore`）用
**单线程执行器**：写库本身串行，多线程只会让"第 3 个文档先写完"这种顺序问题
变得不可复现。

### D6 上下文组装：先组装、后编号；裁剪之后才生成编号

顺序是 `去冗余 → 预算裁剪 → 首尾放置 → build_context`。

- 先编号再移动 → `[n]` 指向被移动过的块（D1 的错位）；
- 先裁剪再去冗余 → 白裁（重复块占掉了预算）；
- 裁剪后不重新编号 → 答案引用到已被裁掉的 `[n]`（悬空引用）。

`trim_to_budget` 的裁剪策略是**前缀裁剪**：超出预算就从此处整段丢弃，不"挑小的塞
进来"。理由是事后可解释——报告中能说清"第 3 条之后没进 context"，而不是
"为什么第 3 条没进、第 5 条进了"讲不出来。

### D7 治理三条：降权不丢弃，标注不裁决

| 问题 | 手段 | 为什么不是另一个选择 |
|---|---|---|
| 来源质量 | 白名单 1.0 / 未知 0.6；解析失败 ×0.8 | 不引入二次模型打分：入库成本 ×数量级，且分数无法解释 |
| 内容过期 | `valid_until < now` → ×0.5 | **不丢弃**：丢掉会让"最新政策是什么"直接无答案，比给一条旧政策更糟——它连"我有一条旧的"这个信息都丢了 |
| 同主题冲突 | `conflicts_with` + 答案里显式提示 | **不裁决**：裁决需要业务权威，模型猜错比暴露冲突贵 |

**必须作用在精排之后。** cross-encoder 的 score 是 logit，可能为负；负数乘 0.6
会变大（-3 × 0.6 = -1.8），降权反而把低质量内容抬上去。代码里负数走除法
（`-3 / 0.6 = -5`），由 `test_governance.py` 守着。

### D8 检索惰性装配：启动不加载模型

`build_runtime` 会加载 bge-m3（本机 18–20s）。放进 lifespan 意味着容器要等 20s 才
通过 healthcheck，而 `start_period` 只有 10s → 部署失败。

所以：启动只装配 LLM 网关，检索在第一个 `/v1/answer` 请求时加载（`runtime.py`）。
代价明确写进 RUNBOOK §6：**上线后打一次预热请求**，别让第一个真实用户吃冷启动。
`/health` 的 `rag` 字段报 `cold/ready`，让"首请求 20s"这件事可观测而不是靠猜。

不做后台预热任务：`create_task` 的预热会在测试、CLI、多 worker 场景各起一份，
失败信息只留在没人看的后台异常里。

### D9 `/v1/answer` 与 `/v1/draft` 是两个端点，不合并

- `/v1/draft` 写文案：调性、长度，产物本身不需要逐句出处；
- `/v1/answer` 回答事实：每条结论**必须**带 `[n]`，响应给出 `chunk_id / 文档 / 页码 / 摘录`。

合并的后果是二选一：让写 slogan 也背上引用义务（模型开始给"更懂你的选择"标出处），
或者让回答事实可以不标。capstone 要的是"先 `/v1/answer` 拿带出处的结论，
再 `/v1/draft` 写成文案"。

### D10 温度显式下发

`answer_temperature` 默认 0.2，评测显式设 0。不显式下发时，各厂商默认值不同
（0.7 / 1.0），换个模型答案风格就变，**引用行为也跟着变**（尤其是"没资料就编编号"）。
抽样若随采样波动，同一次改动跑两遍会得到两个结论。

## 3. 反向验证（每条决策对应一次"故意做错"，确认会被检出）

| 决策 | 故意做错 | 期望结果 | 实测 |
|---|---|---|---|
| D1 | 映射起点写成 0 | `[1]` 指向第二段，`off_target` | `test_citation.py::test_numbering_and_mapping_come_from_the_same_loop` |
| D2 | 只引邻居不引 gold | `off_target`（不是 ok） | `test_citation.py::test_off_target_when_no_cited_chunk_is_the_gold_one` |
| D2 | 引对一条 + 混入邻居 | `ok` 且 `precision=1/3` | `test_citation.py::test_mixed_citations_pass_with_precision_recorded` |
| D3 | 全引 context | 不计通过 | `test_citation_accuracy.py::test_blanket_citation_does_not_pass` |
| D4 | 删掉 `X-Principal-Id` | 按匿名拒绝，不是放行 | `test_acl.py::test_anonymous_does_not_match_labeled_documents`、`test_unlabeled_is_denied_by_default` |
| D5 | 删掉一个 chunk 的 `source_uri` | 判 `source_missing`，不是静默通过 | `test_citation.py::test_missing_source_is_detected_via_prefix` |
| D6 | 裁剪后不重编号 | 出现悬空引用编号 | `test_context_assemble.py::test_trimmed_context_still_numbers_continuously` |
| D7 | 负分乘权重 | 低质量块上升 → 单测红 | `test_governance.py::test_negative_scores_go_worse_after_downweighting` |
| D10 | 不传 temperature | 假网关断言 opts 含温度 | `test_answer.py::test_answer_pins_temperature_explicitly` |

## 4. 未采纳的方案

| 方案 | 没采纳的理由 |
|---|---|
| 引用准确率按"条条命中" | 标注只有 69/425，会把标注不全算成模型引错（见 D2 实测表） |
| 引用不判对错，只判"有没有 `[n]`" | 等于奖励"爱写方括号"，追溯不到原文的引用没有价值 |
| 过期文档直接过滤 | 用户看到"资料未覆盖"，而库里其实有一条旧政策 | 
| 冲突自动裁决（取权威源） | 业务权威不在模型手里；猜错比暴露冲突贵 |
| 用 pgvector 自带的行级安全（RLS）做 ACL | 检索发生在召回之前/之后需要不同的过滤点；RLS 只能过滤行，不能参与 over-fetch 决策 |
| 让框架管切分 | 切分依赖 PDF 解析细节，属检索实现；框架再造一套 = 两份真相（W5-D9） |

## 5. 遗留

1. **`/v1/draft` 未接检索**：目前只有 `/v1/answer` 走完整链路。
2. **`normalize_query` 未接进 pipeline**：空/超长 query 静默返回 0 分，而不是抛 `InvalidQueryError`。
3. **ACL 只有主体解析，没有真鉴权**：`X-Principal-Id` 可伪造，真鉴权在 W10。
4. **冲突检测靠入库时打标**：`conflicts_with` 由入库管线写，框架不自己算相似度。
