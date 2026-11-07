# RAG 全链路调优清单（W6 D5）

> **怎么用**：这份清单不是"最佳实践列表"，是**排查顺序**。看到现象 → 先用"判据"确认
> 落在哪一层 → 只动那一层的旋钮 → 跑对应的验证命令 → 数字变了才保留。
>
> 三层纪律（与 `docs/w4/root-cause-playbook.md` 同源）：
> 1. **先证明是这层**，再动这层。凭感觉动 prompt 是最贵的路径——改完指标不动，再改一轮。
> 2. **一次只动一个旋钮**。同时开混合+精排+改写，之后任何变化都无法归因。
> 3. **每个旋钮都要有"关掉会怎样"的对照**。没有对照的数字不是证据。

---

## 0. 先记住这张总表

| 现象 | 先看哪个判据 | 层 | 主旋钮 | 验证命令 |
|---|---|---|---|---|
| 召回不到 / 召回垃圾 | `Recall@1`、top1 是否明显无关 | 召回 | `RETRIEVAL_MODE`、`RETRIEVAL_RECALL_MULTIPLIER` | `retrieval_metrics.py --configs dense` |
| 召回到了但顺序不对 | `Recall@10` 高而 `R@1` 低 | 召回 / 重排 | 先别开精排，see §3 | `retrieval_metrics.py --configs dense,dense+rerank` |
| 答案没引用 | `audit.verdict = no_citation` | 溯源 | `CITATION_RULES`、`answer_temperature` | `citation_accuracy.py --replay` |
| 引用了但编号指错段 | `verdict = off_target`；`gold_in_context` 仍为真 | 溯源 / 生成 | prompt 的编号约束 | `citation_accuracy.py` 失败样本区 |
| 引用了不存在的编号 | `verdict = unknown_index` | 生成 | context 编号是否超范围 | 同上 |
| 点开引用找不到原文 | `verdict = source_missing`、`source_missing=True` | 入库 | 抽取管线的 `source_uri` | 入库自检 `missing_source_chunks()` |
| 答案跑偏 / 中间内容被忽略 | 答案只用了头尾的块 | 上下文 | `retrieval_attention_ordering`、`retrieval_token_budget` | `make eval` |
| context 里同一段出现两三次 | `assembled.duplicate_of` 非空 | 上下文 | `retrieval_dedupe_threshold` | `make eval` + token 数 |
| 答案里出现旧政策 | `metadata.expired=True` 的块进了 context | 知识 | 治理权重（§7） | 构造新旧对照题 |
| 两处资料互相矛盾，答案挑了一个 | `conflict` 字段为 None 但有矛盾对 | 知识 | `metadata.conflicts_with` | 构造矛盾对照题 |
| 延迟高 | `diagnostics.recall_ms/rerank_ms/rewrite_ms` 分段 | 服务 | 预算与候选数（§8） | 门禁 p95 |
| 权限过滤后结果不足 k | `diagnostics.filtered_out` 接近候选数 | ACL | `recall_multiplier`（over-fetch） | `pytest -k acl` |

> **ACL 过滤在精排之前**（`pipeline.py`）：无权限的块不进精排。除了省算力，
> 更重要的是**侧信道**——精排耗时随"无权限块数量"变化，放在后面等于让没有权限的
> 调用方从响应时间推断库里有多少他看不到的文档。过滤后不足 k 由 over-fetch 补偿。
---

## 1. 入库与切分（一切的下游都受它制约）

**判据**：`missing_source_chunks()` 返回非空；或切出来的块里同一段话被切断（答案引到半句）。

| 项 | 现状 | 怎么判断要动 |
|---|---|---|
| 切分归属 | **在 rag_mvp**（`pre_query`/chunking），框架不重写 | 框架里再造一套切分 = 两份真相（ADR W5-D9） |
| 块大小 | 425 块 / 单 PDF，平均约 500–800 字 | 答案总是"差半句" → 块太小；引用总是"指向一大段无关内容" → 块太大 |
| `source_uri` | 必须有值，缺失写 `missing-source:` 占位 | 溯源链路的地基，缺了整条链作废（W6 最容易返工处 #1） |
| `valid_until` | 可空，ISO 日期字符串 | 政策/价格类文档应标注，否则过期治理不生效 |

**命令**

```bash
# 入库自检：哪些块没有出处（必须在入库时跑，不是抽样时才发现）
.venv/bin/python -c "
from agent_framework.knowledge import missing_source_chunks
"   # 实际调用见 evals/scripts 里的入库脚本
```

**坑**：`source_uri` 写成相对路径 `docs/x.md` 时 `source_domain()` 返回空串，治理会把它
当"本地自有稿件"（权重 1.0）。这不是 bug，是刻意的——但如果你把网页内容扒下来存成
本地文件，它就绕过了质量降权。**外部来源要保留原始 URL。**

---

## 2. 召回层

**判据**：`Recall@10` 明显高于 `Recall@1`，且 top1 是与问题无关的块（看 `top_chunk_ids`）。

| 旋钮 | 默认 | 调整方向与代价 |
|---|---|---|
| `RETRIEVAL_MODE` | `dense` | `sparse` 不需要模型但中文分词差（见 §9）；`hybrid` 实测负收益 |
| `RETRIEVAL_TOP_K` | 8 | 只是"给下游几条"，不改变召回质量 |
| `RETRIEVAL_RECALL_MULTIPLIER` | 3 | 召回池 = `top_k × 3`。ACL 过滤后不足 k 时调大它，不是调 `top_k` |
| `RETRIEVAL_BUDGET_MS` | 800 | 生产预算。**评测不要用它**，否则测的是超时不是质量 |
| embedding 模型 | `BAAI/bge-m3`（1024 维，L2 归一化） | 换模型必须全量重建索引 + 更新 `model_snapshot`（ADR-009） |

**命令**

```bash
# 单档基线（先确认现状，不要一上来就加组件）
.venv/bin/python evals/scripts/retrieval_metrics.py --configs dense --report /tmp/dense.json
```

**坑**：改 `retrieval_mode` 或换 embedding 模型后**必须重建索引**，否则 dense 用新模型
编码 query、库里是旧模型的向量——相似度全是噪声，但不会报错，只会"召回质量变差"。

---

## 3. 重排层（默认关，开之前先读这段）

**判据**：只有 `Recall@10` 已经够高、`R@1` 才是瓶颈时才考虑。W5 实测它让 R@1 掉了 7.1pp。

**为什么默认关**：cross-encoder 偏好"主题邻近的段落"，而 golden 要的是"恰好回答这一个
问题的那一段"，于是它把第一名换成同页邻居。两套候选池口径（20 / 100）逐位一致，
说明不是池子大小的问题。

**要开就三个值一起给**（缺一个都会静默降级，`rerank_preflight()` 会在启动日志里告警）：

```bash
RETRIEVAL_RERANK=true
RETRIEVAL_BUDGET_MS=8500           # ≥ rerank_timeout + 500ms 召回余量
RETRIEVAL_DEVICE=cpu               # macOS 必须；否则 MPS 断言直接杀进程
# 可选：RETRIEVAL_RERANK_TIMEOUT_MS=8000（默认已 8s，本机 p95 5.4s）
```

**验证**

```bash
# `deg` 列必须为 0。有降级的精排 = 没开精排，但日志上看起来是开的
.venv/bin/python evals/scripts/retrieval_metrics.py \
  --configs dense,dense+rerank --rerank-timeout-ms 90000 --device cpu
```

**坑**：`RETRIEVAL_BUDGET_MS` 保持 800 而只设 `RETRIEVAL_RERANK=true` 时，精排会在
`min(超时, 剩余预算)` 处被掐断，**28/28 全降级**，指标等于没开。这是 W5 踩过的坑，
`deg` 列就是为了暴露它。

---

## 4. 改写层（默认关）

**判据**：问题是口语化/指代不明（"这个怎么配"），且 `rewrite_ms` 可接受。

**实测**：两次运行 R@1 分别为 .7857 / .7143（不改是 .7857），p95 15.2s，成本
0.0036 元/次。**没有一次比不改更好**，所以默认关。

| 旋钮 | 默认 | 说明 |
|---|---|---|
| `RETRIEVAL_REWRITE` | `false` | 开后每次检索多一次 LLM 调用 |
| `RETRIEVAL_REWRITE_TIMEOUT_MS` | 2500 | **本机实测需要 8–12s**，2500 必降级 → 要么调大，要么别开 |

**命令**

```bash
.venv/bin/python evals/scripts/retrieval_metrics.py \
  --configs dense,dense+rewrite --rewrite-timeout-ms 15000
```

**坑**：改写是温度 0.3 的采样，**运行间方差大**。一次跑出"改写更好"不能作为保留依据，
至少两次，且要看两次的方差是否覆盖了差异。

---

## 5. 上下文组装层（默认全开）

三件事：去冗余 → 预算裁剪 → 首尾放置。**顺序不能换**：先裁剪再去冗余会白裁；
先编号再移动会让 `[n]` 指向错的块。

| 旋钮 | 默认 | 现象 → 调整 |
|---|---|---|
| `RETRIEVAL_TOKEN_BUDGET` | 3000 | 答案总被截断（`finish_reason=length`）→ 别调大，先看裁剪是否把关键块丢了 |
| `RETRIEVAL_CONTEXT_RESERVE_TOKENS` | 600 | 留给"规则+问题+答案"；不留给会出现半句答案 |
| `RETRIEVAL_DEDUPE_THRESHOLD` | 0.9（4-gram Jaccard） | 同一段反复出现 → 调到 0.85；误合并了不同段 → 调到 0.95 |
| `RETRIEVAL_ATTENTION_ORDERING` | `true` | 只改"给模型看的顺序"，不改检索名次，不影响任何检索指标 |
| `RETRIEVAL_CITATION_MAX_CHARS` | 800 | 单块进 context 的字符上限；太大时前几块吃掉整个预算 |

**判据**：`assembled.dropped_over_budget` 非空 = 有块被裁；被裁的块里有 gold 就会
同时表现为"召回成功但答案没引用"（`citation_miss`）——这是最容易被误判成生成问题的
一种，**先看这个字段再看 prompt**。

---

## 6. 溯源与引用层

**判据（四级，报告里直接给）**

| verdict | 含义 | 该改什么 |
|---|---|---|
| `ok` | 至少一条引用命中标注段 | — |
| `no_citation` | 一条 `[n]` 都没有 | prompt 的引用规则、`answer_temperature` |
| `unknown_index` | 引了不存在的编号 | 编号是否超范围、模型是否在编 |
| `source_missing` | 引用的块没有出处 | **回入库层**，不是生成层 |
| `off_target` | 引用的段**全都不在**标注集内 | 若 `gold_in_context` 为真 → 生成层；否则 → 召回层 |

**两个必须一起看的数**（只看前一个会被"全引一遍"骗过去）：

- **引用准确率** = 至少一条引用命中标注段的题目占比（门槛 90%）
- **引用精确率** = 命中标注段的引用 / 全部引用（不设门槛，但要看趋势）

**为什么精确率不设门槛**：标注只覆盖 69/425 个块，precision 的上限由标注完整性决定。
拿它当门禁，等于让"标了多少"决定"验收过不过"。

**为什么准确率允许"混入非标注段"**：首次抽样 `golden-001` 引了 5 条，2 条是标注段、
3 条是同页相邻块。按"条条命中"判它失败，会把**标注不全**算成**模型引错**，
于是人去改 prompt，改完指标不动。

**命令**

```bash
# 实跑（需模型 + DB + 密钥）
.venv/bin/python evals/scripts/citation_accuracy.py --record /tmp/cit.json
# 复算（不需模型、不需密钥；CI 与别人复核都用这个）
.venv/bin/python evals/scripts/citation_accuracy.py --replay /tmp/cit.json
```

---

## 7. 知识治理层（三条，全挂在 chunk 元数据上）

**共同的取舍：降权而不是过滤，标注而不是裁决。**

| # | 问题 | 元数据 | 手段 | 判据 | 调哪个数 |
|---|---|---|---|---|---|
| 1 | 来源质量参差 | `quality`（0–1）、`parse_failures` | 白名单 1.0 / 未知 0.6；解析失败 ×0.8 | 同一 query 下低质量源不得排在高质量源之前 | `GovernancePolicy(trusted_domains=...)` |
| 2 | 内容过期 | `valid_until` | `valid_until < now` → ×0.5（**不丢弃**） | 新政策必须排在旧政策之前 | `expiry_decay` |
| 3 | 同主题冲突 | `conflicts_with: [chunk_id]` | 冲突组一起进 context + 答案里显式提示 | 矛盾必须被指出，而不是挑一个 | 入库时的相似度阈值 |

**为什么过期是降权不是过滤**：丢弃后"最新政策是什么"会直接无答案，用户看到的是一句
"资料未覆盖"——比给一条旧政策更糟，因为它连"我有一条旧的"这个信息都丢了。

**为什么冲突不自动裁决**：裁决需要业务权威（哪份是正式版），模型猜错比暴露冲突贵。
目标是让答案**诚实标注不确定**，与 W4 "解析失败给 None 不给补分"是同一条原则。

**坑（最隐蔽的一个）**：治理必须作用在**精排之后**。cross-encoder 的 score 是 logit，
可能是负数；负数乘 0.6 会**变大**（-3 × 0.6 = -1.8），降权反而把低质量内容抬上去。
代码里负数走除法（`-3 / 0.6 = -5`），这条必须有单测守着。

**验证**

```bash
.venv/bin/python -m pytest packages/agent_framework/tests/test_governance.py -q
```

---

## 8. 服务与延迟层

| 指标 | 从哪看 | 正常范围（本机） |
|---|---|---|
| 召回 | `diagnostics.recall_ms` | 150–300ms（dense，425 块） |
| 精排 | `diagnostics.rerank_ms` | 开则 4.7–5.4s（20 候选，CPU） |
| 改写 | `diagnostics.rewrite_ms` | 开则 8–12s |
| 整链 | `diagnostics.total_ms` | dense 默认档 p95 约 160–175ms |
| 首请求 | `/health` 的 `rag` 字段 | `cold` → `ready` 需 18–20s（模型加载） |

**为什么检索是惰性装配**：模型加载 18–20s，放进 lifespan 会让 healthcheck（start_period
10s）超时，部署直接失败。代价是第一个真实请求吃冷启动 → **上线后打一次预热请求**
（RUNBOOK §6）。

**坑**：`/health` 显示 `rag=cold` 时测出来的 p95 全是冷启动，不要写进基线。

---

## 9. 已知的负收益（别重新发明一遍）

W5 用 28 条正例 / 2 条负例、425 块语料实测，两套口径交叉验证：

| 组件 | 实测 | 结论 |
|---|---|---|
| 混合检索（BM25+RRF） | R@1 .5714（dense .7857）；修好分词后 .6429 | RRF 假设"两路各有独立信息"，这里不成立 |
| 精排（bge-reranker） | R@1 −7.1pp、p95 160ms→5.4s | 两套候选池口径逐位一致，是算法性质不是参数问题 |
| 查询改写（multi-query） | 两次都不如不改，p95 +94 倍 | 温度 0.3 的运行间方差已覆盖差异 |
| sparse 分词器 | 标点被当 token，`？` 单独成一个词 | 修掉后 R@1 .2500 → .5357，但仍输 dense |

**这三条都是"做了、量化了、默认关掉、开关保留"**，不是"没做"。要开必须重新 A/B，
并给出"为什么这次的语料不一样"。

---

## 10. 最小改动验证流程（每次改动都走这五步）

```bash
# 1. 单测（秒级）：逻辑对不对
.venv/bin/python -m pytest packages/agent_framework/tests -q -o addopts=""

# 2. 检索指标（分钟级）：召回有没有变差
.venv/bin/python evals/scripts/retrieval_metrics.py --configs dense --report /tmp/after.json

# 3. 引用准确率（十分钟级）：溯源还成立吗
.venv/bin/python evals/scripts/citation_accuracy.py --json /tmp/cit.json

# 4. 端到端评测（分钟级）：成功率和延迟过门禁吗
.venv/bin/python evals/scripts/run_eval.py --week 2026-W49 --judge

# 5. 静态检查：别让类型/风格问题混进去
.venv/bin/ruff check . && .venv/bin/mypy packages apps
```

**顺序不能反**：先跑端到端会看到一堆互相掩盖的信号；先跑单测能在 1 秒内否掉一半改动。

---

## 11. 已知缺口（写在这里，别以为是"没有这个问题"）

1. **`/v1/draft` 还没接检索**：只有 `/v1/answer` 走完整链路（`/v1/draft` 仍是纯生成）。
2. **ACL 默认关且语料无标注**：开了等于全拒。策略默认"拒绝未标注"，别改成放行来自欺欺人。
3. **`normalize_query` 未接进 pipeline**：空/超长 query 目前静默返回 0 分结果，而不是抛错。
4. **增量索引的 embedder 是注入的**：`IncrementalIndexer` 不自己建模型，调用方负责给
   `embed(texts) -> vectors`，否则每次 sync 都会重新加载 bge-m3。
5. **`PgVectorStore` 是同步的**：异步门面在 `rag_mvp/contrib`，别在事件循环里直接调 store。
