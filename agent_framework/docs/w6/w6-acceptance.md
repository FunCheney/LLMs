# W6 验收记录（Gate 6 · 溯源与增量）

- 日期：2026-11-07
- 范围：`knowledge/{citation,incremental,governance}.py`、`retrieval/{acl,context_assemble,pipeline}.py`、`apps/growth_agent/{answer,runtime,routes_answer,app}.py`
- 数据：`evals/history/2026-W49-citation.json`（30 条抽样摘要）、`2026-W49-citation-run.json`（可 replay 的完整 run）、`2026-W49-eval.json`（端到端）、`evals/golden/citation_sample_v1.jsonl`（抽样集）
- 全量：**715 passed / 25 skipped**（10.6s）、ruff 全绿、mypy 61 文件 0 问题

## 0. 九项对照

| # | 验收项 | 结论 | 证据 |
|---|---|---|---|
| 1 | 溯源贯通 | **pass** | `/v1/answer` 返回 `chunk_id / document_id / page_start–end / section_path / excerpt / locator`；`test_answer_endpoint_returns_citations` 断言响应结构 |
| 2 | 引用准确率 ≥90% | **pass（口径需连同精确率一起读）** | 28/28 = 1.0000；精确率 0.4286（36/84）；2/2 应拒答题未编引用；`--replay` 逐位复现 |
| 3 | 增量索引免全量重建、幂等 | **pass** | 真库 3 条：`test_incremental_e2e.py`（新增 A 不影响 B；SKIPPED 时 embedder 零调用；重复删除返回 0） |
| 4 | 事务安全（不留半新半旧） | **pass** | `replace_document` 单事务删旧写新；e2e 断言更新后旧 chunk 全消失（写 4 / 删 3） |
| 5 | ACL 最小实现、越权不进 context | **pass（默认关，语料无标注）** | 13 条单测；`test_acl_runs_before_rerank` 断言无权限块**不进精排**；默认 `allow_unlabeled=False` |
| 6 | 上下文质量（无悬空引用、success_rate 过门禁） | **pass（离线档）** | `test_trimmed_context_still_numbers_continuously`；`run_eval` 54 条 50 passed / 0 failed / 4 skipped，success_rate 100%；回归门禁通过（success_rate + latency_p95） |
| 7 | 调优清单可照着排查 | **pass** | `docs/w6/rag-tuning-checklist.md`（现象 → 层 → 旋钮 → 命令，11 节） |
| 8 | 知识治理三条 | **partial** | 单测 15 条覆盖质量/过期/冲突；**但语料里没有过期文档与矛盾文档**，端到端没跑过真样本 |
| 9 | 合流（history 接上、端到端可跑） | **pass** | `evals/history/2026-W49.json`、`docs/log.md` W49 条目；`/v1/answer` 与 `/v1/draft` 同时可用 |

**7 pass / 1 partial / 0 fail。**

## 1. 引用准确率：为什么 1.0000 不能当成"引用很准"

| 指标 | 值 | 怎么读 |
|---|---|---|
| 引用准确率 | **1.0000**（28/28） | 每条答案至少有一条引用落在标注段上 |
| gold 进 context | 28/28 | 检索段没拖后腿——**这也说明这个数字主要在证明链路通，不是在区分模型** |
| 引用精确率 | **0.4286**（36/84 条引用命中标注段） | 按题平均为 0.4855；两个口径差 6pp，说明引得多的大题命中率更低 |
| 未命中引用的去向 | 48 条：**11 条同页相邻、37 条在别的页** | 不能一句"切分边界问题"带过 |
| 全量引用 | 0 条 | 没有"把整个 context 都引一遍"的样本 |
| 应拒答 | 2/2 | 语料未覆盖的题回答"资料未覆盖"，没有编引用 |
| 延迟 | p50 14.7s / p95 22.2s | 含 LLM 生成，不是检索延迟（检索 p95 约 160ms） |

**精确率 0.4286 的含义要掰开看**：未命中的 48 条里只有 11 条是"与标注段同页的相邻块"，
**37 条在别的页**。所以不能解释成"只是切分边界问题"。

它也不等于"模型引错了"——标注是**必答段**（每题 2–7 个），不是"全部相关段"。
一道"query 执行循环怎么工作"的题本来就会跨页取多段。真正能判"每条引用是否支撑
它所在的那句"的，是 W7 的 judge（逐句 faithfulness），不是 chunk 集合比对。

**为什么两个 precision 都要报**：micro（36/84）回答"全部引用里有多少指对"；
macro（0.4855）回答"典型一条答案有多干净"，但它给引 2 条和引 8 条的答案同样权重，
会把这 6pp 的差距平均掉。只报一个，都会漏掉"长答案引得多、命中少"这个事实。

**结论**：Gate 6 #2 通过，但要带着一句话——这个指标在当前语料上**已饱和**（28/28 全过），
它的区分度来自精确率与失败归因，而不是准确率本身。下一期若要提高区分度：
① 把标注从"必答段"扩成"相关段"；② 用逐句 judge 判引用支撑度。

## 2. 口径修正记录（本期最重要的一次返工）

第一版判定把"混入非标注段"判成失败，跑出 **0/2**：

| case | 引用 | 其中命中标注 | 首版判定 | 真相 |
|---|---|---|---|---|
| `golden-001` | 5 | 2（c330、c348） | `off_target` ✗ | 另外 3 条是同页相邻块（c329 与 c330 同为 p152） |
| `golden-002` | 5 | 3 | `off_target` ✗ | 同上 |

按"条条命中"判，等于用**标注是否穷尽**衡量**引用是否正确**。改法是拆成两个数
（准确率 + 精确率）并补一条反制（全量引用不算通过），见 ADR-W6-01 D2/D3。

这次返工记在这里的原因：如果没有先跑 2 条样本就下结论，"引用准确率 0%"会被当成
"模型不会引用"，然后去改 prompt——而那是最贵且无效的一条路。

## 3. 增量索引（真库）

`PGVECTOR_DSN=postgresql://rag:rag@localhost:5433/rag` 下跑通（表 `rag_chunks_incremental_e2e`）：

| 步骤 | 断言 | 结果 |
|---|---|---|
| 入库 A（3 段） | `CREATED`，写 3 | ✅ |
| 再入库 B（2 段） | A 的块数仍为 3 | ✅ 免全量重建 |
| 同内容再同步 A | `SKIPPED`，embedder 调用次数不变 | ✅ 成本真的省了 |
| A 内容变（3→4 段） | `UPDATED`，写 4 / 删 3，旧块全消失 | ✅ 无残留 |
| `model_snapshot` m1→m2 | `UPDATED`，note 写明"模型变了" | ✅ 强制重建 |
| 删除 A | 清 4 条；再删返回 0 | ✅ 幂等 |
| 切分写错 document_id | `ValueError` | ✅ 拒绝写入（否则删除传播漏删） |

## 4. ACL

| 项 | 结论 |
|---|---|
| 未标注默认 | **拒绝**（`allow_unlabeled=False`） |
| 开关默认 | **关**（语料无 `acl` 字段，开了等于全拒） |
| 过滤位置 | **精排之前**（省算力 + 消除侧信道），过滤后不足 k 由 over-fetch 补偿 |
| 主体来源 | 请求头 `X-Principal-Id` / `X-Principal-Groups`；缺失按匿名（仍拒绝），不是"当没开" |
| 日志纪律 | 只记 `filtered_out` 数量，不记被滤掉的内容 |
| 真鉴权 | **没有**，W10 |

## 5. 服务接线

| 项 | 状态 |
|---|---|
| `/v1/answer` | ✅ 新增；检索→组装→编号→生成→引用体检全链路 |
| `/v1/draft` | ✅ 仍可用（**未接检索**，纯生成） |
| 惰性装配 | ✅ 启动不加载模型；首请求 18–20s；`/health` 的 `rag` 字段报 `cold/ready` |
| 释放 | ✅ `close_runtime` 先关检索（线程池+模型）再关网关（httpx） |
| 错误语义 | ✅ 模型/索引不可用 → `ConfigError`(500)，**不是**"没检索到资料" |

## 6. 端到端门禁

```
54 条：passed 50 / failed 0 / interrupted 0 / skipped 4 → success_rate 100.0%（mode=offline）
judge 均分 1.922（50 条）
RAGAS（9 条）：faithfulness 0.0 · relevancy 0.0543 · context_precision 1.0
回归门禁：通过（success_rate 1.0、latency_p95 2.48ms）
```

两点必须一起说，否则这些数字会被误读：

1. **离线档是夹具回放**：`success_rate` 只证明执行管道通；`quality` 块保持
   `pending-online`。faithfulness 0.0 与 W48 完全一致，是夹具属性（夹具答案不依赖检索
   结果），**不是本期引入的劣化**。
2. **generation 侧没有线上读数**：`/v1/answer` 还没被真实流量压过，p95 只有抽样脚本的
   端到端 14.7s（含 LLM），不能当服务延迟基线。

## 7. 未通过项与遗留

| 项 | 状态 | 下一步 |
|---|---|---|
| Gate 6 #8 知识治理端到端 | partial | 造 3 类样本（低质量源 / 2024 旧政策 / 两篇矛盾）跑一遍真链路；现在只有单测 |
| `/v1/draft` 接检索 | 未做 | 接上后 Gate 5 #5 的 p95 门禁才能真正覆盖检索段 |
| `normalize_query` 接入 | 未做 | 主要价值是把空/超长 query 变成 `InvalidQueryError` |
| ACL 真鉴权 | 未做 | W10 |
| 逐句引用支撑判定 | 未做 | W7 judge（当前只判到 chunk 集合层） |
| 镜像 `--extra pgvector` | 未做 | Dockerfile 缺 psycopg；镜像构建仍卡 docker.io 502 |
