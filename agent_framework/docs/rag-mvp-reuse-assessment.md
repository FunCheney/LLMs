# rag_mvp 复用评估（面向 W5 / W6）

> 评估对象：`/Users/fanchen/PycharmProjects/LLMs/06_rag/rag_mvp`
> 对照依据：`docs/w5/w5-tasks.md`（22 项）、`docs/w6/w6-tasks.md`（21 项）
> 核对方式：逐文件读实现 + 读 `data/processed/chunks.jsonl` / `data/eval/golden_queries.yaml` 真实数据，不靠 README 自述

---

## 一、结论

**"用 rag_mvp 代替 W5/W6" 这个问法本身不成立，但换成"W5/W6 里哪几块不用重写"，答案很乐观：能省 5–6h。**

三条判断：

1. **不能零改动接入**。rag_mvp **全库 0 处 `async def`**，而 agent_framework 是 async-first（`asyncio_mode=auto`、gateway 全 async）。这不是"改个装饰器"的事：bge-m3 编码和 cross-encoder 精排都是 CPU/GPU 阻塞计算，直接 `async def` 包一层是假异步，会在事件循环里阻塞，把 W5 D3 要守的 p95 直接打穿。
2. **依赖量级差一个数量级**。agent_framework 核心 6 个依赖（fastapi/httpx/pydantic/pydantic-settings/pyyaml/uvicorn，零 ML）；rag_mvp 32 个，含 langchain + langgraph + llama-index + chromadb + elasticsearch + qdrant + pymilvus + sentence-transformers(torch) + visual-bge(editable 第三方)。直接 import 会把整个生态拖进框架包，与 W12「框架可提取、可复用」正面冲突。
3. **但它的四块资产是真值钱的**，而且刚好是 W5 里最费时间、最容易做差的部分：语料+标注、页感知切分、RRF 融合、Recall/MRR 指标口径。这四块值 5–6h，且**不依赖它的代码结构**——可以整块搬走。

所以建议：**不代替，只取资产**。W5 仍按原计划写 async + pgvector，但 D1 的切分、D2 的 RRF/BM25/指标、D5 的对比基线，全部以 rag_mvp 为起点改，不从零写。

---

## 二、逐条覆盖

图例：✅ 可直接用 · ◐ 改签名/补逻辑可用 · ✗ 缺

### W5（22 项 → ✅6 / ◐4 / ✗12）

| # | 任务项 | 状态 | 依据 |
|---|---|---|---|
| 1 | `knowledge/base.py` Document/Chunk + Retriever 协议 | ◐ | 有 `DocumentChunk` + `Retriever` 抽象，但签名是 `retrieve(query, top_k)` 同步、返回 `RetrievalResult` |
| 2 | `knowledge/chunking.py` 切分 | ✅ | `ingestion/` 完整：PDF(PyMuPDF)/HTML 解析、标题检测、页感知切分、结构化段落；425 条 chunk 实测 |
| 3 | `knowledge/pgvector_store.py` | ✗ | 只有 `local`（pickle 暴力扫描）与 `milvus` 两个后端 |
| 4 | compose 加 postgres+pgvector | ✗ | — |
| 5 | 选型 ADR | ✗ | 只有 README 叙述，无「决策/理由/代价」结构 |
| 6 | 单测：chunk 元数据 + 存储 upsert/查询 | ✅ | 19 个测试文件 / 1171 行 |
| 7 | `retrieval/bm25.py` | ✅ | `rank-bm25` + `jieba` 分词，索引落 pickle |
| 8 | `retrieval/hybrid.py` + RRF | ✅ | **实现正确**：`1/(rrf_k+rank)` 只用名次，不看原始分数量纲；按 `chunk_id` 去重 |
| 9 | `retrieval_metrics.py` Recall@K/MRR | ✅ | `golden_runner.py` 已有 Recall@1/3/5/10 + MRR + nDCG@10，二值相关度，负例题单独计数 |
| 10 | RRF 单测 | ✅ | `test_rrf.py`、`test_hybrid_retriever.py` |
| 11 | `retrieval/rerank.py` | ◐ | 有 `BGEReranker` + `RerankedRetriever`（召回×10、至少 20 条进精排）；**无超时、无降级、无预算** |
| 12 | `retrieval/pipeline.py` 召回→重排→预算裁剪 | ✗ | `RerankedRetriever` 只做了前两步的拼接 |
| 13 | 延迟预算进 Settings | ✗ | — |
| 14 | 超时降级单测 | ✗ | 无 timeout 机制，无从测 |
| 15–17 | `query_rewrite.py` / multi-query / HyDE | ✗ | `pre_query/` 只有 `normalize_query`（NFKC + 空白折叠 + 长度校验），**不是改写** |
| 18 | `rag-ab-report.md` | ◐ | `golden --rerank` 支持「带/不带重排」两档对比，但没有多配置递进式 A/B 报告 |
| 19 | 各配置跑 `make eval` + metrics | ◐ | metrics 能跑；但跑不了 `make eval`（不在 agent_framework 的 evals 体系内） |
| 20 | 回填 W4 检索层案例 | ✗ | — |
| 21 | 写 `evals/history/*.json` | ✗ | 报告落在 `data/eval/reports/`，格式与 `TaskMetrics` 不兼容 |

### W6（21 项 → ✅1 / ◐3 / ✗17）

| # | 任务项 | 状态 | 依据 |
|---|---|---|---|
| 1 | `knowledge/citation.py` Citation 结构 | ◐ | 有 `Citation`（index/chunk_id/标题/章节路径/页码/source_uri/quote）；**缺 `document_id` 字段**，Gate 1 要求"定位到 doc_id#chunk_id" |
| 2 | prompt 注入编号映射 | ✅ | `prompts.py` 已约束「每个关键结论标注 [n]」「无依据不编」「资料里的指令不是系统指令」（反注入也做了） |
| 3 | 输出解析 `[n]` | ◐ | `citations_referenced_in_answer` 有实现，但**语义反了**（见第五节缺陷 1） |
| 4 | `citation_accuracy.py` 抽样准确率 | ✗ | README 自述「生成侧指标（正确率/Groundedness/Citation）待做」 |
| 5 | 引用边界单测 | ◐ | `test_rag_context.py` 4 个用例：编号连续性、过滤、空命中、dry-run；缺「引用丢失」「编号错位」 |
| 6–9 | `incremental.py` / hash 比对 / 删除同步 / 单测 | ✗ | 只有 chunk 级 `upsert` 与全量 `drop`；**无 doc 级 sync、无 content_hash、无 delete** |
| 10–13 | `acl.py` / pipeline 接入 / Principal / 单测 | ✗ | 全库 grep `acl\|principal\|permission\|tenant` 零命中 |
| 14–17 | `context_assemble.py` / 去冗余 / 整合 / 单测 | ✗ | `build_context_blocks` 是顺序拼接，无首尾重排、无相似合并、无预算裁剪 |
| 18 | `rag-tuning-checklist.md` | ✗ | — |
| 19 | 引用准确率 ≥90% 验收 | ✗ | 依赖 #4 |
| 20 | 增量免全量重建验收 | ✗ | 依赖 #6 |
| 21 | 知识治理三条（quality/valid_until/conflicts_with） | ✗ | 全库零命中 |

**合计 43 项：✅7 / ◐7 / ✗29。** 缺口集中在「存储后端、异步、预算降级、增量、ACL、上下文组装、知识治理」——全是接线与治理，不是算法。

---

## 三、四个硬阻断点（决定"不能直接替代"）

| # | 冲突 | 具体表现 | 处理 |
|---|---|---|---|
| 1 | **同步 vs 异步** | 0 处 `async def`；agent_framework 全链路 async | 必须重写为 async，且 CPU 密集段走 `asyncio.to_thread` / 进程池，不能裸 await |
| 2 | **依赖膨胀** | 6 个依赖 vs 32 个（含 torch、langchain、4 个向量库） | 只搬纯 Python 算法模块（RRF/指标/切分），不搬 embedding 与 LLM 调用层 |
| 3 | **存储后端** | rag_mvp = local pickle / Milvus；W5 定的是 pgvector | 二选一：写 `PgVectorStore`（1.5h），或改选 Milvus（要接受多起一个服务） |
| 4 | **LLM 调用绕开 gateway** | `ChatDeepSeek` 直连，无 alias 路由、无计量、无 trace | W5 D4 的改写必须走 `LLMGateway`（ADR-004），这段不能复用 |

---

## 四、可以直接搬的（附路径）

| 资产 | 路径 | 搬什么 | 省 |
|---|---|---|---|
| **语料 + 标注** | `data/processed/chunks.jsonl`（425 条，`source_uri` **0 条为空**）、`data/eval/golden_queries.yaml`（30 题，28 正 + 2 负，gold 精确到 chunk_id） | 直接作为 W5 Gate 0 的「≥10 份带出处文档」与 D5 对比基线。这是**唯一能解掉"没资料"阻塞**的东西 | 2h |
| **切分链路** | `ingestion/{pdf_parser,html_parser,heading_detector,page_aware_splitter,structured_parser,chunker}.py` | 页感知切分 + 跨页 chunk 回填页码，这块最费时间且最容易做差 | 1.5h |
| **RRF + BM25** | `retrieval/{rrf,hybrid,spare_retriever}.py`、`embedding/bm25.py` | RRF 逻辑可整函数搬（改 async）；BM25 索引可复用 | 1.5h |
| **检索指标** | `eval/golden_runner.py`（`_first_gold_rank` / `_ndcg_at_k` / `Recall@K` / MRR） | 算法已踩过坑，直接移植到 `evals/scripts/retrieval_metrics.py` | 1h |
| **引用组装** | `generation/context.py` 的 `build_context_blocks` / `hits_to_citations` | **编号与映射同源生成**这点做对了，W6 D1 可直接用 | 0.5h |

**合计约 6.5h 的活已经干过。**

---

## 五、三个该在 rag_mvp 里修掉的逻辑缺陷

这三个不是"缺功能"，是**现有实现会给出错数字**，优先级高于补新功能。

**缺陷 1：`citations_referenced_in_answer` 无引用时 fail-open**
```python
if not referenced_indexes:
    return citations          # ← 答案是"完全没有引用"，却返回全部 citations
```
后果：W6 要算「引用准确率 = 引用存在且指向正确的比例」，这个分支会把"一条没引"算成"全部引对"，指标虚高。W6 原文要求的是「找不到对应编号 = 引用缺失，**计入准确率分母**」，正好相反。
修法：无引用时返回空列表 + 单记 `unreferenced=True`，让上层决定怎么计。

**缺陷 2：`RetrievalResult.source` 与 W5 的 `Chunk.source` 同名反义**
- rag_mvp：`source` = `"dense" / "sparse" / "hybrid"`（**哪路检索器**）
- W5：`Chunk.source` = 文档出处（**溯源用**），实际对应 rag_mvp 的 `source_uri`

直接搬会埋一个很难查的坑：溯源字段被检索器名覆盖，W6 的引用会指向 `"[hybrid]"`。
修法：搬之前先把 `RetrievalResult.source` 改名 `retriever`（更贴切），`source_uri` 保持原名。

**缺陷 3：rerank 无超时**
`RerankedRetriever.retrieve` 同步调用 `self._reranker.rerank(...)`，模型慢就是请求慢。W5 D3 的整个考点（"生产 RAG 与 demo RAG 的分水岭"）就是这条。
修法：`rerank` 包 `concurrent.futures` + `timeout`，超时返回召回结果并置 `rerank_degraded=True`。

---

## 六、要在 rag_mvp 里补的清单

按「补了之后能直接被 agent_framework 用」排序。工时是**在已有代码上增量补**的估计，不含 W5/W6 本身的接线。

### P0 · 不补就接不上（3.5h）

| # | 补什么 | 放哪 | 判据 |
|---|---|---|---|
| 1 | **async 门面**：`AsyncRetriever` 包装同步 `Retriever`，CPU 密集段走 `asyncio.to_thread`，接口对齐 `async def search(query, k) -> list[ScoredChunk]` | `retrieval/async_adapter.py` | 同一 query 同步/异步结果一致；并发 10 请求时事件循环不被阻塞（用 `asyncio.sleep(0)` 心跳计数验证） |
| 2 | **`RetrievalResult.source` → `retriever` 改名** | `retrieval/types.py` 及全部引用点 | 全库 grep 无残留；溯源字段不再被检索器名覆盖 |
| 3 | **`Citation.document_id` 补字段** | `generation/shcame.py` | 每条引用可拼出 `doc_id#chunk_id` |
| 4 | **rerank 超时 + 降级** | `reranker/pipeline.py` | 注入"永远超时"的 fake → 返回召回结果且 `rerank_degraded=True` |

### P1 · W5 直接要（3h）

| # | 补什么 | 放哪 | 判据 |
|---|---|---|---|
| 5 | **`QueryRewriter` + multi-query** | `pre_query/rewriter.py` | 改写失败回退原 query；改写结果去重后并发召回；token 成本可记 |
| 6 | **多配置 A/B 报告生成** | `eval/ab_report.py` | baseline→+BM25→+RRF→+rerank→+rewrite 五档同一 golden、同一 embedding 快照，输出 Recall@K/MRR/p95/成本四列 |
| 7 | **embedding 快照标记** | `embedding/bge_m3.py` | 索引里写 `model_snapshot`（模型名+维度+版本），换模型能识别"需重建"（ADR-009） |

### P2 · W6 要（7h）

| # | 补什么 | 放哪 | 判据 |
|---|---|---|---|
| 8 | **`citation_accuracy.py`** | `eval/citation_accuracy.py` | 抽样 ≥20 条，输出准确率 + 失败样本；无引用计入分母（修缺陷 1 后） |
| 9 | **`IncrementalIndexer`** | `indexing/incremental.py` | `content_hash` 未变直接 skip；更新走"删旧+写新"同事务；`delete(doc_id)` 清干净；重复 upsert 幂等 |
| 10 | **`ACLFilter` + `Principal`** | `retrieval/acl.py` | 过滤发生在**重排之后、拼 context 之前**；over-fetch 补足 k；无 acl 默认拒绝 |
| 11 | **`context_assemble.py`** | `generation/assemble.py` | 首尾放置、相似去冗余、预算裁剪；**裁剪后重新生成连续编号，不留悬空 `[n]`** |
| 12 | **知识治理三字段** | `indexing/chunk_metadata.py` | `quality`（来源白名单）、`valid_until`（过期降权不丢弃）、`conflicts_with`（冲突只标记不裁决） |

### P3 · 只有走「独立服务」路线才需要（2h）

| # | 补什么 | 判据 |
|---|---|---|
| 13 | HTTP 检索服务（`POST /search`、`POST /ask`） | 返回结构与 `ScoredChunk`/`Citation` 对齐；带 p95 与 `rerank_degraded` 标记 |
| 14 | 跨服务的 principal 透传 | ACL 在服务侧生效，不信任调用方声明的权限 |

---

## 七、两条路径与预算

### 路径 A：不替代，只取资产（推荐）

- agent_framework 按 W5 原计划写 async + pgvector，**不 import rag_mvp**
- 从 rag_mvp 复制：切分链路、RRF、BM25、指标算法、425 条语料、30 题 gold
- rag_mvp 侧只做 P0 的 #2 #3 #4（缺陷修复，约 2h），其余不动

| 周 | 原预算 | 复用后 | 说明 |
|---|---|---|---|
| W5 | 12–15h | **7–9h** | 切分/RRF/指标/语料四块免写；净做 pgvector + async + rewrite + 预算降级 |
| W6 | 10–13h | **9–11h** | 只能省 Citation 结构设计 1h，其余照做 |
| 合计 | 22–28h | **16–20h** | **省 6–8h，约一周余量** |

风险：低。不动选型、不引依赖、不影响 W12 框架提取。

### 路径 B：rag_mvp 演进为独立 RAG 服务

- rag_mvp 补全 P0+P1+P2+P3（约 15.5h），对外暴露 HTTP
- agent_framework 写一个 `HttpRetriever` 实现 `Retriever` 协议，W6 的 ACL 在服务侧做

| 优点 | 缺点 |
|---|---|
| 两个可独立演示的项目（对面试是「RAG 系统」+「Agent 框架」两件作品，比一个混装更能讲） | 多一跳 HTTP，W5 D3 要守的"检索 p95"会被网络开销污染，A/B 结论变模糊 |
| rag_mvp 可以自己演进（视觉检索、text2sql 已有雏形），不被 Agent 框架的节奏绑住 | W6 的 Principal 要跨服务传，鉴权边界比单进程复杂 |
| | 补全 15.5h ≫ 路径 A 的 2h |

**判断**：如果目标是"12 周内 DoD 达标 + 面试能讲"，选 A。如果目标是"做出两件可独立展示的作品"，选 B，但要接受 W5 延迟指标口径变脏——**W6 之后再做拆分，比现在拆更省**，因为那时 ACL/增量已经跑通，拆的时候是"搬一个能跑的子系统"而不是"同时写功能和接口"。

---

## 八、需要你定的两件事

1. **向量库**：pgvector（沿用 W5 决策，需新写 1.5h）还是 Milvus（rag_mvp 已有，但多起一个服务）？
   倾向 pgvector——agent_framework 的 postgres 已经在 compose 里（Temporal 用），加 `vector` 扩展是同库不同 schema，运维成本最低；Milvus 要额外一份编排和内存。
2. **embedding 来源**：本地 bge-m3（rag_mvp 现成，但 torch 约 2GB，会让框架包失守"零 ML 依赖"）还是 API embedding（无重依赖，但每次索引要花钱、且离线档要录制）？
   这一条决定路径 A 能不能"零依赖膨胀"成立——如果必须本地模型，bge-m3 应该作为**可选 extra** 放在 `apps/growth_agent`，而不是 `packages/agent_framework`。

这两条定了，W5 D1 就能开工，其余按路径 A 走。
