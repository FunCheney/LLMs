# ADR-W5-01 · 检索栈：pgvector + bge-m3 本地 + 混合默认关、精排保留开关

- 状态：已采纳（2026-W45）
- 影响范围：`knowledge/base.py`、`retrieval/{fusion,pipeline,query_rewrite}.py`、`apps/growth_agent/src/growth_agent/rag.py`、`ops/{docker-compose.yml,pgvector/init.sql}`、`Settings` 的 `retrieval_*` / `vector_store` / `pgvector_*`
- 数据来源：`06_rag/rag_mvp/data/eval/reports/golden_w5_4way.json`（2026-11-05T14:50，28 条正例 / 425 chunk）、`docs/w5/rag-ab-report.md`

## 1. 问题

W5 要在"能检索、可度量"的前提下把 RAG 接进 capstone。有四个必须现在定、
以后改代价很大的选择：

1. 向量库用哪个（自研 / pgvector / Milvus）；
2. embedding 走厂商 API 还是本地模型；
3. 混合检索（BM25 + 稠密 + RRF）与精排（cross-encoder）到底开不开；
4. 语料切分放框架还是放检索实现。

第 3 不是"没实现所以没开"——**两者都实现了、都过了单测，是数据说它们在这批语料上
是负收益**。把它写成 ADR 而不是留在代码注释里，是因为"我们试过、量化过、然后关掉"
和"我们没做"在后人眼里长得很像，而这两种情况的返工成本完全不同。

## 2. 决策

### D1 协议先行：`Chunk` / `ScoredChunk` / `Retriever` 定义在框架，实现在应用

`knowledge/base.py` 只有 pydantic + 标准库，零 ML 依赖。换向量库、换 embedding 供应商
只改 `apps/growth_agent/.../rag.py` 一个文件，不动框架、不动检索实现。

这条的代价是**两边各有一套数据结构**（rag_mvp 的 `RetrievalResult` vs 框架的
`ScoredChunk`），桥接层要写一遍字段映射。收益是 W12 提取框架时不用把 pgvector /
torch 一起带走：任何引用框架的项目不该为了用 `SessionStore` 就下 2GB 模型。

### D2 `source_uri` 与 `retriever` 必须是两个字段

rag_mvp 里两者共用 `source` 一个名字（存的是 `"dense" / "hybrid"` 这类检索器名）。
直接搬过来，W6 的引用会指向 `"[hybrid]"`，`citation_accuracy` 会 100% 算错，
而报错信息里看不出任何异常。框架侧因此拆成：

- `Chunk.source_uri` —— 文档出处（URL / 仓库相对路径），**引用写它**；
- `ScoredChunk.retriever` —— 谁召回的这一条，**归因写它**。

### D3 向量库用 pgvector，不用 Milvus，不自研

| 方案 | 结论 | 理由 |
|---|---|---|
| pgvector | **采纳** | 与业务表同库同事务；ACL / 质量 / 有效期这些治理维度天然是 jsonb，`metadata @> '{"acl":...}'` 一条 SQL 就能过滤；HNSW 索引在万级 chunk 上够用；运维面只多一个容器 |
| Milvus | 不选（实现保留） | 独立集群（etcd + minio）或 Lite 单机，与"单机 capstone"不匹配；元数据过滤要在两套系统间做 join |
| 自研 HNSW | 不做 | ROADMAP §7 P3-5：除非投基础设施岗，否则不碰 |

`PgVectorStore` 的语义（`vector_store/pgvector.py`，23 条单测 + 11 条真库集成）：

- `add` 只进缓冲，`save` 才落库；**`save` 失败不清缓冲**——清了等于让调用方
  以为数据还在，下次 save 静默丢一批；
- `count` 数的是持久化行数，不是缓冲长度（否则"写完 425 条"是自我安慰）；
- `upsert` 幂等（同 chunk_id 覆盖），`drop` 用 `TRUNCATE` 保表结构；
- 额外提供 `delete_document(document_id)`：W6 的增量索引是"按文档成组删旧写新"，
  只有 `drop` 的话每次更新都要全量重建。

`CREATE EXTENSION vector` 需要超级用户，所以**建表与扩展放 `ops/pgvector/init.sql`**，
不放应用层——应用层执行 DDL 会在受管数据库上直接失败。

### D4 embedding 用本地 `BAAI/bge-m3`，1024 维，L2 归一化

- 归一化后点积 == 余弦，本地库与 pgvector 的 `1 - cosine_distance` 同量纲，
  切后端不会让任何阈值逻辑失效（这是"换库不改语义"的前提）；
- 成本为 0，A/B 可以随便跑（本周实测跑了 6 档 × 28 题 × 2 轮）；
- 中文语料上 bge-m3 的检索质量明显好于 BM25（下文数据）；
- **模型信息必须随向量落库**（`model_snapshot` 列，ADR-009）：换模型后旧向量不可比，
  没有这个字段就只能全量重建才知道"为什么分数整体变了"。

真机验证：425 条真实语料写入 pgvector 用时 47s；同一 query 在 `local` 与 `pgvector`
两个后端 **Top-5 完全一致、分数小数点后 6 位相同**（`scripts/compare_vector_backends.py`）。
后端等价不是"应该等价"，是实测等价。

### D5 RRF 只用名次，不用原始分数

`1/(60+rank)`。BM25 分无上界且随语料规模浮动，余弦分在 [0,1]，cross-encoder 是 logit。
三者混排等于让语料规模决定排序结果。名次在所有检索器里同义，分数不是。

`retrieval/fusion.py` 有单测覆盖：同分并列稳定、去重键 `chunk_id`、空路不拖低另一路。

### D6 **默认关混合检索**（`retrieval_mode="dense"`）

六档实测（28 正例 / 425 chunk，同 embedding 快照、同 top_k）：

| 配置 | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| **dense** | **.7857** | **.9643** | **1.0000** | **.8628** | **.7416** |
| sparse | .2500 | .6786 | .7143 | .4122 | .4288 |
| hybrid | .5714 | .8929 | .9643 | .7063 | .6619 |
| dense+rerank | .7143 | .9286 | .9643 | .8146 | .7232 |
| sparse+rerank | .6429 | .8571 | .8929 | .7432 | .6925 |
| hybrid+rerank | .7143 | .9286 | .9643 | .8146 | .7232 |

dense 单路全场最优。所以 `Settings.retrieval_mode` 的默认值从 `hybrid` 改成 `dense`。

**为什么混合会变差**（两层原因，都实测过，不是"RRF 写错了"）：

**① 分词器把标点当 token（真实缺陷，已量化）。** `rag_mvp/embedding/tokenizer.py`
只过滤空 token，于是 `？`、`，`、`.` 都是有效 token。后果很具体：

- 查询「如何配置 MCP 服务器？」的 jieba 切分是 `['如何','配置','mcp','服务器','？']`；
- gold chunk `p157-p157-c343`（19.7 配置 MCP 服务器）里 `服务器`×1、`mcp`×2——
  因为正文是 JSON 配置块，关键词只出现一次；
- 而 `p069-p069-c149`（工具系统的核心问题）里有 `？`×12、`如何`×7——
  它是一段"六个根本问题"的列表。**BM25 因此把它排到了第 1 位。**

修这个缺陷的收益是可测的（同一批语料、同一份 gold、只换分词器）：

| 分词口径 | sparse R@1 | R@10 | MRR |
|---|---|---|---|
| 现状（只过滤空 token） | .2500 | .7143 | .4227 |
| 去掉纯标点 token | **.5357** | .8571 | .6329 |
| 再去掉疑问/虚词停用词 | .5357 | .8571 | **.6433** |

R@1 直接翻倍——所以"sparse 很差"有一半是分词器的锅，不是 BM25 的锅。

**② 但修好分词器之后 hybrid 仍然是负收益。** 同一实验里：

| 配置 | R@1 | R@10 | MRR |
|---|---|---|---|
| dense 单路 | **.7857** | **1.0000** | **.8628** |
| hybrid（现状分词） | .5714 | .9286 | .6921 |
| hybrid（去标点） | .5714 | .9643 | .7074 |
| hybrid（去标点+停用词） | .6429 | .9643 | .7427 |

这是 RRF 的固有性质，不是实现 bug：正确答案在 dense 里排第 1、在 sparse 里排
第 30 时，`1/(60+1)=0.0164` vs `1/(60+30)=0.0111`；而一个"两路都排第 5"的邻居页
拿到 `2×1/(60+5)=0.0308`——**两路都中庸，赢过一路极好**。RRF 假设两路各自
提供独立信息；当一路明显弱于另一路时，它不是"取长补短"，是"取平均"。

具体案例：`golden-018` 的 gold chunk `p157-p157-c343` 在 dense 里排第 1，
在 hybrid 里掉出 top10，而 hybrid 的 top5 全是它前后的邻居页
（c347 / c335 / c346 / c348）——融合把"正确的那一页"换成了"看起来相关的一圈页"。

不选"调 RRF 权重 / 调 k"来救：没有证据支持任何特定权重，28 条样本上调权重等于
对评测集过拟合。**分词器缺陷是另一件事**（影响 sparse 档本身，也影响任何未来的
混合方案），记为 rag_mvp 的待修项：修完要重跑本节全部四行，混合才有翻盘资格。

### D7 **默认关精排**（`retrieval_rerank=False`）

两个独立理由，任一条都够：

1. **延迟**：cross-encoder 精排 20 条约 2.5s（本机 CPU 实测 5.4s），而默认总预算是
   800ms。开着它等于每次都走 `budget_exhausted` 降级——降级路径天天跑，等于没开，
   还多花 CPU。
2. **质量是负收益**：dense 加精排后 R@1 从 .7857 掉到 .7143、MRR 从 .863 掉到 .815，
   还多出一个 `golden-022` 的 top10 miss。

**这个结论在两套口径下都成立**（框架 pipeline 实测，28 正例）：

| 口径 | 候选池 | R@1 | R@10 | MRR | 检索 p50 / p95 |
|---|---|---|---|---|---|
|---|---|---|---|---|---|
| dense 单路 | 30 召回 | **.7857** | **1.0000** | **.8628** | 175ms |
| dense+rerank | 100 条全局重排 | .7143 | .9643 | .8146 | **20.9s** |
| dense+rerank | 20 条（生产口径） | .7143 | .9643 | .8146 | 4.7s / 5.4s |

**两行精排的数字一模一样（四位小数全同、连 miss 的题都一样）**，这一点比指标本身
更有信息量：它否掉了我原先的猜测——"负收益是因为 `candidate_k=100` 把 top10 打乱了"。
候选池缩到 20 条，损伤一点没少。所以问题不在池子大小，在 cross-encoder 自己在这个
语料上的判断。

逐题看更清楚（dense vs 20 条精排，比较首个 gold 的名次）：

| 变化 | 题数 | 明细 |
|---|---|---|
| 变好 | **1** | `golden-013`：2 → 1 |
| 变差 | **6** | `golden-002/004/008`：1 → 2；`golden-014`：2 → 3；`golden-025`：5 → 7；`golden-022`：8 → 出局 |
| 不变 | 21 | |

1 好 6 坏。`golden-022`（"CLAUDE.md 和 Memory 该如何选择？"）是最干净的证据：dense 把
`p115-p115-c249` 排在第 8，精排后它掉出 top10，而精排把同一页的邻居 `c250`、`c229`
顶上来了——**精排偏好"主题邻近段落"，而这道题要的是"恰好回答这个问题的那一段"**。
这是 cross-encoder 的典型失效模式：它按"问题与段落的语义贴合度"打分，
而"贴合度"和"这段是不是答案"在对比型/操作型问题上不是一回事。

结论：**能力保留，默认关**（`retrieval_rerank=False`）。

"保留"和"关掉"要分清：在这批语料上它是负收益，不代表在别的语料上也一样。换语料、
换模型（例如拿本语料做 hard negative 微调 cross-encoder）都可能翻盘——删掉代码等于
删掉实验入口。而"默认关"是因为开着的代价是确定的（+5s p95、−7pp R@1、多一个 top10
miss），收益是零。

**打开它必须三个值一起给**，少一个都会静默失效：

```bash
RETRIEVAL_RERANK=true
RETRIEVAL_BUDGET_MS=8500     # ≥ rerank_timeout(8000) + 500ms 召回余量
RETRIEVAL_DEVICE=cpu         # macOS 必设；不设则 MPS 多线程断言直接杀进程
```

为什么这么苛刻：精排的超时取 `min(rerank_timeout, 剩余预算)`，预算给 800 时剩余只有
约 500ms，精排 20 条要 4.7s → 每次都在 `asyncio.wait_for` 里被掐掉 → 降级回召回顺序。
指标看起来和"没开精排"一模一样，人就会得出"精排无效"的结论，而真相是参数没给够。
（2026-11-05 第一轮精排档就是 28/28 全降级，靠评测脚本的 `deg` 列才发现。）

所以 `growth_agent/rag.py` 里加了 `rerank_preflight()`：启动时检查预算与设备，
不自洽就打 `retrieval.rerank_misconfigured` 警告，文案直接写出该改成多少。
`retrieval_rerank_timeout_ms` 的默认值也从 3000 改成 8000——3000 在 CPU 上等于
"强制不生效"，那不是一个保守的默认值，是一个假的默认值。

### D8 **默认关查询改写**（`retrieval_rewrite=False`）

改写是一次 LLM 调用：有 token 成本、有延迟、有失败可能。它必须先用单变量 A/B
证明"这点召回提升值这个钱"。实测（28 正例，多查询 3 条，真机 `qwen3.7-plus`）：

| 项 | 数值 |
|---|---|
| R@1 | .7857 / .7143（两次运行） vs dense .7857 → 不变或更差 |
| R@3 | .8571 vs .9286 → 更差 |
| MRR | .8482 / .8089 vs .8628 → 更差 |
| 改写自身 p50 | **8.7s** |
| 整条检索 p95 | 15.2s（dense 160ms，**+94 倍**） |
| 超时降级 | 3/28、4/28 |
| 成本 | 0.0036 元/次（25 次调用共 0.0888 元） |

**为什么改写会让排序变差**：multi-query 把 N 条改写各自召回后 RRF 融合，
与 D6 是同一个机制——原查询本来是最准的一条，把它的第 1 名与三条改写的第 5 名
平均，结果就掉到第 2、第 3 名。改写只有在"原查询召不回"时才有价值，
而本语料 dense 的 R@10 已经是 1.000——**没有可捞的东西**。

成本侧记一个坑：请求 `max_tokens=160`，实测输出均值 **402 token/次**——
思考型模型把思考 token 计入 output，`max_tokens` 不约束总输出量。
成本预算必须按实测 token 算。

**开关必须真的接进装配。** `retrieval_rewrite` 曾经是个死开关（`config.py` 定义了字段、
`pipeline.py` 支持 `rewriter=` 注入，但 `rag.py` 从没读过它），设了没有任何效果。
死开关比"没实现"更危险：它让人以为"我开过了，没用"，从而错误地否掉一个可能有效的
组件。现在 `rag.py` 会按开关建 gateway + `MultiQueryRewriter` 并注入 pipeline，
`RagRuntime` 负责 `aclose()` 释放连接池。

### D9 切分归 rag_mvp，框架不造 `knowledge/chunking.py`

任务书 D1 列了 `knowledge/chunking.py`，实际**不做**，理由：

- 切分与解析强耦合：`rag_mvp/ingestion/` 有 PDF / HTML / 结构化解析、
  `page_aware_splitter`、`heading_detector`、`chunk_metadata`，加起来几千行。
  框架里再写一个"通用切分器"只能是玩具版，而两套切分器产出的 chunk_id 规则不一致，
  W6 的引用定位会错位；
- 切分产物（`chunks.jsonl` 的 `content_hash` / `model_snapshot` / 页码）是索引的输入，
  它属于"语料摄取"这条流水线，不属于"框架"；
- 框架真正需要的只有**数据契约**（`Chunk` 的 `source_uri` / `page_start` / `section_path`），
  这个已经在 `knowledge/base.py` 定死了。

所以：**框架定义 chunk 的形状，rag_mvp 负责把文档切成这个形状**。桥接层
（`rag_mvp/contrib/agent_framework.py`）做字段映射与 `source_uri` 兜底校验。

### D10 精排与召回分开装配，不合成 `RerankedRetriever`

`RerankedRetriever` 的问题不是设计丑，是**超时后拿不到任何东西**：精排超时时
整条链路只能失败，而正确的行为是"退回召回顺序，仍然给出结果"。
所以 `AsyncReranker` 单独持有，由 `RetrievalPipeline` 按剩余预算施加超时，
降级时记 `rerank_degraded` 并保留 RRF 顺序。这也是为什么 rag_mvp 的
`retrieval/factory.py` 被改成只构建 base retriever。

## 3. 后果

**好的**：

- 默认档跑的是实测最优配置，不是"看起来最完整"的配置；
- 两个后端实测等价，切 pgvector 不需要重新调任何阈值；
- 降级路径有反向验证（注入"永远超时的精排" → 返回 RRF 结果 + `rerank_degraded` 日志）；
- 关掉的东西都有数字依据，不是"感觉没用"。

**代价 / 未决**：

- 框架 venv 跑 dense 需要 `uv sync --extra local`（torch 约 583MB）；容器镜像仍未带
  `--extra pgvector`，镜像内 `VECTOR_STORE=pgvector` 会 ImportError（W6 待修）；
- 28 条正例的样本量下，R@10 差 1 条不显著；R@1 差 6 条、MRR 差 15.7pp 是系统性的；
- sparse 路的真正问题是中文分词，换成 jieba / 专用 analyzer 后**必须重测 D6**，
  混合检索可能翻盘；
- HyDE 未实现（只做 multi-query）：同时上两个变量无法归因，这是刻意的。

## 4. 验收口径的修订

原检查点写"Recall@K / MRR 提升 ≥10%"。dense 的 R@10 已经是 1.000，**数学上没有
10% 的空间**。改为：

> 在 R@10 触顶的前提下，用 **R@1 / MRR** 作主指标；混合与精排若为负收益，
> 如实记录并默认关闭。

这不是降低标准——把"提升 10%"换成一个已经饱和的指标才是降低标准。R@1 从
baseline（sparse .250）到 dense（.786）是 **+214%**，MRR 从 .412 到 .863 是 **+109%**，
都远超 10%，只是不该再用 R@10 去衡量它。
