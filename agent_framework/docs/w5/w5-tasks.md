# W5 执行清单（生产 RAG I · 混合检索与重排 · P0）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> 本周在 W2 的 compose 底座上加 **pgvector**（ROADEMAP §7 默认档：用 pgvector，不造轮子），让 capstone 第一次"基于自有资料回答问题"。
> 评测口径完全复用 W3 的 `make eval` / `TaskMetrics` / golden `references`——**RAG 的每一次调优都必须能在评测集上量化**。

## 承接面：W1–W4 已交付

| 已有 | 位置 | W5 的接缝 |
|---|---|---|
| `docker-compose.yml`（app+redis） | `ops/` | 本周加 `postgres + pgvector` service，别另起一套编排 |
| `SessionStore`（会话消息） | `session/` | 检索结果作为 context 注入，**不要塞进会话历史** |
| `Message.provider_meta`（ADR-007） | `llm/types.py` | 检索引用走 content 或新结构，**别塞 provider_meta**（那是透传给上游的） |
| `golden_v1.jsonl` 的 `references`/`difficulty` | `evals/golden/` | Recall@K/MRR 与 A/B 实验的标注来源 |
| `make eval`、`TaskMetrics`、`classify_failures` | `evals/` | D5 的 A/B 对比必须用同一套指标，不许另算 |
| `root-cause-playbook.md` 的"检索层"归因 | `docs/w4/` | 本周终于能真正触发 `retrieval_miss` 归因 |
| `LLMGateway`（编排/计量） | `llm/client.py` | 检索/重排是在调用 gateway **之前**拼 context，不是改 gateway |

> W5 检查点（ROADEMAP）：**Recall@K / MRR 量化提升 ≥ 10%**，有对比报告；capstone 能基于自有资料回答问题。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接符号 | 推进的 DoD |
|---|---|---|---|
| D1 | `knowledge/` 抽象 + `pgvector` 选型落地 | compose、`Settings` | 质量（自有资料） |
| D2 | `retrieval/hybrid.py`（BM25+稠密+RRF） | golden references | 质量（Recall/MRR） |
| D3 | `retrieval/rerank.py`（bge-reranker + 延迟预算） | `first_token_ms` 口径 | 延迟、质量 |
| D4 | `retrieval/query_rewrite.py`（multi-query/HyDE） | gateway | 质量 |
| D5 | A/B 对比实验 + 归因回填 | `make eval`、playbook | 证据包、可归因 |

**唯一主线**：先建立"能检索、可度量"的最小闭环，再逐层加混合/重排/改写，**每层都用 A/B 证明有提升再保留**。不要一口气堆四个组件——那等于没法归因（W4 刚学的方法）。

---

## Gate 0 · 开工前（30 min）

- [ ] 复核 `make eval` 全绿、门禁生效（W4 的护栏先立住，再动 RAG）
- [ ] 准备 capstone 的「品牌/竞品资料库」原始文档（≥10 份，含可引用的出处）
- [ ] 确认 embedding 方案：走某厂商 embedding API 还是本地模型——**记进 ADR，别散在代码里**
- [ ] 通读 W4 `root-cause-playbook.md` 的"检索层"一节（本周会真正用到）
- [ ] `pgvector` 镜像与连接串写进 `ops/.env.example`，密钥只走 env

---

## Day 1（2.5h）· 向量库选型 + 知识层抽象

**目标**：定稿存储与抽象。**先定 Retriever 协议，再接 pgvector**——顺序反了会被具体库绑死（W12 提取框架时会痛）。

### 你要做的事
- [ ] `knowledge/base.py`：`Document`/`Chunk` 数据结构 + `Retriever` 协议
- [ ] `knowledge/chunking.py`：切分策略（按标题/长度，保留出处元数据）
- [ ] `knowledge/pgvector_store.py`：`PgVectorStore`（实现 Retriever）
- [ ] `ops/` compose 加 `postgres`（pgvector 镜像）+ 迁移脚本
- [ ] `docs/w5/adr-w5-retrieval-stack.md`：选型理由 / embedding 方案 / 维度
- [ ] 单测：chunk 元数据完整、存储 upsert/查询

### 引导框架

```python
# knowledge/base.py —— 协议先行
@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source: str            # 出处 URL/文档 id —— W6 溯源要用，今天就得带上
    metadata: dict         # 标题/章节/版本/acl（W6 用）

class Retriever(Protocol):
    async def search(self, query: str, k: int) -> list["ScoredChunk"]: ...

@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    retriever: str         # 来自哪路检索，RRF 融合与归因要用
```

**算法步骤**
1. 切分时把 `source`/`metadata` 写进每个 Chunk——**今天不带，W6 溯源就得全量重建索引**（这是 ROADEMAP W6 检查点的伏笔）
2. `PgVectorStore` 只实现 `Retriever` 协议 + upsert；业务代码只依赖协议，不 import pgvector 具体类型
3. embedding 调一次外部 API/本地模型，结果连同 `model_snapshot` 存库（ADR-009：模型换了我们能发现旧向量漂移）
4. compose 用独立 `postgres` service + 初始化 SQL 建 `vector` 扩展与表；连接走 env

**自检**
- [ ] `docker compose up` 后能 upsert 一个文档并 `search` 返回带 `source` 的 chunk
- [ ] `pytest -k knowledge` 离线档（用内存 fake store）不依赖真实数据库
- [ ] chunk 的 `source` 字段 100% 非空（写个校验断言）

### 🔍 交给我的锚点
`Retriever`/`Chunk` 定义、ADR、compose diff。我会看：协议是否先于实现、`source` 有没有今天带上、业务层有没有泄漏 pgvector 类型。

---

## Day 2（3h）· 混合检索 + RRF 融合

**目标**：BM25（关键词）+ 稠密（语义）两路召回，用 RRF 融合。**这是"Recall@K 提升 ≥10%"的主要来源**。

### 你要做的事
- [ ] `retrieval/bm25.py`：BM25 检索（可用轻量实现，别引重依赖）
- [ ] `retrieval/hybrid.py`：`HybridRetriever`（两路并发 + RRF）
- [ ] `evals/scripts/retrieval_metrics.py`：Recall@K / MRR 计算
- [ ] 单测：RRF 排序稳定性、两路去重、空结果

### 引导框架

```python
# retrieval/hybrid.py —— RRF 是本周技术核心
def reciprocal_rank_fusion(rankings: list[list[ScoredChunk]],
                           k: int = 60) -> list[ScoredChunk]:
    """score = Σ 1/(k + rank)；按 chunk_id 去重合并，不依赖原始分数量纲。"""
    ...

class HybridRetriever:
    def __init__(self, dense: Retriever, sparse: Retriever) -> None: ...
    async def search(self, query: str, k: int) -> list[ScoredChunk]:
        # 两路并发 -> RRF -> 取 top k
        ...
```

**算法步骤**
1. 两路**并发**召回（asyncio.gather），各自 top N（N > k）；顺序执行会白白加延迟
2. RRF 用 `1/(k+rank)` 融合，**不看原始分数量纲**——这正是选它的理由（BM25 分和余弦分不可比），写进 ADR
3. 按 `chunk_id` 去重；合并后 `retriever` 字段记"来自哪路/两路都中"（归因用）
4. Recall@K/MRR 用 golden 的 `references` 当 ground truth：`references` 命中 top-K 即算召回
5. **先跑纯 BM25、纯稠密、混合三档各一次**，D5 才有对比基线（今天就要存三份结果）

**自检**
- [ ] `pytest -k retrieval` 覆盖：RRF 去重、两路空/一路空、排序稳定
- [ ] 用 golden 跑出三档的 Recall@K，混合档 > 任一单路（这是 ≥10% 的雏形）
- [ ] `retrieval_metrics.py` 输出可复现（钉 embedding 快照 + 温度无关）

### 🔍 交给我的锚点
RRF 实现、三档 Recall@K 对比、ADR。我会看：两路是否并发、RRF 有没有误用原始分数、去重键对不对。

---

## Day 3（2.5h）· 重排 + 延迟预算控制

**目标**：加 bge-reranker 精排，但**必须守住延迟预算**——重排很容易把 p95 拖爆，这是"生产 RAG"和"demo RAG"的分水岭。

### 你要做的事
- [ ] `retrieval/rerank.py`：`Reranker` 协议 + 实现（bge-reranker 或 cross-encoder）
- [ ] `retrieval/pipeline.py`：`RetrievalPipeline`（召回→重排→预算裁剪）
- [ ] 延迟预算配置（写进 `Settings`，不硬编码）
- [ ] 单测：预算超时降级、重排失败回退到 RRF 结果

### 引导框架

```python
# retrieval/pipeline.py —— 关键是有"降级路径"
class RetrievalPipeline:
    def __init__(self, hybrid: HybridRetriever, reranker: Reranker,
                 *, budget_ms: float) -> None: ...
    async def run(self, query: str, k: int) -> list[ScoredChunk]:
        # 1 召回 2 重排（带超时）3 超预算就用 RRF 结果，绝不拖垮请求
        ...
```

**算法步骤（今天最容易写错的地方）**
1. 重排放在召回**之后**、只重排 top N（如 30），不要对全库重排
2. 给重排设 `asyncio.wait_for(..., timeout=budget_ms)`；**超时/异常一律回退到 RRF 结果**并记 `*_degraded` 日志（复用 W1 ADR"软策略降级"的思路）
3. 延迟预算从 `Settings` 读，进 `RequestLog` 的分段耗时（W2 trace），让 p95 可归因到"检索 vs 重排"
4. 重排模型同样记 `model_snapshot`（ADR-009）；本地模型要评估冷启动对首请求 p95 的影响

**自检**
- [ ] 注入一个"重排永远超时"的 fake → pipeline 返回 RRF 结果且日志有 `rerank_degraded`（反向验证降级）
- [ ] `make eval` 里 `latency_p95` 加进重排后劣化在预算内（不超 W4 门禁阈值）
- [ ] 重排档的 Recall@K 与 RRF 档对比有提升

### 🔍 交给我的锚点
pipeline、降级日志、带重排/不带重排的延迟对比。我会看：**超时是否真降级**（不是硬等）、预算是否可配、p95 有没有守住门禁。

---

## Day 4（2.5h）· Query 改写（multi-query / HyDE）

**目标**：让"用户一句话"变成更好的检索 query，进一步提升召回。**改写本身是一次 LLM 调用，要计成本**（W11 成本看板盯着它）。

### 你要做的事
- [ ] `retrieval/query_rewrite.py`：`QueryRewriter` 协议 + 两种实现
- [ ] multi-query：一次生成 3–5 个改写 query，分别召回后合并
- [ ] HyDE：先生成"假设答案"再检索
- [ ] 单测：改写失败回退原 query、去重、成本计量

### 引导框架

```python
# query_rewrite.py —— 走 gateway，钉 cheap alias（改写不值得用 strong）
class QueryRewriter(Protocol):
    async def rewrite(self, query: str) -> list[str]: ...

class MultiQueryRewriter:
    def __init__(self, gateway: LLMGateway) -> None: ...  # model=cheap, temperature=0.3
```

**算法步骤**
1. 改写走 `LLMGateway.chat` + `cheap` alias（成本敏感），**禁止裸 HTTP**（ADR-004）；温度可略高但要记录
2. 改写结果去重后并发召回，合并进 RRF——注意别让改写把 k 撑爆（总召回上限要控）
3. 改写**失败/超时一律回退原 query**并记 `rewrite_degraded`（和 D3 同一套降级纪律）
4. 改写的 token 成本与延迟进 trace/日志；**D5 要回答"这点成本换来的 Recall 提升值不值"**
5. HyDE 作为可选开关，默认关——先用 multi-query 跑通，别同时上两个变量（否则无法归因）

**自检**
- [ ] 注入改写抛错 → 回退原 query 且检索仍返回结果（反向验证）
- [ ] 改写调用能在 Langfuse trace 里看到独立 span，含成本
- [ ] 单变量对比：开 vs 关 multi-query 的 Recall@K（D5 用）

### 🔍 交给我的锚点
QueryRewriter、成本/延迟数据、单变量对比。我会看：是否走 gateway、失败是否降级、有没有把成本记进日志。

---

## Day 5（2.5h）· A/B 对比实验 + 归因回填

**目标**：用 W3 评测集做一份**单变量 A/B 对比报告**，证明 Recall@K/MRR 提升 ≥10%，并回填 W4 的归因 playbook。

### 你要做的事
- [ ] `docs/w5/rag-ab-report.md`：各配置 A/B 表（baseline / +BM25 / +RRF / +rerank / +rewrite）
- [ ] 每个配置跑 `make eval` + `retrieval_metrics.py`，记录 Recall@K/MRR/p95/成本
- [ ] 回填 `docs/w4/cases/` 或新建检索层案例（现象→根因→改动→量化）
- [ ] 更新 `evals/history/YYYY-Www.json`；周五合流仪式

### 引导框架

**A/B 表模板（单变量，每格都要有数字）**
| 配置 | Recall@5 | MRR | 检索 p95(ms) | 单次检索成本 | 备注 |
|---|---|---|---|---|---|
| 纯稠密 |  |  |  |  | baseline |
| +BM25/RRF |  |  |  |  | 混合 |
| +rerank |  |  |  |  | 是否守住预算 |
| +multi-query |  |  |  |  | 成本换召回值不值 |

**算法步骤**
1. **严格单变量**：每次只加一个组件，用同一份 golden、同一 embedding 快照
2. 每格跑三次取稳定值（避免单次抖动），记录口径与时间戳（W1 Day5 教训）
3. 目标：混合档相对 baseline 的 Recall@K 提升 ≥10%；若重排/改写性价比为负，**如实记录并默认关掉**（比"全开"更专业）
4. 用 W4 的 `root-cause-playbook` 把某个 `retrieval_miss` 案例从"信号缺失"升级为"真实归因+修复"
5. 合流：合 main → `make eval` 过门禁 → 写 history json → `docs/log.md`

### 🔍 Gate 5 · 周末验收（全绿才算过 W5）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 协议先行 | 业务依赖 `Retriever` 协议，不 import pgvector 类型 |
| 2 | 溯源伏笔 | 每个 Chunk 的 `source` 100% 非空（W6 不用重建） |
| 3 | 混合有效 | Recall@K 相对 baseline 提升 ≥10%，有对比报告 |
| 4 | RRF 正确 | 用排名不用原始分数，去重键正确 |
| 5 | 延迟守门 | 重排/改写超时降级，`latency_p95` 过 W4 门禁 |
| 6 | 成本可查 | 每次改写/重排成本进日志，A/B 报告含成本列 |
| 7 | 归因闭环 | 至少一篇检索层根因案例（量化） |
| 8 | 合流 | history 接上，端到端仍可跑 |

---

## 本周最容易返工的 6 处

1. **Chunk 今天不带 `source`** → W6 溯源要全量重建索引（ROADEMAP W6 检查点直接挂）
2. **RRF 误用原始分数** → BM25 分与余弦分不可比，融合结果没意义
3. **重排不设超时/不降级** → 一个慢模型把 p95 拖爆，"生产"变"demo"
4. **一口气上四个组件** → 无法归因，违反 W4 刚建立的方法
5. **改写用 strong 模型 / 裸 HTTP** → 成本失控 + 违反 ADR-004/009
6. **A/B 多变量同时开** → 说不清是哪个组件带来提升，报告不可信

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 知识层 | `knowledge/{base,chunking,pgvector_store}.py`、`ops/` compose diff |
| 检索 | `retrieval/{bm25,hybrid,rerank,pipeline,query_rewrite}.py` |
| 评测 | `evals/scripts/retrieval_metrics.py`、`docs/w5/rag-ab-report.md` |
| ADR | `docs/w5/adr-w5-retrieval-stack.md` |
| 案例 | 检索层根因案例（回填 `docs/w4/cases/` 或 `docs/w5/`） |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W5 条目） |

---

## P3 挂载点（见 ROADEMAP §7）

- **向量库/模型**（P3-5）：默认档 = 用 pgvector 不造轮子（本周已完成）；自研 HNSW / LoRA 微调属深入档，且明确"除非投基础设施/算法岗"——**W4 决策点若没解锁就不要碰**
- 本周专注 P0 主线 RAG，别被"自研检索"分心
