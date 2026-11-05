# 复制路线：把 rag_mvp 的算法段搬进 agent_framework

> 对比文档：`docs/rag-mvp-refactor-plan.md`（改造路线，15h）
> 本文只回答一件事：**不改造 rag_mvp，只复制代码，具体怎么复制。**

---

## 一、先说结论

能复制，但**复制粒度必须是「文件」而不是「目录」**。

`cp -r rag_mvp/src agent_framework/...` 会立刻踩两个坑：

**坑 1：import 全是顶层名。** 待搬文件里有 26 处 `from retrieval.xxx` / `from ingestion.xxx` / `from schemas import DocumentChunk`，没有一处带包前缀。因为它们靠 editable 安装把整个 `src/` 塞进了 `sys.path`。搬进来之后这些名字在 `agent_framework` 里不存在，**26 处全部 ModuleNotFoundError**。

**坑 2：目录结构和 W5 落点表对不上。** W5 定的落点是：

```
knowledge/{base,chunking,pgvector_store}.py
retrieval/{bm25,hybrid,rerank,pipeline,query_rewrite}.py
```

而 rag_mvp 是 `reranker/`、`embedding/bm25.py`、`vector_store/`、`ingestion/` —— **没有一个目录名能直接对上**。所以「复制」实际是「把 29 个文件里的算法段落，重新编入 6 个目标文件」。

这个重排不是负担，是好事：它强迫你只留有用的部分（见第三节）。

---

## 二、能搬的规模（脚本核过）

| 目标文件 | 来源文件 | 行数 | 处理方式 |
|---|---|---|---|
| `knowledge/base.py` | `schemas.py` + `retrieval/types.py` | 74 | **改**：`DocumentChunk`→`Chunk`，`source`→`retriever` |
| `knowledge/chunking.py` | `page_aware_splitter` + `text_splitter` + `paragraph_splitter` + `chunker` + `heading_detector` | 777 | 直接搬，只改 import |
| `knowledge/ingest.py` | `document_loader` + `pdf_parser` + `html_parser` + `storage` + `chunk_validator` | 424 | 直接搬，只改 import |
| `knowledge/metadata.py` | `indexing/chunk_metadata.py` | 26 | 直接搬 |
| `retrieval/bm25.py` | `embedding/bm25.py` + `tokenizer.py` | 131 | **拆**：扔掉 pickle 落盘与 `VectorRecord` |
| `retrieval/hybrid.py` | `rrf.py` + `hybrid.py` | 121 | 直接搬（`base.py` 重写为 async Protocol） |
| `retrieval/rerank.py` | `reranker/{base,bge,pipeline}.py` | 149 | **改**：加 timeout + 降级 |
| `retrieval/pipeline.py` | `retrieval/retriever.py` | 82 | **重写**：dense 换 pgvector |
| `retrieval/query_rewrite.py` | `pre_query/normalizer.py` | 20 | 只留 NFKC；multi-query 新写 |
| `retrieval/context_assemble.py` | `generation/context.py` + `shcame.py` | 117 | **改**：修引用编号缺陷 |
| `evals/scripts/retrieval_metrics.py` | `eval/golden_runner.py` 指标段 | ~315 | **拆**：抽纯函数，适配本项目 golden |
| `apps/growth_agent/.../rag/embedding.py` | `embedding/{base,bge_m3,embedding_types}.py` | 144 | **不进 core 包**（torch） |

合计 **2399 行 / 29 个文件**。

但真正值得照抄的是 **约 1100 行**，其余按 W5 自己写更快——理由见下节。

---

## 三、三类文件：搬 / 拆 / 不搬

### ✅ 直接搬（约 1050 行，改 import 就能用）

| 文件 | 行数 | 为什么值得搬 |
|---|---|---|
| `page_aware_splitter.py` | 256 | 页感知切分是真功夫：按 PDF 页边界切、回填 `page_start/page_end`、超长段落二次切。自己写至少 4h，还容易切出跨页断句 |
| `heading_detector.py` | 169 | 章节标题识别（字号/编号模式/层级），`section_path` 靠它 |
| `text_splitter.py` + `paragraph_splitter.py` | 167 | 句子/段落切分与合并阈值 |
| `chunker.py` | 185 | 拼装 chunk + 生成 `chunk_id`（`<doc>-p001-p002-c003`） |
| `rrf.py` | 74 | RRF 用名次不用原始分数量纲——这一条写错的人很多 |
| `context.py` | 77 | 引用编号重排逻辑（**但有缺陷，见第五节**） |
| `golden_runner.py` 指标段 | ~120 | Recall@1/3/5/10 + MRR + 二值 nDCG，纯函数，含理想排序的 IDCG 计算 |
| `chunk_validator.py` | 185 | 规则校验（长度/页码/chunk_id 唯一性），W6 `citation.py` 可作基础 |
| `bge.py`（reranker） | 73 | cross-encoder 打分，**但必须加 timeout** |

### ◐ 拆后搬（约 250 行，扔掉一半）

**`embedding/bm25.py`** 表面是 106 行，实际耦合了四件事：分词、BM25 打分、pickle 索引落盘、`VectorRecord`/`load_chunks_jsonl` 的 IO。只保留前两件（约 60 行），落盘改走 pg 表或进程内缓存。

**`ingest.py` 那 424 行**里，`pdf_parser`/`html_parser` 值得搬，但 `storage.py` 的落盘格式要按 W5 的 pg 表改。

### ✗ 不搬（约 1100 行，搬了反而要删）

| 文件 | 行数 | 不搬的理由 |
|---|---|---|
| `vector_store/{local,milvus,factory,config}.py` | ~400 | W5 定的是 pgvector。搬来就得删，还带走 milvus 依赖 |
| `retrieval/{factory,retriever}.py` | 82+ | 工厂注册 + DenseRetriever，pgvector 版要重写 |
| `generation/{generator,pipeline,prompts}.py` | ~180 | **直连 `ChatDeepSeek`，绕开 gateway**。搬进来等于把 W1 的 alias/计量/trace 全废掉 |
| `eval/{smoke_runner,smoke_schema,matchers}.py` | ~440 | 是 rag_mvp 自己的 smoke 框架，和我们四态机 + golden 完全不兼容 |
| `embedding/visual_bge.py` | — | 依赖 git submodule，未接线 |
| `text_2_sql/` | 0 | 空目录 |
| `rag_mvp/cli.py` | 186 | 按文件路径加载 `scripts/`，那是开发期工具 |

---

## 四、目标目录（含分层）

```
packages/agent_framework/src/agent_framework/
├── knowledge/                    # 知识层：与"存什么"有关
│   ├── base.py                   # Chunk / ScoredChunk / Document（协议，零依赖）
│   ├── chunking.py               # ← 搬 777 行
│   ├── ingest.py                 # ← 搬 424 行
│   ├── metadata.py               # ← 搬 26 行
│   ├── pgvector_store.py         # 新写（接口参考 vector_store/base.py）
│   ├── citation.py               # W6
│   └── incremental.py            # W6
└── retrieval/                    # 检索层：与"怎么找"有关
    ├── base.py                   # async Protocol（重写，19 行）
    ├── bm25.py                   # ← 拆搬 60 行
    ├── hybrid.py                 # ← 搬 121 行
    ├── rerank.py                 # ← 搬 149 行 + timeout
    ├── pipeline.py               # 重写（dense 走 pgvector）
    ├── query_rewrite.py          # ← 搬 20 行 + multi-query
    ├── context_assemble.py       # ← 搬 117 行
    └── acl.py                    # W6

apps/growth_agent/src/growth_agent/rag/
└── embedding.py                  # ← 搬 144 行，torch 只在这里
```

**关键：`embedding.py` 放在 apps 而不是 core。** bge-m3 拉 `sentence-transformers` + `torch`（约 2GB），放进 core 包会让「框架零 ML 依赖」失守，以后任何项目引用框架都得装 torch。core 只定义 `EmbeddingProvider` 协议，apps 注入实现。

---

## 五、三个必须顺手修的缺陷

这三个是**搬之前发现、搬进来会一起带过来**的问题。复制时修，只改一次；不修，W6 的指标会假绿。

### 1. `RetrievalResult.source` 语义反了

rag_mvp 里 `source` = 检索器名（`dense` / `sparse` / `hybrid`）；W5 要的 `Chunk.source` = 文档出处。

直接搬的结果：溯源字段被检索器名覆盖，引用会指向 `"[hybrid]"`，W6 的 `citation_accuracy` 全部算错。

**修法**：搬进 `knowledge/base.py` 时立即改名 —— 检索器名 → `retriever`，文档出处 → `source`（取 `source_uri`）。

### 2. 答案里一条 `[n]` 都没有时，引用准确率算成 100%

`citations_referenced_in_answer` 的逻辑是「取答案中出现的编号集合」，没有编号时该集合为空，而后续判定退化成「全部通过」。W6 要求这种情况**算引用缺失并计入分母**。

**修法**：空集合直接返回 `0/len(citations)`，并在报告里单列 `no_citation_count`。

### 3. `RerankedRetriever` 没有超时

W5 D3 的考点原话是「重排超时降级是生产 RAG 与 demo RAG 的分水岭」。rag_mvp 的 rerank 是同步阻塞调用，模型卡住就整体卡住。

**修法**：`asyncio.wait_for(rerank, timeout=…)`，超时返回**未重排的召回结果**并记 `rerank_degraded=True`，不是抛错。

---

## 六、操作顺序（可脚本化）

### 步骤 1：建骨架 + 拷文件（10 分钟）

```bash
SRC=/Users/fanchen/PycharmProjects/LLMs/06_rag/rag_mvp/src
DST=/Users/fanchen/PycharmProjects/LLMs/agent_framework/packages/agent_framework/src/agent_framework

mkdir -p $DST/knowledge $DST/retrieval
# 切分链路 5 个文件先并到一个临时文件，再人工分段
cat $SRC/ingestion/{heading_detector,text_splitter,paragraph_splitter,chunker,page_aware_splitter}.py \
  > $DST/knowledge/chunking.py
```

**注意 `chunking.py` 的合并方式**：5 个文件里有些是 `@dataclass`、有些是模块级常量，直接 `cat` 会出现重名（`_split_sentences` 在 `text_splitter` 和 `paragraph_splitter` 里都有）。做法是保留原文件名作为分节注释，重名的加前缀。

如果嫌合并麻烦，**保留子目录也是可以的**：`knowledge/chunking/{splitter,chunker,heading}.py`。W5 落点表写的是 `chunking.py`，但那是示意，不是硬约束——**合并的意义是少一层 import，不值得为此改 5 个文件的函数名**。

### 步骤 2：改 import（26 处，脚本 + 手查）

```python
# 在 agent_framework 根目录跑
import re, pathlib
MAP = {
    "retrieval.types": "agent_framework.knowledge.base",
    "schemas":         "agent_framework.knowledge.base",
    "retrieval.base":  "agent_framework.retrieval.base",
    "retrieval.rrf":   "agent_framework.retrieval.hybrid",
    "retrieval.hybrid":"agent_framework.retrieval.hybrid",
    "retrieval.retriever": "agent_framework.retrieval.pipeline",
    "ingestion.storage":   "agent_framework.knowledge.ingest",
    "ingestion.document_loader": "agent_framework.knowledge.ingest",
    "ingestion.chunker":   "agent_framework.knowledge.chunking",
    "ingestion.text_splitter": "agent_framework.knowledge.chunking",
    "ingestion.page_aware_splitter": "agent_framework.knowledge.chunking",
    "validation.chunk_validator": "agent_framework.knowledge.ingest",
    "embedding.tokenizer": "agent_framework.retrieval.bm25",
    "embedding.base":      "agent_framework.knowledge.base",
    "embedding.embedding_types": "agent_framework.knowledge.base",
    "indexing.chunk_metadata": "agent_framework.knowledge.metadata",
    "generation.shcame":   "agent_framework.retrieval.context_assemble",
    "pre_query.normalizer":"agent_framework.retrieval.query_rewrite",
}
pat = re.compile(r'^(\s*from\s+)(' + '|'.join(map(re.escape, MAP)) + r')\b')
for p in pathlib.Path("packages/agent_framework/src/agent_framework").rglob("*.py"):
    t = p.read_text("utf-8")
    n = pat.sub(lambda m: m.group(1) + MAP[m.group(2)], t)
    if n != t: p.write_text(n, "utf-8"); print("fixed", p.name)
```

脚本改完必须**手工过一遍**，因为三种情况脚本处理不了：

1. `from retrieval.types import RetrievalResult` → 名字也要改成 `ScoredChunk`
2. 同一个文件里 `from .base import Reranker` 这种相对导入，合并后层级变了
3. `grep -rn "^\s*import "` 漏掉的三处（`pickle`/`pathlib` 不用改，但 `from vector_store.vector_store_types import VectorRecord` 要删）

**验收判据**：`grep -rE "^\s*(from|import)\s+(retrieval|ingestion|embedding|indexing|vector_store|reranker|generation|eval|validation|schemas|pre_query)\b" packages/` 输出必须为空。

### 步骤 3：异步边界（1 小时）

**不要改被复制代码的内部逻辑。** 在 `retrieval/base.py` 定义 async Protocol，用一层薄适配包住同步实现：

```python
# retrieval/base.py
class Retriever(Protocol):
    async def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]: ...

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-cpu")

class SyncRetrieverAdapter:
    """把同步 Retriever 包成 async，CPU 密集段走专用线程池。"""
    def __init__(self, inner: "_SyncRetriever") -> None:
        self._inner = inner
    async def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_EXECUTOR, self._inner.retrieve, query, top_k)
```

两个细节会决定 p95：

- **`max_workers=2` 而不是默认值**。默认 `min(32, cpu+4)` 会让 8 核机器上跑 12 个 bge-m3 线程，彼此抢 GIL，2 路并发就能把 p95 拖到比同步更差。
- **模型加载后 `torch.set_num_threads(1)`**。不设的话 2 个池线程 × 8 核 OMP = 16 个计算线程互相争抢。

**验收判据**：并发 4 路跑 20 次，p95 不高于串行的 1.3 倍。高于这个数说明线程池或 OMP 配错了。

### 步骤 4：搬测试（30 分钟）

rag_mvp 的 `test/` 有 35 个测试文件。只搬对应被复制模块的（切分、RRF、指标、BM25 约 12 个文件），import 同步骤 2 改。

**别搬 smoke 相关测试**（约 8 个）——它们测的是 rag_mvp 自己的 smoke 框架，和我们四态机不兼容。

### 步骤 5：跑通 Gate（见下）

---

## 七、工时

| 阶段 | 工时 |
|---|---|
| 步骤 1–2：建骨架 + 拷 29 文件 + 改 26 处 import | 0.8h |
| 步骤 3：异步边界 + 线程池配置 | 1.0h |
| 三个缺陷修复（第五节） | 1.0h |
| 步骤 4：测试迁移 + 补 3 条反向验证 | 0.5h |
| **搬运小计** | **3.3h** |
| W5 剩余（pgvector 1.5h + multi-query 1h + 五档 A/B 1.5h + 接线门禁 1h） | 5h |
| W6 剩余（citation 1.5 + incremental 2 + acl 1.5 + 治理 1.5 + 清单 1.5） | 8h |
| **W5 + W6 合计** | **16.3h** |

对比改造路线 15h：**复制贵 1.3h**。贵在两个地方：26 处 import 要手工核、5 个切分文件要合并改名。而改造路线这些工作在家里（rag_mvp）已经做过一次（185 处 import 脚本替换，虽然是脚本但一次到底）。

---

## 八、何时该选复制

工时只差 1.3h，所以决策不看工时，看两个问题：

**问 1：rag_mvp 以后还维护吗？**

- **不维护了**（只是这次的学习产物）→ **选复制**。搬完就可以把它当参考资料冻结，不用为它维护 extras、不用管 submodule、不用在两个仓库间同步 lock。
- **还要演进**（视觉检索、text2sql 那些空目录是打算做的）→ **选改造**。否则你在 agent_framework 里改的每一处，rag_mvp 那边都停在旧版，半年后两个版本谁也说不清哪个对。

**问 2：rag_mvp 要不要作为独立作品出现？**

- **要**（简历上两件作品）→ 改造，做成独立服务或独立库。
- **不要**（它就是 agent_framework 的素材库）→ 复制，代码搬完原目录归档。

按现在的信息（`text_2_sql/` 是空目录、`visual_bge.py` 未接线、submodule 没初始化），**rag_mvp 其实是一个"写了一半就转向"的项目**。如果它不打算继续，复制是对的；如果那些空目录是待办清单，改造是对的。

---

## 九、复制路线一定会踩的三件事

1. **`chunking.py` 合并时函数重名** —— `text_splitter` 和 `paragraph_splitter` 都有 `_split_sentences`，直接 cat 会静默覆盖。必须先 grep 重名再合。

2. **`golden_runner.py` 不能整份搬** —— 它 import 了 `retrieval.factory.create_retriever`（那个要删）。只抽 `_ndcg_at_k` / `_first_gold_rank` / `_recall_at` / `_mrr` 四个纯函数，报告格式化另写（我们要并进 `evals/metrics.py` 的 `TaskMetrics`，不是独立 JSON）。

3. **gold 标注格式不一样** —— rag_mvp 的 30 题带 `gold_chunk_ids`（chunk 级），我们的 `golden_v1.jsonl` 带 `references`（URL/repo 级）。**两套并存，别试图合并**：`evals/golden/golden_v1.jsonl` 继续做端到端任务评测，新加 `evals/golden/retrieval_v1.jsonl` 专做检索层 Recall/MRR。指标口径不同，混在一份里会让「检索好了」和「答得好」互相掩盖。

---

## 十、复制路线的真实代价

工时差 1.3h 是小事，真正的代价是**代码分叉后没有归途**：

- rag_mvp 里发现的 bug（比如第五节那三个）修在 agent_framework，rag_mvp 那边还是坏的
- 以后要改切分策略，得决定改哪一份
- 如果哪天要拿 rag_mvp 当作品展示，它跑的是旧代码

**缓解办法**：复制完在 rag_mvp 的 README 顶部加一行——

> ⚠️ 本仓的切分/检索/指标算法已迁至 `agent_framework/packages/agent_framework/src/agent_framework/{knowledge,retrieval}/`，本仓不再演进。保留原因：语料与 gold 标注（`data/`）。

这样至少下一个人不会在错的地方改代码。
