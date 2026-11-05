# 包化与接线改造记录（阶段 0–4）

> 目的：让 `rag_mvp` 成为一个**可安装、可被其他项目 import** 的普通包，
> 并且能被 `agent_framework` 当成检索层直接用。
> 上游方案见 `agent_framework/docs/rag-mvp-refactor-plan.md`。

---

## 一、改之前是什么状态

```
$ uv build
rag_mvp-0.1.0-py3-none-any.whl
  rag_mvp/__init__.py
  rag_mvp/cli.py                 ← wheel 里只有这两个文件
```

`src/retrieval/`、`src/ingestion/`、`src/embedding/` 等 11 个真正的包**全部没进 wheel**。
原因：`uv_build` 按 `project.name = "rag-mvp"` 归一化出 `rag_mvp`，只认 `src/rag_mvp/` 为包，`src/` 下其他顶层目录一律忽略。

它当时能跑，靠的是 editable 安装的副作用：

```
$ cat .venv/lib/python3.12/site-packages/rag_mvp_src.pth
/Users/.../06_rag/rag_mvp/src
```

整个 `src/` 被塞进 `sys.path`，于是 `import retrieval` / `import schemas` / `import eval` 全都能用。
**代价是 11 个通用顶层名（`eval`、`schemas`、`validation`、`generation`…）永久占据进程命名空间**，
任何依赖它的项目都可能被静默覆盖——而且不报错，胜负取决于 `.pth` 加载顺序。

---

## 二、做了什么

### 1. 物理包化

```
src/                      src/rag_mvp/
├── __init__.py      →    （删除，它是空的且会让 src 变成包）
├── schemas.py       →    schemas.py
├── retrieval/       →    retrieval/
├── ingestion/       →    ingestion/
├── embedding/       →    embedding/
├── indexing/        →    indexing/
├── vector_store/    →    vector_store/
├── reranker/        →    reranker/
├── generation/      →    generation/
├── pre_query/       →    pre_query/
├── eval/            →    eval/
├── validation/      →    validation/
└── text_2_sql/      →    text_2_sql/
```

用 `git mv` 完成，89 个文件的移动被 git 识别为 rename，历史不丢。

### 2. 改写 185 处 import（73 个文件）

`src/` 121 处、`test/` 45 处、`scripts/` 19 处，全部 `from retrieval.xxx` → `from rag_mvp.retrieval.xxx`。
脚本机械替换后自检残留为 **0**。

### 3. 手工修脚本抓不到的 13 处

正则会漏掉**字符串形态的模块名**，这类最危险——IDE 和静态检查都看不见，只在运行时炸：

| 位置 | 形态 | 数量 |
|---|---|---|
| `retrieval/__init__.py` | `_LAZY_EXPORTS` 元组里的 `("retrieval.base", "Retriever")` | 8 |
| `vector_store/__init__.py` | 同上 | 3 |
| `embedding/__init__.py` | 同上 | 1 |
| `test/test_hybrid_retriever.py` | `monkeypatch.setattr("retrieval.factory.BGEM3EmbeddingProvider", …)` | 1 |

最后那条还暴露了一个**跨行盲区**：字符串被拆到 `setattr(` 的下一行，单行 grep 搜不到。第一轮全量测试就是被它挡住的（`ModuleNotFoundError: No module named 'retrieval'`）。

### 4. `visual-bge` 拆进 extra

```toml
[project.optional-dependencies]
visual = ["visual-bge"]
```

它原本在默认依赖里，导致 `uv lock` 阶段就要执行该包（submodule）的 `setup.py` 求依赖，而那个解析不过——
**任何新环境的 `uv sync` 都会直接失败**。这不是"重"，是"装不上"。

拆走后 `uv lock` 通过，默认 `uv sync` 不再安装它。

---

## 三、验收判据与实测

| 判据 | 结果 |
|---|---|
| wheel 里包含 11 个子包 | ✅ `embedding 18 / ingestion 19 / retrieval 13 / generation 10 / vector_store 10 / eval 8 / pre_query 6 / reranker 6 / indexing 4 / text_2_sql 4 / validation 4` 文件 |
| wheel 里**没有**顶层污染包 | ✅ 非 `rag_mvp/` 前缀的条目为 0 |
| `import rag_mvp` 可用 | ✅ `from rag_mvp.retrieval.types import RetrievalResult` 通过 |
| `import retrieval` 必须失败 | ✅ `ModuleNotFoundError: No module named 'retrieval'` |
| 逻辑未被改坏 | **52 passed / 1 failed** |

剩下那 1 条 `test_build_milvus_index` 是 **pymilvus 连不上服务**（`_InactiveRpcError`），需要本地 Milvus 实例。
已用 `git worktree` 检出包化前的 HEAD 复跑同一批测试，**同样两条以同样错误失败**，确认非本次引入。

---

## 四、顺手修掉的一个既有缺陷

`test_hybrid_retriever.py::test_hybrid_retriever_uses_rrf` 原本 patch 的是
`retrieval.factory.BGEM3EmbeddingProvider`，但 `factory.py` 里这个符号是**函数内惰性导入**的，
模块级根本不存在 → 该测试一直以 `AttributeError` 失败，**等于从未真正验证过 RRF 合并逻辑**。

改成 patch 源模块 `rag_mvp.embedding.BGEM3EmbeddingProvider` 后才真正跑起来。
这条属于原代码的测试缺陷，不是包化造成的（worktree 复现确认）。

---

## 五、阶段 1：依赖分层（32 → 4 + 5 extras）

分层依据是 **AST 扫描出的真实 import**，不是"当初写了什么"。原来声明的 32 个依赖里，
**19 个从未被任何代码 import**（`chromadb`、`elasticsearch`、`fastapi`、`langgraph`、
`llama-index-*`、`qdrant-client`、`sqlalchemy`、`asyncmy`、`pymysql`、`unstructured`、
`cryptography`、`greenlet`、`omegaconf`、`loguru` 等），属于探索期残留。

| 层 | 内容 | 被谁用 |
|---|---|---|
| core（默认） | `jieba` `pyyaml` `python-dotenv` `rank-bm25` | 分词、数据集解析、凭证、BM25 |
| `local` | `sentence-transformers`（连带 torch） | `embedding/bge_m3.py`、`reranker/bge.py` |
| `pdf` | `pymupdf` | `ingestion/pdf_parser.py` |
| `milvus` | `pymilvus` | `vector_store/milvus.py`（惰性导入） |
| `llm` | `langchain-deepseek` | `generation/generator.py`（探索期，将被 gateway 取代） |
| `visual` | `numpy` `torch` `visual-bge` | `embedding/visual_bge.py`（未接线） |

**实测效果**（`uv sync` 默认行为）：

```
Would uninstall 64 packages      # 含 torch / sentence-transformers / pymilvus / pymupdf
Would install 6 packages
```

lock 从 **286 包降到 114 包**（少 172 个，含 spaCy / kubernetes / opencv / onnxruntime /
matplotlib 这些被 `unstructured`、`qdrant-client` 拖进来的传递依赖）。

## 六、阶段 2：agent_framework 侧只落协议

`packages/agent_framework/src/agent_framework/` 新增两个包，**零新增依赖**（只用了已有的 pydantic）：

- `knowledge/base.py`：`Chunk` / `ScoredChunk` + `EmbeddingProvider` / `VectorStore` 协议
- `retrieval/base.py`：`Retriever` / `Reranker` / `QueryRewriter` 协议

三条写进注释、也被测试锁住的约定：

1. **`chunk.source_uri`（文档出处）与 `scored.retriever`（谁召回的）是两个字段。**
   rag_mvp 用 `source` 一个名字表达两件事，直接搬会让引用指向 `"[hybrid]"`。
2. **分数不可跨检索器比较。** dense 是余弦（0–1）、sparse 是 BM25 原始分（无上界、随语料浮动）、
   rerank 是 cross-encoder logit。混排等于让语料规模决定排序——这正是 RRF 只用名次的原因。
3. **空结果 ≠ 失败。** 返回 `[]` 是内容问题（补语料/改查询），抛异常是系统问题（重试/降级）。
   两者混成空，W4 的根因归因会把 rerank 超时误判成"语料缺失"。

## 七、阶段 3：桥接层 `contrib/agent_framework.py`

把同步检索接成 async，三件事都不是"加个 async def"能解决的：

| 问题 | 处理 |
|---|---|
| 阻塞 | **专用** `ThreadPoolExecutor(max_workers=2)`，不用 asyncio 默认 executor（那是 min(32, cpu+4) 个线程，CPU 密集任务互抢 GIL，2 路并发就比同步还慢） |
| 字段同名反义 | `RetrievalResult.source` → `ScoredChunk.retriever`；`DocumentChunk.source_uri` → `Chunk.source_uri` |
| 过滤条件 | **不支持就报错**，不静默忽略——W6 的 ACL 走 filters，忽略等于越权 |

另外两个细节：

- `metadata_to_document_chunk` 缺页码时填 **0**，而 0 是无效页码（会写出"第 0 页"）。桥接层统一转 `None`。
- `configure_local_threads(1)`：不设的话 2 个工作线程 × 8 核 OMP = 16 个计算线程互抢缓存，
  现象是"并发越高越慢"，且极难归因。

**超时抛 `RetrievalTimeoutError`，不返回空列表**（理由同第六条约定）。注意
`ThreadPoolExecutor` 无法取消已提交任务，超时只保证调用方脱身，不保证 CPU 立刻释放——
这点写进了异常消息，避免下一个人以为超时后线程就停了。

---

## 八、包化带来的一个副作用：pickle 里焊死了旧模块路径

包化后第一次跑真实检索就炸了：

```
ModuleNotFoundError: No module named 'embedding'
```

不是文件坏了，是 **pickle 存的是 `__module__ + __qualname__`，不是类本身**。
`data/index/{bm25,local}.pkl` 里记的还是 `embedding.bm25.BM25Index`、
`vector_store.vector_store_types.VectorRecord` 这些包化前的顶层名。

处理方式是**加载时映射**，不是往 `sys.modules` 塞别名：

```python
class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in LEGACY_TOPLEVEL:
            module = f"rag_mvp.{module}"
        return super().find_class(module, name)
```

塞别名会把"顶层包污染"重新固定进进程，正是包化要消除的东西，还会让
`import retrieval` 重新变得可用、掩盖真正的问题。映射只在 `load` 时生效，
新写出的索引带新前缀，不触发映射。

改动落在 `rag_mvp/compat.py` + `BM25Index.load` / `LocalVectorIndex.load` 两处。
两个旧索引实测都能读：`local.pkl 425 条 / dim=1024`、`bm25.pkl 425 条`。

**长期方向**：索引不该用 pickle 存对象。它把实现细节（模块路径、类布局）焊进数据文件，
任何重构都会让历史数据失效。换格式时优先选 JSON / parquet 这类只描述数据的格式。

---

## 九、实测：四种检索模式（语料 425 chunks，查询「MCP 是什么」）

| 模式 | #1 | #2 | #3 | 建模 | 查询 |
|---|---|---|---|---|---|
| sparse（BM25） | ❌ p111「简短描述项目是什么、做什么。」 | p126 | ✅ p85 | ~0s | — |
| dense | ✅ p152 MCP 定义 | ✅ p159 MCP 核心机制 | ✅ p152 | 20.5s | 1.47s |
| hybrid（RRF） | ✅ p159 | ✅ p152 | ❌ p111 | 18.0s | 0.65s |
| hybrid + rerank | ✅ p152 | ✅ p159 | ✅ p159 | 29.2s | 3.10s |

**hybrid 反而比 dense 差**，原因值得记住：RRF 只用名次。BM25 把 p111 排第 1
（被疑问词"是什么"带偏），dense 里它进不了前 3；但 sparse 第 1 名的 RRF 贡献
`1/61=0.0164` 已经超过 dense 第 2 名的 `1/62=0.0161`，于是它挤进了融合结果。

结论：**混合检索不会自动变好，它会把单路的系统性偏差原样带进融合结果。**
解药在查询侧（停用词、疑问句改写）或精排——上表最后一行就是精排把 p111 挤出去的证据。
这条直接成为 W5 D5 的 A/B 报告起点。

---

## 十、阶段 4：pgvector 后端

### 选型结论

**postgres + pgvector，不用 Milvus。** 理由不是性能，是运维面：项目里已有 postgres
（Temporal 用它），加一个 `vector` 扩展就是同库不同表；Milvus 要多一份常驻服务、
多一份内存、多一套备份策略，而它的收益（亿级向量、原生分区）在万级 chunk 上用不到。

### 实现要点（每条都对应一个会静默出错的点）

| 点 | 做法 | 不这么做的后果 |
|---|---|---|
| 装扩展 | `ops/pgvector/init.sql`，容器首启以 postgres 身份执行 | `CREATE EXTENSION` 需要超级用户，应用层做不到 → 运行时 "permission denied to create extension"，看起来与向量无关 |
| 传向量 | 文本字面量 `%s::vector`，不装 `pgvector` 这个 Python 包 | 少一个依赖，也免掉驱动注册；注册漏了会报一个和向量毫无关系的错 |
| 分数方向 | `1 - (embedding <=> %s)` | `<=>` 是余弦**距离**，直接用会让"最不相关"排第一，而分数仍是正数 |
| 缓冲语义 | `add()` 只进内存，`save()` 才落库，失败**不清空** | 逐条往返让导入耗时线性上升；失败清空则一次网络抖动丢整批 |
| `count()` | `SELECT count(*)`，数已持久化的行 | 数缓冲会让"导入了 100 条实际 0 条"看不出来 |
| `upsert()` | `ON CONFLICT (chunk_id) DO UPDATE` | 重复执行产生重复数据，检索结果里出现同一段文本 3 次 |
| 表名 | 构造期正则校验 | 表名要拼进 SQL，`x; DROP TABLE chunks` 会被当合法表名执行 |
| 索引 | `hnsw (embedding vector_cosine_ops)` + `doc_id` btree + `metadata` gin | `vector_cosine_ops` 与查询侧 `<=>` 不一致时索引不被使用，表现为"加了索引还是全表扫" |

额外加了 `delete_document()`（不在 `VectorWriter` 契约里）：W6 的增量索引
"更新一个文档"必须能按 `document_id` 成组删除，否则只能全量重建，耗时随语料线性增长。
空 `document_id` 直接报错——拼出来是 `WHERE document_id = ''`，静默删除是更坏的默认值。

### 验收：不只是单测绿

| 层次 | 数量 | 证明什么 |
|---|---|---|
| 单元（假连接） | 23 | SQL 拼得对、缓冲语义、表名校验、字面量格式 |
| 集成（真库） | 11 | `<=>` 方向、`ON CONFLICT` 幂等、`@>` 命中 jsonb、TRUNCATE 保留结构 |
| 端到端（真库+框架） | 6 | `RetrievalPipeline → AsyncRetriever → DenseRetriever → PgVectorStore` 全程字段不串 |

**换后端等价性实测**（真实语料 425 chunks、真实 bge-m3、查询「MCP 是什么」）：

```
rank local chunk                        score     pgvector chunk                     score
1    claude-code-0001-p152-p152-c330    0.678577  claude-code-0001-p152-p152-c330    0.678577
2    claude-code-0001-p159-p160-c348    0.630113  claude-code-0001-p159-p160-c348    0.630113
3    claude-code-0001-p152-p152-c329    0.572487  claude-code-0001-p152-p152-c329    0.572487
4    claude-code-0001-p159-p159-c347    0.565310  claude-code-0001-p159-p159-c347    0.565310
5    claude-code-0001-p157-p158-c344    0.564612  claude-code-0001-p157-p158-c344    0.564612
```

5/5 同 chunk、分数小数点后 6 位完全相同。脚本留在 `scripts/compare_vector_backends.py`，
以后换任何后端都先过这一关——**等价性不验，"指标变好"就分不清是改动起效还是后端差异**。

### 与本地 pickle 后端的差异（不是 bug，是特性）

`local` 后端 `save()` 是整文件覆盖，天然全量；pgvector 走 `upsert`，天然支持增量。
这也正是 pgvector 必须先做的原因：**增量索引在 pickle 后端上做不干净。**

---

## 十一、仍未处理

- `test_build_milvus_index` 需要外部 Milvus 服务。CI 里应加 `-m "not milvus"` 或探测可用性后 skip——
  **本次没动**，因为把"环境不具备"改成 skip 容易变成掩盖失败的习惯，先如实留着。
- `generation/generator.py` 直连 `ChatDeepSeek`，绕开 `LLMGateway`（无 alias、无计量、无 trace）。
  它已在 `llm` extra 里隔离，正式接线时应整体替换为 gateway 调用。
- `visual_bge.py` 未接线，仍在 `visual` extra。
- 精排的**单独超时降级**由 framework 侧 `RetrievalPipeline` 承担（超时后回退 RRF 顺序），
  `RerankedRetriever` 本身仍只有整链路超时。
- **`filters` 未接通到底**：`PgVectorStore.search(filters=...)` 已能按 jsonb 包含过滤，
  但 `DenseRetriever.retrieve` 只调 `store.search(vec, top_k)`，不转发 filters。
  桥接层对非空 filters 是显式 `NotImplementedError`（正确做法——静默忽略等于越权），
  W6 做 ACL 时补 `DenseRetriever → store` 这段接线。
- pgvector 后端**不用连接池**：检索是低频调用且已有自己的线程池限流，再加一层池会让
  "到底是哪层在排队"不可归因。连接断了重连，代价可接受；但高并发场景要重新评估。

---

## 十二、以后怎么用

`agent_framework` 侧（推荐，走协议）：

```python
from agent_framework.knowledge import Chunk, ScoredChunk
from agent_framework.retrieval import Retriever

# 装配在 apps 层完成，框架只依赖协议
from growth_agent.rag import build_retriever
retriever: Retriever = build_retriever()          # 启动阶段调一次
hits = await retriever.retrieve("MCP 是什么", top_k=5)
```

`rag_mvp` 侧（直接用）：

```python
from rag_mvp.retrieval.types import RetrievalResult
from rag_mvp.schemas import DocumentChunk
from rag_mvp.ingestion.pipeline import build_chunks
```

注意：`rag_mvp.retrieval` / `rag_mvp.embedding` / `rag_mvp.vector_store` 三个包用 `__getattr__`
做惰性导出（避免 `import` 就加载 torch）。**patch 这些符号时要指向源模块**，不是访问点。

`contrib` 是可选适配层，不被本体依赖：删掉整个目录，切分、索引、CLI、评测照常工作。
