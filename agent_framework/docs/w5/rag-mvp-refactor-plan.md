# rag_mvp 改造方案（目标：被 agent_framework 直接 import 使用）

> 前置评估见 `docs/rag-mvp-reuse-assessment.md`。  
> 本文只回答一个问题的落地路径：**把 rag_mvp 改造到 agent_framework 能直接 `import` 并当 RAG 层用，要改什么、按什么顺序、每步怎么验证。**

---

## 零、开工前的两个前置条件

这两条不是"优化"，是**改造能不能被验证**的前提。第一条与 RAG 无关但优先级更高。

### 前置 A：`agent_framework/.github/workflows/ci.yml` 是死文件，GitHub 从来没读过它

```
$ ls -a /Users/fanchen/PycharmProjects/LLMs | grep github
（无）

$ git rev-parse --show-toplevel
/Users/fanchen/PycharmProjects/LLMs          ← 仓库根在这里
```

GitHub Actions 只扫描**仓库根**的 `.github/workflows/`。这个仓库的根是 `PycharmProjects/LLMs`（`agent_framework/`、`06_rag/`、`07_agent/` 都是它的子目录），而根下没有 `.github`。

所以 `agent_framework/.github/workflows/ci.yml` 里那四步（离线快照 / 门禁对比 / 门禁必须能红 / judge 一致性）**一次都没被执行过**。

这不代表质量链是空的——本地的 `make eval` 是真跑的（496 passed、门禁 2 可比 0 违规都是实测），但**远端没有自动化**：改坏了没人拦，除了你自己记得跑。

**修法**（15 分钟）：把文件移到仓库根 `.github/workflows/agent-framework.yml`，加路径过滤和默认工作目录：

```yaml
on:
  push:
    paths: ["agent_framework/**", ".github/workflows/agent-framework.yml"]
  pull_request:
    paths: ["agent_framework/**"]

defaults:
  run:
    working-directory: agent_framework
```

顺带要确认的：`uv sync` 的 lock 在 `agent_framework/uv.lock`，`working-directory` 设对之后 `actions/setup-uv` 的 cache 键要改成 `agent_framework/uv.lock`，否则缓存永远命中不了。

### 前置 B：`visual-bge` 卡住 `uv lock`，必须先拆

```
$ uv lock
× Failed to build `visual-bge @ file:///.../06_rag/rag_mvp/third_party/FlagEmbedding/research/visual_bge`
├─▶ Failed to resolve requirements from `setup.py` build
```

submodule 本身是初始化好的（`git submodule status` 能看到 commit `fd1a2bd`），卡住的是这个 submodule 里的包**在 lock 阶段就要执行 `setup.py` 求依赖**，而它的 build 依赖解析不过。

只要 rag_mvp 的 pyproject 里还有对它的引用，**agent_framework 的 `uv sync` 就会连带失败**——这跟选不选改造无关，是任何一个新环境都装不上。

**修法**：把它挪进 `[project.optional-dependencies]` 的 `visual` extra，并从默认依赖里删掉。`visual_bge.py` 本来就是个未接线的实验文件，W5/W6 都用不到。

---

## 一、先说三个实测事实（它们决定改造顺序）

### 事实 1：现在的 rag_mvp 装成依赖后是**不可用**的

```
$ uv build --out-dir /tmp/ragbuild
$ unzip -l rag_mvp-0.1.0-py3-none-any.whl
  rag_mvp/__init__.py
  rag_mvp/cli.py
  rag_mvp-0.1.0.dist-info/...
```

**wheel 里只有 2 个文件。** `src/retrieval/`、`src/ingestion/`、`src/embedding/` 等 11 个真正的包全部没被打进去。

原因：`uv_build` 按 `project.name = "rag-mvp"` 归一化出 `rag_mvp`，只认 `src/rag_mvp/` 这个目录为包，`src/` 下其他顶层目录一律忽略。

后果：`pip install rag-mvp` 之后 `import retrieval` 直接 ModuleNotFoundError，`rag-mvp search` 一执行就崩。它今天能跑，**完全靠 editable 安装的副作用**（见事实 2）。

### 事实 2：editable 安装会**把整个 `src/` 塞进 sys.path**

```
$ cat .venv/lib/python3.12/site-packages/rag_mvp_src.pth
/Users/fanchen/PycharmProjects/LLMs/06_rag/rag_mvp/src

$ ./.venv/bin/python -c "import retrieval, schemas; print(retrieval.__file__)"
/Users/fanchen/PycharmProjects/LLMs/06_rag/rag_mvp/src/retrieval/__init__.py
```

也就是说：一旦 agent_framework 以 editable 方式依赖 rag-mvp，它的进程里会多出 11 个顶层模块名 + `schemas`：

```
retrieval  ingestion  embedding  indexing  vector_store  reranker
generation  pre_query  eval  validation  schemas  text_2_sql
```

而 **W5 明确要建 `retrieval/` 目录，W6 要建 `knowledge/`** — `retrieval` 直接撞名。

最坏的地方是**它不会报错**。谁赢取决于 `.pth` 加载顺序（按文件名排序），也就是说：

- 本地开发可能是对的结果，CI 上是错的结果
- 或者反过来
- 报错信息会指向一个看起来完全无关的文件

这类 bug 的排查成本远高于改造本身。

### 事实 3：185 处 import 依赖"顶层可导入"这个前提

| 目录         | 行数                  |
| ---------- | ------------------- |
| `src/`     | 121                 |
| `test/`    | 45                  |
| `scripts/` | 19                  |
| **合计**     | **185**（分布在 73 个文件） |

按模块：`retrieval` 42 · `ingestion` 26 · `embedding` 26 · `vector_store` 26 · `schemas` 16 · `eval` 14 · `generation` 13 · `indexing` 8 · `reranker` 7 · `pre_query` 5 · `validation` 2。

全是 `from retrieval.types import ...` 这种顶格绝对导入，没有任何相对导入。**好消息**：这是机械替换，脚本能做完。

---

## 二、结论：改造顺序必须是「先包化，再适配」

```
阶段 0  包化            sys.path 污染消失，wheel 完整       1h   ✅
阶段 1  依赖分层        框架包仍是零 ML 依赖                0.5h ✅
阶段 2  协议定义        agent_framework 侧零依赖协议       0.5h ✅
阶段 3  桥接层          async 门面 + 类型映射              1.5h ✅ 到此 W5 D1/D2 可跑
阶段 4  pgvector        按 W5 决策换后端                    1.5h ✅
阶段 5  W5 余项         rewrite / A-B / 预算降级            3h   ← 下一步
阶段 6  W6              citation/增量/ACL/assemble/治理     7h
```

**最小可用 = 阶段 0–3（3.5h）**，之后 W5 的 D1/D2 就能在 agent_framework 里跑真实检索。

包化有个附带好处：改完后 `src/` 下**只剩 `rag_mvp/` 一个目录**，于是 `rag_mvp_src.pth` 把 `src/` 加进 sys.path 也变得完全无害——因为那里已经没有别的东西可污染了。

---

## 三、阶段 0：包化（1h）

### 操作

```bash
cd /Users/fanchen/PycharmProjects/LLMs/06_rag/rag_mvp
git checkout -b refactor/package-layout          # 可回退
mkdir -p src/rag_mvp

for m in retrieval ingestion embedding indexing vector_store reranker \
         generation pre_query eval validation text_2_sql; do
  git mv "src/$m" "src/rag_mvp/$m"
done
git mv src/schemas.py src/rag_mvp/schemas.py
git rm src/__init__.py                            # 顶层 src 不该是包
touch src/rag_mvp/py.typed                        # 让 mypy/pyright 认类型
```

### 改 import（脚本，别手改 185 处）

```python
# scripts/_migrate_imports.py（一次性脚本，跑完可删）
import re, pathlib

PKGS = ("retrieval ingestion embedding indexing vector_store reranker "
        "generation pre_query eval validation schemas text_2_sql").split()
pat = re.compile(rf'^(\s*)(from|import)\s+({"|".join(PKGS)})\b', re.M)

for root in ("src", "scripts", "test", "example"):
    for p in pathlib.Path(root).rglob("*.py"):
        s = old = p.read_text("utf-8")
        s = pat.sub(lambda m: f"{m.group(1)}{m.group(2)} rag_mvp.{m.group(3)}", s)
        # 已是 rag_mvp. 的不重复加
        s = s.replace("rag_mvp.rag_mvp.", "rag_mvp.")
        if s != old:
            p.write_text(s, "utf-8")
            print("rewrote", p)
```

`cli.py` 已验证**不用改**：它用 `importlib.util` 按**文件路径**加载 `scripts/xxx.py`，`PROJECT_ROOT = Path(__file__).parents[2]`——文件留在 `src/rag_mvp/` 位置，层级没变。

但要注意一个改造前就存在的缺陷：**`scripts/` 不在 wheel 里**，所以 `rag-mvp` 这个 console script 只在源码目录能用，装成依赖后必然找不到脚本。处理二选一：

- 把 `scripts/` 收进包内（`src/rag_mvp/cli_scripts/`），CLI 才真正可分发；
- 或从 `[project.scripts]` 删掉，承认它是开发期工具。  
  推荐后者——W5/W6 用不到这些脚本，它们只是 Phase 1–4 的构建入口；留着反而让包体积和依赖变复杂。

### `pyproject.toml` 保持 `pythonpath = ["src"]`

包化后 `src/` 里只有 `rag_mvp/`，这条配置不再有副作用，留着能让 `pytest` 不依赖安装状态。

### 验收（四条，缺一不可）

```bash
# 1 wheel 里有 11 个子包
uv build --out-dir /tmp/ragbuild2
unzip -l /tmp/ragbuild2/*.whl | grep -c "rag_mvp/"     # 期望 > 60，不再是 2

# 2 非 editable 安装后能 import
uv pip install /tmp/ragbuild2/*.whl --target /tmp/t && \
  PYTHONPATH=/tmp/t uv run python -c "import rag_mvp.retrieval; print('ok')"

# 3 顶层名不再可见（这是本次改造的核心目的）
uv run python -c "import retrieval" 2>&1 | grep -q ModuleNotFoundError && echo "干净"

# 4 原有能力没退化
uv run pytest -q                                      # 期望原 19 个测试文件全绿
rag-mvp search "MCP 是什么" -k 3                      # CLI 仍可用
```

---

## 四、阶段 1：依赖分层（0.5h）

现在 `dependencies` 是 32 个一锅端，`sentence-transformers`(torch) 和 4 个向量库全在里面。agent_framework 核心包只有 6 个依赖且零 ML——**不能因为接 RAG 就把 torch 拖进框架**。

改造 `pyproject.toml`：

```toml
[project]
dependencies = [
    "pyyaml>=6.0.3",        # golden 数据集
    "rank-bm25>=0.2.2",     # BM25
    "jieba>=0.42.1",        # 中文分词
    "pymupdf>=1.28.2",      # PDF 解析
    "python-dotenv>=1.2.3",
]

[project.optional-dependencies]
# 本地模型（重，约 2GB）
embedding-local = ["sentence-transformers>=6.1.0", "huggingface-hub>=1.32.0"]
rerank-local    = ["sentence-transformers>=6.1.0"]
# 云端 embedding（无 torch）
embedding-api   = ["httpx>=0.28.1"]
# 向量库
pgvector        = ["psycopg[binary]>=3.2", "pgvector>=0.3"]
milvus          = ["pymilvus>=3.0.2"]
# Agent 框架桥接（只走 async 门面时不需要任何额外依赖）
agent           = ["httpx>=0.28.1"]
```

**这样 agent_framework 只装 `rag-mvp[pgvector]`（约 5 个依赖），torch 完全不进来。**

顺带处理三个当前依赖里不该有的：`langchain`/`langgraph`/`llama-index`/`chromadb`/`elasticsearch`/`qdrant-client`/`fastapi`/`sqlalchemy`/`asyncmy`/`pymysql`/`unstructured` —— 这些是探索期留下的，检查后删掉。`generation/generator.py` 用了 `langchain-deepseek`，但那部分**本来就要被删掉**（见阶段 3：LLM 调用必须走 gateway）。

### 一个必须先拆的地雷：`visual-bge` 是 submodule 里的 editable 包

`pyproject.toml` 现在有：

```toml
[tool.uv.sources]
visual-bge = { path = "third_party/FlagEmbedding/research/visual_bge", editable = true }
```

而 `third_party/FlagEmbedding` 是 git submodule（未初始化时目录是空的），`visual_bge` 还有自己的 `setup.py`。实测后果：

```
$ uv lock --offline
× Failed to build `visual-bge @ file:///.../third_party/FlagEmbedding/research/visual_bge`
├─▶ Failed to resolve requirements from `setup.py` build
```

也就是说：**任何新环境（包括 agent_framework 的 `uv sync`）想在没 `git submodule update --init` 的情况下装这个包，都会直接失败。** 这一条比 torch 更硬——torch 只是重，这个是不装不上。

处理：把 `visual-bge` 从 `dependencies` 挪进 `[embedding-visual]` extra，并从默认 `[tool.uv.sources]` 移除。视觉检索不在 W5/W6 范围（`embedding/visual_bge.py` 也只是一个未接线的实验文件），留着它只会让核心包不可安装。

> 附带收益：删掉 langchain 后，`generation/generator.py` 的 `ChatDeepSeek` 直连也没了，ADR-004「禁止裸 HTTP 走模型」自然满足。

---

## 五、阶段 2：协议定义（agent_framework 侧，0.5h）

**agent_framework 不 import rag_mvp。** 依赖方向必须是这样：

```
apps/growth_agent ──> rag_mvp ──┐
        │                        │ 实现
        └──> agent_framework ────┘（只定义协议）
```

`packages/agent_framework/src/agent_framework/knowledge/base.py`（零依赖，只有 dataclass + Protocol）：

```python
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source: str                      # 文档出处（URL / 文件路径）
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    retriever: str                   # 哪一路召回：dense / sparse / hybrid / rerank
                                     # 注意：叫 retriever 不叫 source，避免与 Chunk.source 混


class Retriever(Protocol):
    async def search(self, query: str, k: int) -> list[ScoredChunk]: ...
```

这两个 dataclass 与 rag_mvp 的 `DocumentChunk`/`RetrievalResult` 字段几乎一一对应，映射是纯函数，不需要业务逻辑。

---

## 六、阶段 3：桥接层（1.5h）—— 到这一步就能用

放 `rag_mvp/contrib/agent_framework.py`。**不 import agent_framework**，用结构化兼容（duck typing），这样两边谁都不绑死谁。

```python
"""rag_mvp 与 agent_framework 的桥接层。

放在 contrib 下而不是主包内：不装 agent_framework 时其余功能照常可用。
本模块不 import agent_framework —— 靠结构化兼容（字段名对齐），
避免两个包形成循环依赖。
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from rag_mvp.retrieval.base import Retriever as SyncRetriever


class AgentRetriever:
    """把同步 Retriever 暴露成 agent_framework 需要的 async 协议。"""

    def __init__(self, inner: SyncRetriever, *, max_workers: int = 2) -> None:
        self._inner = inner
        # 专用池而不是默认 executor：bge-m3/cross-encoder 是 CPU 密集，
        # 默认池 min(32, cpu+4) 个线程会让多个请求互相抢 GIL、p95 反而变差。
        # 限到 2 个线程，让排队发生在池里，可预测。
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                       thread_name_prefix="rag-embed")

    async def search(self, query: str, k: int):
        loop = asyncio.get_running_loop()
        hits = await loop.run_in_executor(
            self._pool, functools.partial(self._inner.retrieve, query, k)
        )
        return [_to_scored(h) for h in hits]

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def _to_scored(hit):
    """RetrievalResult -> (Chunk, score, retriever) 三件套。"""
    doc = hit.chunk
    return _Scored(
        chunk=_Chunk(
            doc_id=doc.document_id,
            chunk_id=hit.chunk_id,
            text=hit.text,
            source=doc.source_uri,      # ← 关键：source 取 source_uri，不是 hit.source
            metadata=dict(hit.metadata),
        ),
        score=hit.score,
        retriever=hit.source,           # ← 原来的 source 是检索器名，落到 retriever
    )
```


`_Chunk` / `_Scored` 用 `NamedTuple` 或直接 import agent_framework 的类型（若已装）。推荐后者 + `TYPE_CHECKING`，运行时仍是 duck typing。

### 必须同时做的两件事

**1. `torch.set_num_threads(1)`**  
`embedding/bge_m3.py` 加载模型后加一行。不设的话每个请求线程内部又开 N 个 OMP 线程，2 个池线程 × 8 核 = 16 个计算线程互相争抢，延迟会抖得没法看。

**2. 类型映射只在一处**  
`_to_scored` 是两种数据结构的唯一转换点。**不要**在 rag_mvp 里到处加 agent_framework 字段——那样等于把两个包焊死。

### 验收

```bash
# 同一个 query，同步与异步结果逐条一致（顺序、chunk_id、score）
uv run python -c "
import asyncio
from rag_mvp.contrib.agent_framework import AgentRetriever
from rag_mvp.retrieval.factory import create_retriever, RetrieverConfig
r = create_retriever(RetrieverConfig(mode='hybrid'))
a = AgentRetriever(r)
sync = r.retrieve('MCP 是什么', 5)
async_ = asyncio.run(a.search('MCP 是什么', 5))
assert [h.chunk.chunk_id for h in sync] == [c.chunk_id for c in async_]
print('一致 ok')
"
```

- [ ] 10 路并发时事件循环不被阻塞（每 10ms 一次心跳，最大间隔 < 100ms）
- [ ] `chunk.source` 取到的是 `source_uri`，不是 `"hybrid"`
- [ ] 不装 agent_framework 时 `import rag_mvp.contrib.agent_framework` 仍成功

---

## 七、阶段 4：pgvector（1.5h）—— ✅ 已完成

实现 `vector_store/pgvector.py`，遵守现有 `VectorReader`/`VectorWriter` 语义（`add` 只进缓冲、`save` 持久化、`count` 数已持久化、`upsert` 幂等、`drop` 清空）—— 这五条约定 `base.py` 里已写明，照做就能和 local/milvus 平替。

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (
    chunk_id     text PRIMARY KEY,
    doc_id       text NOT NULL,
    text         text NOT NULL,
    metadata     jsonb NOT NULL,
    embedding    vector(1024),
    content_hash text,                 -- 增量用（阶段 6）
    model_snapshot text                -- ADR-009：换模型能发现
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks (doc_id);
```

`factory.py` 加一个 `store_type == "pgvector"` 分支即可，上层零改动——这正是现有抽象的价值。

**实际落地与计划的三处偏差**（都记在 `rag_mvp/REFACTOR-NOTES.md` 第十节）：

1. 建表放进了 `ops/pgvector/init.sql` 而不是全部依赖应用层 `ensure_schema`：`CREATE EXTENSION` 需要超级用户，应用连库是普通用户，运行时会撞 permission denied。
2. 多了 `delete_document()`：W6 增量索引"更新一个文档"必须能按 `document_id` 成组删除，否则只能全量重建。
3. 端口用 5433 而非 5432，避开本机可能已有的 postgres——连错库的现象是"表不存在"，排查成本很高。

**验收**：单元 23 + 集成 11（真库）+ 端到端 6（真库+框架）；真实语料 425 chunks 上
`local` 与 `pgvector` 的 Top-5 完全一致（同 chunk、分数小数点后 6 位相同），
校验脚本 `rag_mvp/scripts/compare_vector_backends.py` 保留备用。

---

## 八、阶段 5–6：W5/W6 剩余项

沿用评估文档里的 P1/P2 清单，不重复。只补两条**改造后才有意义**的调整：

| 项                   | 调整                                                                                                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `QueryRewriter`     | 现在 `generation/generator.py` 直连 DeepSeek。改造时**整个删掉这个文件**，改写改走 agent_framework 的 `LLMGateway`（rag_mvp 只暴露 `rewrite` 接口，注入一个 `Callable[[str], Awaitable[str]]`）——这样 rag_mvp 不绑死任何模型供应商 |
| `citation_accuracy` | 依赖阶段 0 的 `Citation.document_id` 补字段 + 修掉 `citations_referenced_in_answer` 的 fail-open，否则指标一定偏高                                                                                       |

---

## 九、总预算与取舍

| 阶段         | 工时      | 做完能干什么                   |
| ---------- | ------- | ------------------------ |
| 0 包化       | 1h      | rag_mvp 第一次成为「可安装的库」     |
| 1 依赖分层     | 0.5h    | 框架包仍是零 ML                |
| 2 协议       | 0.5h    | agent_framework 有 RAG 抽象 |
| 3 桥接       | 1.5h    | **W5 D1/D2 可跑真实检索**      |
| 4 pgvector | 1.5h    | 按 W5 原决策落地               |
| 5 W5 余项    | 3h      | Gate 5 全过                |
| 6 W6       | 7h      | Gate 6 全过                |
| **合计**     | **15h** | W5 + W6 双双过门禁            |

对比「不改造、只搬资产」路线（W5 7–9h + W6 9–11h = 16–20h）：**改造反而快 1–5h，而且得到的是可复用的库，不是一次性复制。**

多出来的 3.5h（阶段 0–3）买到三样东西：

1. `import rag_mvp` 在任何项目里都能用（现在不能）
2. 命名污染这个隐性炸弹被拆除
3. W6 之后的 RAG 演进（视觉检索、text2sql）不再被 agent_framework 的节奏绑住

### 时间顺序上的一个建议

阶段 0–3 建议**在 W5 Gate 0 之前做掉**（1 天内），不要边做 W5 边改 rag_mvp。理由：包化会动 73 个文件的 import，如果同时还在 agent_framework 里写新代码，两边的报错会互相掩盖——`ModuleNotFoundError` 到底是包化没改干净，还是新代码写错了，很难分清。先把 rag_mvp 单独改到「测试全绿 + wheel 完整」，再开始接线。

---

## 十、接线方式（已实测可行）

agent_framework 是 uv workspace，`members = ["packages/*", "apps/*"]`。rag_mvp 在 workspace 目录之外，**不能作为 member**，但可以作为 path 依赖。实测：

```toml
# apps/growth_agent/pyproject.toml（只有 app 层依赖它，core 包不依赖）
[project]
dependencies = ["agent-framework", "rag-mvp[pgvector,embedding-local]"]

[tool.uv.sources]
agent-framework = { workspace = true }
rag-mvp = { path = "../../../06_rag/rag_mvp", editable = true }
```

`uv lock` 实测输出：

```
[[package]]
name = "rag-mvp"
version = "0.1.0"
source = { editable = "/tmp/extpkg" }
```

工作区外的绝对/相对路径 editable 依赖，uv 是支持的。要点：

| 要点                | 说明                                                                                                                                                                                                                            |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **依赖方向**          | 只有 `apps/*` 依赖 rag-mvp；`packages/agent_framework` 只定义协议。这样框架包能单独发版，RAG 也能单独演进                                                                                                                                                 |
| **editable 的副作用** | editable 会生成 `.pth` 把 rag-mvp 的 `src/` 加进 sys.path——**包化之后那里只剩 `rag_mvp/`，所以无害**；包化之前则会塞进 11 个顶层名。这也是为什么阶段 0 必须在阶段 3 之前                                                                                                       |
| **相对路径 vs 绝对**    | 用相对路径（`../../../06_rag/rag_mvp`），否则换机器就断                                                                                                                                                                                      |
| **CI**            | **本来就是同一个 git 仓库**：根是 `PycharmProjects/LLMs`（remote `FunCheney/LLMs`），`agent_framework/` 与 `06_rag/rag_mvp/` 都是子目录。所以 `actions/checkout@v4` 一次就把两边都拿到，path 依赖天然可解析，**不存在"两个仓库同步"的成本**。（初稿以为要双 checkout，那是把 `06_rag` 误当成独立仓库了） |

---

## 十一、开工前要你拍板的两件事

1. **向量库**：pgvector（沿用 W5，需阶段 4 的 1.5h）还是 Milvus（已有，跳过阶段 4）？  
   我仍倾向 pgvector：Temporal 已经占着 postgres，加 `vector` 扩展是同库不同 schema；Milvus 要多一份编排和常驻内存，而它的收益（亿级向量）在这个项目里用不上。
2. **embedding 走本地还是 API**？  
   本地 bge-m3：离线可跑、零 API 费用，但要 `[embedding-local]` extra（约 2GB torch），且首请求冷启动会拉高 p95（W5 D3 要测这个）。  
   API embedding：框架彻底零 ML，但要 `.env` 有 key，且离线档必须录制响应（W8 的录制回放机制届时才建，会有一个空窗期）。

第 2 条建议：**先用本地 bge-m3 过 W5/W6（离线档必须能跑），W8 建好录制回放后再评估切 API**。理由是 W5 的 D5 要做单变量 A/B，embedding 换供应商等于换了一个变量，会把「混合检索带来多少提升」这个问题搅浑。

---

拍板后我从阶段 0 开始动，每阶段跑完验收再进下一步。
