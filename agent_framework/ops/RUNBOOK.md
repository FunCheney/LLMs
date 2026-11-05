# 运行手册（RUNBOOK）

> 这份文件回答一个问题：**这台机器上，怎么把项目跑起来、怎么确认它真的跑起来了、起不来时先看哪里。**
>
> 约定：命令后面标注 `[实测]` 表示本机真跑过并看到了预期输出；`[未验证]` 表示没跑通或没跑过，
> 不要当成已知可用。新增服务、改端口、改环境变量名时，**改这份文件**，不要只留在聊天记录里。

---

## 0. 先确认你要跑哪一种

| 形态 | 用途 | 需要什么 | 起多久 |
|---|---|---|---|
| **A. 本机裸进程** | 日常开发、改代码即时生效、跑测试 | 只要 `uv` | 秒级 |
| **B. 容器全套** | 验证部署形态、跑 redis 会话、压测 | Docker + 能拉镜像 | 首次构建分钟级 |
| **C. 只跑检索** | 调检索/评测、看召回结果 | sparse 只要 `uv`；dense/hybrid 要 pgvector 容器 + torch | 模型加载约 18–20s |

三种形态**互不替代**：A 的 venv 里没有 torch，所以 dense/hybrid 跑不了（sparse 可以，见 §4.5）；
B 的镜像同样没装 `local` extra，原因相同——"连数据库"和"跑本地模型"是两个独立的部署决策。

### 端口总表

| 端口 | 谁 | 出现在 | 备注 |
|---|---|---|---|
| 8000 | agent-server | 形态 A / B | 框架服务，`/v1/chat`、`/v1/tasks` |
| 8100 | growth-agent | 形态 A / B | 产品服务，`/v1/draft` 在这里 |
| 6379 | redis | 形态 B | 会话存储；裸进程默认 memory，不用它 |
| 5433 → 容器 5432 | pgvector | 形态 B / C | **宿主是 5433**，写错就连到本机别的 postgres |
| 7233 | temporal | `--profile temporal` | W9 才用 |

查占用：`lsof -nP -iTCP:8000 -iTCP:8100 -sTCP:LISTEN`。

---

## 1. 前置条件

```bash
cd /Users/fanchen/PycharmProjects/LLMs/agent_framework
uv sync                        # workspace 全部成员 + dev 依赖 [实测]
uv run python -c "import agent_framework, growth_agent"   # 两个包可导入
```

密钥放在**仓库根**的 `.env.local`（不是 `ops/.env`，两者用途不同，见 §5）：

```bash
cp .env.example .env.local     # 首次；然后填 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY 等
```

没有密钥也能起服务、也能跑全部离线测试——只有真实模型调用会失败。想完全离线，请求里带 `"model": "fixture"`。

---

## 2. 形态 A：本机裸进程（推荐日常用）

### 2.1 启动

两个**独立终端**各跑一个：

```bash
# 终端 1：框架服务 :8000
uv run agent-server

# 终端 2：产品服务 :8100
uv run growth-agent
```

**必须清掉代理变量**，否则出网请求会走一个可能已经死掉的本地代理端口：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY uv run agent-server
```

> 现象：报 `upstream connect failed: Connection refused (os error 61)`。这句话**不来自本项目代码**，
> 看起来像上游挂了，实际是代理。这一条是本机真踩过的坑。

### 2.2 确认起来了 `[实测]`

```bash
curl -s --noproxy '*' localhost:8000/health
# {"status":"ok","session_backend":"memory",...}

curl -s --noproxy '*' localhost:8100/health
# {"status":"ok","framework_version":"0.1.0"}
```

`--noproxy '*'` 不是可有可无的：curl 会读同一个代理变量，连 localhost 都可能被转出去。

### 2.3 真实调用一次 `[实测]`

```bash
# 离线回放（不花钱、不联网，约 30ms）
curl -s --noproxy '*' -X POST localhost:8100/v1/draft \
  -H 'Content-Type: application/json' \
  -d '{"brief":"测试离线回放链路","model":"fixture"}'

# 真实供应商（实测 qwen3.7-plus，18.0s，990 output tokens）
curl -s --noproxy '*' -X POST localhost:8100/v1/draft \
  -H 'Content-Type: application/json' \
  -d '{"brief":"为小红书写一条咖啡机种草文案，突出静音研磨"}'
```

响应头带 `X-Model-Alias`，说明实际命中了哪个路由。

**`/v1/draft` 不依赖 agent-server**：它在 growth-agent 进程内直接调 `LLMGateway`（`routes_draft.py`），
不是 HTTP 转发。所以只想起产品端时，起一个就够。compose 里的 `depends_on` 是历史遗留的保守配置。

同样重要的是：**`/v1/draft` 目前完全不碰检索**。检索装配层 `growth_agent.rag.build_runtime()`
已经写好但还没有调用方——W6 接引用溯源时才会接进 lifespan。

### 2.4 停止

前台启动的直接 `Ctrl-C`。后台起的话：

```bash
pkill -f "agent_framework.server.app"
pkill -f "growth_agent.app"
```

> 别用 `sh -c 'uv run agent-server &'` 这种写法：子 shell 退出时进程会被一起回收，
> 表现是"刚起就没了、端口没人监听"，很容易误判成启动失败。

---

## 3. 形态 B：容器全套

### 3.1 启动

```bash
cp ops/.env.example ops/.env     # 首次，然后按需改值（不要提交）
make up                          # = cd ops && docker compose up -d --build
make ps                          # 等 healthcheck 变 healthy
curl -s --noproxy '*' localhost:8000/health
```

`make up` 实际会起 **4 个**服务（`docker compose config --services` 的实测输出）：

```
pgvector  redis  agent-server  growth-agent
```

pgvector 也在里面，因为它在 compose 里没挂 profile。只想跑对话链路、不想占一个数据库进程时，
点名起前三者 `[未验证]`：

```bash
cd ops && docker compose up -d redis agent-server growth-agent
```

### 3.2 日志与停止

```bash
make logs                        # 跟踪 agent-server，grep first_token_ms / trace_id 用
cd ops && docker compose logs -f growth-agent
make down                        # 停容器，保留数据卷
```

### 3.3 镜像构建的两个坑（都已在文件里修掉，动 Dockerfile 时别改回去）

1. **`rag-mvp` 在构建上下文之外**。它是 `06_rag/rag_mvp` 的路径依赖，而 `build.context` 是 `agent_framework/`。
   不额外挂上下文，`uv sync` 会报：
   ```
   Failed to determine installation plan
     Caused by: Distribution not found at: .../06_rag/rag_mvp
   ```
   这行字看着像网络或缓存问题，实际是文件根本没进镜像。修法：compose 里 `additional_contexts: ragmvp`。
   **相对路径以 `ops/` 为基准**（不是以 `build.context` 为基准），写 `../../06_rag/rag_mvp`。

2. **runtime 阶段必须拷 `/06_rag`**。`.venv` 里是指向 `/06_rag/rag_mvp` 的 editable 指针；
   只从 builder 拷 `.venv + packages + apps` 的话，构建能过、启动时 `import rag_mvp` 直接失败。
   Dockerfile 里 `COPY --from=builder /06_rag /06_rag` 就是为这一步。

`06_rag/rag_mvp/.dockerignore` 也存在：不加的话每次构建要把 1.4G 的 `.venv` 传上去。

### 3.4 当前状态 `[未验证]`

本机跑到「buildx 成功加载 `ragmvp` 上下文 + 模拟目录下 `uv sync --frozen` 装成功」，
**完整镜像没构建完**：拉 `python:3.12-slim` 时 `docker.io` 返回 502（`ghcr.io` 的 uv 镜像正常）。
这是镜像仓库访问问题，不是代码问题；网络正常时应能过。

沙箱/受限环境里构建还会撞 buildx 写 `~/.docker` 的权限限制，把它指到工作区内：

```bash
mkdir -p .tmp/buildx && BUILDX_CONFIG=$PWD/.tmp/buildx docker compose -f ops/docker-compose.yml build agent-server
```

---

## 4. 形态 C：只跑检索

框架的 venv **没有 torch**（`uv run python -c "import torch"` → `ModuleNotFoundError`），
但 **sparse（BM25）不需要模型**，所以在框架侧也能直接跑通整条流水线；
dense/hybrid 要 embedding 与向量库，走 `rag_mvp` 自己的环境（它有 torch）。

顺序建议：先 §4.1 起库 → §4.2 建索引 → §4.3 验等价 → 再按需 §4.4/§4.5 查询。
只跑 sparse 的话可以跳过 4.1–4.3。

### 4.1 起向量库 `[实测]`

```bash
docker compose -f ops/docker-compose.yml up -d pgvector
docker compose -f ops/docker-compose.yml ps pgvector      # 等 healthy
```

端口映射 **5433 → 容器内 5432**。刻意避开本机 5432，否则会和本机已有的 postgres 连错库，
症状是"表不存在"，排查成本很高。

### 4.2 建索引 `[实测，425 条约 47s]`

```bash
cd /Users/fanchen/PycharmProjects/LLMs/06_rag/rag_mvp
uv run --extra pgvector python scripts/build_vector_index.py \
  --store-type pgvector --pgvector-dsn "postgresql://rag:rag@localhost:5433/rag"
```

### 4.3 查一条 / 对比两个后端 `[实测]`

```bash
# 同一 query 分别走本地 pickle 与 pgvector，比 chunk 与分数
cd /Users/fanchen/PycharmProjects/LLMs/06_rag/rag_mvp
PGVECTOR_DSN="postgresql://rag:rag@localhost:5433/rag" \
uv run --extra pgvector python scripts/compare_vector_backends.py --query "MCP 是什么" --top-k 5
```

最近一次结果：5/5 同 chunk、分数小数点后 6 位完全相同。**换后端后先跑这条**，
否则"指标变好"分不清是改动起效还是后端差异。

### 4.4 不用模型也能查（sparse）`[实测]`

```bash
cd /Users/fanchen/PycharmProjects/LLMs/06_rag/rag_mvp && uv run python -c "
from rag_mvp.retrieval.spare_retriever import SparseRetriever
r = SparseRetriever(index_path='data/index/bm25.pkl')
for h in r.retrieve('MCP 是什么', top_k=3): print(h.chunk_id, round(h.score, 3), h.source)
"
```

> 文件名是 `spare_retriever.py`（拼写如此，历史原因，未重命名）。照抄上面的 import 路径，别自作主张改。

实测输出（注意第 1 条命中 p111，这正是 W5 记录的 BM25 对疑问词的系统偏差）：

```
claude-code-0001-p111-p111-c233 6.068 sparse
claude-code-0001-p126-p126-c271 5.901 sparse
claude-code-0001-p085-p085-c177 5.663 sparse
```

### 4.5 走完整流水线（含预算、降级、后面接 ACL/引用的位置）

**sparse 模式在框架 venv 里就能跑**，不需要 torch `[实测]`：

```bash
cd /Users/fanchen/PycharmProjects/LLMs/agent_framework
RETRIEVAL_MODE=sparse VECTOR_STORE=local uv run python -c "
import asyncio
from growth_agent.rag import build_runtime
rt = build_runtime()
out = asyncio.run(rt.pipeline.run('MCP 是什么', top_k=3))
for s in out.chunks: print(s.chunk.chunk_id, round(s.score, 3), s.retriever)
print('degraded =', out.diagnostics.degraded if out.diagnostics else None)
rt.close()
"
```

实测输出 `degraded = ()`，即无降级；三跳 `AsyncRetriever → SparseRetriever → BM25` 正常。

**dense / hybrid 需要 torch** `[未验证]`：

```bash
uv sync --package growth-agent --extra local      # 约 2GB，可先加 --dry-run 看清单
RETRIEVAL_MODE=dense VECTOR_STORE=pgvector \
PGVECTOR_DSN="postgresql://rag:rag@localhost:5433/rag" \
uv run python -c "
from growth_agent.rag import build_runtime
build_runtime().close()
"
```

首次会加载 bge-m3，约 18–20 秒。装配慢是刻意的：放进请求路径会让第一个请求吃掉整个加载时间，
p95 被一次冷启动污染，之后所有延迟结论都不可信。

---

## 5. 配置文件：三份 env 各管一段

| 文件 | 谁读它 | 什么时候生效 | 注意 |
|---|---|---|---|
| `.env.local`（仓库根） | `Settings`（`env_file=".env.local"`） | `uv run ...` 裸进程 | 路径相对**当前工作目录**，所以要在 `agent_framework/` 下启动 |
| `ops/.env` | compose 的 `env_file` | 容器 | 值要用**服务名**：`redis://redis:6379/0`、`pgvector:5432` |
| `.env.example` / `ops/.env.example` | 人 | 模板 | `cp` 之后填值；`.env*` 已在 `.gitignore` 里 |

**两套配置不会互相污染**，但也意味着：改了 `ops/.env` 对裸进程无效，反之亦然。

### 变量命名铁律

`Settings` 没有 `env_prefix`，**字段名大写就是环境变量名**。写错不会报错，只会静默用默认值——
表现是"明明配了 pgvector 还在读 pickle"。常用的几个：

| 变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `RETRIEVAL_MODE` | sparse / dense / hybrid | `hybrid` | 默认值需要 torch，裸机不装会 `ImportError` 并给出提示 |
| `VECTOR_STORE` | local / pgvector / milvus | `local` | 只影响装配 |
| `PGVECTOR_DSN` | 连接串 | 空 | 本机 `localhost:5433`；容器 `pgvector:5432` |
| `RETRIEVAL_TOP_K` | int | 8 | |
| `RETRIEVAL_BUDGET_MS` | float | 800 | 整链路预算，不是单段超时 |
| `RETRIEVAL_RERANK` | true / false | false | 开精排必须同时调大预算，否则每次都在降级 |
| `RAG_ROOT` | 路径 | 自动推导 | 装成 wheel 后推导失效，必须显式指定 |
| `SESSION_BACKEND` | memory / redis | memory | 裸进程用 memory，别为跑通对话去起 redis |
| `LOG_LEVEL` | INFO / DEBUG | INFO | |

---

## 6. 排障速查

| 症状 | 真因 | 处置 |
|---|---|---|
| `upstream connect failed: Connection refused (os error 61)` | shell 里有指向死端口的 `HTTP_PROXY` | `env -u HTTP_PROXY -u HTTPS_PROXY uv run ...` |
| curl 连 localhost 也超时 / 返回奇怪页面 | curl 也走代理 | 加 `--noproxy '*'` |
| 服务刚起就没了，端口没人监听 | 用 `&` 起在子 shell 里，随 shell 退出被回收 | 独立终端，或 `nohup` |
| `No module named 'torch'` | 框架 venv 没装 `local` extra | `uv sync --package growth-agent --extra local`；或 `RETRIEVAL_MODE=sparse` |
| `Distribution not found at .../06_rag/rag_mvp` | 容器构建没带 ragmvp 上下文 | 见 §3.3，检查 `additional_contexts` 的相对层级 |
| 容器里 `No module named 'rag_mvp'` | runtime 阶段没拷 `/06_rag` | 见 §3.3 |
| `connection refused ... :5433` | pgvector 容器没起 | `docker compose -f ops/docker-compose.yml up -d pgvector` |
| `relation "rag_chunks" does not exist` | `init.sql` 只在**空数据卷**时执行 | `down -v` 重建，或手工 `psql -f ops/pgvector/init.sql` |
| 连上了但表在别的地方 | 连到了本机 5432 的另一个 postgres | 连接串端口写 5433；`select current_database()` 确认 |
| 每次都 `budget_exhausted` 降级 | 精排 ~2.5s 装不进 800ms 预算 | 调大 `RETRIEVAL_BUDGET_MS`，或关掉 `RETRIEVAL_RERANK` |
| `docker compose build` 报权限/`~/.docker` | 沙箱限制 buildx 写家目录 | `BUILDX_CONFIG=$PWD/.tmp/buildx` |
| 拉基础镜像 502 | 镜像仓库网络问题 | 换网络重试；`ghcr.io` 那个 uv 镜像通常正常 |
| `pytest` 报建不了临时目录 | 受限环境 | `--basetemp=.pytest_tmp` |

---

## 7. 数据与清理

| 东西 | 位置 | 删了会怎样 |
|---|---|---|
| 语料 chunk | `06_rag/rag_mvp/data/processed/chunks.jsonl`（425 条） | 要重新切分 |
| 本地向量索引 | `06_rag/rag_mvp/data/index/local.pkl` | 重新 `build_vector_index.py` |
| BM25 索引 | `06_rag/rag_mvp/data/index/bm25.pkl` | 同上 |
| pgvector 数据 | docker 卷 `agent-platform_pgvector-data`（下划线） | 425 条向量丢失，需重建索引 |
| redis 会话 | docker 卷 `agent-platform_redis-data` | 会话全丢，无影响 |

```bash
make down                                              # 停容器，数据保留
docker compose -f ops/docker-compose.yml down -v       # 连数据卷一起删（会丢向量）
make clean                                             # 只删缓存，不碰 .venv 与 .env*
```

---

## 8. 尚未验证的事（不要当成可用）

- `make up` 的完整镜像构建（卡在 docker.io 502）
- `docker compose --profile temporal up -d`（Temporal，W9 才用）
- 容器内的 dense/hybrid 检索（镜像 `uv sync --frozen --no-dev`，**没带任何 extra**）
- 容器内连 pgvector：镜像里同样没有 psycopg（`pgvector` extra 未装），
  所以容器里把 `VECTOR_STORE` 改成 `pgvector` 会 `ImportError`。要支持得给 Dockerfile 加
  `--extra pgvector`（约 3MB），**这是 W6 接检索前必须先改的一处**
- `ops/.env` 里 `AGENT_SERVER_URL` 目前没有任何代码读取（`/v1/draft` 是进程内调用），
  属于 W1 遗留的保守配置，接服务间 HTTP 调用时再确认是否要留

---

## 9. 一分钟自检（改完代码 / 换机器后跑这个）

不需要起服务的部分：

```bash
cd /Users/fanchen/PycharmProjects/LLMs/agent_framework
uv run ruff check . && uv run mypy                    # 静态
uv run pytest --basetemp=.pytest_tmp                  # 离线全量（零网络）
make smoke                                            # 10 条 smoke eval（fixture）
RETRIEVAL_MODE=sparse uv run python -c \
  "import asyncio; from growth_agent.rag import build_runtime; \
   rt = build_runtime(); print([s.chunk.chunk_id for s in asyncio.run(rt.pipeline.run('MCP 是什么', top_k=3)).chunks]); rt.close()"
# ['claude-code-0001-p111-p111-c233', 'claude-code-0001-p126-p126-c271', 'claude-code-0001-p085-p085-c177']
```

需要起服务的部分，按 §2.1 开两个终端后再：

```bash
curl -s --noproxy '*' localhost:8000/health
curl -s --noproxy '*' localhost:8100/health
```

全绿再动手做别的。**任何一条红着往下做，后面所有"指标变好"都不可信。**
