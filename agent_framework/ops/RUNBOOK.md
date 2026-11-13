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
| 80 / 443 | caddy | 生产档（`-f docker-compose.prod.yml`） | 唯一对公网开放的入口，其余端口绑 127.0.0.1 |

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

前台启动的 `Ctrl-C` 即可。后台起的、或者找不到终端时，**按端口找 PID 再 kill**：

```bash
# 看谁占着
lsof -nP -iTCP:8000 -iTCP:8100 -sTCP:LISTEN

# 停框架服务（8000）
kill $(lsof -nP -iTCP:8000 -sTCP:LISTEN -t)
# 停产品服务（8100）
kill $(lsof -nP -iTCP:8100 -sTCP:LISTEN -t)

# 确认停干净：无输出 = 已停
lsof -nP -iTCP:8000 -iTCP:8100 -sTCP:LISTEN
```

**不要写 `pkill -f "agent_framework.server.app"`**：这两个服务的入口是 console script
（`agent-server` / `growth-agent`），进程命令行是 `.venv/bin/agent-server`，**不含模块名**。
这条命令实测匹配到 **0 个进程**——它会静默成功，让人以为已经停了，实际端口还占着。

`pkill -f "agent-server"` 能匹配到，但它同时会匹配命令行里含这个字符串的**其他进程**，
包括正在执行这条命令的 shell 本身。按 PID 停是唯一不会误伤的方式。

> 也别用 `sh -c 'uv run agent-server &'` 这类写法起服务：子 shell 退出时进程会被一起回收，
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
| `RETRIEVAL_MODE` | sparse / dense / hybrid | `dense` | dense 需要 torch；容器镜像里没有，所以要显式写 `sparse`。hybrid 实测负收益，别当折中开 |
| `VECTOR_STORE` | local / pgvector / milvus | `local` | 只影响装配 |
| `PGVECTOR_DSN` | 连接串 | 空 | 本机 `localhost:5433`；容器 `pgvector:5432` |
| `RETRIEVAL_TOP_K` | int | 8 | |
| `RETRIEVAL_BUDGET_MS` | float | 800 | 整链路预算，不是单段超时 |
| `RETRIEVAL_RERANK` | true / false | `false` | 能力保留、默认关。要开见下方「开精排」 |
| `RETRIEVAL_DEVICE` | cpu / mps / cuda | 自动 | **macOS 开精排必须 `cpu`** |
| `RAG_ROOT` | 路径 | 自动推导 | 装成 wheel 后推导失效，必须显式指定 |
| `SESSION_BACKEND` | memory / redis | memory | 裸进程用 memory，别为跑通对话去起 redis |
| `LOG_LEVEL` | INFO / DEBUG | INFO | |

### 开精排：三个值缺一不可

精排（cross-encoder）保留成了真开关，默认关。但**只改 `RETRIEVAL_RERANK=true` 是没用的**——
精排超时取 `min(rerank_timeout, 剩余预算)`，预算 800ms 时剩余约 500ms，而 20 条精排要 4.7s，
于是每次都在 `asyncio.wait_for` 里被掐掉、静默降级回召回顺序。现象是"开了，指标一模一样"。

```bash
export RETRIEVAL_RERANK=true
export RETRIEVAL_BUDGET_MS=8500    # ≥ rerank_timeout(8000) + 500ms 召回余量
export RETRIEVAL_DEVICE=cpu        # macOS 必设；不设会让进程直接退出（MPS 多线程断言）
```

启动时会看到：

```
retrieval.rerank_misconfigured: 检索预算 800ms 不够精排用：……
```

配置自洽时这行不会出现。运行时每次降级都会打 `retrieval.degraded stages=rerank_degraded`，
用它判断"精排到底有没有真跑"。

**值不值得开**：这批语料上实测是负收益（R@1 .7857→.7143、MRR .8628→.8146、p95 +5s，
两套口径结论一致），所以默认关。换语料或换模型后要重新跑一次 A/B 再决定——
命令见 `docs/w5/rag-ab-report.md`。

---

## 6. 排障速查

| 症状 | 真因 | 处置 |
|---|---|---|
| `upstream connect failed: Connection refused (os error 61)` | shell 里有指向死端口的 `HTTP_PROXY` | `env -u HTTP_PROXY -u HTTPS_PROXY uv run ...` |
| curl 连 localhost 也超时 / 返回奇怪页面 | curl 也走代理 | 加 `--noproxy '*'` |
| 服务刚起就没了，端口没人监听 | 用 `&` 起在子 shell 里，随 shell 退出被回收 | 独立终端，或 `nohup` |
| `pkill -f "agent_framework.server.app"` 执行了，端口还占着 | 入口是 console script，命令行里没有模块名；这条 pkill 匹配 0 个进程 | `kill $(lsof -nP -iTCP:8000 -sTCP:LISTEN -t)` |
| `docker compose down` 之后 `5433` 仍在监听 | 是引擎在代理端口映射，不是漏停的进程 | 看 `docker ps` 是否还有 `agent-platform-pgvector-1` |
| `No module named 'torch'` | 框架 venv 没装 `local` extra | `uv sync --package growth-agent --extra local`；或 `RETRIEVAL_MODE=sparse` |
| `Distribution not found at .../06_rag/rag_mvp` | 容器构建没带 ragmvp 上下文 | 见 §3.3，检查 `additional_contexts` 的相对层级 |
| 容器里 `No module named 'rag_mvp'` | runtime 阶段没拷 `/06_rag` | 见 §3.3 |
| `connection refused ... :5433` | pgvector 容器没起 | `docker compose -f ops/docker-compose.yml up -d pgvector` |
| `relation "rag_chunks" does not exist` | `init.sql` 只在**空数据卷**时执行 | `down -v` 重建，或手工 `psql -f ops/pgvector/init.sql` |
| 连上了但表在别的地方 | 连到了本机 5432 的另一个 postgres | 连接串端口写 5433；`select current_database()` 确认 |
| 精排"开了"但指标和没开一样 | 预算或超时不够，每次在 `wait_for` 里被掐断 | 启动日志搜 `rerank_misconfigured`，按它给出的值改；或看 `retrieval.degraded stages=rerank_degraded` |
| `budget_exhausted` 频繁出现 | 整链路预算装不下精排（20 条要 4.7s） | 见 §5「开精排：三个值缺一不可」 |
| Mac 上开精排后进程直接退出 | 没设 `RETRIEVAL_DEVICE=cpu`，MPS 多线程断言 | `export RETRIEVAL_DEVICE=cpu` |
| `docker compose build` 报权限/`~/.docker` | 沙箱限制 buildx 写家目录 | `BUILDX_CONFIG=$PWD/.tmp/buildx` |
| 拉基础镜像 502 | 镜像仓库网络问题 | 换网络重试；`ghcr.io` 那个 uv 镜像通常正常 |
| `pytest` 报建不了临时目录 | 受限环境 | `--basetemp=.pytest_tmp` |

---

## 7. 停止与清理

按"影响范围"分四档，从轻到重。**先看第 1 档够不够，别一上来就 `down -v`。**

### 7.1 四档停法

| 档 | 命令 | 停什么 | 数据 | 常用场景 |
|---|---|---|---|---|
| 1 | 见 §2.4 | 本机两个服务进程 | 不涉及 | 收工、改完代码重起 |
| 2 | `make down` | 全部容器（保卷） | **保留** | 不用容器了，但想保住向量 |
| 3 | `docker compose -f ops/docker-compose.yml down -v` | 容器 + 数据卷 | **删** | 重建索引、清干净环境 |
| 4 | 退出 OrbStack / Docker Desktop | 整个引擎 | 保留 | 释放内存与 CPU |

**第 1 档：停本机进程**

一条命令停两个，端口本就空闲时也不会报错：

```bash
for p in 8000 8100; do
  pid=$(lsof -nP -iTCP:$p -sTCP:LISTEN -t 2>/dev/null)
  if [ -n "$pid" ]; then kill $pid && echo "已停 $p (pid $pid)"; else echo "$p 本就空闲"; fi
done

lsof -nP -iTCP:8000 -iTCP:8100 -sTCP:LISTEN   # 无输出 = 已停干净
```

只停一个的话：

```bash
kill $(lsof -nP -iTCP:8000 -sTCP:LISTEN -t)   # 框架
kill $(lsof -nP -iTCP:8100 -sTCP:LISTEN -t)   # 产品
```

**第 2 档：停容器、留数据** `[实测]`

```bash
make down                                        # 移除容器与网络，保留数据卷
docker compose -f ops/docker-compose.yml stop    # 只停不删，下次 start 更快
```

两条都实测过：

- `stop → start` 后 `select count(*) from rag_chunks` 仍是 **425**；
- `down → up` 后同样仍是 **425**（容器和网络重建了，卷没动）。

停完之后应该是这个状态：

```bash
docker ps -a                                     # 空
lsof -nP -iTCP:8000 -iTCP:8100 -iTCP:5433 -sTCP:LISTEN   # 空
docker volume ls                                 # agent-platform_pgvector-data 还在
```

要重开会话继续做，`docker compose -f ops/docker-compose.yml up -d pgvector` 约 6 秒到 healthy。
想临时腾资源又不想等重建，用 `stop`；想彻底收干净，用 `make down`。

**第 3 档：连数据一起删**

```bash
docker compose -f ops/docker-compose.yml down -v
```

删掉 `agent-platform_pgvector-data` 与 `agent-platform_redis-data`。**向量没了，要按 §4.2 重建**（425 条约 47s）。
只有这两种情况该用它：`init.sql` 改了需要重跑，或者索引数据已经脏了想从零重建。

**第 4 档：退掉整个 Docker 引擎**

macOS 上先用 `docker ps` 确认没有别的容器在跑，再从菜单栏退出 OrbStack / Docker Desktop。
引擎一停，5433 也一并释放——有时"端口还被占着"就是它，不是我们的进程。

### 7.2 各停法之后怎么确认

```bash
lsof -nP -iTCP:8000 -iTCP:8100 -sTCP:LISTEN     # 服务：无输出
lsof -nP -iTCP:5433 -sTCP:LISTEN                # 向量库：无输出（或只剩 OrbStack）
docker ps                                       # 容器：无相关条目
```

**`5433` 上显示 `OrbStack` 是正常的**：那是容器的端口映射在代理，不是漏停的进程。
判断标准是 `docker ps` 里还有没有 `agent-platform-pgvector-1`。

### 7.3 数据在哪

| 东西 | 位置 | 删了会怎样 |
|---|---|---|
| 语料 chunk | `06_rag/rag_mvp/data/processed/chunks.jsonl`（425 条） | 要重新切分 |
| 本地向量索引 | `06_rag/rag_mvp/data/index/local.pkl` | 重新 `build_vector_index.py` |
| BM25 索引 | `06_rag/rag_mvp/data/index/bm25.pkl` | 同上 |
| pgvector 数据 | docker 卷 `agent-platform_pgvector-data`（下划线） | 425 条向量丢失，需重建索引 |
| redis 会话 | docker 卷 `agent-platform_redis-data` | 会话全丢，无影响 |

```bash
make clean          # 只删 .pytest_tmp / 各工具缓存，不碰 .venv 与 .env*
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

### W10 补记：上线相关（2026-11-13）

- **生产档 compose 没跑过**：`docker-compose.prod.yml` 的 `!override` 语法需要
  Compose ≥ 2.24.4，本机有没有这个版本我没验证；`docker compose config` 也没跑过
- **caddy 没起过**：`Caddyfile` 的语法、证书申请、`SITE_ADDRESS` 缺失时的拒绝行为
  全部**未经运行验证**，只按文档写的
- **公网可访问没验证**：没有域名、没有服务器。§10.3 那次实测是本机 127.0.0.1
- **真实厂商 key 下的 `/v1/article` 全链路**（四次模型调用 + 三次改写）没跑过；
  端点级端到端用的是脚本化网关（`apps/growth_agent/tests/test_article_route.py`）
- **两位真实用户**：还没有。ROADEMAP §3.3 要求的"朋友开始用"这一环未发生，
  W11 要的"真实反馈"因此**目前没有任何输入**

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

---

## 10. 上线（W10）：从"能起"到"别人能用"

> 这一节与前面九节的区别：前面回答"怎么跑起来"，这一节回答"**别人怎么访问到它，
> 以及怎么证明它现在是对外可用的那个东西**"。

### 10.1 三档环境，别混用

| 档 | 命令 | 谁在访问 | 鉴权 |
|---|---|---|---|
| 开发 | `uv run uvicorn growth_agent.app:app --port 8100` | 只有你 | `AUTH_MODE=off`（默认） |
| 预演 | `cd ops && docker compose up -d --build` | 内网/你自己 | `.env` 里的 `AUTH_MODE` |
| 生产 | `cd ops && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build` | 公网（经 caddy） | **强制 `key`**（写死在 prod 覆盖层） |

生产档的两个硬约束：

1. **只有 443 对公网开**。`docker-compose.prod.yml` 把 8000/8100/6379/5433 全部
   绑到 `127.0.0.1`。基础文件里那些 `"8000:8000"` 是给开发者的，生产必须覆盖掉。
2. **框架服务（8000）不对外**。它没有我方鉴权——`AUTH_MODE` 只挂在 growth-agent 上。
   把 8000 映射到公网等于开了个后门，而它在 `docker compose ps` 里看起来和 8100 一样正常。

### 10.2 上线前必须做的三件事

```bash
# 1) 配密钥（`.env` 不要进 git，`.gitignore` 已含）
cd ops && cp .env.example .env
# 编辑 .env：
#   AUTH_MODE=key
#   API_KEYS=sk-live-xxxx=<你的名字>:admin
#   SITE_ADDRESS=growth.example.com      # 生产档必填，缺了 caddy 拒绝启动
#   ACME_EMAIL=you@example.com

# 2) 本地预演（不花证书、不暴露公网）
docker compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null   # 语法
docker compose up -d --build
./ops/deploy_check.sh http://localhost:8100 sk-live-xxxx            # 5 项全绿

# 3) 起生产档并复检（证书申请需要域名已解析到这台机器）
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
./ops/deploy_check.sh https://growth.example.com sk-live-xxxx
```

`deploy_check.sh` 五项，任何一项红都不要继续：

| # | 断言 | 失败的含义 |
|---|---|---|
| 1 | `/health` 200 | 进程没起来 / 端口没通 |
| 2 | 无 key → **401** | **鉴权没开**（比内容问题严重得多） |
| 3 | 错 key → 401 + `AUTH_INVALID` | 校验逻辑坏了，或所有 key 都被当成有效 |
| 4 | 正确 key → 200 | 鉴权通过了，但上游/超时有问题（`502/504` 会单独标出来） |
| 5 | 响应头有 `X-Request-Id` | 多实例时无法串联同一次调用 |

第 2 项是这张表的核心：**"没设防"和"设了防但没人试"在监控上长得一样**。

### 10.3 一次真实的首次上线记录 `[实测：本机进程，非公网]`

2026-11-13 在本机用生产同构的参数起进程验证：

```bash
AUTH_MODE=key API_KEYS="sk-local-test=alice:admin" \
RETRIEVAL_MODE=sparse SESSION_BACKEND=memory \
uvicorn growth_agent.app:app --host 127.0.0.1 --port 8199
```

```
$ ./ops/deploy_check.sh http://localhost:8199 sk-local-test
  ✓ /health 200        {"status":"ok","service":"growth-agent",...,"rag":"cold"}
  ✓ 无 key → 401        code=AUTH_REQUIRED
  ✓ 错 key → 401        code=AUTH_INVALID
  ✓ 带 key → 200        request_id=req_826f6bb204a3
  ✓ X-Request-Id 存在
通过 5 项，失败 0 项
```

401 的响应体（`request_id` 有值，说明 RequestID 中间件在最外层）：

```json
{"code":"AUTH_REQUIRED","message":"缺少凭据：请在 X-API-Key 或 Authorization 中提供",
 "request_id":"req_71f1570dc97b"}
```

**没做过的部分**（不要当成已可用）：镜像构建、caddy 证书申请、公网 DNS、
真实厂商 key 下的全链路（`/v1/article` 四次模型调用 + 三次改写）。

### 10.4 回滚

镜像 tag 是唯一可靠的锚点。**上线前先记下当前 tag**：

```bash
docker compose ps --format '{{.Service}} {{.Image}}' | tee /tmp/before-release.txt
```

回滚 = 把 `.env` 里的 `IMAGE_TAG` 调回上一个 + `up -d`。不重建镜像：
重建会带着当前（可能有问题的）源码，那不是回滚，是"重启了另一个 bug"。

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-build
```

回滚后**必须复跑 `deploy_check.sh`**：容器起来了不等于新代码没生效
（镜像 tag 变了但 compose 用了缓存的情况真实存在）。

---

## 11. 密钥轮换

两套密钥，轮换方式完全不同——混了会停机。

### 11.1 我方 API key（调用方的 key）

`API_KEYS` 支持多个条目，所以可以**双写过渡**，不需要停机：

```bash
# 1) 同时放新旧两个 key（同一个人两个 key，subject 相同）
API_KEYS="sk-live-old=alice:admin, sk-live-new=alice:admin"
docker compose up -d --no-deps --force-recreate growth-agent

# 2) 通知调用方换（把新 key 给对方）
# 3) 观察一周：确认 old 的请求量归零
#    日志里每一次 401 都带 mask 后的 key，按 mask 前缀聚合就能看出谁还在用旧的
grep '我方鉴权失败' app.log | sed 's/.*presented=//' | sort | uniq -c

# 4) 从 API_KEYS 里删掉旧 key，再 recreate 一次
```

**不要**直接删旧 key：那一次 recreate 会把所有还拿着旧 key 的调用方同时打成 401，
而他们的第一反应是"服务挂了"。

### 11.2 上游厂商 key

```bash
# 1) 新 key 写进 .env（旧的不动）
# 2) 重启（网关在启动时读 Settings，运行期不重读）
docker compose up -d --no-deps --force-recreate growth-agent
# 3) 确认用的是新的：打一次真实调用，看日志里的 provider 与计费是否正常
# 4) 观察 30 分钟，确认 UPSTREAM_AUTH 为 0
# 5) 去厂商控制台把旧 key 作废
```

顺序不能反：先作废旧 key 再换新 key，中间那段时间所有请求都是 `502 UPSTREAM_AUTH`，
而值班看到 502 的第一反应是"上游挂了"。

---

## 12. W10 新增的排障入口

### 12.1 状态码先分流，再看 code

| 你看到 | 是谁的问题 | 先查 |
|---|---|---|
| **401** `AUTH_REQUIRED` | 调用方没带凭据 | 调用方的 header |
| **401** `AUTH_INVALID` | 凭据不对 | `API_KEYS` 里有没有这个 key |
| **502** `UPSTREAM_AUTH` | **厂商 key 失效/欠费** | 厂商控制台，不是我们的 key 表 |
| **400** `CONTENT_BLOCKED` | 输入被护栏拦了 | `violations` 字段；`docs/w10/threat-model.md` |
| **422** `CONTENT_BLOCKED` | 产物不合格（缺引用/禁用词） | `violations`；多半是 prompt 或证据质量问题 |
| **422** `CONTENT_UNQUALIFIED` | 文章阶段失败（无证据/引用体检不过） | `stage` 字段：`evidence` 说明检索没回来 |
| **202** `APPROVAL_REQUIRED` | 动作等人批 | `approval_id`；用 `/v1/approvals` 列表 |

**401 与 502 的区别是这套系统里最容易搞混的一处**，也是 ADR-005 的全部内容：
401 是"你不对"，502 是"我不对"。分错了，值班会去查错的那一端。

### 12.2 `CONTENT_BLOCKED` 突然变多

两种原因，处理方式相反：

1. **真的在被扫**：`violations` 里出现多个不同 `kind`，来源 IP 集中。
   → 保留拦截；`run_redteam.py` 跑一遍确认规则没被改坏。
2. **误报变多**：集中在某一类正常业务输入上（例如用户在讨论"系统提示词"这个主题）。
   → 修规则，**不是**关护栏。关护栏的成本在 `threat-model.md` §5 写了。

判断依据不是感觉，是 `RequestLog` 里的 `violations` 分布：同一条规则占比 > 80%
且集中在少数 principal 上，多半是误报；`kind` 分散则是被扫。

### 12.3 审批卡住（`approval_backlog` 告警）

`behavior.py` 的规则是"10 分钟内 3 条待审没人管"。这**不是安全事件**，是流程卡住：

```bash
curl -s --noproxy '*' localhost:8100/v1/approvals -H "X-API-Key: $KEY" | python -m json.tool
```

待审请求 15 分钟过期且**默认拒绝**。所以"没人管"的结果不是"事后补批"，
而是"这个动作没发生"——这一点在告诉用户"稍等一下"之前要先想清楚。
