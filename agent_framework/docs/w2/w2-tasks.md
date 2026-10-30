# W2 执行清单（运行底座 + 可观测基线 + 评测冒烟集 · P0）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——**架构、骨架、验收标准由我给，代码由你写，我不代写**。每完成一个 Gate 把仓库状态给我，我逐条 review。
> 契约真相源仍是 `docs/architecture.md` 与其中的 ADR-001~012；本文件只讲"做什么、怎么搭骨架、怎么自查、何时交给我"。

## 承接面：W1 已交付（你站在这些上面，不要重写）

| 已有 | 位置 | W2 的接缝 |
|---|---|---|
| `LLMGateway.precheck/chat/stream/aclose`、`ChatCall`/`ChatOutcome` | `llm/client.py` | 会话层只负责"攒消息"，调用仍走 gateway；**不在会话里重试** |
| `ChatCall(messages, task, tools, model, opts)` | 同上 | 队列/工作流最终都还原成一次 `ChatCall` |
| `RequestLog`（18 字段，含 `first_token_ms`/`usage_estimated`/`retry_count`）+ `configure_logging`/`get_request_logger` | `observability.py` | 本周只**新增** `trace_id/span_id`，不改已有字段含义 |
| `error_code_for`/`http_status_for`/`install_error_handlers`/`error_body` | `llm/errors.py` + `server/errors.py` | 背压/队列满的新异常必须走同一张映射表 |
| `RequestIDMiddleware`（纯 ASGI，contextvar `request_id_var`） | `server/middleware.py` | trace_id 用同样的 contextvar 手法贯穿 |
| `models.yaml`（providers/models/routes/tasks）+ `build_gateway` | `config.py` + `bootstrap.py` | 并发限额按 **alias/provider** 读，别写死数字 |
| `RecordingTransport`、`LogCapture`、mock 夹具（24 个）、`bench_baseline.py` | `evals/scripts/` + `tests/fixtures/` | smoke eval 直接复用这两套离线能力，不另造 runner |
| 端到端 `/v1/chat`、`/v1/draft`；278 passed / 16 skipped | `server/` + `apps/` | 本周所有压测对这两个端点做，别新开端口 |

> W2 结束时（ROADEMAP 检查点）：**100 并发错误率 < 1%**；能从一条请求追到每次 LLM 调用耗时；capstone 可一键 `docker compose up`。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接的 W1 符号 | 推进的 DoD 维度 |
|---|---|---|---|
| D1 | `session/`（SessionStore）+ `workflow/`（分层定稿） | `Message`、`LLMGateway.stream` 直连边界 | 可靠性（断点续跑）、可观测 |
| D2 | `resilience/`（并发限流）+ `queue/`（长任务队列） | ADR-003、`error_code_for` | 可用性（注入故障不崩）、延迟 |
| D3 | `observability.py` 增 `trace_id/span_id` + `tracing.py`（exporter 协议） | `RequestLog`、`configure_logging` | 可观测（完整 trace） |
| D4 | `ops/`（Dockerfile/compose）+ `evals/scripts/locustfile.py` | `create_app`、`/health`、`/v1/chat` | 可用性、延迟（p95） |
| D5 | `evals/golden/smoke_v1.*` + `run_smoke.py` + CI | `LogCapture`/`RecordingTransport`/mock 夹具 | 质量（golden set）、工程（CI 门禁） |

**唯一主线**：把 W1 那个"单请求直连"的纵向切片，变成"能扛并发、能追全链路、能一键起来、能自动证明没坏"的服务底座。**不要在这周碰 RAG / 多智能体 / 前端**。

---

## Gate 0 · 开工前（30 min）

- [ ] 复核 W1 状态：`uv run pytest --basetemp=.pytest_tmp` = 278 passed / 16 skipped；`ruff`/`mypy` 干净
- [ ] `uv run agent-server` + `growth-agent` 两个 `/health` 都 ok（本周要在其上加压）
- [ ] 本机可起 Redis（`docker run -p 6379:6379 redis:7` 或 brew）；确认能连
- [ ] 读 `legacy/temporal_demo/` 的旧 demo（W1 移过去的），**只取思路，不复制代码**
- [ ] 决定本周 Temporal 依赖是否加回 `pyproject.toml`（`w1-tasks.md` T1.2 记着"W2 再加回"）
- [ ] 通读 `architecture.md` §7「W1 明确不做」与 §8「扩展点」——本周只做 §8 里标注 W2/W3 的那几格

---

## Day 1（3h）· 会话状态 + 长任务分层

**目标**：让"多轮对话"有地方存，并**定稿**一个问题——哪些请求走直连、哪些走工作流。这天的产出一半是决策文档，不是代码。

### 你要做的事
- [ ] `session/base.py`：`SessionStore` 协议 + `Turn` 数据结构
- [ ] `session/memory.py`：`InMemorySessionStore`（测试/离线用，零依赖）
- [ ] `session/redis.py`：`RedisSessionStore`（真存储）
- [ ] `session/__init__.py`：工厂按 `settings` 选实现
- [ ] `docs/w2/adr-w2-session-boundary.md`：**分层决策记录**（下面给格式）
- [ ] 单测：多轮 append/get、会话隔离、aclose

### 引导框架

**落点**：`packages/agent_framework/src/agent_framework/session/`

```python
# session/base.py —— 只给协议与数据，实现留给你
from typing import Protocol
from agent_framework.llm.types import Message

class SessionStore(Protocol):
    async def get_history(self, session_id: str) -> list[Message]:
        """取该会话归一化后的历史消息（已是内部 Message，不含厂商字段）。"""
        ...
    async def append(self, session_id: str, messages: list[Message]) -> None:
        """追加一轮；必须支持幂等（同一 request_id 重复 append 不翻倍）。"""
        ...
    async def aclose(self) -> None: ...
```

**算法步骤（Redis 实现）**
1. key 设计：`session:{session_id}:turns`，用 list；value 是 `Message` 的 JSON（`model_dump` 在这里可用，因为是**内部存储**不是对外响应——与 Day4「禁止 dump LLMResponse 到响应」是两件事）
2. `append` 用 `RPUSH` + 以 `request_id` 做去重标记（setnx + TTL），这就是幂等
3. `get_history` 用 `LRANGE 0 -1` 后反序列化；**不要把 `provider_meta` 丢掉**（ADR-007，回传 DeepSeek reasoning 要用）
4. 连接池走 `redis.asyncio`，`aclose` 关池；TTL 可配，别用永久 key

**分层决策文档（今日核心，必须写）**

| 请求类型 | 路径 | 理由 |
|---|---|---|
| `/v1/chat` 流式 / 非流式 | **直连 `LLMGateway`**（W1 已定） | 首 token 延迟敏感，ADR-008 的 committed 状态机不适合套工作流 |
| 分钟级"深度调研"任务（W9 才成型） | **Temporal workflow** | 需要可续跑、幂等、分步重试 |
| 本周只做**接口缝**：`workflow/research_workflow.py` 定义 activity 签名 + 占位 workflow | — | 真实编排逻辑 W9 写，别提前 |

> ⚠️ 硬边界（review 必查）：**不许把 `/v1/chat` 塞进 Temporal**（`architecture.md` §7 明令）。也不要在 `SessionStore` 里调 `LLMGateway.chat`——会话层只存取，编排归调用方。

**自检**
- [ ] `uv run pytest -k session` 全绿且**零网络**（InMemory 实现）
- [ ] 起本地 Redis 跑一次：append 两轮 → get_history 返回 4 条 Message，顺序正确
- [ ] 重复 append 同一 `request_id` 不产生重复条目

### 🔍 交给我的锚点
`SessionStore` 协议签名、`adr-w2-session-boundary.md` 全文、`pytest -k session` 输出。我会看：幂等是否真做了、`provider_meta` 有没有被丢、分层有没有越界。

---

## Day 2（3h）· 并发控制 + 背压 + 长任务队列

**目标**：100 并发不崩、不静默失败；过载时给调用方一个**可据此退避**的信号。

### 你要做的事
- [ ] `resilience/concurrency.py`：`ConcurrencyLimiter`（按 alias/provider 分信号量）
- [ ] `resilience/errors.py`：`ServerBusyError`——并接入 `llm/errors.py` 的映射表
- [ ] `queue/base.py`：`TaskQueue` 协议（enqueue / get_status / get_result）
- [ ] `queue/inmemory.py`：`InMemoryTaskQueue`（离线可测）
- [ ] `server/routes_task.py`：`POST /v1/tasks`（提交→返回 task_id）/ `GET /v1/tasks/{id}`
- [ ] 单测：限流触发、队列满、状态机流转

### 引导框架

**边界铁律（今天最容易翻车）**：ADR-003 说"重试只在 client 实现"。**入口层的过载拒绝绝不能被 `LLMGateway` 当成 retryable 再打一遍**——否则过载时重试风暴会把 100 并发放大成 300。

```python
# resilience/concurrency.py
class ConcurrencyLimiter:
    def __init__(self, limits: dict[str, int]) -> None:
        """limits 的 key 用 alias 或 provider，值从 models.yaml 读，别硬编码。"""
        ...
    async def acquire(self, key: str, timeout_s: float) -> None:
        """拿不到信号量且超时 -> raise ServerBusyError（retry_after 有值）。"""
        ...
```

```python
# resilience/errors.py —— 关键：retryable = False，但 retry_after 要透传
class ServerBusyError(LLMError):
    retryable = False          # 入口过载，调用方按 Retry-After 退避，框架不再自重试
    retry_after = 2.0
```

**算法步骤**
1. `acquire()` 在**调 gateway 之前**执行；超时抛 `ServerBusyError`
2. 在 `server/errors.py` 的 `install_error_handlers` 里给 `ServerBusyError` 补一行 → `503` + `Retry-After` 头（复用 `error_body`，别写第二套 handler）
3. `POST /v1/tasks`：只做校验 + `enqueue` + 立即返回 `{task_id, status: "queued"}`，**不在 HTTP 请求里跑长任务**
4. 消费者（本周可以是后台 asyncio task）从队列取 → 还原成 `ChatCall` → 调 `LLMGateway.chat` → 写回 result
5. 状态机：`queued → running → succeeded | failed`，每次跃迁落一条带 `trace_id` 的日志（D3 用）

**自检**
- [ ] `locust` 或自写并发脚本：100 并发打 `/v1/chat`，错误率 < 1%，无 5xx 风暴
- [ ] 故意把限额设成 1 → 第 2 个并发收到 `503` 且响应头有 `Retry-After`；日志里 `retry_count` 不因此增长
- [ ] 断网跑 `pytest -k "queue or concurrency"` 全绿

### 🔍 交给我的锚点
`ServerBusyError` 的 `retryable`/`retry_after` 定义、映射表新增行、并发脚本输出。我会重点看：**过载有没有被误重试**（注入一次满载看 `retry_count`）。

---

## Day 3（2.5h）· 可观测三件套收口：trace 贯穿

**目标**：`request_id`（一次 HTTP）之上补 `trace_id`（一次用户任务，可能跨多次 LLM/工具调用）与 `span_id`，让"任意任务可回放完整 trace"（DoD 可观测项）有骨架。**Langfuse 本周不接，只留缝**（W3 接）。

### 你要做的事
- [ ] `observability.py`：新增 contextvar `trace_id_var` + `span_id_var`，`RequestLog` 增加 2 个字段（**不改已有 18 字段的语义**）
- [ ] `RequestIDMiddleware`：入口生成/透传 `trace_id`（与 `request_id` 同手法）
- [ ] `tracing.py`：`Span` 数据结构 + `span()` 异步上下文管理器 + `TraceExporter` 协议（本周 `NoopExporter`）
- [ ] 在 `LLMGateway.chat`/`stream` 每次实际调用处包一个 `span("llm.call", ...)`，记录 `first_token_ms`/`latency_ms`/tokens
- [ ] 单测：trace_id 贯穿嵌套 span、contextvar 在并发请求间不串

### 引导框架

```python
# tracing.py
from contextlib import asynccontextmanager
class TraceExporter(Protocol):
    async def export(self, spans: list["Span"]) -> None: ...   # W3 换 LangfuseExporter

class Span:
    trace_id: str; span_id: str; parent_span_id: str | None
    name: str; started_at: float; latency_ms: float
    attributes: dict  # model/tokens/cost/error_code

@asynccontextmanager
async def span(name: str, **attributes):
    """进入置 span_id_var，退出算 latency 并交给 exporter；异常也要 export。"""
    ...
```

**算法步骤**
1. `trace_id` 在中间件层生成，一次用户任务内所有下游调用共享；`span_id` 每次 `span()` 新建
2. `span()` 用 `try/finally` 保证异常路径也 export（否则失败调用的 trace 会缺）
3. exporter 走 `get_request_logger()` 的输出通道思路，但**本周只 Noop**，不要在库里硬连 Langfuse
4. 在 `LLMGateway` 里包 span 的位置要选"真正发起 HTTP 的那层之外"，让 `retry_count`/`fallback_from` 也能进 attributes

**自检**
- [ ] 一次 `/v1/chat` 流式请求，日志里能 grep 到同一 `trace_id` 的 `llm.call` span，含 `first_token_ms`
- [ ] 并发 10 个请求，各请求 `trace_id` 互不串（contextvar 隔离测试）
- [ ] 注入一次上游失败，失败 span 仍被 export（不是静默丢）

### 🔍 交给我的锚点
`span()` 实现、`RequestLog` 新字段定义、一条真实请求的日志片段。我会看：`finally` 是否 export 了异常、有没有改动 W1 已有字段语义。

---

## Day 4（3h）· 容器化 + 压测基线

**目标**：`docker compose up` 一键起 app + redis（+ 可选 temporal），交出第一组延迟/成本基线报告——这是 W11 成本优化、W12 趋势图的第一批数据点。

### 你要做的事
- [ ] `ops/Dockerfile`（多阶段构建，uv 装依赖）、`.dockerignore`
- [ ] `ops/docker-compose.yml`：`agent-server` + `growth-agent` + `redis`（+ `temporal` 可选 profile）
- [ ] `ops/.env.example`：4 个 key + `REDIS_URL`，密钥只走 env
- [ ] `evals/scripts/locustfile.py`：压 `/v1/chat`（stream + 非 stream 两类）
- [ ] 报告落 `evals/history/YYYY-Www-w2-load.md`（含 p50/p95/错误率/并发数）

### 引导框架

**压测脚本要点**（复用 W1 的 mock 夹具做"离线档"，真机另跑一次"在线档"）
```python
# locustfile.py 骨架
class ChatUser(HttpUser):
    wait_time = between(1, 2)
    @task(3)
    def chat_blocking(self):
        self.client.post("/v1/chat", json={
            "messages": [{"role": "user", "content": "..."}],
            "model": "fixture",            # 离线档不花钱；在线档换真实 alias
        })
    @task(1)
    def chat_stream(self):
        with self.client.post("/v1/chat", json={..., "stream": True},
                              stream=True, catch_response=True) as r:
            ...  # 统计首帧到达时间
```

**算法步骤**
1. 先跑**离线档**（model=fixture）确认工具链与端点没被压崩，再跑在线档记真实延迟
2. 100 并发 × 60s，采集 p50/p95/错误率；`first_token_ms` 从 `evals/scripts/bench_baseline.py` 的 LogCapture 口径取，**别自己 time.monotonic**（否则和 W1 基线两套定义）
3. 报告里必须记**时间戳**（W1 Day5 已发现 DeepSeek 峰谷计价差 2 倍）
4. compose 里 healthcheck 指向 `/health`，别压在没就绪的容器上

**自检**
- [ ] `docker compose up -d` 后两个 `/health` 都 200，`docker compose ps` 无 restart 循环
- [ ] 离线档 100 并发错误率 < 1%
- [ ] 报告写入 `evals/history/`，字段可被 W11 读（对齐 W44 json 的 metrics 结构）

### 🔍 交给我的锚点
compose/Dockerfile、locustfile、load 报告。我会看：压测口径是否复用 W1 的 `first_token_ms`、有没有记录时间戳。

---

## Day 5（2.5h）· 10 条 smoke eval + CI 门禁

**目标**：建最小评测集并接进 CI——**先能跑（10 条），W3 再扩到 50+**（ROADEMAP 原话）。本周的 eval 只回答"没坏"，不回答"多好"。

### 你要做的事
- [ ] `evals/golden/smoke_v1.jsonl`：10 条 case（文本 / tool_call / 错误映射 / 流式各覆盖）
- [ ] `evals/scripts/run_smoke.py`：批量 runner（复用 `LogCapture` + mock 夹具，零网络档 + 真机档）
- [ ] `.github/workflows/ci.yml`：`ruff` + `mypy` + `pytest` + `run_smoke` 门禁
- [ ] `Makefile`：`make test` / `make eval` / `make up`（可选但强烈建议）
- [ ] 更新 `docs/log.md` W2 条目；周五合流仪式

### 引导框架

**case 数据结构**（与 `tests/canonical_input.py` 对齐，别另立一套输入）
```json
{"id": "smoke-01", "kind": "chat", "model": "fixture",
 "messages": [...], "expect": {"text_contains": [...], "finish_reason": "STOP",
 "usage_estimated": false}}
```
- 3 条文本、3 条 tool_call、2 条错误映射（其中 1 条必须是"上游鉴权失败 → 502 不是 401"，ADR-005）、2 条流式
- runner 判据直接复用 conformance 的断言风格，**不许另写宽松判据**

**算法步骤**
1. runner 加载 jsonl → 每条还原成 `ChatCall` → 走 `LLMGateway` → 用 `expect` 断言
2. 离线档强制 `model=fixture`（零网络、可进 CI）；真机档由 `-m live` 触发、无 key 自动 skip（复用 W1 `test_live.py` 的三重闸门）
3. CI 门禁：`run_smoke` 任一 case 失败即 exit 非 0；阈值本周只设"全过"，W4 再加"指标不得劣化"
4. 合流仪式（ROADEMAP §3.2）：合 main → 跑全量评测 → 写 `evals/history/YYYY-Www.json`（对齐 W44 结构）→ 更 `docs/log.md`

**自检**
- [ ] `make eval`（或 `uv run python evals/scripts/run_smoke.py`）离线全绿
- [ ] CI 里故意改坏一个 case 的 expect → 门禁变红（反向验证）
- [ ] `evals/history/YYYY-Www.json` 里 latency/cost 有本周压测填入的真实值（或明确标 pending）

### 🔍 Gate 2 · 周末验收（全绿才算过 W2）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 会话可用 | 多轮 append/get 幂等，`provider_meta` 不丢 |
| 2 | 分层不越界 | `/v1/chat` 仍直连，无任何 workflow 包装；决策文档在 |
| 3 | 并发达标 | 100 并发错误率 < 1% |
| 4 | 过载不自重试 | 满载 → 503 + Retry-After，`retry_count` 不增长 |
| 5 | trace 贯穿 | 一条请求可追到每次 LLM 调用耗时与 tokens |
| 6 | 一键部署 | `docker compose up` 起 app+redis，`/health` 就绪 |
| 7 | 评测门禁 | 10 条 smoke 全绿且接进 CI，反向验证能拦住 |
| 8 | 合流 | 端到端仍可跑，`evals/history` 与 `docs/log.md` 已更新 |

---

## 本周最容易返工的 6 处

1. **过载被 client 当 retryable 重试** → 100 变 300，压测直接崩。`ServerBusyError.retryable` 必须 False
2. **会话层丢 `provider_meta`** → DeepSeek 第二轮 400（ADR-007），W1 已踩过
3. **把 `/v1/chat` 塞进 Temporal** → 违反 architecture §7，首 token 延迟被拖垮
4. **压测自己 `time.monotonic`** → 与 W1 基线两套 `first_token_ms` 定义，W11 曲线对不上
5. **`RequestLog` 改了已有字段语义** → W1 的成本/日志看板口径全乱；只许加字段
6. **smoke 判据另写一套宽松逻辑** → CI 假绿；必须复用 conformance 断言风格

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 会话层 | `packages/agent_framework/src/agent_framework/session/{base,memory,redis}.py` |
| 分层决策 | `docs/w2/adr-w2-session-boundary.md` |
| 并发/队列 | `.../resilience/{concurrency,errors}.py`、`.../queue/{base,inmemory}.py` |
| 任务端点 | `.../server/routes_task.py` |
| trace | `observability.py`（+2 字段）、`.../tracing.py` |
| 部署 | `ops/{Dockerfile,docker-compose.yml,.env.example}` |
| 压测 | `evals/scripts/locustfile.py`、`evals/history/YYYY-Www-w2-load.md` |
| 评测 | `evals/golden/smoke_v1.jsonl`、`evals/scripts/run_smoke.py`、`.github/workflows/ci.yml` |
| 记录 | `docs/log.md`（W2 条目）、`evals/history/YYYY-Www.json` |

---

## P3 挂载点（默认档必做 / 深入档由你判断，见 ROADEMAP §7）

- **高并发规模**（P3-6）：本周的 100 并发压测就是**默认档已完成**的证据；压到 1000 并发+K8s/HPA 属深入档，W12 之后再说
- **前端**（P3-1）：本周不做，最佳挂载点 W11 末 / W12
- 决策点提醒：ROADEMAP §7 的三个决策点是 **W4 末 / W8 末 / W12 末**，W2 末不用急着判断深入档
