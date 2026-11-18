# `server/question.md` 逐题解答（下半场：第 10–21 问 + 场景题 + 实战题）

> 上半场（原文第 1–9 问：中间件、路由、gateway、session、队列、字段注释、架构图）
> 见 `docs/server-question-answers.md`。本文接下去，覆盖原文第 10–21 问、
> 四道场景题、三道实战题。
>
> 结论以当前代码为准。**函数名比行号可靠**——行号只作近似定位。
> 凡是"未实现"的地方，本文直说未实现 + 给出接入方案，不包装成"已支持"。

---

## 速查：这十二问的一句话结论

| 问 | 一句话 | 依据 |
|---|---|---|
| 10 队列为何不继承 base | `TaskQueue` 是 `Protocol`，结构化类型，**靠 mypy 检查而不是继承** | `queue/base.py:81`、`server/app.py:49` |
| 11 memory 怎么起作用 | 框架只给四块积木，**不自动注入**；消费方是产品 `/v1/answer` | `memory/*`、`growth_agent/answer.py` |
| 12 RAG 怎么起作用 | 框架给"流水线 + 引用治理"，**索引和模型都在产品侧** | `retrieval/*`、`knowledge/*`、`growth_agent/rag.py` |
| 13 接 GitHub MCP | 今天接不了：**没有真传输**。要写一个 `MCPTransport`，四步接上 | `skills/mcp/types.py:203` |
| 14 多轮对话 | 两档：无状态 `/v1/chat`；有状态 `/v1/answer` + 会话 + 滚动摘要 | `session/*`、`memory/working.py` |
| 15 幻觉 | 五层，第一层最有效：**没资料就不生成** | `answer.py:208`、`knowledge/citation.py` |
| 16 上下文管理 | 五槽 + 预留输出；**槽位内按分裁、槽位间不挪用** | `context/budget.py`、`context/schema.py` |
| 17 token 成本 | `(in×p_in + out×p_out)/1000000`，6 位小数，**只够说量级** | `observability.compute_cost` |
| 18 functionCall | 框架**只翻译不执行**；自研 vs MCP 靠内部命名 `server__tool` | `llm/types.py`、`skills/mcp/types.py:61` |
| 19 MCPClient 连接与鉴权 | 有协议、**没有真传输，也就没有握手/鉴权** | `skills/mcp/client.py` |
| 20 resource / prompt | **未实现**；要做必须落成 `trust=external` 的 ContextItem | `MCPTransport` 只有两个方法 |
| 21 MCP + Skill 方案 | Skill 是"六层契约 + 可见面 + 沙箱 + 审计"，MCP 只是工具来源 | `skills/spec.py:172` |

---

## 原文第 10 问：`InMemoryTaskQueue` 为什么没有实现 `queue/base.py` 里的类

**因为 `base.py` 里那个"类"根本不是基类，是 `Protocol`。**

```python
# queue/base.py:81
class TaskQueue(Protocol):
    async def enqueue(...) -> TaskRecord: ...
    async def get(self, task_id: str) -> TaskRecord: ...
    async def aclose(self) -> None: ...
```

```python
# queue/inmemory.py:32 —— 注意：没有基类
class InMemoryTaskQueue:
```

`typing.Protocol` 是**结构化类型**（structural subtyping）：只要方法名、签名对得上，
就被认为是该类型，不需要写 `(TaskQueue)`。这是刻意的选择，不是漏写：

| 写法 | 代价 |
|---|---|
| `class InMemoryTaskQueue(TaskQueue)` | 每个第三方实现都要 `import` 框架的基类；换 Temporal 时 MRO 里挂着一段永远不执行的 InMemory 逻辑；Protocol 继承后若想 `isinstance` 还得加 `@runtime_checkable` |
| 不继承（当前） | 少了一层"显式声明"，靠类型检查器兜 |

**那谁来保证它真的满足契约？** 不是靠"第一次调用 500"，有两个兜底：

1. **mypy 的结构化检查**。`server/app.py:49` 的签名是
   `task_queue: TaskQueue | None = None`，`:88` 处
   `app.state.task_queue = task_queue or InMemoryTaskQueue(...)` 这句赋值会让 mypy
   去核对 `InMemoryTaskQueue` 是否满足 `TaskQueue`——少一个方法就报错。
   当前门禁 `mypy 134 files 0 issues`，说明这条检查是过的。
2. **`tests/test_queue.py` 的行为测试**：提交后成功、LLM 错误标 `failed` 带 code、
   崩了标 `internal`、未知 id 抛 `not_found`、队列满抛 `busy` + `Retry-After`、
   关闭后再入队抛错。

一个容易混淆的点：`TaskQueue` 不是"任务队列的抽象"，而是"**返回任务状态的抽象**"。
它连 `run` 都没有——真正的执行在构造时注入的 `TaskHandler`
（`Callable[[ChatCall], Awaitable[dict]]`）。所以换实现（Redis 队列 / Temporal）
时，要换的是"状态存哪、谁来消费"，不是"怎么执行一次 LLM 调用"。

---

## 原文第 11 问：memory 系统是怎么在 agent 框架中起作用的

**先纠正一个常见预期**：框架**不会**自动记住任何东西。`/v1/chat` 是无状态的，
它连 session 都不读。记忆是一套**可组合的积木**，由产品决定何时用、写到哪。

### 11.1 框架给了四块积木

| 积木 | 位置 | 职责 |
|---|---|---|
| `MemoryItem` / `MemoryRecord` / `MemoryStore(Protocol)` | `memory/long_term.py` | 数据形态与存储契约。`MemoryItem` 是"给上层看的活跃记忆"，`MemoryRecord` 多两个治理字段（`dedupe_hash`、`superseded_by`） |
| `LongTermMemory` | `memory/long_term.py:231` | 写入五道闸 + 召回过滤（TTL / 置信度） |
| `WorkingMemory` | `memory/working.py:165` | 会话视图（[摘要] + 最近 N 条）、压缩、工具结果截断与句柄 |
| `MemoryExtractor` | `memory/extract.py:160` | 从对话里提候选（走 cheap 档，key 白名单） |

存储实现**不在框架里**：`MemoryStore` 是协议，产品用 pgvector 实现
（`growth_agent/memory.py:PgVectorMemoryStore`）。框架零 ML 依赖——
embedding 在 store 里，这是它能被别的项目直接用的前提。

### 11.2 谁真的在消费它

```
memory/                 ← 框架：机制
   ↑ 注入（构造函数）
growth_agent/memory.py  ← 产品：装配（表名、TTL、recall_k、阈值）
   ↑
growth_agent/answer.py  ← 产品：三处调用
   ① recall_for_query(question)   → memory 槽位
   ② session_view(session_id)     → history 槽位
   ③ record_turn(...)             → 回答落地后写
```

框架的 `server/` 侧**没有**任何一处读记忆：`app.state.session_store` 至今没有消费方
（见上半场第 5 问）。所以"框架有记忆功能"这句话只对一半——
**它有记忆的机制，没有记忆的行为**。

### 11.3 一次写入的完整链路

```
is_explicit_instruction("记住：以后别用行业领先")   ← 字面正则，零成本
        ▼ 命中才继续
MemoryExtractor.extract（cheap 档，temp=0，≤5 条）
        ▼  LLM 输出 JSON → parse_extraction（坏输出返回空，不抛）
MemoryItem × n
        ▼
LongTermMemory.write：
   闸1 trust=external 且未确认 → rejected
   闸2 confidence < 0.6        → rejected
   闸3 dedupe_hash 命中        → duplicate
   闸4 同 key 不同值           → superseded（旧记录保留，只写 superseded_by）
   闸5 upsert 单事务           → written
        ▼
MemoryStore.upsert（pgvector：agent_memories 表，HNSW cosine）
```

### 11.4 为什么这些决策放在框架、那些放在产品

| 决策 | 放在哪 | 理由 |
|---|---|---|
| key 白名单（brand_tone / forbidden_words / past_topics） | 框架 `extract.py` | 受控词表让召回端的 prompt 稳定，换产品也只换词表内容 |
| 置信度阈值、TTL 分档 | 框架（可配） | 属于"记忆策略"，与存储实现无关；换 store 不该改策略 |
| 表名、DSN、recall_k、槽位预算 | 产品 | 用几张表、放哪个库是部署决策 |
| 注入到哪个槽位、信任级 | 产品（`ContextItem`） | 同一个记忆在不同产品里可能是 system 也可能是 user |

**最关键的一条**：记忆块的信任级是 `TrustLevel.USER`，绝不进 system。
混进 system 就被模型当成系统指令——这正是 W10 注入防护要挡的事，
只是提前在 Context 层立住了（`context/schema.py` 里 external 进 system 直接抛
`ValueError`，不是裁掉）。

---

## 原文第 12 问：RAG 系统是怎么在 agent 框架中起作用的

和记忆同构：**框架给流水线，不给索引**。

### 12.1 分层

```
  knowledge/          引用与治理：Citation / build_context / audit_citations / conflict_note
  ─────────────────────────────────────────────────────────────
  retrieval/          召回与组装：Retriever(Protocol) / RetrievalPipeline / assemble / ACL / 改写 / 融合
  ─────────────────────────────────────────────────────────────
  （框架到此为止，下面全是产品）
  ─────────────────────────────────────────────────────────────
  growth_agent/rag.py           接 rag_mvp：pgvector + bge-m3 + AsyncRetriever
  growth_agent/answer.py        串起来：run → assemble → build_context → 生成 → 体检
```

`Retriever` 是 `Protocol`（`retrieval/base.py`），只要 `retrieve(query, top_k)`
返回带名次的结果就能接进来。框架里**没有**任何 `import faiss/pgvector/torch`——
`api_audit.py` 会盯着这一点。

### 12.2 流水线做了什么（顺序不能换）

```
rewrite（默认关）→ recall（over-fetch ×3）→ ACL 过滤 → fusion → rerank（默认关）
        │                                                      │
        └──────────── 每段独立降级，degraded 里记名字 ────────────┘
                              ▼
                        diagnostics（各段耗时 / candidates / filtered_out）
```

两个默认值是实测定的，不是拍脑袋：

- `retrieval_mode = "dense"`：W5 的 A/B 里 hybrid 的 R@1 `.5714`、R@5 `.8571`，
  单路 dense 是 `.7857` / `.9643`（`docs/w5/rag-ab-report.md`）。融合没带来收益，
  就没有默认开着的理由。
- `retrieval_rewrite = False`、`retrieval_rerank = False`：默认关。改写 p95 15.2s
  （+94×，28 次里 3–4 次超时降级），精排 20 条候选 p95 5.4s 且 R@1 −7pp——
  两个都是"开着更慢、还可能更差"的档。

### 12.3 谁可以调用它

只有产品层：`/v1/answer`、`/v1/article`（共用同一条 `RetrievalPipeline`，
理由是两边的 `[n]` 要能互相对照）。**框架的 `/v1/chat` 不调检索**——
所以你在 chat 里问"我们资料里怎么写的"，永远拿不到引用。

### 12.4 框架真正提供的价值在哪

不在召回质量（那是 embedding 和语料的事），在三件"每个项目都会写错一次"的事：

1. **编号只有一处**（`build_context`）——编号与引用面板同源，不会两套 `[n]`；
2. **裁剪在编号之前**（`assemble`）——编号发出后就不能再删段；
3. **引用可体检**（`audit_citations`）——`no_citation / unknown_index /
   source_missing / off_target` 四档，而不是只有一个"看起来对"。

---

## 原文第 13 问：如何加载并使用三方 MCP 服务（比如 GitHub）

### 13.1 先说实话：今天接不了

框架里**没有真实传输**，也没有 `mcp` SDK 依赖
（`pyproject.toml` 依赖只有 fastapi/httpx/pydantic/pyyaml/uvicorn；extras 只有 `langgraph`）。
现有三个实现：

| 实现 | 文件 | 用途 |
|---|---|---|
| `InMemoryMCPTransport` | `skills/mcp/inmemory.py` | 测试/离线：把 `{method: handler}` 当线缆 |
| `RecordedToolTransport` | `skills/replay.py` | 回放：录制过的工具结果 |
| （真实 stdio / SSE） | — | **不存在**。`types.py:206` 的注释写"W11 接"，实际没接 |

所以"接 GitHub"的第一步不是配置，是**写一个 transport**。

### 13.2 四步接上

**第 1 步：实现传输（唯一需要懂 JSON-RPC 的地方）**

```python
# 新文件：skills/mcp/stdio.py（示意）
class StdioTransport:
    """JSON-RPC 2.0 over stdio。只搬运，不解释。"""
    def __init__(self, command: list[str], env: dict[str, str]) -> None: ...

    async def list_tools(self) -> dict[str, Any]:
        return await self._rpc("tools/list", {})

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})
```

三条硬约束（都来自现有设计，别绕过）：

- 返回**原始 `result`**，不要在这里转成 `ToolSpec` / `ToolOutcome`——
  翻译只允许发生在 `MCPClient` 一处；
- 传输出错**不要重试**（重试归网关，工具侧重试会让"同一动作执行两次"变默认行为）；
- token/鉴权信息只放在 transport 的 `env`/`headers` 里，**不进 `ToolSpec`、不进日志**。

**第 2 步：包成 client**

```python
client = MCPClient(StdioTransport(["npx", "-y", "@modelcontextprotocol/server-github"],
                                  env={"GITHUB_TOKEN": token}),
                   server_name="github", timeout_s=8.0)
```

`list_tools()` 会把 `search_repositories` 变成内部名 **`github__search_repositories`**
（`internal_name`）。名字转换只在这一处发生——模型看到的名字和 server 认识的名字
必然不同，两处各换一次，改名那天就有一半调用打到不存在的工具。

**第 3 步：注册进产品装配（一行）**

```python
# growth_agent/tools.py:build_tool_runtime
clients = {
    "search": SearchServer().client(),
    "fetch": FetchServer().client(),
    "workspace": workspace.client(...),
    "github": github_client,          # ← 新增
}
```

同时把 `api.github.com` 加进 `TOOLS_NETWORK_ALLOW`——沙箱的网络闸默认**全拒**，
漏了这一步，工具会在沙箱层被拦，报的错看起来像"工具不可用"。

**第 4 步：写 skill 声明**

```yaml
name: github_repo_lookup
description: 查 GitHub 仓库的元信息与最近提交
mission: |
  只负责读取公开仓库信息。不负责克隆代码、不提 PR、不碰私有仓库。
instructions: |
  1) 用 github__search_repositories 找到仓库全名
  2) 用 github__get_file_contents 读关键文件
entrypoint: github.search_repositories
permissions: [net.read]
stages: [research]
dangerous: false
```

### 13.3 三个必踩的坑

1. **工具名冲突**：两个 server 都有 `search`。前缀命名空间解决，`ToolAudit.server`
   记录是哪一个——别靠"描述里写了 GitHub"来区分。
2. **schema 缺失就放水**：MCP server 不返回 `inputSchema` 时，`MCPClient._to_spec`
   给的是**严格空对象**（`additionalProperties: false`），不是宽松的
   `{"type": "object"}`。后者等于"参数随便传"，A14 那条入参校验就形同虚设。
3. **超时与重试的归属**：`asyncio.wait_for(8s)` 在 client 层；超时返回
   `tool_timeout`，**不重试**。如果 transport 自己带重试，一次模型调用会变成
   "候选链 3 × 重试 2 × transport 重试 N"——W11 INC-5 里期望 6 实测 18 就是这么来的。

---

## 原文第 14 问：多轮对话是怎么设计的

### 14.1 两档，不是一个

| 档 | 端点 | 谁持有历史 | 适合 |
|---|---|---|---|
| 无状态 | `/v1/chat` | **调用方**（每次带全量 messages） | 前端自己管上下文、网关型用法 |
| 有状态 | `/v1/answer` | 服务端（`session_id`） | 产品自己的问答链路 |

`/v1/chat` 无状态是刻意的：它要保证"同样的请求体 → 同样的行为"，
一旦在服务端藏历史，压测与回放的结论都不可解释。

### 14.2 有状态那档的两层记忆

```
session_id
   │
   ├── SessionStore（短）  原始轮次，append 幂等（request_id 去重），max_turns 截断
   │        │
   │        └── WorkingMemory.view(session_id) = [摘要] + 最近 8 条原文
   │                ▲
   │                └── compress(): 把 history[covered:-8] 压进滚动摘要
   │                     触发：pending > 6000 tok 或 未压缩条数 > 20
   │
   └── LongTermMemory（长）跨会话，靠显式指令触发提取
```

三个细节，每个都是踩过的：

1. **幂等键是 `request_id`**：前端双击、SDK 重试都不会把历史翻倍。
   `request_id` 来自 `RequestIDMiddleware`，所以"同一次请求"在整条链路上是同一个号。
2. **摘要是 `role=user` + 带标签**：`[会话摘要（系统生成，覆盖前 N 条消息，非用户原话）]`。
   用 `system` 会被模型当指令，不带标签会被当用户原话——两种都错。
3. **压缩是幂等的**：每轮都调，未达阈值直接 `SKIPPED`。这样"什么时候压"不需要
   调用方判断，也不会有"漏压一次就再也不压"的状态机。

### 14.3 一个坑（面试可以主动说）

产品侧 `build_memory_runtime(settings, gateway, embedder=...)` 没传 `session_store`，
于是它自己 `build_session_store(settings)` 建了一份；框架 `server/` 侧那份
`app.state.session_store` 至今没有消费方。**两个实例，不共享历史。**
现在不影响功能（只有产品那份被用），但"框架的 session 能被产品复用"这个假设是错的。

---

## 原文第 15 问：如何解决幻觉问题

先划边界：这套机制治的是**"没有依据却说得像有依据"**（可溯源的事实型幻觉），
治不了"资料本身就是错的"。后者要靠治理（来源可信度、时效），不是靠 prompt。

### 15.1 五层，从上游往下

| 层 | 做法 | 位置 | 拦住什么 |
|---|---|---|---|
| ① 不生成 | 零召回返回固定话术，**不调模型** | `answer.py:208` | 没资料时编造（最有效的一层） |
| ② 强制编号 | `CITATION_RULES` + `build_context` 唯一编号 | `knowledge/citation.py` | 结论与出处脱钩 |
| ③ 引用体检 | `audit_citations` 四档判据 | 同上 `:330` | 编编号、引占位块、引到无关段 |
| ④ 输出闸 | `OutputGuard`：`dangling_markers` / `unpublished_claims` | `guardrails/output.py` | "读者点开是空的"、"有事实信号却没标号" |
| ⑤ 评测兜底 | golden + 引用抽样（precision / blanket_citation） | `evals/` | 指标回退 |

### 15.2 第 ① 层为什么最强

> 没有 context 时让模型"自由发挥"，等于把幻觉写进溯源链路，
> 而抽样时它还会被记成"引用缺失 1 条"——归因直接跑偏到生成层。

返回一句固定话术 + `grounded=false`，归因就老老实实指向检索层（该补语料）。
这是"用产品行为换归因正确"，比任何 prompt 技巧都值。

### 15.3 第 ③ 层的判据为什么是"至少一条命中"

首次抽样时 `golden-001` 引了 5 条，其中 2 条是标注段，另外 3 条是同页相邻块。
按"条条命中"判它失败，但答案确实指向了正确出处。根因是**标注本身是部分标注**
（425 个块只标了 69 个）——用"标注是否穷尽"衡量"引用是否正确"，
模型引对了没标的地方反而扣分，而这类失败会被归因到生成层，改 prompt 永远改不好。

所以拆成两个量：`verdict` 只回答"有没有引对"，`precision` 回答"引的东西有多干净"。
只要前者，模型学会"把整个 context 都引一遍"就能稳过（`blanket_citation` 检查盯的就是这个）。

### 15.4 明确不做的两件事

- **不让模型自评"我有没有编"**：那是它最擅长伪装的地方，且不可复算。
- **不把 LLM judge 当唯一判据**：W4 的人工一致性 kappa 加权 `.343` / 名义 `.200`
  （24 条标注），只够当"规则基线的下限对照"，不够当门禁。

---

## 原文第 16 问：上下文是如何管理的

### 16.1 一张预算表

```python
# context/schema.py:144 ContextBudget
window = 8000            # 模型输入窗口 - reserve_output
system = 400
retrieved = 3000
memory = 600
history = 2000
tool_results = 1000
reserve_output = 600     # 不可被任何输入槽位侵占
```

`validate()` 在分配前强制：五槽之和 ≤ `window`，且每个槽位 ≥ 0。
破了这条，"预算"就是五个互不相干的数字，总输入照样爆窗口——
这种错要在**分配前**炸，不能留到"第一次真实请求 400"。

回答链路的实际取值（`answer.py:_answer_budget`）把 `tool_results` 显式设 0：
那条链路没有工具结果，省下的 1000 token 给检索。

### 16.2 两条铁律（写死，不提供"自定义裁剪策略"）

1. **槽位内**：超预算时按 `(score, authority, id)` 从低到高裁。
   确定性不是为了好看，是 replay 复现的前提——引用抽样就是靠逐位复现
   才查出判据错误的。
2. **槽位间不挪用**：每个槽位独立裁到自己的上限，剩余额度不借给别的槽位。
   挪用会让"retrieved 里第 6 条为什么没进"的解释依赖当轮 memory 的占用，
   Debug Report 就从"配置问题"变成"运行时问题"，不可复算。

### 16.3 三道闸（顺序固定）

```
① 上游已出局（excluded != None）→ 照抄进报告，decided_by=upstream
② 过期（is_expired）           → filtered_expired
③ 信任级违规                   → 抛 ValueError，**不是裁掉**
```

第 ③ 条值得单独说：`system` 槽位出现 `trust=external` 的内容时直接抛，
因为那是**攻击面**而不是预算问题。静默裁掉等于把一次注入尝试降级成"预算不足"，
日志里再也看不见。

### 16.4 排布（builder 只排，不判）

```
system（+ extra_instructions）
history / tool_result   ← 中段
user:
   记忆块（带来源，无编号）
   已编号资料（retrieved_text，**不可拆分单元**）
   问题（永远最后，尾部高注意力）
```

`retrieved_text` 不做内部重排：编号 `[n]` 的完整性高于槽位排序，
重排会让 `[n]` 指错段。

### 16.5 太长的时候

四步，顺序不能反：**先砍 → 再压 → 再挪 → 最后才扩窗口**。

1. 砍：`assemble` 去重（阈值 0.9）、按预算裁；
2. 压：`history` 滚动摘要（6000 tok / 20 条 / 保留 8 条）；
3. 截：工具结果超 2000 字符截断 + 句柄（`truncate_tool_result` / `refetch`）；
4. 挪：首尾重排（`attention_ordering`）、`context/degradation.py` 分层缓解；
   事后归因用 `context/drift.py` 的三指标（目标漂移 / 证据丢失 / 重复劳动）。

**不要**把所有东西塞进 system：既会被模型当指令，又把注入面扩大到最大。

---

## 原文第 17 问：token 成本如何计算

### 17.1 口径

```python
# observability.py:234
cost = (usage.input_tokens * pricing.input + usage.output_tokens * pricing.output) / pricing.per
return round(cost, 6)
```

三条都是刻意的：

- `per` 一律 1000000（`config.py` 强制校验）。不假设单位，避免"某家按 1K 计价"把成本算小 1000 倍；
- 保留 **6 位小数**：单次成本常在 0.0001 量级，圆到 2 位会变成 0，
  而"成本恒为 0"会让 W11 的熔断/预算彻底失效；
- 价格写在 `models.yaml` 的 alias 上（不是 wire 名）：换模型快照时价格跟着别名走，
  12 周的指标曲线才可比。

例（`models.yaml`）：`qwen3.7-plus` 输入 2.0 / 输出 8.0 CNY per 1M；
`deepseek-flash` 1.5 / 4.5。混合币种（CNY/USD）是允许的，
但报告必须按币种分组——`ChatOutcome` 里带 `currency` 就是为这个。

### 17.2 用量从哪来

`Usage(input_tokens, output_tokens, total_tokens, estimated)` 由 adapter 归一。
厂商没返回 usage 时标 `estimated=True` 做粗估——**不标 0**，否则成本看板会静默漏账。

### 17.3 预算侧为什么用 Decimal

```python
AutonomyBudget(max_cost=Decimal("0.50"))
BudgetMeter.charge_cost(...)   # Decimal 累加
```

`0.1 + 0.1 + 0.1 != 0.3`（float）。预算判定用浮点会出现"看起来没超但账单超了"，
所以整个 budget 链路用 `Decimal`，只有最终展示才转字符串。

### 17.4 必须承认的三个不精确

1. `cache_read` 价已经在 `models.yaml` 里，但 `compute_cost` **暂未使用**
   （注释写明"W11 成本优化时接入"）——所以命中缓存时算出来的成本偏高；
2. 阶梯/峰谷计价取**最保守的常规档**，单值字段表达不了分时；
3. token 数是 `1.6 chars/token` 的**粗估**（不引 tiktoken，它在模型间不通用）。
   预算是"防爆窗"，不是"精确计费"。

所以对外只能说成本量级，不能说"这次请求花了 0.0031 元"。

---

## 原文第 18 问：functionCall 是怎么实现的，如何区分自研工具和 MCP 工具

### 18.1 分工：框架只做"翻译"，不做"执行"

```
调用方 / planner
     │  ChatCall{tools=[ToolSpec...], tool_choice=...}
     ▼
LLMGateway → adapter      ← 把 ToolSpec 翻成厂商格式（这一步是"协议"）
     │
     ▼  LLMResponse.message.tool_calls = [ToolCall(id, name, arguments)]
     │
     ├─ 框架到此为止：/v1/chat 把 tool_calls 原样回给你
     │
     ▼  谁想执行谁执行
ActionRuntime.submit(ToolCall)     ← 这才是"执行"
```

`/v1/chat` 的 `ChatRequest.tools` 是**调用方声明的**，框架不校验它是否真实存在——
校验发生在执行层（A14 按 JSON Schema 校验入参）。

### 18.2 协议层的三个细节

1. **`ToolSpec.parameters` 是 JSON Schema**，直接喂给厂商；缺 schema 时给严格空对象，
   不给宽松对象（见第 13 问的坑 2）。
2. **流式按 `index` 拼分片**：`tool_calls` 在 SSE 里是逐片下发的
   （`{index:0, function:{name:"...", arguments:"{\"q"}}`），adapter 用
   `_ToolCallSlot` 按 index 归并 `arguments` 字符串，最后一次性 `json.loads`。
   不按 index 归并，多工具并发时参数会串。
3. **参数解析失败不抛**：`ToolCall.arguments` 置 `{}`，原因写
   `arguments_error`（`llm/types.py:125`）。抛异常会让一次模型跑偏变成一次 500。

### 18.3 自研 vs MCP：靠"内部命名空间"，不靠来源标记

```python
# skills/mcp/types.py:61
def internal_name(server: str, tool: str) -> str:
    return f"{tool_name_part(server)}__{tool_name_part(tool)}"
```

| 问题 | 答案 |
|---|---|
| 模型怎么知道这是哪个 server 的工具？ | 它不知道，也不需要知道。它只看到 `github__search_repositories` |
| 框架怎么知道该调谁？ | `SkillSpec.entrypoint = "github.search_repositories"` → 执行器按 `server` 找 `clients[server]`，按 `tool` 调 |
| 内置 server 与外部 MCP 有区别吗？ | **执行路径上没区别**：`SearchServer.client()` 返回的也是 `MCPClient`，跑的是 `InMemoryMCPTransport`。唯一区别在 `ToolAudit.server` 的值 |
| 业务代码会 grep 到 `mcp` 吗？ | 不会。`test_mcp_boundary.py` 直接扫源码，`skills/mcp/` 之外出现 `import mcp` 就红 |

这套设计的好处：**换协议不改上层**。今天内部 server 走内存传输、外部走 stdio，
上层看到的是同一种 `ToolSpec` / `ToolOutcome`。坏处也要认：内部工具被
"MCP 化"了一层，多了一次 JSON 序列化——这是为可替换性付的税，量级可接受
（本地 server 是微秒级）。

---

## 原文第 19 问：`MCPClient` 如何连接 server，鉴权握手怎么做

### 19.1 现状（诚实版）

```python
class MCPClient:
    def __init__(self, transport: MCPTransport, *, server_name, timeout_s=8.0, ...)
```

- **构造不连接**：构造只绑定 transport 与名字；
- **`list_tools()` 才第一次碰网络**（`tools/list`），且**不缓存**——
  server 加了工具，再调一次就能看到。缓存的收益是省一次往返（本地是微秒级），
  代价是"新工具要等缓存过期"，而检查点要求的正是"加了就能用"；
- **没有 `initialize` 握手**：`MCPServerInfo(name, version, protocol_version,
  list_changed)` 这个结构已经定义了，但**没有任何代码填充它**——因为只有内存传输，
  没有版本协商这回事；
- **没有鉴权**：token 无处可放，也没有 header 可设。

### 19.2 握手该发生在哪（写给要动手的人）

```
transport.connect()
   → initialize {protocolVersion, capabilities, clientInfo}   ← 版本协商
   → notifications/initialized
   → 之后才允许 tools/list
```

三条约束：

1. 握手失败 = **transport 抛异常**，`MCPClient` 只在 `list_tools` 里把它转成
   `warnings.append("server:list_failed")` + 空列表——一个 server 挂了不该让
   整个工具集消失；
2. `protocolVersion` 不匹配要在握手期判，不要留到第一次 `tools/call`；
3. `capabilities.tools.listChanged` 决定要不要做热重载（`MCPServerInfo.list_changed`）。

### 19.3 鉴权的三个位置选择

| 放哪 | 优点 | 缺点 |
|---|---|---|
| transport 的 env / headers（**推荐**） | 与 MCP 协议解耦，换 server 只改配置 | 配置泄露面在 env |
| `ToolSpec.description` | 无处可放，别这么干 | 会进 prompt，直接泄露 |
| 审计/日志 | 绝对不行 | 一旦泄露就是全量泄露 |

代码里已经有配套：`redact.sanitize` 在写日志前清洗，`mask_secret` 把密钥显示成
`sk-***wxyz`（前缀判厂商、尾 4 位对账）。所以"能把 token 放进 transport"这件事
是有护栏的，不是裸奔。

### 19.4 错误规范化（这段是现成可讲的）

| 情况 | code | 是否重试 |
|---|---|---|
| 名字不在缓存（刷新后仍没有） | `tool_not_found` | 否（模型编的名字，重试也一样） |
| `asyncio.wait_for` 超时 | `tool_timeout` | **否**（重试会让"同一动作执行两次"变默认） |
| 传输层异常 / `isError: true` | `tool_error` | 否（工具自己的问题） |
| JSON-RPC error / 返回体畸形 | `tool_protocol_error` | 否（我们的集成问题，要人看） |

`call_tool` **永不抛异常**：工具失败是正常路径（搜索为空、页面 404 都是常态），
抛异常会逼调用方写 `try/except`，而那个 except 迟早顺手吞掉真正的编程错误。
原始错误体只进日志（可能带内网主机名或上游 HTML）。

---

## 原文第 20 问：MCP server 提供的 resource 和 prompt 应该怎么用

### 20.1 现状：没实现

`MCPTransport` 只有两个方法（`list_tools` / `call_tool`），
`MCPClient` 也只映射 tools。所以今天既读不到 resource，也拿不到 prompt。

### 20.2 为什么没做（这是个设计判断，不是偷懒）

MCP 的 resource/prompt 是"**服务端渲染好的上下文**"，而我们这边的上下文
必须经过 `ContextItem` 的统一治理：谁给预算、谁给信任级、谁给版本号、谁负责编号。
直接塞进 messages 等于绕过整条链路——预算层看不见它，审计记不到来源，
引用面板也没法解释"这条是谁给的"。

### 20.3 要做的话，映射成这样

| MCP 能力 | 内部形态 | 关键约束 |
|---|---|---|
| `resources/list` + `resources/read(uri)` | `ContextItem(kind=TOOL_RESULT, trust=EXTERNAL, source=uri, version=…)` | **必须 external**：这样它永远进不了 system（`allocate` 会抛）；不参与 `[n]` 编号 |
| `prompts/get(name, args)` | 只能产出"用户消息模板"，且要过 `InputGuard` | 不得覆盖 system；外部模板按不可信文本处理（包 untrusted 封皮） |
| `resources/subscribe` | 暂不做 | 订阅是长连接语义，与"每请求装配"冲突 |

三条红线：

1. **resource 不进引用编号**。引用必须指向"我们的知识库块"，
   否则 `[3]` 会指向"某个 server 说什么"，无法核对、无法复算。
2. **prompt 不得进 system**。它是外部文本，进 system 就绕过了注入闸。
3. **审计必须记 `server` + `uri`**。排障时"这条上下文哪来的"必须能回答，
   否则出了问题只能盲猜是检索还是工具。

---

## 原文第 21 问：MCP 和 Skill 在本项目中的设计方案

### 21.1 一句话

**Skill 是"给人看的契约"，MCP 是"给机器用的传输"**。前者定义"什么时候用我、
边界在哪、做完怎么算完"，后者只负责"把 tool 调用送到对端"。

### 21.2 Skill 的六层（`skills/spec.py:172`）

| 层 | 字段 | 硬约束 |
|---|---|---|
| 1 身份 | `name` / `description` | 名字同时是合法工具名；`description ≤ 300` 字符 |
| 2 使命边界 | `mission`（必填） | 必须写"不负责什么"。只写负责什么 = 万能 Skill |
| 3 领域原则 | `principles` | "引用必须给原文"这类硬规矩 |
| 4 执行步骤 | `instructions`（必填） | 模型照这个做 |
| 5 工具依赖 | `parameters` / `entrypoint` / `permissions` / `sandbox` / `stages` | `entrypoint` 必须 `server.tool`；有 entrypoint 就必须声明 permissions |
| 6 完成标准 | `done_criteria` / `verify` / `dangerous` | `dangerous=true` → 走审批门 |

三个"看起来多余、其实救命"的校验：

- **`description ≤ 300`**：渐进式加载只注入这一段，太长就是拿 token 换没用的信息；
- **`mission` 必填**：没有边界的 Skill 会被模型拿去干相邻的活，失败表现是
  "答得很像但不该由它答"——比报错难查十倍；
- **无 `permissions` 却声明 `entrypoint` → 报错**：那是想绕过白名单。

### 21.3 运行时的四道关（`ActionRuntime.run`）

```
幂等（idempotency_key，先于一切校验）
  → 业务规则（BusinessRule / NumericCapRule）
  → 可见面（stages + principal，**先于授权**）
  → 授权（PermissionGuard.check）
  → 审批（仅 dangerous，fail-closed：pending 不执行）
  → 执行（SkillExecutor：沙箱 → 传输 → 截断 → 审计）
  → 验真（VerifySpec：回读状态算不算生效）
```

两个顺序是刻意的：

- **幂等在最前**：重复请求不该重新走审批，也不该因为"这次参数刚好不合规"
  返回一个不同答案；
- **可见面在授权之前**：被可见面裁掉的工具被强行调用，要报
  `tool_not_visible`（模型不该知道它存在）而不是 `forbidden`（主体缺权限）。
  两个码在值班队列里不是同一类工单。

### 21.4 安全模型四件套

| 件 | 拦什么 | 默认值 |
|---|---|---|
| 可见面（`stages` + principal） | 模型"知道不该知道的工具" | 空 stages = 全阶段可见（lint 会提示显式声明） |
| 权限（`permissions`） | 越权调用 | 空 = 纯提示词 Skill，不允许有 entrypoint |
| 沙箱（`write_roots` / `command_prefixes` / `network_allow`） | 写源码、跑任意命令、任意出网 | 工作区 `./.agent_workspace`；命令白名单短；网络**全拒** |
| 审计（`ToolAudit`） | "这个动作谁批的" | 全量记录：action_id / tool / skill / risk / principal / args_digest / approval |

### 21.5 MCP 在这一层的定位

```
SkillSpec.entrypoint = "github.search_repositories"
        │  split(".")
        ▼
clients["github"]  →  MCPClient(StdioTransport(...), server_name="github")
        │
        ▼  call_tool("github__search_repositories", args)
     JSON-RPC → GitHub MCP server
```

所以"接一个新的工具来源"= 加一个 client + 写一份 skill yaml，
**不改执行器、不改 ActionRuntime、不改路由**。这就是为什么值得把内部工具也
"MCP 化"：路径统一了，加东西才是加法。

---

# 场景题

## 场景 1：RAG 召回不准怎么办

**排查顺序：从便宜到贵，每一层都有可看的字段。**

| 步 | 检查什么 | 看哪个字段/文件 | 典型症状 |
|---|---|---|---|
| 1 | 资料到底在不在库里 | 直接查 `rag_chunks` 条数 | 空库跑出"没检索到"——不是召回问题 |
| 2 | 是不是被 ACL 滤掉了 | `diagnostics.filtered_out`、`acl_applied` | 有候选但进 context 的少 |
| 3 | 是不是被预算裁了 | `assembled.dropped_over_budget`、`audit.tokens` | 召回对但模型没看到（**这不是召回问题**） |
| 4 | 召回本身好不好 | `diagnostics.candidates`、`recall_ms` | 候选就少 → 切分/embedding/索引 |
| 5 | 改写是否帮倒忙 | `diagnostics.queries`（多查询） | 改写出同义但跑偏的查询 |
| 6 | 精排有没有开/超时 | `diagnostics.reranked`、`rerank_ms`、`degraded` | 开了但降级 → 实际没精排 |
| 7 | 生成有没有用上 | `audit.candidates` vs `audit.cited` | 引了 1 条 / 候选 8 条 → 提示词问题 |

**这个顺序的意义**：第 2、3 步最容易被误判成"召回不准"，而它们改起来最便宜
（一条配置 / 一个预算），改完往往就够了。

**项目里的实证结论**（可直接引用，但要说清条件）：

- hybrid 的 R@1 `.5714` / R@5 `.8571` 都低于 dense 的 `.7857` / `.9643` → 默认
  `dense`，不迷信融合（`docs/w5/rag-ab-report.md`）；
- 语义改写默认关：p95 15.2s（+94×）、1000 次约 3.55 元，而 R@1 两次分别
  `.7857` / `.7143`，**一次都没比不改好**；
- 精排默认关：20 条候选 p95 5.4s、R@1 `.7143`（dense `.7857`），1 题变好 / 6 题变差。

**优化优先级**：先补语料/修切分（收益最大且可解释）→ 再看 ACL 与预算 →
然后才动 embedding/模型 → 最后才调融合与改写参数。

---

## 场景 2：Agent 为什么会死循环，怎么解决

### 2.1 死循环的四种真实形态

| 形态 | 表现 | 对应的闸 |
|---|---|---|
| 步数耗尽 | 一直在调工具，每步都"有点用" | `max_steps` |
| 同一动作重复 | 同样的工具 + 同样的参数连发 | `idempotency_key`（会被判 `duplicate`）+ 动作窗口预警 |
| 计划重复 | 重新规划出同一个计划 | `max_replans` + `plan_loop` |
| 无进展 | 每步都不同，但目标没靠近 | `no_progress` + Drift 三指标 |

### 2.2 五道机制（本项目都实现了）

```python
AutonomyBudget(max_steps=8, max_tokens=60_000,
               max_cost=Decimal("0.50"), max_seconds=120, max_replans=2)

BudgetMeter.exhausted() -> StopReason | None   # 步数 → token → 成本 → 墙钟，顺序固定
```

1. **预算四维**：`max_steps` 数的是**执行步数**不是模型轮次（一轮发 5 个工具不能算 1 步）；
2. **中止原因是枚举**：`completed / max_steps / max_tokens / max_cost / max_seconds /
   max_replans / plan_loop / no_progress / error`，**没有 `unknown`**——
   一个说不出原因的中止等于这次运行不可归因；
3. **动作窗口**：`REPEAT_ACTION_WINDOW` 内同样的 `(tool, args_digest)` 出现多次 →
   `loop_warning`。**记录必须在幂等检查之前**：否则第二次就被判 `duplicate`，
   这个信号永远不会出现（幂等生效的同时把预警能力关掉了）；
4. **幂等**：防的是"重试导致重复副作用"，不是防循环。两者都要，但别混为一谈；
5. **Drift 三指标**：目标漂移 / 证据丢失 / 重复劳动——事后归因用，不参与实时止损。

### 2.3 一个诚实的设计取舍

`ReActPlanner` 没有重规划，失败只能"再试一次"，试到预算耗尽为止。
**这是刻意保留的缺陷**（它是对照组，补上就不叫朴素 ReAct 了）。
所以"我们怎么防死循环"的答案是**预算 + 环路检测**，不是"我们的规划器很聪明"。

---

## 场景 3：工具调用失败怎么处理

### 3.1 先把"失败"分类

| 失败类型 | 谁的问题 | 处理 | 重试 |
|---|---|---|---|
| `tool_not_found` | 模型编了工具名 | 回给模型"没有这个工具"，让它改 | 否 |
| `tool_timeout` | 对端慢 | 降级（换工具/跳过）+ 上报 | **否**（工具侧重试 = 重复副作用） |
| `tool_error`（`isError:true`） | 工具自己 | 同上 | 否 |
| `tool_protocol_error` | 我们的集成（JSON-RPC 畸形） | 报错、告警，别让模型背锅 | 否 |
| 参数不合规 | 模型 | 按 JSON Schema 拒，附具体哪个字段 | 否 |
| 需要审批但没人批 | 流程 | 返回 `pending`，**不执行**（fail-closed） | 等人 |

### 3.2 三段兜底

```
调用层：超时（8s）+ 错误规范化（4 个 code）+ 结果截断（2000 字符 + 句柄）
   ▼
编排层：预算（步数/token/成本/墙钟）+ 幂等（同一动作不重复执行）+ 降级（换工具/换路径）
   ▼
业务层：审批门（dangerous → pending）+ 行为监控（频率异常告警）+ 人工介入
```

关键点：**每一层只做自己那件事**。工具层不重试（重试归网关），
网关不判业务（业务归 ActionRuntime），编排层不吞异常（要能看见失败）。

### 3.3 已知缺口（面试主动说，比被问出来好）

- 工具结果的句柄是**进程内**的（`ResultTruncator._handles`），
  多副本部署时 `refetch` 可能取不到——接口不变，未来换工具注册表；
- 没有跨进程的任务续跑（`workflow/` 只有接口缝，没接 Temporal）。

---

## 场景 4：上下文太长怎么优化

按"收益 / 风险"排序，不要一上来就换大窗口模型：

| 优先级 | 动作 | 在本项目里的位置 | 收益 | 风险 |
|---|---|---|---|---|
| 1 | 去重 + 预算裁剪 | `assemble`（阈值 0.9） | 高 | 无（可解释） |
| 2 | 工具结果截断 + 句柄 | `ResultTruncator` | 高 | 需要 refetch 能力 |
| 3 | 历史滚动摘要 | `WorkingMemory.compress` | 高 | 摘要有信息损失，要用 drift 监控 |
| 4 | 首尾重排 | `attention_ordering` / `degradation` | 中 | 无 |
| 5 | 槽位重分配 | `ContextBudget` | 中 | 破 `validate` 会直接炸（这是好事） |
| 6 | 换大窗口模型 | `models.yaml` | 低（贵） | 成本线性上涨，延迟上涨 |

**三件不要做的事**：

1. 把所有内容塞进 system（会被当指令 + 注入面最大）；
2. 跨槽位挪用预算（Debug Report 从此不可复算）；
3. 用"更聪明的摘要器"掩盖没做好的裁剪（摘要是有损的，裁剪是无损的）。

---

# 实战题

## 实战 1：给你一个业务，设计 Agent 方案

> 题目要求：需求分析 → 技术方案 → 核心流程 → 异常处理 → 优化方向。
> 这里用本项目真实业务（增长营销内容生产）走一遍，所有组件都能指到代码。

### 1 需求分析

| 问题 | 回答 | 对设计的影响 |
|---|---|---|
| 产物是什么 | 三平台文案（公众号/小红书/知乎）+ 事实型问答 | 两条链路：`/v1/draft`（文案，不需逐句出处）、`/v1/answer`（事实，必须带 `[n]`） |
| 内容从哪来 | 内部知识库（PDF/网页）+ 公开检索 | 检索必须分 ACL；公开检索走工具而不是进知识库 |
| 谁验收 | 市场团队（人） | 高风险动作要审批；文案要有禁用词/调性约束 |
| 错了会怎样 | 编数据 → 品牌风险 | 溯源是硬要求，不能只靠 prompt |
| 多轮吗 | 会追问、会改口（"别说行业领先"） | 需要会话 + 长期记忆 |

**结论**：这不是"一个万能 Agent"，是**两条产物链路 + 一套共享底座**。
先定产物，再定编排——反过来一定过度设计。

### 2 技术方案

```
入口      FastAPI：/v1/answer（事实） /v1/draft（文案） /v1/article（流水线）
  │
编排      Plan-and-Execute（研究类）/ 单次调用（文案类）
  │        预算：8 步 / 60k tok / 0.5 元 / 120s
  │
能力      检索（dense，top_k=8）· 记忆（工作 + 长期）· 工具（search/fetch/workspace）
  │
底层      LLMGateway（缓存 → 路由 → 候选链 → 重试 → 熔断）+ Skills（沙箱 + 审计 + 审批）
```

三条"为什么这么选"：

- **为什么不用 LangGraph**：我需要的守卫（步数/预算累计/环路）必须能看见全局，
  在 LangGraph 里只能寄生在某个节点内部，"这个节点一定会跑到"成了正确性前提；
- **为什么检索与生成不合成一个端点**：合成后要么 slogan 也得标出处，
  要么事实可以不标——两种都糟；
- **为什么记忆默认关（`MEMORY=off`）**：它是实验变量，关掉必须能跑通，
  否则"做对照"这件事本身做不成。

### 3 核心流程（以 `/v1/answer` 为例）

```
鉴权 → 校验 → 惰性装配（检索 + 记忆）→ 检索（改写/召回/ACL/精排）
→ 组装（去重/预算/首尾）→ 编号 → 零召回短路 → 记忆槽位 → 生成
→ 引用体检 → 落地（追加/压缩/提取）→ 响应（text + citations + audit + retrieval）
```

每个箭头的失败语义都在上半场第 9 问的架构图里，这里不重复。

### 4 异常处理

| 异常 | 表现 | 处理 |
|---|---|---|
| 上游 429/5xx/超时 | 网关错误 | 候选链 + 重试 + 熔断；映射 429/502/504 |
| 检索降级 | `degraded=[...]` | 继续回答，但 `audit` 会显示候选变少 |
| 零召回 | `grounded=false` | 固定话术，**不调模型** |
| 记忆不可用 | WARNING | 不带记忆继续（增强不是依赖） |
| 工具超时 | `tool_timeout` | 不重试，让编排层换路径 |
| 高风险动作 | `pending` | fail-closed，等人批 |

### 5 优化方向

1. **成本**：难度分流（简单问题走 cheap 档）+ 语义缓存（阈值 + TTL + 资产版本）；
2. **质量**：引用抽样 → 定位是检索还是生成；不要笼统改 prompt；
3. **延迟**：首 token 打点（`committed` 状态机），区分"冷启动"与"模型慢"；
4. **可运维**：`/health` 报 `rag: cold|ready`；`/v1/memory` 提供记忆运维面。

---

## 实战 2：RAG 效果不好，怎么排查和优化

比场景 1 更细，重点是**怎么证明"不是检索的错"**。

### 1 需求分析

"效果不好"必须先拆成可测的量，否则永远在猜：

| 模糊说法 | 可测的量 | 口径 |
|---|---|---|
| 找不到 | Recall@k（标注集） | 标注段有没有进 top-k |
| 找错了 | Precision@k / 人工抽检 | 进 top-k 的有多少相关 |
| 找到了没用上 | `audit.candidates` vs `audit.cited` | 引用数远小于候选数 = 生成问题 |
| 引错了 | `audit.verdict` 分布 | `off_target` / `unknown_index` 各多少 |
| 答得不对但引对了 | 人工抽检 | **这是语料问题**，不是链路问题 |

### 2 技术方案（排查树）

```
        ┌─ 候选为空？ ──是→ 库空 / embedding 维度错 / ACL 全滤 / 索引未建
        │
召回 ───┼─ 候选少？ ────是→ over-fetch 不够 / 切分过大 / 查询与文档语种不一致
        │
        └─ 候选多但排序差 → 精排没开 / 精排降级 / 融合权重错

        ┌─ 进了 context 但被裁 → 预算太小（改 token_budget，不是改召回）
组装 ───┤
        └─ 编号错乱 → 有人在 assemble 之前编号（顺序被破坏）

生成 ───┬─ 引了不存在的编号 → prompt 没给编号规则 / 模型太小
        └─ 有资料但不引 → strict_claims 没开 / 提示词没要求

评测 ───┴─ 指标波动 → 标注集太小（69/425）/ judge 不一致（kappa .343）
```

### 3 核心流程（一次可复现的排查）

```
1) 固定一个 query（写进 golden）
2) 打印 diagnostics：queries / candidates / filtered_out / degraded / 各段耗时
3) 打印 assemble：dropped_over_budget / duplicate_of
4) 打印 audit：verdict / cited / candidates
5) 对照期望，定位到"哪一层第一次偏离"
```

关键是**逐层打印**，不是"重跑一遍看看"。重跑只会得到另一个不知道原因的结果。

### 4 异常处理

| 现象 | 常见误判 | 真实原因 |
|---|---|---|
| "召回不准" | 换 embedding 模型 | 其实是 ACL 把文档滤了（`filtered_out` 很大） |
| "模型没看资料" | 改 prompt | 其实是预算裁掉了（`dropped_over_budget > 0`） |
| "引用不对" | 换大模型 | 其实编号被重复分配（两处 `build_context`） |
| "效果不稳定" | 调温度 | 标注集太小 / 评测脚本没固定随机 |

### 5 优化方向

1. 语料：去重、补元数据、修切分（**收益最大**）；
2. 索引：确认维度/距离函数与 embedder 一致（`1 - cosine` 不是 `cosine`）；
3. 检索：先 dense 跑满，再考虑 hybrid（本项目实测 hybrid 更差）；
4. 生成：只改"引用规则"部分，不要重写整个 prompt（否则无法归因）；
5. 评测：标注集扩到 ≥200 条再谈"提升 X pp"。

---

## 实战 3：工具调用失败、超时、死循环怎么做异常兜底

### 1 需求分析

三类问题要分开治，因为**归因不同、值班的人不同**：

| 类别 | 归属 | 该做的 |
|---|---|---|
| 失败（返回错误） | 业务/工具 | 让模型看见错误并改路径 |
| 超时（没返回） | 基础设施 | 快速失败 + 降级，**不重试** |
| 死循环（一直在跑） | 编排 | 预算强制中断 + 原因码 |

### 2 技术方案

```
         ┌──────────── 失败 ────────────┐
ToolOutcome(code)                      │
  not_found  → 回模型"无此工具"，让它改
  tool_error → 回模型错误摘要（不回原始堆栈）
  protocol   → 告警，人看
         └────────── 超时 ─────────────┘
  wait_for(8s) → tool_timeout → 编排层换路径 / 跳过

         ┌────────── 死循环 ────────────┐
  预算四维（步数/token/成本/墙钟） → 原因码
  动作窗口（同 tool+args 重复）    → loop_warning
  幂等（idempotency_key）          → duplicate（不重复执行）
  计划重规划次数（max_replans）     → plan_loop
         └──────────────────────────────┘
```

### 3 核心流程

```
模型提议 ToolCall
   ▼
ActionRuntime.run
   ① 记动作（预警）→ ② 幂等命中？→ 返回 duplicate（不执行）
   ③ 业务规则 → ④ 可见面 → ⑤ 授权 → ⑥ 审批（dangerous 才走）
   ⑦ 执行：沙箱检查 → 传输（超时 8s）→ 结果截断 → 审计
   ⑧ 验真：回读状态是否真的生效
   ▼
编排层：charge_tokens / count_step → exhausted() → 原因码 → 停
```

### 4 异常处理（表格化，便于值班）

先分清两层：`ActionResult.status`（动作层：`verified / unverified / failed /
pending / rejected / forbidden / invalid_action / duplicate / tool_not_visible`）
与 `ToolOutcome.code`（工具层：`tool_not_found / tool_timeout / tool_error /
tool_protocol_error / invalid_arguments / permission_denied / sandbox_denied / replay_miss`）。
同一件事在两个层各有名字，值班时看 status，定位时看 code。

| 触发 | status / code | 人要不要介入 | 日志里看什么 |
|---|---|---|---|
| 工具超时 | `failed` / `tool_timeout` | 连续 3 次才要 | `server` / `tool` / `latency_ms` |
| 工具报错 | `failed` / `tool_error` | 否 | `ToolAudit.status` |
| 协议错误 | `failed` / `tool_protocol_error` | **要** | 原始 message（已 sanitize） |
| 参数不合规 | `invalid_action` / `invalid_arguments` | 否 | `args_digest`（**不记原文**） |
| 越权 | `forbidden` / `permission_denied` | 要（可能是攻击） | `missing: [...]` |
| 不可见 | `tool_not_visible` | 要（可能是注入） | `visibility:*` |
| 待审批 | `pending`（**不执行**） | 要 | `approval_request` |
| 重复调用 | `duplicate` | 否 | `duplicate_of`（首次 action_id） |
| 步数/token/成本耗尽 | `partial` + `StopReason` | 否 | `BudgetMeter.snapshot()` |

### 5 优化方向

1. **把"失败"变成正常路径**：`call_tool` 不抛异常、错误用 JSON 结构回给模型，
   模型才有机会换工具（抛异常它只会重复同一个调用）；
2. **超时值按工具分档**：搜索 8s、抓取 15s、写操作 30s——统一 8s 会让慢工具永远失败；
3. **把重试收敛到一处**：网关重试上游，工具层不重试，编排层只做"换路径"；
4. **审计要能回答三个问题**：谁发起的、参数是什么（digest）、谁批的。

---

## 这些答案的边界（面试被追问时主动说）

| 事情 | 现状 | 影响 |
|---|---|---|
| 真实 MCP 传输（stdio/SSE） | 未实现 | 接三方 MCP 必须先写 transport |
| MCP `resources` / `prompts` | 未实现 | 只能当工具用 |
| Temporal / 跨进程续跑 | 未实现，只有接口缝 | 长任务只能在单进程内完成 |
| `cache_read` 计价 | 未接入 `compute_cost` | 命中缓存时成本算高 |
| 工具结果句柄 | 进程内 | 多副本时 `refetch` 可能失效 |
| 引用标注集 | 69 / 425 块 | `precision` 只能看趋势，不能当绝对值 |
| judge 一致性 | kappa 加权 .343 | 只能当基线，不能当门禁 |

---

## 维护说明

改下面任何一处，都要回来改本文对应段落：

- `MCPTransport` 协议加方法（→ 第 19、20 问）
- `MCPClient` 的错误码 / 超时（→ 第 19 问）
- `SkillSpec` 字段或校验（→ 第 21 问）
- `ActionRuntime.run` 的判定顺序（→ 第 18、21 问、实战 3）
- `ContextBudget` 的槽位与默认值（→ 第 16 问）
- `compute_cost` 或 `models.yaml` 的 pricing（→ 第 17 问）
- `AutonomyBudget` / `StopReason`（→ 场景 2、实战 3）

行号会漂，**函数名不会**。
