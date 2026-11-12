# W7 使用指南：在本系统里用上上下文预算与记忆

> 面向三种人：**只想让长会话不爆**的、**要开长期记忆**的、**要在自己链路里
> 复用预算/组装**的。三种接入深度，各占一节，互不依赖。
>
> 设计依据：`docs/w7/context-budget-spec.md`；记忆与知识库的边界：
> `docs/w7/memory-vs-knowledge.md`；实测数字：`docs/w7/context-cost-ab.md`

---

## 0. 一分钟版

```bash
# 1) 记忆要落库（与知识库同一个 pgvector，不同表）
docker compose -f ops/docker-compose.yml up -d pgvector   # 端口 5433，用户/库都是 rag
export PGVECTOR_DSN='postgresql://rag:rag@localhost:5433/rag'

# 2) 打开记忆（默认 off：不配置就是 W6 的行为，一条 SQL 都不多跑）
export MEMORY=governed

# 3) 起服务（产品层带 /v1/answer + /v1/memory 的是 growth-agent），带 session_id 的多轮问答
#    自动带摘要历史 + 记忆注入
.venv/bin/growth-agent        # 或 `uv run growth-agent`；框架自带的 agent-server 只有 /v1/chat
curl -s localhost:8000/v1/answer \
  -H 'content-type: application/json' \
  -d '{"question":"我们品牌的调性是什么？","session_id":"brand-2026"}'
```

响应里多两个字段：

```json
{"memory_keys": ["brand_tone"], "history_messages": 6, "...": "..."}
```

`memory_keys` 空 = 没开记忆或没召回到；`history_messages` = 本轮注入了多少条
历史（摘要 + 最近原文）。**这两个字段就是 Gate 7 两档实验的观测口径**——
不用看日志就知道记忆有没有真的进 prompt。

---

## 1. 配置项（全部有默认值，`MEMORY=off` 时一个都不生效）

| 环境变量 | Settings 字段 | 默认 | 作用 | 调错的症状 |
|---|---|---|---|---|
| `MEMORY` | `memory` | `off` | `off` 关 / `governed` 开 | 忘了设 → 响应 `memory_keys` 恒空 |
| `PGVECTOR_DSN` | `pgvector_dsn` | 无 | 记忆库连接串 | 未设 + `MEMORY=governed` → 启动即 `ConfigError`（不静默降级） |
| `MEMORY_TABLE` | `memory_table` | `agent_memories` | 记忆表名 | 配成 `rag_chunks` 会被**构造期拒绝**（表级隔离） |
| `MEMORY_RECALL_K` | `memory_recall_k` | 3 | 每次召回条数 | 调大 → 挤爆 `memory` 槽位（600 token，约 3–4 条） |
| `MEMORY_MIN_CONFIDENCE` | `memory_min_confidence` | 0.6 | 写入与召回的下限 | 调太低 → 推断类噪声记忆进 prompt |
| `WORKING_MEMORY_TOKEN_THRESHOLD` | `working_memory_token_threshold` | 6000 | 摘要触发阈值 | 调太大 → 摘要来不及，长会话仍增长 |
| `WORKING_MEMORY_SUMMARIZE_AFTER` | `working_memory_summarize_after` | 20 | 或"满 N 条消息"触发 | 同上（两道阈值取先到者） |
| `WORKING_MEMORY_KEEP_RECENT` | `working_memory_keep_recent` | 8 | 视图里保留的最近原文条数 | 太小 → 模型记不住最近 3 轮；太大 → 摘要省下的又被吃掉 |
| `MEMORY_TOOL_RESULT_MAX_CHARS` | `memory_tool_result_max_chars` | 2000 | tool result 截断长度 | 太小 → 截断后信息不足，只能靠 `refetch` |

预算相关的槽位不在环境变量里，而在代码里（`answer.py::_answer_budget`）：
system 400 / retrieved 跟随 `RETRIEVAL_TOKEN_BUDGET` / memory 600 / history 2000 /
tool_results 0。要改预算就改那一处，**不要在别处加"临时上限"**——两份预算
必然对不上。

---

## 2. 接入方式 A：HTTP（不动代码）

### 2.1 `/v1/answer` 带 `session_id`

```bash
# 第一轮：用户给偏好（"记住…" 会触发提取，写入长期记忆）
curl -s localhost:8000/v1/answer -H 'content-type: application/json' -d '{
  "question": "记住：我们品牌调性专业克制，标题不要用「最」「第一」这类词。",
  "session_id": "brand-2026"
}'

# 第二轮：同一个 session_id → 自动带"摘要 + 最近原文"
curl -s localhost:8000/v1/answer -H 'content-type: application/json' -d '{
  "question": "那这次活动标题怎么写？",
  "session_id": "brand-2026"
}' | jq '{text, memory_keys, history_messages}'

# 换个 session_id（跨会话）→ 靠长期记忆召回，而不是靠历史
curl -s localhost:8000/v1/answer -H 'content-type: application/json' -d '{
  "question": "给保温杯写 3 个标题",
  "session_id": "campaign-2026-11"
}' | jq '{text, memory_keys}'
```

三种情况的差别，一句话记住：

- **同 session**：工作记忆（摘要 + 最近原文）起作用，跨轮连贯；
- **换 session**：长期记忆（`memory_keys`）起作用，跨会话记得偏好；
- **不传 `session_id`**：完全不落历史、不注入记忆 = W6 行为（回归基线）。

### 2.2 `/v1/memory` 查看与遗忘

```bash
# 看某个 query 会召回什么（排障"它为什么记得/不记得"）
curl -s 'localhost:8000/v1/memory?query=品牌调性' | jq '.entries[] | {key, value, confidence}'

# 忘掉某个 key（污染治理：先查再删，删的是整条 key）
curl -s -X DELETE localhost:8000/v1/memory \
  -H 'content-type: application/json' -d '{"key":"brand_tone"}' | jq '.deleted'
```

两个路由的失败语义不同，这是刻意的：

- `/v1/answer` 记忆不可用 → **降级**（打 WARNING，照常回答，`memory_keys=[]`）；
- `/v1/memory` 记忆不可用 → **500 `CONFIG_ERROR`**（运维主动查记忆时，静默空列表
  会让人以为"它没记过"，比报错更危险）。

---

## 3. 接入方式 B：在自己的链路里复用预算与组装

框架层不依赖产品层，任何链路都能用（`/v1/chat`、队列 worker、W9 workflow）：

```python
from agent_framework.context import (
    ContextBudget, ContextBuilder, ContextItem, ContextKind, TrustLevel, allocate,
)

budget = ContextBudget(window=8000, system=400, retrieved=3000,
                       memory=600, history=2000, tool_results=0)

items = [
    ContextItem(id="sys", content=system_prompt, kind=ContextKind.SYSTEM,
                source="my_app", trust=TrustLevel.SYSTEM, score=1.0),
    # 历史：score 必须按"新旧"递增，否则裁剪会先裁掉最新的几轮
    *[ContextItem(id=f"h{i}", content=m.content or "", kind=ContextKind.HISTORY,
                  source="session", message=m, score=(i + 1) / len(history))
      for i, m in enumerate(history)],
    # 记忆：trust=USER 且走 memory 槽位，永远不会进 system
    *[ContextItem(id=item.key, content=item.value, kind=ContextKind.MEMORY,
                  source=item.source_session, score=item.confidence,
                  trust=TrustLevel.USER)
      for item in memory_items],
]

allocated = allocate(items, budget)
messages = ContextBuilder().build(allocated, question=question,
                                  retrieved_text=numbered_context)

# 排障：任何一个 id 为什么没进上下文，一句话能问出来
allocated.explain("h3")        # "budget_squeezed:slot=history used=1980 cost=210"
allocated.as_dict()            # 进 trace / 成本看板
```

三条不许动的约束：

1. **`score` 的语义是"槽位内相关性"**，历史按新旧给分，不要全部 0.5；
2. **不要自己算 token**，用 `allocate` 的 `used` 与 `tokenizer` 同一口径；
3. **不要绕过 `allocate` 直接拼 messages**——那样 Debug Report 里就查不到
   "这条为什么没进来"，等于把这一层又退回到 W6 之前。

---

## 4. 接入方式 C：只想要记忆，不要 RAG

`MemoryRuntime` 是产品层的装配点，但你也可以直接用框架的两个对象：

```python
from agent_framework.memory import LongTermMemory, MemoryExtractor, WorkingMemory

# 1) 存储：实现 MemoryStore 协议（name / upsert / search / delete / all_records）
#    框架侧零 ML 依赖，embedding 在 store 实现里做（见 ADR-W7-06）
store = MyMemoryStore(...)          # 生产用 growth_agent.memory.PgVectorMemoryStore
memory = LongTermMemory(store, min_confidence=0.6)

# 2) 写入：候选 → 五道闸（污染 / 置信度 / 幂等 / 冲突顶替 / 原子提交）
outcome = await memory.write(candidates)   # outcome.as_dict() 可进审计日志

# 3) 召回：按语义取活跃记忆（过期 / 被顶替 / 低置信在这里过滤）
items = await memory.recall("品牌调性是什么", k=3)

# 4) 会话内：摘要 + 保留最近原文
working = WorkingMemory(session_store, summarizer=make_gateway_summarizer(gateway))
view = await working.view(session_id)      # [摘要] + 最近 N 条
result = await working.compress(session_id)  # created / updated / skipped
```

写一个自己的 `MemoryStore` 只要 5 个方法，注意两条契约：

- `upsert` **单次调用是一个提交单元**（顶替记录 + 新记录一起成功或失败），
  调用方靠它做"新值 + 顶替旧值"的原子写；
- `search(query: str, top_k)` 收**文本**不收向量（embedding 是你的实现细节），
  与知识库的 `Retriever` 同语义。

---

## 5. 演示与实验（离线可跑）

```bash
# 20 轮长会话：naive 会爆窗、budget 压住但丢早期信息、summary 两头兼顾
.venv/bin/python scripts/demo_long_session.py            # 演示档 window=2000
.venv/bin/python scripts/demo_long_session.py --window 8000   # 生产档（20 轮撞不穿）

# 两档对照实验（真实调用，需要密钥；本机实测约 5 分钟）
.venv/bin/python evals/scripts/context_cost_ab.py \
  --json evals/history/2026-W50-ab-run.json

# 只复算已记录的结果（不调 LLM，CI 用）
.venv/bin/python evals/scripts/context_cost_ab.py --replay evals/history/2026-W50-ab-run.json
```

---

## 6. 运维要点

| 事项 | 做法 | 为什么 |
|---|---|---|
| 建表 | 首次构造 `PgVectorMemoryStore` 时自动 `CREATE TABLE IF NOT EXISTS` + hnsw 索引 | 幂等，多副本同时启动不会冲突 |
| 清理实验数据 | `TRUNCATE TABLE agent_memories` 或 `DELETE /v1/memory` | 实验脚本用独立表 `agent_memories_w7ab`，别动生产表 |
| 删除传播 | `forget()` / `forget_expired()` 一条 SQL 同时删记录与向量 | 不允许"删了记录但向量还在"的半删状态 |
| 多副本 | **工作记忆的摘要状态是进程内的**，多副本各自摘要 | 要一致就把状态挪进 SessionStore（W11 待办） |
| 备份 | 记忆与知识库同库不同表，备份策略一致 | 恢复时不要只恢复一半 |

---

## 7. 排障清单

| 症状 | 先查什么 | 常见原因 |
|---|---|---|
| 响应 `memory_keys` 恒空 | `MEMORY` 是否 `governed`；`/v1/memory?query=…` 有没有召回 | 没开开关 / 置信度闸 0.6 太高 / 表里真没有 |
| "它忘了我刚说的" | `history_messages` 是否为 0 | 没传 `session_id` / 摘要阈值太大导致视图为空 |
| 长会话仍越来越慢 | `payload_tokens` 与 `used` 对照 | `keep_recent` 太大 / 阈值太大 / 历史 item 的 `score` 全相同 |
| 第 6 条资料没进上下文 | `allocated.explain("chunk-id")` | 返回 `budget_squeezed` = 预算问题；`not_in_candidates` = 上游没送进来 |
| 记忆写进去了但召回不到 | 语义相似度 / 用的是不是真 embedder | 实验替身（词面哈希）召回偏保守；生产必须用 bge-m3 |
| 记忆把选题当调性顶替了 | 提取的 `key` 归类 | 见 `extract.py` 的 key 边界规则（调性 ≠ 写作手法） |
| 摘要调用失败 | 日志 `memory.record_turn: 压缩失败` | 正常降级：返回旧摘要，不影响回答；查 cheap 档限流 |

---

## 8. 已知缺口（用之前先知道）

1. **`/v1/draft` 仍是两消息结构**，没接预算与记忆；
2. **工作记忆压缩状态进程内**，多副本不同步；
3. **W6 检索块的裁剪原因尚未并入 Debug Report**（`assemble` 的
   `dropped_over_budget` 是另一套字段），查"第 6 块为什么没进"要翻两块报告；
4. **记忆没有跨租户鉴权**，只有表级隔离 + 单用户单团队（W10）。
