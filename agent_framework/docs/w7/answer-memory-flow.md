# `/v1/answer` 与 `/v1/memory` 数据流（W6 / W7）

> 一句话：`/v1/answer` 是「检索资料 + 召回记忆 → 生成一条带 `[n]` 出处的答案」；
> `/v1/memory` 是「看记忆库 / 删一条记忆」。两者共用同一个 `MemoryRuntime`
> 和同一张 `agent_memories` 表，但**故意不共用失败语义**。
>
> 框架侧（gateway 内部：缓存 / 路由 / 候选链 / 重试 / 熔断）见
> `packages/agent_framework/data_flow.md`。本文只讲产品层这两条链路，
> 以及它们跟框架的接缝。

---

## 1. `/v1/answer` —— 八步，顺序不能换

```
  POST /v1/answer  {question, top_k?, token_budget?, session_id?}
        │
        │  RequestID（最外层）→ APIKeyAuth → 校验（1–2000 字）
        ▼
  ① routes_answer.answer()                        routes_answer.py:115
        │   settings = app.state.settings
        ▼
  ② await ensure_answerer(app)                    runtime.py:73
        │   命中 app.state.answerer？→ 直接返回
        │   否 → app.state.rag_lock 内二次检查
        │        asyncio.to_thread(_build_answerer_blocking)
        │          build_runtime()（加载 bge-m3，18–20s）
        │          MEMORY=governed → build_memory_runtime()（复用同一份 embedder）
        ▼
  ③ pipeline.run(question, top_k, acl_check)      retrieval/pipeline.py:164
        │   改写 → 召回（over-fetch）→ ACL 过滤 → 融合 → 精排
        │   产出 chunks + diagnostics(degraded / 各段耗时)
        ▼
  ④ assemble(...)                                 context_assemble.py:233
        │   去冗余 → 预算裁剪 → 首尾重排   ← 必须在编号之前
        ▼
  ⑤ build_context(assembled.chunks)               knowledge/citation.py:170
        │   编号 [1..n]，产出 index → Citation   ← 唯一决定编号的地方
        ▼
  ⑥ context.citations 为空？
        │   是 → NO_CONTEXT_ANSWER，不调模型，grounded=False   answer.py:208
        ▼ 否
  ⑦ 记忆槽位 + 消息组装                            answer.py:226 / 279
        │   recall_for_query(question)  → 最多 3 条，置信度 ≥0.6
        │   session_view(session_id)    → [摘要] + 最近原文
        │   无记忆无历史 → 退回 W6 的两条消息结构（回归基线不变）
        ▼
  ⑧ gateway.chat(ChatCall)                        llm/client.py
        │   缓存 → 路由 → 候选链 → 重试 → adapter（熔断包在外层）
        ▼
  ⑨ 落地 + 体检                                    answer.py:260 / 268
        │   record_turn：append → compress → 显式指令才提取（见第 4 节）
        │   audit_citations(text, context) → verdict / reasons
        ▼
  AnswerResponse {request_id, text, model, grounded, citations,
                  audit, retrieval, conflict, memory_keys, history_messages}
```

九步里有一步不是「步骤」而是**短路**（⑥），这也是最容易漏讲的一步：
零召回时模型根本没被调用，`model` 是空串，`grounded=false`。

| 步骤 | 谁负责 | 失败会怎样 |
|---|---|---|
| ① 校验 | pydantic | `question` 空/超长 → **400 `BAD_REQUEST`**（不是 422） |
| ② 装配 | `ensure_answerer` | 缺 torch → 500 `CONFIG_ERROR`，提示 `uv sync --extra local` |
| ③ 检索 | `RetrievalPipeline` | 各段独立降级，`degraded` 里记名字，**不抛** |
| ④ 组装 | `assemble` | 预算内裁掉，`dropped_over_budget` 计数 |
| ⑤ 编号 | `build_context` | 无 |
| ⑥ 零召回 | `answer.py` | 固定话术，`diagnostics` 保留（归因指向检索层） |
| ⑦ 记忆 | `MemoryRuntime` | 召回失败只记 WARNING，返回 `[]`，**不挡回答** |
| ⑧ 生成 | `LLMGateway` | 上游故障按 `ERROR_HTTP` 映射 502/504/429 |
| ⑨ 体检 | `audit_citations` | 不抛；结果进 `audit.verdict` |

### 1.1 三个「顺序不能换」

1. **`assemble` 在 `build_context` 之前。**
   编号一旦发出就不能再裁：裁掉 `[3]` 之后答案里的 `[3]` 就指向了空。
   所以先按预算决定「哪些段进上下文」，再给这些段编号。
2. **ACL 在精排之前**（`pipeline.py:200` 的注释）。
   - 算力：本机精排 20 条要 4.7s，其中一半无权查看就是纯浪费；
   - 侧信道：放在精排之后，一个毫无权限的调用方也能从 p95 变化推断
     「库里有多少他看不到的文档」。
3. **零召回在调模型之前。**
   没有 context 时让模型自由发挥，等于把幻觉写进溯源链路，
   而抽样时还会被记成「引用缺失 1 条」——归因跑去了生成层，
   实际该改的是语料或检索。

---

## 2. 两种 `no_citation`：别都当成引用问题

`audit_citations` 的 `verdict` 有四种取值（`citation.py:395`）：
`unknown_index` / `source_missing` / `no_citation` / `off_target`，全对才是 `ok`。
但零召回时也会给出 `no_citation`——这两件事的归因完全不同：

| 场景 | `grounded` | `audit.verdict` | 真正的问题在哪 | 该改什么 |
|---|---|---|---|---|
| 检索没召回 | `false` | `no_citation` | 检索层 / 语料 | 补语料、查 embedding、看 `retrieval.degraded` |
| 召回到了，模型没标 | `true` | `no_citation` | 生成层 | 改 prompt、换模型、查 `audit.tokens` 是否被裁太狠 |

判断只需要一个字段：**先看 `grounded`，再看 `verdict`**。
只看 `verdict != ok` 就一律归因成「引用写错了」，是最常见的误判。

另外两个字段在运行时**恒为空**，别拿它们当指标：

- `audit.unmatched`：运行时没有 ground truth，`expected=None`，所以永空；
- `audit.precision`：同上，为 `None`（响应里不出现）。
  两者只有抽样脚本 `evals/scripts/citation_sample.py` 才有值——
  共用一套 `CitationAudit` 结构，不是让评测另造一个 DTO。

---

## 3. 记忆在 `/v1/answer` 里出现三次

| 出现点 | 代码 | 进哪个槽位 | 有没有 `[n]` | 失败语义 |
|---|---|---|---|---|
| **召回**（长期记忆） | `answer.py:233` | `memory`（600 token） | 没有，也**不允许**引 | WARNING + `[]` |
| **会话视图**（工作记忆） | `answer.py:235` | `history`（2000 token） | 没有 | WARNING + `[]` |
| **落地**（写会话 + 提取） | `answer.py:261` | — | — | 只记日志，不影响响应 |

三条都是「增强」，不是「依赖」。`MEMORY=off` 是合法配置
（W7 两档实验的对照组），关掉之后回答链路必须照常跑通。

记忆块的信任级是 `TrustLevel.USER`，**不进 system**（`answer.py:319`）：
混进 system 会被模型当成系统指令，这正是 W10 注入防护要挡的那件事，
只是提前在 Context 层立住了。

---

## 4. 记忆写入：`record_turn` 的三件事 + 五道闸

```
  answer 已拿到 text
        │  memory is not None  且  session_id 存在  且  MEMORY=governed
        ▼
  ① working._store.append(session_id, [user, assistant], request_id)
        │   幂等键 = request_id：重试 / 前端双击不会把历史翻倍
        ▼
  ② working.compress(session_id)
        │   可压缩区 = 历史[已覆盖 : -keep_recent(8)]
        │   触发：pending tokens > 6000  或  未压缩条数 > 20
        │   未达阈值 → SKIPPED（幂等，每轮都调不会重复压）
        ▼
  ③ is_explicit_instruction(question)？
        │   字面正则（记住 / 以后要 / 从今往后 / 务必 / 禁止使用）
        │   否 → 结束（这一步就是成本闸：不命中的轮次零 LLM 调用）
        ▼ 是
  ④ extractor.extract（cheap 档，temperature=0，max_tokens=500，≤5 条）
        │   key 白名单：brand_tone / forbidden_words / past_topics
        │   输出解析失败 → 0 条候选，**不抛**
        ▼
  ⑤ long_term.write(candidates)                   long_term.py:231
```

五道闸，任一条不过就换去向，全部记进 `WriteOutcome.actions`：

| 闸 | 判据 | 去向 | 为什么排这个位置 |
|---|---|---|---|
| 1 信任 | `trust=external` 且未被用户确认 | `rejected` | 排在幂等之前：被拒的候选不该占用 dedupe 位 |
| 2 置信度 | `confidence < 0.6`（`MEMORY_MIN_CONFIDENCE`） | `rejected` | 推断出来的偏好比"记住…"更不可靠，阈值一刀切 |
| 3 幂等 | `dedupe_hash` 命中已有活跃记录 | `duplicate` | 同一句偏好说十遍不该存十条 |
| 4 冲突 | 同 `key`、不同 `value` | `superseded` | 旧记录**不物理删**，只写 `superseded_by`——审计要能回放"以前是什么" |
| 5 提交 | `store.upsert(to_write)` | `written` | 单事务：新记录与被顶替记录一起提交或一起回滚，不留半条 |

召回侧（`long_term.recall`，`long_term.py:306`）与写入侧的闸**不是同一套**：

```
  store.search(query, top_k=max(k*3, k))     ← over-fetch，因为下面要过滤
        │
        ├─ alive_at(now)      = active 且未过期（TTL 按 key 分档）
        ├─ confidence >= 0.6
        └─ 取前 k（默认 3）条 → MemoryItem
```

过滤放在 `LongTermMemory` 而不是 store 侧：TTL、置信度是**记忆策略**，
属于框架；store 只负责存取。换成别的存储实现时，策略不该跟着变。

---

## 5. `/v1/memory` —— GET 看，DELETE 删

```
  GET  /v1/memory?query=品牌调性
        │
        ▼
  await ensure_memory(app)                        runtime.py:110
        │  app.state.memory 已是 MemoryRuntime？→ 直接用
        │  否 → await ensure_answerer(app)   ← 注意：会连带装配整条回答链路
        │        （embedder 在检索运行时里，没有它就召不回记忆）
        │  仍为 None（MEMORY=off / sparse 无 embedder）→ ConfigError（500）
        ▼
  long_term.recall(query, k=memory_recall_k)      routes_memory.py:55
        │  store.search → 过滤 active / TTL / 置信度 → 前 k 条
        ▼
  {request_id, query, entries:[{key, value, confidence,
                                source_session, expires_at}]}


  DELETE /v1/memory  {"key": "brand_tone"}
        │
        ▼
  await ensure_memory(app)                        ← 同上，可能触发冷启动
        ▼
  long_term.forget(lambda r, now: r.key == key)   routes_memory.py:84
        │  all_records() 里挑 active 且 key 相同的 → store.delete(ids)
        │  物理删除（连向量一起），与 superseded 语义相反
        ▼
  {request_id, deleted: n}
```

| 维度 | `/v1/answer` | `/v1/memory` |
|---|---|---|
| 会调模型吗 | 会（除零召回） | **不会** |
| 会走检索流水线吗 | 会（改写/召回/精排） | **不会**，直接查记忆表 |
| 会写会话吗 | 有 `session_id` 时写 | **不写** |
| 记忆不可用 | 降级成「不带记忆」，WARNING | **500 `CONFIG_ERROR`** |
| 为什么 | 记忆是增强，不能挡回答 | 运维主动查记忆时，「静默空列表」比报错更危险——会让人以为没记过 |

DELETE 的删除范围有个容易漏的点：`forget` 只挑 `record.active` 的，
所以被顶替的旧记录（`superseded_by != None`）**留在表里**。
「删掉这条记忆」删的是当前生效那条，历史审计痕迹还在——
这是刻意的，不是漏删。

---

## 6. 存储：同库、不同表

```
  PostgreSQL
  ├── rag_chunks          ← 知识库（W5/W6）：文档段 + 页码 + ACL
  └── agent_memories      ← 记忆（W7）：key/value/confidence/TTL/superseded_by
        HNSW 向量索引（cosine），key 二级索引
```

隔离放在**表级**，不是靠 metadata 字段区分：`PgVectorMemoryStore.__init__`
里硬挡住 `table == "rag_chunks"`（`memory.py:140`）。配成同一张表不会立刻
报错，症状是「引用里出现了一条谁都没写过的资料」——记忆项混进了检索候选，
编号指到了记忆上。

Embedder 是**借来**的：`build_memory_runtime` 复用检索链路已加载的 bge-m3
（`runtime.py:48`），不再实例化第二份。sparse 模式没有 embedder，
所以 `MEMORY=governed` + `RETRIEVAL_MODE=sparse` 时记忆静默关闭并打 WARNING——
这是配置组合问题，不是 bug。

---

## 7. 五处最容易答错

1. **`GET /v1/memory` 可能等 18–20 秒。**
   因为它会 `ensure_answerer` 装配整条检索链路。运维第一次查记忆时
   以为"记忆库慢"，实际是模型冷启动。`/health` 的 `rag: cold|ready`
   就是用来区分这件事的。

2. **`MEMORY=off` 时 `/v1/memory` 是 500，不是空列表。**
   空列表会让运维以为"这个用户没记过东西"，实际是没开。
   这也是为什么 `ensure_memory` 的报错信息里带了当前 `MEMORY=` 的值。

3. **`/v1/answer` 写的会话和框架 `app.state.session_store` 不是同一个实例。**
   `build_memory_runtime(settings, gateway, embedder=…)` 没传 `session_store`，
   于是自己 `build_session_store(settings)` 建了一个（`memory.py:389`）。
   框架侧那份由 `server` 路由装配，至今没有消费方。**不要假设两者共享历史。**

4. **冲突提示的位置：注释与实现不一致。**
   `answer.py:221` 的注释写"冲突提示放在问题之前"，实现是把 `note`
   追加到 `user_content` 末尾（`render_prompt` 把「问题：…」放在最后），
   也就是**问题之后**。影响很小，但两处说法必须统一，否则下次改的人
   会照注释去调顺序。

5. **`record_turn` 用的是 `working._store.append`（私有符号）。**
   `memory.py:301` 带 `# noqa: SLF001 - 同包内部接线`。这是当前唯一的
   "借私有符号"点，改 `WorkingMemory` 的内部结构时要一起看。

---

## 8. 复现（产品服务在 :8100）

```bash
# 1) 无记忆的单轮问答：应看到 grounded=true、citations 非空、audit=ok
curl -s localhost:8100/v1/answer -H 'content-type: application/json' \
  -d '{"question":"MCP 协议最近有什么进展"}' | jq '{text, grounded, verdict: .audit.verdict}'

# 2) 带会话的多轮：memory_keys / history_messages 用来核对"记忆真的进 prompt 了吗"
curl -s localhost:8100/v1/answer -H 'content-type: application/json' \
  -d '{"question":"记住：以后文案不要用行业领先","session_id":"s-1"}' \
  | jq '{memory_keys, history_messages}'

# 3) 查记忆（MEMORY=governed 才有；否则 500 带 "记忆未开启"）
curl -s 'localhost:8100/v1/memory?query=品牌调性' | jq

# 4) 删一条（物理删，返回 deleted 条数）
curl -s -X DELETE localhost:8100/v1/memory \
  -H 'content-type: application/json' -d '{"key":"brand_tone"}' | jq
```

期望：`MEMORY=off`（默认）时只有第 1 条能跑；第 2 条会返回
`memory_keys=[]`、`history_messages=0`——不是坏了，是没开。

---

## 维护说明

改下面任何一处，都要回来改本文：

- 中间件顺序（`app.py:100-101`）——先鉴权后 RequestID，RequestID 在最外层
- `KnowledgeAnswerer.answer` 的步骤顺序（尤其 `assemble` 与 `build_context` 的先后）
- `_answer_budget` 的槽位切分（400 / retrieved / 600 / 2000）
- `record_turn` 的三步与 `LongTermMemory.write` 的闸门
- `ensure_memory` 的"先 ensure_answerer"这条依赖
- 记忆表名与知识库表名的隔离断言

行号会漂，**函数名比行号可靠**。
