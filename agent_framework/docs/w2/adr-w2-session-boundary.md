# ADR-W2-01 · 会话层与长任务的边界划分

- **状态**：Accepted（W2 Day1 定稿）
- **决策者**：架构（架构文档 §7/§8）+ 实现（session/ 与 server/routes_*）
- **关联**：ADR-003（重试只在 client）、ADR-007（`provider_meta` 必须回传）、ADR-008（committed 状态机）、`docs/architecture.md` §7「W1/W2 明确不做」

## 1. 问题

W1 交付的是一个**单请求直连**的纵向切片：`/v1/chat` 收 `ChatCall` → `LLMGateway` 选路/重试/降级 → 归一化成 `LLMResponse` 或 SSE 帧。这条链路上**没有"历史"这个概念**：每次请求自带全部 messages，服务端无状态。

W2 要加两件事，而它们互相冲突：

1. **多轮对话要能存**（否则前端每次都要重传全量历史，token 成本随轮数平方增长）；
2. **分钟级长任务要能续跑**（调研类任务跨多次 LLM 调用，中途失败不能从头再来）。

冲突点在于：一旦引入"状态"，最自然的想法是"把整条链路都变成一个 workflow"——这正是 `architecture.md` §7 明令禁止的。所以本周必须先写清楚：**哪一段有状态、哪一段必须保持无状态**。

## 2. 决策

三条路径，边界画在"谁拥有编排权"上，而不是"谁调用了 LLM"：

| 请求类型 | 路径 | 谁拥有编排权 | 理由 |
|---|---|---|---|
| `/v1/chat`（流式 / 非流式） | **直连 `LLMGateway`**，会话历史由调用方从 `SessionStore` 取出后拼进 `ChatCall` | 调用方（HTTP 请求生命周期内） | 首 token 延迟是硬指标；ADR-008 的 committed 状态机要求"一旦吐字就不能重放"，而 workflow 的默认语义是"失败就重跑整个 activity"，两者直接冲突 |
| `/v1/tasks`（本周：单次 LLM 调用的异步化） | **进程内队列**（`queue/TaskQueue`）+ 后台消费者 | 队列（跨请求生命周期） | 只解决"HTTP 超时前必须返回"这一件事，不引入编排语义；状态机 `queued → running → succeeded/failed` 三态，落日志带 `trace_id` |
| 分钟级「深度调研」（W9 才成型） | **Temporal workflow**（本周只留接口缝） | workflow（跨进程、可续跑） | 需要 activity 级重试、超时、补偿、断点续跑；这些是 Temporal 的本职，不该在 W2 手搓 |

**一句话**：`/v1/chat` 永不进 Temporal；`SessionStore` 永不调 `LLMGateway`。

## 3. 为什么 `SessionStore` 只存取、不编排

`SessionStore` 的签名被刻意压到三个方法：

```python
async def get_history(self, session_id: str) -> list[Message]: ...
async def append(self, session_id, messages, *, request_id=None) -> bool: ...
async def aclose(self) -> None: ...
```

反例（被明确否决的设计）：给 store 加一个 `chat(session_id, text)` 方法，内部"取历史 → 调 gateway → 写回"。看起来省事，实际会把三个东西焊死：

1. **重试归属**：ADR-003 说重试只在 `LLMGateway` 里。store 一旦调 gateway，就会出现"store 重试 + gateway 重试"两层退避，`retry_count` 语义立刻失真，W11 的成本看板会看到翻倍的 attempt。
2. **幂等语义**：`append` 的幂等键是 `request_id`，代表"这一轮已经写过了"。如果 store 负责调 LLM，那"写回"和"调用"就不是一个原子动作，重试时会出现"消息写了两次、模型只调了一次"。
3. **实现分裂**：InMemory 和 Redis 两个实现要各自保证"取-调-写"的顺序，而单测只覆盖 InMemory —— 生产路径无人验证。

所以编排权留在调用方：HTTP 路由、队列消费者、将来的 workflow activity，各自决定"要不要带历史""要不要落库"。

## 4. 幂等怎么做（D1 的实现约束）

- **幂等键来源三档**：显式入参 → `request_id` contextvar（中间件写入）→ 本地 uuid。
  第三档不是"兜底凑数"：脚本/评测/队列消费者直接调 `append` 时没有请求上下文，缺它就会产出 `request_id=""`，第二次 append 被误判成重复，表现为"多轮对话第二轮丢失"。
- **InMemory**：`dict[session_id, set[request_id]]` + `asyncio.Lock`（单进程内无竞态）。
- **Redis**：`SET session:{sid}:req:{rid} NX EX ttl` 抢标记，抢到才 `RPUSH`。
  **不能**用"`LRANGE` 扫一遍看有没有这个 rid"：那是 O(n)，且两个并发请求会同时得出"没有"，各写一次。`SET NX` 是原子的，天然只有一个赢家。
- **序列化**：`Message.model_dump(mode="json")` 存 list 成员。`provider_meta` 必须原样带上（ADR-007）——丢了它，DeepSeek 的第二轮会 400（W1 已踩过）。
  这里用 `model_dump` 与 Day4 的红线不冲突：红线是"不许把 `LLMResponse.model_dump()` 当 HTTP 响应体"，这是内部存储格式。

## 5. 队列的边界（D2）

`POST /v1/tasks` 只做三件事：校验 → `enqueue` → 返回 `{task_id, status:"queued"}`。**不在 HTTP 请求里跑长任务**：那等于把 HTTP 超时当成任务超时，客户端一断开任务就被取消。

队列满 → `ServerBusyError(retryable=False, retry_after=3.0)` → `503 + Retry-After`。
`retryable=False` 是本周最容易翻车的一行：入口过载如果被 `LLMGateway` 当成可重试，100 并发会被放大成 300 并发，压测必崩（`docs/w2/w2-tasks.md`「最容易返工的 6 处」#1）。

消费者从队列取到 `ChatCall` → 调 `LLMGateway.chat` → 写回结果。消费者**不碰** `SessionStore`：落库是调用方的事，队列只保证"调用被执行一次"（至少一次 + 幂等键由调用方给）。

## 6. 后果

**收益**
- `/v1/chat` 的延迟特性完全不变（W1 基线可直接对比）；
- 会话层零依赖可测（InMemory），Redis 只换实现不换契约；
- W9 接 Temporal 时只需新增 `workflow/`，不改 `session/` 与 `server/routes_chat.py`。

**代价（明确接受）**
- 调用方要自己拼历史：`/v1/chat` 增加了"取历史 + 追加历史"两步。这是有意的——把隐式状态变显式，才能在日志里看到"这次请求带了几轮历史"。
- 队列是进程内的（`InMemoryTaskQueue`）：多副本部署时任务不跨实例。**W2 接受**，因为本周的目的是"HTTP 不再被长任务拖死"，分布式队列（Redis Stream / SQS）留到真正多副本时再换，签名不变。

## 7. 自查清单（review 用）

- [ ] `grep -r "LLMGateway" packages/agent_framework/src/agent_framework/session/` 只应命中 `__init__.py` 的注释，**不得有 import**
- [ ] `grep -r "SessionStore" packages/agent_framework/src/agent_framework/llm/` 应为空
- [ ] `/v1/chat` 路由里没有 workflow / queue 调用
- [ ] 同一 `request_id` 连续 append 两次，`get_history()` 长度不变
- [ ] Redis 实现 `get_history` 后，`provider_meta` 仍在（ADR-007）
- [ ] 满载时 `retry_count` 不因 503 增长
