# A2A 认知笔记：Agent 间协作协议，与 MCP 分层

- 日期：2026-11-12（W9 D4）
- 档位：**认知 + 笔记（默认档）**，不做跨进程 handoff demo。理由见 §6
- 与 W8 的关系：MCP 在 `docs/w8/adr-w8-mcp-boundary.md`，本文只讲它上面那一层
- 相关：`multi_agent/handoff.py`（我们自己的 handoff 消息）

> 本文所有协议事实来自公开规范与综述（a2a-protocol.org、Linux Foundation A2A 项目、
> 2026 年的一份协议综述），**没有一条来自我们自己的实现**——因为按默认档我们没有跑
> A2A 的跨进程 demo。凡是我们自己的判断，都标了「我们的取舍」。

---

## 1. 一句话分层

| | MCP | A2A |
|---|---|---|
| 连接 | Agent ↔ **工具/数据** | Agent ↔ **Agent** |
| 对方是什么 | 无状态工具服务器 | 有状态的对等 Agent |
| 工作单元 | Tool call | **Task**（带生命周期） |
| 发现方式 | `tools/list` | **Agent Card**（`/.well-known/agent-card.json`） |
| 治理 | Anthropic + 社区 | Linux Foundation（2025 年由 Google 捐出） |

一句好记的：**MCP 给 Agent 能力，A2A 给 Agent 同事。**

这两个协议不竞争。真实系统里两者都用：编排器用 A2A 把子任务派给别的 Agent，
每个 Agent 自己再用 MCP 去读库、调 API。

```
用户请求
  ↓
编排器（本项目 W9 的 Supervisor）
  ├─ MCP  → search server          （W8 已实现）
  ├─ MCP  → workspace server       （W8 已实现）
  ├─ A2A  → 外部研究 Agent          （W9 只做认知，未实现）
  └─ A2A  → 外部合规 Agent          （同上）
```

---

## 2. 三个原语

### 2.1 Agent Card —— "我是谁、我会什么"

一份 JSON，发布在 `/.well-known/agent-card.json`，声明身份、endpoint、
skills（每个 skill 带 input/output schema 与模态）、认证方案、协议版本。

两个容易踩的事实：

- **路径变过**：早期草案是 `/.well-known/agent.json`，后来是 `agent-card.json`。
  按旧文章写会 404，而 404 会被误读成"对方没上线"。**发现失败要能区分
  "不存在"与"路径不对"。**
- **卡片可以被签名**（对规范化后的卡片做 JWS）。不验签的卡片等于
  "对方自称是谁"——这正是 §5 里第一条供应链风险的入口。

### 2.2 Task —— 工作单元，而且是一等公民

```
submitted → working → completed
              ├→ input-required →（客户端补料）→ working
              ├→ auth-required  →（客户端补凭证）→ working
              ├→ failed
              └→ canceled
```

把 Task 做成一等公民（而不是一次 RPC 调用）换来三件事：

1. **长任务可挂起**：`input-required` / `auth-required` 让"等人"成为一个
   协议状态，而不是应用层自己发明的轮询；
2. **三种交付形态共用同一个 task id**：同步请求、SSE 流、webhook 推送；
3. **可取消**：`canceled` 是状态而不是"客户端断开连接"。

> **我们的取舍**：`Handoff` 没有状态机，只有 `(from, to, task, artifacts,
> budget_left, step_no)`。这是刻意的——W9 的角色都在同一个进程里、一次
> `await` 就结束，没有"挂起三天等人批"的场景。真正需要状态机的是 W10 的
> HITL，那时 `STATUS_PENDING` 会长成 `input-required` 的样子。
> 现在假装有状态机，只会让 `Handoff` 变成一个没人读的字段堆。

### 2.3 Message / Part / Artifact —— 内容模型

Message 带 role（user / agent）+ 若干 Part；Part 是 `TextPart` / `FilePart` /
`DataPart`；Artifact 是任务的产出，同样由 Part 组成，可内联也可引用。

> **我们的取舍**：`Handoff.artifacts` 传的是 **id**，不是正文。理由与 A2A 里
> "Artifact 可以按引用交付"一致：正文一旦进了消息，handoff 就成了第二条
> 上下文通道，预算按次数估、按长度花，必然对不上。

---

## 3. 与 MCP 的分层判据（怎么判断该用哪个）

问三个问题：

1. **对方会自己做决定吗？** 会 → A2A；只是执行 → MCP。
2. **对方有内部状态吗？** 有（进度、中间产物、待批状态）→ A2A；无 → MCP。
3. **需要跨组织/跨厂商吗？** 需要 → A2A（这也是它存在的理由）；不需要 →
   一个 MCP server 往往就够了。

反过来，**不该用 A2A 的场景**：只是调一个检索接口、只需要一次请求-响应、
对方根本不需要知道任务的全貌。这种场景上 A2A = 给 REST 套一层 JSON-RPC
再加一个 agent card，纯增成本。

---

## 4. 与 W8（MCP）的边界对照

| 我们已有的 | 对应 A2A 的哪一层 | 差在哪 |
|---|---|---|
| `ToolSpec` / `ToolCall` | 无直接对应（A2A 传的是 Message/Part） | A2A 没有"工具"这个概念，只有"技能" |
| `SkillExecutor` 五道门 | 无 | A2A 的鉴权是**协议层**（OAuth/bearer + auth-required 状态），我们的门是**进程内**的 |
| `Handoff` | Task 的极小版 | 缺状态机、缺 cancel、缺 streaming |
| `SharedState` | 无 | A2A 明确不共享内部状态（"协作但不暴露内部推理与记忆"） |
| 录制回放（W8 `replay.py`） | 无对应 | 跨进程后回放要连"对方的行为"一起录，成本高一个量级 |

最后一行是真实成本：**W8 的离线档之所以能跑，因为工具都在进程内、可录制。
一旦 handoff 跨进程，回放就要录对方的完整行为，离线评测的门槛会跳一级。**
这是我们不做 demo 的第二个理由。

---

## 5. 已知的安全缺口（综述里点名的，不是我们猜的）

1. **Agent Card 供应链**：卡片被篡改 → 你连到了冒充者。缓解：验签（JWS）。
2. **Registry poisoning**：发现机制被污染 → 以为在派活给 A，实际是 B。
3. **delegation token 窃取**：A2A 本身**不定义**凭证的作用域/有效期/吊销语义
   （规范里明确把这些留给实现或扩展）。所以"拿到凭证"不等于"拿到授权"。
4. **DataPart 里的 prompt injection**：结构化数据的载荷同样会被对方当指令读。
5. **push notification 重放**：webhook 回调需要独立的防重放校验。
6. **没有标准的委托链**（cryptographic delegation chain）：A 授权 B 代表它去找 C，
   这条链在协议层没有表达。

> **我们的取舍**：W9 所有角色都在同一进程、同一 principal 下，
> 上面六条一条都不适用。**但这也意味着我们没有资格说"我们处理过 A2A 的安全问题"。**
> 验收里必须这么写，否则就是把"没做"说成"做了"。

---

## 6. 为什么不做跨进程 demo（档位决定）

`w9-tasks.md` 的第 4 条：若 W8 解锁了 A2A 深入档才做 demo。W8 的结论是
**P3 不解锁**（见 `evals/history/2026-W51.json` 的 `decisions`）。三条理由：

1. **收益低**：一个 echo agent 的跨进程 handoff 只能证明"HTTP 能通"，
   而这件事实在没有信息量；
2. **成本高**：要起第二个进程、要定 auth、要让离线档仍然可跑（§4 最后一行）；
3. **会挤掉 D5**：D5 的三层评测是本周唯一的证据产出，A2A demo 是演示。

**如果真要做，最小可信版本是**：两个进程、共享的 `Handoff` 结构、
一个 `input-required` 挂起 + 恢复、以及一份"对方 Agent 的行为录制"。
这四件里前三件各半天，第四件是一天——它才是真成本。

---

## 7. 面试怎么讲（30 秒版）

> "MCP 解决 Agent 调工具，A2A 解决 Agent 之间派活，是上下两层，不是竞品。
> A2A 的关键设计是把 Task 做成一等公民——有 `submitted/working/input-required/
> auth-required/completed/failed/canceled` 的生命周期，所以长任务、等人审批、
> 流式进度、取消都能用同一套语义表达。我们 W9 的 handoff 只实现了它的极简版：
> 同一进程、一次 await、没有状态机——因为我们的角色不需要挂起三天。
> 但有一个东西我沿用了它的思路：handoff 传的是产物 id 而不是正文，
> 否则消息会变成第二条上下文通道，预算就对不上了。"

---

## 附：不要把这两件事混在一起

D4 同时做了两件事，它们**没有关系**：

- 引入 `langgraph`：**进程内**的编排框架，管的是"节点之间怎么连"；
- 读 A2A 规范：**跨进程**的协作协议，管的是"两个独立 Agent 怎么派活"。

容易被混起来的原因：两者都有"任务/节点状态机"这个词。但前者管的是
一次运行内的调度，后者管的是跨网络的委托与鉴权。把"我们用了 LangGraph"
当成"我们支持 A2A"，是这一层最常见的误解。
依赖决策见 `adr-w9-langgraph-dependency.md`。
