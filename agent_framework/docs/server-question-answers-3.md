# server/question.md 续答（三）：MCP 工具的注入与选择

> 接 `docs/server-question-answers.md`（原文 1–9 问）与
> `docs/server-question-answers-2.md`（原文 10–21 问、场景 1–4、实战 1–3）。
> 本文只覆盖新增的两条：**原文第 22 问**与**场景 5**。
>
> 两问是同一个主题的两面：第 22 问问"工具怎么进到模型眼前"，
> 场景 5 问"多个来源时怎么决定进哪些"。前者是管道，后者是阀门。

---

## 速查：两问的一句话结论

| 问题 | 一句话结论 | 依据 |
|---|---|---|
| 22. 工具如何注入上下文 | 有**两条**真通路，但**框架都不替调用方拼 prompt**；planner 那条算出了可见面却没人读 | `routes_chat.py` 的 `chat()`、`routes_tools.py:71`、`reasoners.py:340` |
| 场景 5. 多 server 如何确定用哪些 | 只有三道过滤（装配了哪些 client → registry 注册了哪些 → stage/principal 可见面），**没有会话级选择**；归属只靠 `entrypoint` 的点号前缀 | `executor.py:278`、`visibility.py:120` |

---

## 原文第 22 问：在本项目中 MCP 工具是如何被注入到上下文中的

### 22.1 先把"注入"这个词拆开

"工具进了上下文"在实际系统里是两种完全不同的东西，混着讲必然答错：

| 形态 | 长什么样 | 模型怎么用 | 优点 | 代价 |
|---|---|---|---|---|
| **结构化** | 请求体的 `tools` 字段（JSON Schema） | 走厂商的 function calling，参数的 schema 由协议保证 | 参数类型、枚举、必填项都强约束 | 每家格式不同；工具多时 token 涨得快 |
| **文本** | system prompt 里的一段"可用工具：…" | 模型自己按自然语言理解，输出要再解析 | 通用、可控、能写"什么时候不该用" | 解析可能失败；约束是软约束 |

本项目两条都有，但**都不是框架自动完成的**。分开看：

### 22.2 通路一：`/v1/chat` 的 `tools`（唯一真正到模型的结构化通道）

链路很短，短到容易以为"框架帮我做了"：

```python
# server/schemas.py:121  ChatRequest.to_call()
tools=[tool.to_spec() for tool in self.tools] if self.tools else None,
```

```python
# llm/providers/openai_compat.py  build_request()
payload["tools"] = [...]        # 原样翻译成厂商格式
```

`ChatRequest.tools` 是**调用方给的**。框架在这里做的只有"翻译"（内部 `ToolSpec` →
厂商 JSON），不做"发现"。

所以"怎么把 MCP 工具注入"在 `/v1/chat` 这条路上答案是：
**先 `/v1/tools` 拿可见面 → 自己转成 `ToolSpec` → 塞进请求体的 `tools`**。
框架没有一步是"自动从 MCP server 拉到 tools 再塞进请求"。

**为什么不做自动**：`/v1/chat` 是无状态的透传端点（第 14 问）。
一旦它自己去查可见面、自己拼 tools，这个端点就带上了"主体"和"阶段"两个隐式参数，
而这两个参数在契约里根本不存在——同一个请求体在不同人的会话里会得到不同的工具集，
缓存和回放全部失效。

### 22.3 通路二：`/v1/tools` 的 `describe`（文本，但注入方是调用方）

```python
# routes_tools.py:62-71
decision = runtime.surface(principal=principal, stage=stage)
return JSONResponse({
    ...
    "describe": decision.describe_all(),   # ← 只有 name + description
    "skills": registry_snapshot(decision.visible),
})
```

`describe_all()` 的输出长这样（每个工具约 20 token）：

```
web_search: 检索公开网页，返回标题/链接/摘要。适用于行业数据、竞品公开信息。
web_fetch: 抓取指定 URL 的正文并转成纯文本。适用于已知链接的深读。
...
```

源码注释写得很直白：**"调用方可以直接拿它拼进 system prompt"**。
也就是说，注入动作在**调用方**，不在框架。

这个划分是刻意的，理由是**措辞权**：

- "可用工具"这句话放在 system prompt 的第几段、用什么语气、
  要不要加"先检索再抓正文"这类流程约束——全是产品措辞，不是框架职责；
- 第 16 问的 `ContextBudget` 槽位表里**没有"工具清单"这一项**。
  框架不知道该给工具描述留多少 token，它只知道怎么算总数。

一个可量化的证据（`docs/w8/w8-acceptance.md` #8）：6 个内置 Skill
**全量注入 `instructions` 是 2778 字符，只注入 `name + description` 是 531 字符**
（19.1%）。渐进式加载的收益就来自这里——先给 19%，模型选中某个工具之后
再去 `/v1/tools/{name}` 取完整正文。

### 22.4 通路三：planner 的 `visible_tools`——**算了，但没人读**

这是本文最需要说清楚的一处，因为它看起来接线了。

```python
# planner/base.py:439
def visible_tools(self) -> tuple[str, ...]:
    decision = self._executor.surface(principal=self._principal, stage=self._stage)
    return decision.names()

# planner/base.py:680  _state()
visible_tools=tuple(sorted(self._visible())),   # ← 填进 AgentState
```

到 `AgentState` 为止都是对的。然后：

```python
# planner/reasoners.py:340  LLMReasoner.next_action
async def next_action(self, state: AgentState) -> Decision:
    body = (
        f"任务：{state.task}\n已做：{list(state.done_goals)}\n"
        f"观察：{list(state.observations)}"
    )
    raw = await self._ask(ACT_SYSTEM, body)
```

**`state.visible_tools` 没有被拼进 body，`ACT_SYSTEM` 里也没有工具清单。**

`ACT_SYSTEM` 全文就三句话：

```
你是执行器。给定当前步骤与已有观察，决定下一步动作。
只输出 JSON：{"action":"tool","tool":"工具名","arguments":{}} 或
{"action":"finish","text":"这一步的结论"}
```

而 `propose_plan` 的 body 里写了一句 `"最多 {max_steps} 步。可用工具见 system。"`——
**system 里没有工具**。这句是悬空的。

**后果**：模型输出 `tool` 字段时是在猜名字。猜错不会崩，会在
`SkillExecutor.execute` 的第 1 步变成 `CODE_NOT_FOUND`（"没有名为 X 的 Skill"），
于是这一轮白跑，`StepRecord.status = failed`，ReAct 循环再试一次——**同一个错误重复到预算耗尽**。

**与注释的冲突**：`StepRunner.visible_tools()` 的 docstring 写：

> 当前可见面。规划器在**提议之前**就据此裁剪，而不是等执行被拒。

实际顺序是"提议 → 执行时才拒"。可见面在执行层有效（`executor.py:210` 处
`if not decision.is_visible(...)` 返回 `tool_not_visible`），但在**提议阶段没有生效**。

**这算不算 bug**：从安全角度不算——越权和不可见的工具在执行层被拦住了，
`tool_not_visible` 也是单独的安全事件码。从效果角度算——可见面本该"省 token + 降选错率"，
现在只剩"执行时拦截"，前两个收益没兑现。

补它要动的范围很小（`next_action` 里拼一行 `f"可用工具：{list(state.visible_tools)}"`），
但会改变 planner 的行为基线，`evals/history/` 里 W9 的 ReAct / P&E 对比数字需要重跑。
所以本文只记录现象，不动代码。

### 22.5 一条 MCP 工具从 server 到模型的完整路径

八步，最后一步才是模型看见：

| # | 位置 | 做了什么 |
|---|---|---|
| 1 | `apps/growth_agent/tools.py:130` | `clients = {"search": SearchServer().client(), ...}`——装配期固定 server 清单 |
| 2 | `skills/mcp/client.py:211` | `list_tools()` → `ToolSpec`，名字加 `server__` 前缀（`_to_spec`） |
| 3 | —— | **到此为止**：结果不写回 `SkillRegistry`（生产代码里 `list_tools()` 只在 `call_tool` 懒填名字映射时被调，见 `client.py:145`） |
| 4 | `skills/library.py:61` | 注册表里是**手写**的 `SkillSpec`，`entrypoint="search.search_web"` 把两边对上 |
| 5 | `skills/visibility.py:120` | `surface(stage, principal)` 按阶段与主体粗筛 |
| 6 | `routes_tools.py:71` | `describe_all()` → `/v1/tools` 的 `describe` 字段，交给调用方 |
| 7 | 模型侧 | 输出 `tool_calls`（`/v1/chat`）或 `Decision`（planner） |
| 8 | `executor.py:278-292` | `entrypoint.partition(".")` → `internal_name(server, tool)` → `client.call_tool()` |

**第 3 步是关键缺口**：`list_tools()` 的返回值从来没有被"注册"。所以
"MCP server 新增了一个工具，框架自动就能用"——**不成立**。
要让新工具可用，得在 registry 里加一条 `SkillSpec`（写 YAML 或加进内置库）。

**第 4 步带出两套 schema 并存**：

| 维度 | `SkillSpec.parameters` | `ToolSpec.parameters` |
|---|---|---|
| 谁写的 | 人手写（YAML / 内置库） | server 返回，`list_tools()` 解析 |
| 用在哪 | A14 参数校验（`validate_arguments`） | 给模型看的 JSON Schema |
| 不一致会怎样 | server 加了必填参数而手写那份没加 → 校验放行 | 调用被 server 拒（`invalid_params`） |

这两份不共用是有理由的：`SkillSpec.parameters` 是**契约**（review 得到、能进 git diff），
`ToolSpec.parameters` 是**上游事实**（随时会变）。但代价是**它们会漂**——
这是接真 MCP 之后第一个要处理的运维问题。

### 22.6 三个缺口（主动说出来）

| 缺口 | 现象 | 补它要动哪 |
|---|---|---|
| `AgentState.visible_tools` 无消费者 | 可见面只在执行层生效，提议阶段模型看不见工具 | `planner/reasoners.py` 的 `next_action` / `propose_plan` 各拼一行；改完要重跑 W9 对比 |
| `list_tools()` 不写回 registry | 上游加工具，框架不会自动认 | `SkillRegistry` 加 `register_from_client()`；要决定"手写 Skill 与上游 Tool 冲突时谁赢" |
| `MCPServerInfo` 空结构 | 没有握手信息（协议版本、能力位） | 先要真传输（`MCPTransport` 只有内存版与回放版），见原文第 19 问 |

---

## 场景 5：有多个 mcpServer 时，如何确定当前会话用到哪些工具

### 5.1 四道过滤，按发生时间排

| 层 | 时机 | 依据 | 裁掉之后的表现 |
|---|---|---|---|
| ① 装配 | 进程启动 | `clients` 字典的 key（`tools.py:130`） | `entrypoint` 前缀找不到 client → `CODE_NOT_FOUND`，detail 是"server 'x' 未装配" |
| ② 注册 | 加载/热重载 | `SkillRegistry` 里有哪些 `SkillSpec` | `CODE_NOT_FOUND`，detail 是"没有名为 x 的 Skill" |
| ③ 可见面 | 每次请求 | `stage` + `principal`（`visibility.py:120`） | `tool_not_visible` + `notes=("visibility:stage"/"visibility:principal",)` |
| ④ 执行复核 | 每次调用 | 同上，`executor.py:210` 再算一遍 | 同 ③（防御"prompt 里的清单与执行层不一致"） |

**为什么 ③④ 要算两遍**：③ 给的是"模型看见什么"，④ 保证"即使它调了也拦住"。
两者用同一个纯函数，所以不会给出不同答案；但如果 prompt 是调用方自己拼的
（见 22.3），就**可能**拼错——④ 是最后一道。

### 5.2 "这个工具属于哪个 server"只有一个事实来源

```python
# executor.py:278
server, _, tool = skill.entrypoint.partition(".")
```

`entrypoint` 写成 `"search.search_web"`，点号左边是 server 名，右边是 server 侧原始工具名。
拼成内部名时：

```python
# mcp/types.py:61  internal_name()
f"{head}{NAMESPACE_SEPARATOR}{tail}"    # → "search__search_web"
```

**`server__tool` 不是独立字段，是 `entrypoint` 算出来的。**
这是刻意的：如果一个工具同时有"entrypoint 字符串"和"内部名字段"，
迟早会出现"改了一个忘了另一个"，而症状是"调用打到了一个不存在的工具名"。

顺带一个不显眼但必要的细节：内部名有 **64 字符上限**（多数厂商的 tool name 上限），
超长时 **截 server 前缀、不截 tool**（`types.py:61` 的 docstring）——
tool 名是模型用来选工具的，截一半会让它拿着 `search_web_fetch_cont` 去猜；
server 前缀截了只是少一截归属信息。

### 5.3 多个 server 有同名工具怎么办

| 层级 | 名字 | 会不会撞 |
|---|---|---|
| 上游 `ToolSpec` | `search__search_web` / `github__search` | 不会，前缀天然隔离 |
| 内部 `SkillSpec.name` | `web_search` / `publish_post` | **会**，全局唯一 |

`SkillRegistry.register()` 重名时直接抛 `SkillSpecError`（`loader.py:81`），
不是静默覆盖：

```
Skill 重名：'web_search' 已存在（来源 builtin），新来源 skills/github.yaml。
要覆盖请显式 replace=True
```

这是对的失败方式——**装配期炸掉，而不是运行期让某个工具凭空消失**。

所以"两个 server 都有 `search`"这个场景在本项目里**撞在 Skill 名上，不撞在工具名上**：
你给 GitHub 的 search 起名 `github_search` 就行，内部名仍是 `github__search`，
两者互不干扰。

### 5.4 想做"按会话选工具"，今天只有三种办法

**没有会话级机制**。`VisibilityPolicy.surface()` 的签名只有 `(skills, stage, principal)`，
`session_id` 根本没进来。三条可行路径，按侵入性从低到高：

| 方案 | 怎么做 | 代价 |
|---|---|---|
| **a. 调用方过滤 `tools`** | 先 `/v1/tools?stage=...` 拿可见面，自己按会话需求再筛一层，塞进 `ChatRequest.tools` | 零框架改动；但 `/v1/chat` 是透传端点，筛错了框架不知道 |
| **b. 用 `stages` 标注** | 给每个 `SkillSpec` 的 `stages` 加上它适用的阶段，靠 `stage` 参数切换 | 已有机制；粒度是"阶段"，不是"会话"——同一阶段的会话看到的完全一样 |
| **c. 给可见面加会话维度** | `surface(skills, stage, principal, session=...)` + 一张 `session → allowed` 映射 | 要改签名、改所有调用点（`executor` / `planner` / `routes_tools`）、并回答"会话过期后映射怎么办" |

**为什么不建议 c**：可见面的两个维度（阶段、主体）都是**稳定的**——
阶段由任务决定，主体由鉴权决定。会话是**易变的**（一次对话可能跨调研和成文）。
把易变维度放进权限计算，结果就是"同一个工具在对话第 3 轮可见、第 7 轮不可见"，
而模型在第 7 轮仍然记得它（历史里还在），于是稳定产出 `tool_not_visible`。
这是可见面设计里最不该碰的一类改动。

如果真实需求是"这个用户只买了 GitHub 的连接器"，那它是**主体**维度
（`principal.groups` 里加一个 group），不是会话维度——用 ②③ 就够，不用改任何签名。

### 5.5 一条排查命令

```bash
curl -s 'localhost:8100/v1/tools?stage=research' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("visible:", d["visible"]); print("hidden:", d["hidden"]); print("ratio:", d["ratio"])'
```

输出里三件事直接可读：

- `visible` / `hidden`：哪些工具在这个 (stage, principal) 下可见；
- `hidden[].reason`：只可能是 `stage` 或 `principal`——**这两个字符串就是归因结论**；
- `ratio`：可见比例，全集为 0 时返回 `0.0`（不是 1.0：没有工具不等于收敛达标）。

如果 `hidden` 里没有你期望的工具，却报 `tool_not_visible`，那问题在 ① 或 ②，
不在这道闸——去看 `runtime.registry.errors`（`/v1/tools` 响应里的 `reload_errors`）
和 `clients` 的 key 是否与 `entrypoint` 前缀逐字一致。

---

## 两问共同的边界

| 边界 | 说明 |
|---|---|
| 工具不进 prompt 是设计，不是遗漏 | 措辞权在调用方；框架只算可见面、只做翻译 |
| 可见面在提议阶段无效 | `visible_tools` 算了没人读，实际生效点是执行层（22.4） |
| 动态发现只到 client 层 | `list_tools()` 不写回 registry，上游加工具要手动加 `SkillSpec` |
| 没有会话级工具选择 | 只有 stage / principal 两个维度；按会话选要走调用方过滤 |
| 无真传输 | `MCPTransport` 只有内存版与回放版（原文第 19 问），所以"多 server"目前都是进程内装配 |

---

## 维护说明

改下面任何一处，都要回来改本文对应段落：

- `ChatRequest.to_call()` 的 `tools` 处理（→ 22.2）
- `VisibilityDecision.describe_all()` 的输出格式（→ 22.3、5.1）
- `LLMReasoner.next_action` / `ACT_SYSTEM`（→ 22.4，**改了就作废这一节**）
- `SkillSpec.entrypoint` 的分隔符（→ 5.2，`.partition(".")` 与 `internal_name` 必须同改）
- `SkillRegistry.register()` 的重名策略（→ 5.3）
- `VisibilityPolicy.surface()` 的签名（→ 5.4）

行号会漂，**函数名不会**。
