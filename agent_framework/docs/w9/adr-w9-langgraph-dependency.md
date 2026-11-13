# ADR-W9-003：把 langgraph 作为可选 extra，且不替换主实现

- 状态：已采纳（W9 D4）
- 日期：2026-11-12
- 相关：`multi_agent/langgraph_variant.py`、`packages/agent_framework/pyproject.toml`（`[project.optional-dependencies]`）、`docs/w9/langgraph-vs-inhouse.md`、ADR-008（客户端层用编排方法而非装饰器链）、ADR-W9-002（多智能体）

---

## 背景

D4 要求"同样的图用 LangGraph 实现一遍"。这里有两个决策要显式化：

1. **要不要引入这个依赖**（它带了 18 个包，含 `langchain-core`、`langsmith`）；
2. **引入之后它是不是主实现**。

第 2 个的答案是不。理由在 `langgraph-vs-inhouse.md` §4：守卫（步数 / 预算累计 /
环路）必须放在"能看见全局"的位置，而框架里没有这个位置——它们只能寄生在某个
节点内部，依赖"这个节点一定会跑到"。

第 1 个的答案是要，但**必须是可以不装的那个**。因为不装它就跑不出来的对照，
等于没做对照：书面推演的"框架更灵活"和实测出来的"框架版把中止原因多出了一个
`recursion_limit`、不在我们的枚举里"，是两种不同质量的证据。

---

## 决策

### 1. 做成 `extras`，不做成主依赖

```toml
[project.optional-dependencies]
langgraph = ["langgraph>=1.0"]
```

```
uv sync --extra langgraph     # 想跑对照时
uv sync                       # 生产路径，不装
```

四条配套约束：

- **主实现不 import 它。** `multi_agent/__init__.py` 不导出 `langgraph_variant`，
  `supervisor.py` 里没有一行 `langgraph`。生产路径装不装都一样。
- **没装时 import 不报错。** `langgraph_variant.py` 把 import 包在 `try/except ImportError`，
  暴露 `LANGRAPH_AVAILABLE`；构造 `LangGraphSupervisor` 时才抛一句能照做的错。
- **测试自动 skip。** `test_multi_agent_langgraph.py` 用 `pytest.mark.skipif(not LANGRAPH_AVAILABLE)`。
  实测（把 `langgraph` 的 import 全部阻断后跑全量）：**1070 passed / 38 skipped**，
  与装着的 **1070 passed / 30 skipped** 相比，只多出那 8 条 skip，**0 失败**。
- **mypy 不受影响。** 根 `pyproject.toml` 的 `ignore_missing_imports = true` 让
  没装 extra 的环境把 `langgraph` 当 `Any`，不会因为缺依赖而红。
- **版本钉在 `>=1.0`。** v1 之前的 `StateGraph` API 与现在不同（`add_node` 的签名、
  `END` 的用法），写死老版本的示例会直接跑不起来。

### 2. 与 ADR-008 的关系：同一条论据，换了一层

ADR-008 当年拒绝的是**客户端层的装饰器链**，理由是：重试 / 计量 / fallback
这些横切关注点一旦散进装饰器，就"说不清扣在哪一层、也保证不了顺序"。

LangGraph 的节点模型是同一类结构：**横切关注点写进节点，顺序由图的边决定。**
把预算扣减写在 `dispatch` 节点里，问题不是"写不出来"，而是：

- 谁保证 `dispatch` 一定在每次工具调用前跑到？靠图的拓扑，而拓扑会被人改；
- 如果将来加一个"直接调用工具"的节点（比如一个 prebuilt 的 react agent），
  预算就不经过 `dispatch` 了，而**代码里没有任何东西会报错**。

这就是 ADR-008 那句话的复现：**横切关注点要有一个唯一的归属点，且归属点必须在
所有路径上都被经过。** 自研版的归属点是 `Supervisor.run()` 的循环顶部——
所有角色调度必然经过它，因为它是唯一的循环。框架版没有这个"必然"。

> 所以这不是"新旧 ADR 冲突"，而是同一条原则在两层各命中一次。
> **ADR-008 不需要修改**；本文是它在编排层的应用记录。

### 3. 引入的代价，如实记

| 代价 | 实测 |
|---|---|
| 包数量 | 18 个（含 `langchain-core 1.8.0`、`langsmith 0.14.17`、`orjson`、`zstandard`） |
| 类型摩擦 | `add_node("dispatch", <async 闭包>)` 过不了 mypy 的 overload，需一处 `type: ignore[call-overload]` |
| 中止原因口径 | 多出 `recursion_limit`，**不在 `StopReason` 枚举里**；混口径的报告没法做趋势 |
| 步数语义 | 框架数 superstep（一轮 = 2 个），与我们数的"角色轮数"不等价 |
| graph state | 装不下 `SharedState`（reducer 语义与乐观锁冲突），只放一个路由字段 |

最后一条是最有信息量的：**框架的 state 是"节点间传的数据"，我们的 `SharedState`
是"带并发语义的工作区"。** 把后者降级成前者，等于用框架的合并语义替代自己的
冲突检测——而这正是 W6/W8 一路建起来的幂等主线。

---

## 后果

### 好的

- 对照是**跑出来的**，不是推演的：8 条用例逐项比对状态 / 轮数 / 上游调用序列 /
  报告正文，两边全等；差异项（`recursion_limit`）也拿到了实证。
- 生产路径零依赖增加，离线档零影响。
- "为什么不换框架"有了可复述的答案，而不是"我们觉得自己写的好"。

### 代价

- **多一个 extra 要维护。** `langgraph` 的 v1 → v2 若改了 `add_node` 签名，
  `langgraph_variant.py` 会先烂。它是**对照工件**，不是产品代码；
  坏了就按当时的 API 重写一遍，成本约 2h，可接受。
- **两套 driver 要靠测试锁住一致性。** 共用 `run_role_actions` / `build_result`
  已经把漂移面压到"只各写 driver"，但对照用例必须一直在 CI 里跑
  （装了 extra 的那一档）。**没装 extra 的 CI 跑不到这 8 条，这一点写在验收里。**
- `uv.lock` 里多了一批包，`uv sync` 的解析时间略增。

### 被否决的方案

| 方案 | 否决理由 |
|---|---|
| langgraph 做为主实现 | 守卫在框架里没有唯一归属点；中止原因口径分裂；为 3 个节点引 18 个包 |
| 不引入依赖，只写文档对比 | 推演的"框架更灵活"没有证据；`recursion_limit` 这类差异推演不出来 |
| 把 langgraph 放进主依赖 | 生产路径平白多 18 个包，且违反"主实现不依赖编排框架" |
| 用 `langgraph==1.2.*` 钉死小版本 | 对照工件不需要这种稳定性承诺；`>=1.0` 让升级时先看到失败 |
