# 自研编排 vs LangGraph：同一张图的两种驱动方式

- 日期：2026-11-12（W9 D4）
- 对照代码：`multi_agent/supervisor.py`（自研，447 行）· `multi_agent/langgraph_variant.py`（LangGraph，330 行）
- 依赖：`langgraph==1.2.20`（extra，见 ADR-W9-003）
- 证据：`tests/test_multi_agent.py`（31 条）· `tests/test_multi_agent_langgraph.py`（8 条，两边并排跑）

---

## 0. 先说这张图

```
START → dispatch ─┬→ researcher ─┐
                  ├→ validator ──┼→ dispatch
                  └→ END         │
```

三条规则（researcher 查证据 → validator 查引用 → 收尾）与三道守卫（步数 / 预算累计 /
环路指纹）**两个实现完全一样**。差别只在「谁在驱动这个循环」。

为了让对比成立，执行与记账是**同一份代码**：
`run_role_actions()` / `build_result()` / `ask_for()` 被两个 driver 共用。
如果各写一份，这个对照就退化成"比较两份重复实现的细节差异"，
而真正要看的"守卫放在哪一层"会被一次重构抹平。

---

## 1. 对比表（每格都跑了，不是列维度）

| 维度 | 自研（`Supervisor.run`） | LangGraph（`LangGraphSupervisor.run`） | 结论 |
|---|---|---|---|
| **守卫写在哪** | 循环顶部，三个 `if` 连排，一个函数里看全 | 全塞进 `dispatch` 节点内部 | 框架不提供预算/环路抽象，守卫只能寄生在节点里。**位置从"外层"变成"节点内"，可读性下降但语义不变** |
| **守卫能否真中断** | `break` + 具名 `StopReason` | 写 `route=END`；节点内的 `break` 无处可写 | 两边都能停（8 条对照用例逐条验过），但自研的"停"是一个值，框架的"停"是一次路由跳转 |
| **中止原因可归因** | 9 个枚举值，全部可归因 | 自研的 9 个照旧；但框架自己会抛 `recursion_limit`，**不在枚举里** | 这是最实的一条差异：混合中止来源 = 混合口径 |
| **步数语义** | 自己数"角色轮数" | 框架数 superstep（每轮 = 2 个） | 两者不等价。`recursion_limit=3` 时框架先炸，业务守卫还没轮到 |
| **业务状态** | `SharedState` 一个对象 | **同一个对象，但没进 graph state** | 见 §2。这是最有信息量的一格 |
| **trace 对接** | 直接产出 `RunTrace` | 同样产出 `RunTrace`（共用 `build_result`） | 因为共用，两边 trace 字段一一对应，`trajectory_eval` 不用写两套 |
| **结果一致性** | — | 状态 / 轮数 / 上游调用次数 / 报告正文**逐字节相同** | 见 §3。换 driver 不改行为，说明这套守卫的语义是可迁移的 |
| **迭代速度** | 加一个角色 = 加一个类 + 一行注册 | 加一个角色 = 加一个节点 + 一条边 | 差距没想象中大：**真正的工时在角色实现，不在接线** |
| **类型摩擦** | 无（自己的 dataclass） | `add_node` 的 overload 过不了 mypy，需一处 `type: ignore` | 框架的 TypedDict 泛型签名为动态注册付了代价 |
| **依赖面** | 0 | 装了 18 个包（含 `langchain-core`、`langsmith`） | 供应链面积是实打实的成本 |
| **黑盒风险** | 无 | 重试 / 缓存 / 中断语义都在框架里，出问题要看框架源码 | 生产事故时"能不能读懂那一层"决定恢复时间 |
| **面试怎么讲** | "我知道守卫该放在哪一层，为什么" | "我会用生态，也知道它的边界在哪" | 两者不冲突，但**只有前者能回答"为什么要这么放"** |

---

## 2. 最有信息量的一格：graph state 装不下 `SharedState`

LangGraph 的 state 通道靠 **reducer** 合并。reducer 的默认语义是
「同一个 key 追加/合并」，而这个语义**不会拒绝一次基于旧快照的写入**。

而 `SharedState` 的核心是乐观锁：

```python
state.commit(role, findings=..., expected_version=snapshot)  # 版本不对就抛
```

把 `SharedState` 交给 graph state，会发生两件事：

1. reducer 把两个角色的 `findings` 合并了——**看起来成功了**；
2. `expected_version` 的校验点没了——因为"合并"这个动作本身就假定冲突不存在。

所以 LangGraph 版的 graph state 只剩一个字段：

```python
class _Route(TypedDict, total=False):
    route: str
```

业务状态留在闭包里。这不是"没用好框架"，是**两套状态机制不在同一个抽象层**：
框架的 state 是"节点间传递的数据"，我们的 `SharedState` 是"带并发语义的工作区"。
把后者降级成前者，等于用框架的合并语义替代自己的冲突检测。

> 反过来说：如果哪天要接 LangGraph 的 checkpointer 做"断点续跑"，
> `SharedState` 就必须可序列化，而"序列化一次快照"与"乐观锁"是同一件事的两面。
> 那时才该把两者对齐——不是现在。

---

## 3. 结果一致性（8 条对照用例）

`tests/test_multi_agent_langgraph.py` 让两个 driver 吃同样的输入，逐项比对：

| 场景 | 比对的量 | 结果 |
|---|---|---|
| 正常流程 | 状态 / 轮数 / 步数 / 上游调用序列 / 报告正文 | 全等 |
| 缺引用 | 状态 / `stop_reason` / 报告正文 | 全等 |
| 互踢皮球 | `stop_reason=plan_loop` / 轮数 / 状态 | 全等 |
| 预算超限 | `stop_reason=max_tokens` / `trace.tokens=300` / 轮数 | 全等 |
| 框架上限 | `stop_reason=recursion_limit`（**不在枚举里**） | 只有框架版有 |
| 守卫优先 | 放大 `recursion_limit` 后回到 `plan_loop` | 框架守卫只是兜底 |

第三、四条是这次对比的关键：**如果框架版把预算算成"每跳重置"，它会跑得更久，
而人眼看报告只会觉得"框架版更勤奋"。** 边跑边比，这类偏差才会露出来。

---

## 4. 结论：W9 的主实现为什么仍是自研

1. **守卫的归属**。预算、步数、环路是"编排层的错"，它们需要一个能看见全局的位置。
   自研版的循环顶部就是那个位置；框架版只能把它们塞进某个节点，依赖"这个节点一定会跑到"。
2. **口径唯一**。自研版中止原因只有 `StopReason` 一个来源。框架版有两个来源
   （我们的枚举 + 框架的 `recursion_limit`），而混口径的报告没法做趋势。
3. **可读的失败**。值班的人看 `detail` 就知道是"预算到顶"还是"两个角色互踢"。
   框架的 `GraphRecursionError` 只告诉你 superstep 用完了。
4. **依赖成本**。为了这张 3 节点的图引入 18 个包，收益不成比例。

**但这不是"框架不好"。** 如果图长到 15 个节点、需要 checkpointer 断点续跑、
需要 subgraph 组合与 streaming，自研的边际成本会迅速超过框架。
W9 这张图只有 3 个节点、且守卫是核心诉求，自研是对的；
W12 若要做"多阶段带人工审批的长流程"，会重新评估。

---

## 5. 这一节留给人问的问题

- "你们用 LangGraph 吗？" → "对照实现过。同一张图 3 个节点，两边行为逐项对齐（8 条用例）。
  主实现用自研，因为守卫要放在编排层最外层；如果节点数上去了或者要 checkpointer，会换。"
- "自研不怕重复造轮子？" → "轮子只有 447 行，其中 200 行是守卫与记账——那部分正是我们不想外包的。
  接线（节点、边）只占几十行。"
- "怎么保证两个实现不漂移？" → "共用执行与记账（`run_role_actions` / `build_result`），
  只各写 driver；8 条对照用例逐项比对报告正文。"
