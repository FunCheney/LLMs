# W9 验收：规划与多智能体

- 日期：2026-11-12
- 依据：`docs/w9/w9-tasks.md` Gate 9 十二项
- 原始数据：`evals/history/2026-W52.json`、`evals/history/2026-W52-eval.json`
- 结论：**10 项通过，2 项判 partial**（#2 规划可用 / #11 稳定性）。两项 partial 都不是"没做"，
  而是"做了，但它证明得没那么多"。

---

## 总览

| # | 验收项 | 判定 | 一句话证据 |
|---|---|---|---|
| 1 | 对照组齐全 | **pass** | ReAct 与 P&E 都实现可跑，D5 里 5 档基线 |
| 2 | 规划可用 | **partial** | 1.00 vs 0.80 确实更高，但归因到窗口不范式；9 样本 |
| 3 | 拆解合理 | **pass** | 四正交角度 + `Gap.failed` + 空引用硬校验 |
| 4 | 三道守卫 | **pass** | 三守卫各配反向用例；两种环路形态都抓 |
| 5 | 预算累计 | **pass** | 250→3 轮 300、1000→10 轮 1000，反证不重置 |
| 6 | 对照实现 | **pass** | 8 条 parity 逐项对齐，差异项有实测 |
| 7 | 选型有依据 | **pass** | 五条链路逐条判定，四条写"不用 Agent" |
| 8 | Agentic Search | **pass** | 四个停止原因可单独断言 + 0.8 去重 |
| 9 | 三层对比 | **pass** | 七档 × 三层数字齐全 |
| 10 | 归因到层 | **pass** | 归到"记忆机制"，"分工"写明无法归因 |
| 11 | 稳定性 | **partial** | 3 次跑完、均值与极差已报；极差 0≠稳定 |
| 12 | 合流 | **pass** | history 已写、eval 0 失败、门禁通过 |

---

## 逐项证据

### #1 对照组齐全 — pass

`planner/react.py`（朴素，`observation_window=3`，无计划、失败只能重试）与
`planner/plan_execute.py`（计划 + 依赖 + 重规划合并）都实现可跑。
`test_planner.py` 26 条覆盖两者；D5 的七档里有 3 档 ReAct、2 档 P&E。

ReAct 刻意保持朴素：**给它补能力会让对照组失真**，这一点写在它的模块 docstring 里。

### #2 规划可用 — **partial**

要求是"≥5 步任务，多 Agent 完成率显著高于朴素 ReAct"。实测（复杂档，5 步，9 样本）：

| | 覆盖 |
|---|---|
| `multi_agent` | **1.00** |
| `react_w3`（朴素 ReAct 默认窗口） | **0.80** |

数字上确实更高。**但判 partial 有三个理由：**

1. **归因不支持"范式优势"**：`react_w8`（只改窗口）同样是 1.00，`plan_execute_w3`
   （只改范式、窗口不变）同样只有 0.80。差异全部由观察窗口解释。
2. **"显著"谈不上**：9 个样本、确定性任务世界、极差 0——这个数字是机制演示，
   不是统计检验。说"显著"是过度解读。
3. **任务集只有一个工具**（`web_search`），"角色化让工具选择更专一"这条机制
   在数据里根本不存在。

所以这一项的真实结论是：**完成率确实能提上去，但提上去的原因是"共享状态不受窗口限制"，
不是"拆成了多 Agent"。**

### #3 拆解合理 — pass

`planner/research_planner.py`：

- 四个**正交**角度：定义与边界 / 规模与数据 / 竞品与替代 / 风险与反例；
- 部分失败**显式标注**：`Gap.failed` 而不是塞一条空证据（第一版就是塞空
  `Evidence`，被 `__post_init__` 的非空校验当场打回——那次是校验救了一个 bug）；
- 报告三段分区：证据（可核对）/ 假设（未验证，不得当结论）/ 证据不足 / 结论（每条带 `[n]`）；
- `unreferenced_claims()` 是**可执行判据**，不是"请注意引用完整性"这种话。

19 条用例，含四条停止条件各自的单独断言。

### #4 三道守卫 — pass

| 守卫 | 反向用例 | 断言 |
|---|---|---|
| 步数 | `test_step_guard_is_a_real_ceiling` | `rounds <= 3`，且 `stop_reason ∈ {max_steps, no_progress}` |
| 无进展 | `test_repeated_empty_role_stops_on_no_progress` | `rounds == 2`、`tool_calls == 0` |
| 预算 | `test_budget_accumulates_across_handoffs` | `stop_reason=max_tokens`、`tokens == 300` |
| 环路 | `test_pingpong_aborts_within_fingerprint` | `stop_reason=plan_loop`、`rounds == 3` < `max_steps=50` |

环路另有一条**反向的反向**：`test_same_role_in_a_row_is_not_a_cycle`——同一角色连任
不算环。这条是踩出来的（见 #6 复盘）。

### #5 预算累计 — pass

```
budget=250：3 轮 × 100 = 300 → 第 4 轮前停（max_tokens）
budget=1000：10 轮 × 100 = 1000 → 第 11 轮前停（max_tokens）
```

如果实现是"每跳重置"，两档都会跑满 `max_steps=50`，轮数相同、而每一跳都显示余额充足。
`test_bigger_budget_runs_longer` 就是这条反证。

`Handoff.budget_left_tokens` 是**读当前余额**算出来的，不是每跳重新赋值。

### #6 对照实现 — pass

`multi_agent/langgraph_variant.py`（langgraph 1.2.20，走 optional extra）。
8 条 parity 用例让两个 driver 吃同样输入，逐项比对：状态 / 轮数 / 步数 /
上游调用序列 / **报告正文**（逐字节相同）。

差异项也拿到了实证：框架自己有 `recursion_limit`，抛异常而不是给具名原因，
**这个原因码不在 `StopReason` 枚举里**——要用它就得扩枚举或接受两套口径。

离线档不受影响：把 langgraph 的 import 全阻断后跑全量，
**1070 passed / 38 skipped / 0 failed**（装着时是 30 skipped）。

### #7 选型有依据 — pass

`docs/w9/adr-w9-planner-choice.md` 对 capstone 五条链路逐条判定：

| 链路 | 判定 |
|---|---|
| 关键词清洗 / 字数裁剪 | 脚本 |
| 品牌客服问答 | Chatbot + 小 RAG |
| 竞品卖点提取 | RAG |
| SEO 流水线 | Workflow（固定图） |
| 深度调研 | Autonomous（复杂时升级多 Agent） |

**五条里四条明确写了"不用 Agent"**——这份 ADR 的价值一半在拒绝。
另附「五条不该用 Agent 的红旗」（能写出判分函数 / 步骤可枚举 / 失败难回滚 /
成本必须可预测 / 同一步要跑 1000 次）。

### #8 Agentic Search — pass

四个停止原因各自有单独用例：

- `STOP_ENOUGH`（证据够）/ `STOP_BUDGET`（检索次数到顶）/
  `STOP_LOW_YIELD`（边际收益不足）/ `STOP_DUPLICATE_QUERY`（同义 query 命中）。

去重是**字符 bigram Jaccard**（阈值 0.8）：中文没有空格，词级 Jaccard 要先分词，
而分词器本身会变成第二个变量。边际收益用**字符数**而不是 token——token 估算
本身有 1.6 字符/token 的粗糙度，两层噪音叠起来就测不出东西了。

设计上有一处关键修正：`SearchOutcome` 让**后端成为唯一执行点**。第一版是
规划器与后端各调一次检索，同一句话搜两遍而预算只扣一半。

### #9 三层对比 — pass

`docs/w9/single-vs-multi-ab.md`，七档配置 × 两档难度，三层都有数字：

- **Outcome**：复杂档 0.80 / 1.00（覆盖 = 每步真值是否真的取回来）；
- **Trajectory**：`tool_selection_acc` 1.00、`invalid_call_rate` 0.00 vs 0.17、
  `steps_used` 5.00 vs 6.00、`path_repeat_rate` 0.00；
- **System**：token 与耗时（**标注为不可跨档比较**，记账口径不同）。

`TaskMetrics` 已回填四个字段 + `trajectory_samples`；`trajectory_eval.py` 18 条口径单测。

> **一处必须说清的落差**：`run_eval` 落盘（`2026-W52-eval.json`）里这四个字段是
> **`null`**，因为 54 条 golden 是 chat / tool_call 形态，跑的是 `LLMGateway`，
> 没有 `RunTrace`。真正填它们的是 `agent_paradigm_ab.py` + `trajectory_eval.py`
> 这条链（结果在 `2026-W52.json` 的 `w9_ab` 块）。
> 「字段存在」不等于「有数」——这行注释已写进 `metrics.py`，免得下次看 history 的人读错。

### #10 归因到层 — pass

要求"优势能指到具体某一层指标，说不出就写无法归因"。本次的答法是：

- **能指到的**：差异在 **Outcome 层**，而机制是**记忆**（窗口 vs 共享状态）。
  证据是三个单变量对照：只改窗口（`react_w3→w8`：0.80→1.00）、
  只改范式（`react_w3→plan_execute_w3`：0.80→0.80）、换成共享状态（`react_w8→multi_agent`：1.00→1.00）。
- **指不到的**：`tool_selection_acc` 与 `steps_used` 在对照之间**没有差异**
  （都是 1.00 / 5.00），所以"角色化让工具选择更专一"这条假设**在本次数据里不成立**；
  "分工本身"的贡献**无法归因**，因为任务集没有需要不同工具面/权限的任务。

这一项判 pass 不是因为结论好看，而是因为**归因做到了可检验、说不清的地方写明了说不清**。

### #11 稳定性 — **partial**

要求"复杂档 3 次跑完，报均值与极差"。做了：每配置 3 次，均值与极差都在报告里。
**七档的极差全部是 0.00。**

判 partial 的理由：这个 0.00 证明的是**可复现**（确定性任务世界，同一输入必得同一输出），
**不是稳定性**。真正的稳定性要看在线档（`LLMReasoner` + 真模型）3 次的极差——
**本次没跑**，因为没有可用 key，也不该为了让表格好看去凑数。

一张"极差 0.00"的表，读者第一眼会读成"系统非常稳定"。它不是那个意思，所以这一项
必须写清楚而不是判 pass 了事。

### #12 合流 — pass

- `evals/history/2026-W52.json`（本周记录）
- `evals/history/2026-W52-eval.json` + `2026-W52-w3-eval.md`（`run_eval --week 2026-W52`）
- golden 54 条：**50 passed / 0 failed / 0 interrupted / 4 skipped**，`success_rate 100%`（offline）
- `regression_gate`：**通过**（基线 2026-W46，对比 success_rate 与 latency_p95）
- 全量：**1096 passed / 30 skipped**；`ruff` 全绿；`mypy` 110 files 0 issues

---

## 复盘：8 个"当时觉得对、实际错"的地方

1. **控制权矩阵第一版是装饰品。** `test_control_matrix_has_no_dead_entries` 红了，
   而它是对的——矩阵当时没有任何运行时查询。修法不是加豁免，是把
   `assert_code_owned` / `owner_of` / `model_may_decide` 接进 `StepRunner` 与 `PlannerBase`。
2. **`PLAN_LOOP` 阈值写成 `>2`**，导致第二次重复那轮**真的执行**——同一个失败花两遍钱。
3. **环路指纹把"同一角色连任"判成环**，两个子问题的正常流程在第 4 轮被误杀，
   报告只剩一半还标着 `plan_loop`。
4. **步数按工具调用计数**，空转角色让守卫永远数到 0，循环转得很欢而守卫不触发。
5. **`_citation_of` 用空白切分找 URL**，遇到 JSON 包络（`"url": "https://…",`）永远找不到，
   于是所有发现都"缺引用"。
6. **A/B 第一次四种编排全 1.00**：不是系统好，是回忆距离（2）小于默认窗口（3），
   机关根本没触发。**没有区分度的实验比没做更危险。**
7. **A/B 第二次仍然全 1.00**：判据错了——答案文本由 `compose` 按任务书合成，
   天生含真值。改成从"证据有没有真的取回来"判。
8. **LCS 口径被我讲错过一次**：我以为它对换序宽容，实际 `[a,b]` vs `[b,a]` 得 0.5。
   测试纠正了我，改的是文档不是断言。

第 6、7 两条是同一类：**评测的失败往往不是数字难看，而是数字好看得没有信息。**
两次都差点带着"全部 1.00"的表格写报告。

---

## 诚实的缺口清单

| 缺口 | 影响 | 什么时候补 |
|---|---|---|
| 在线档一次没跑 | 所有模型侧结论都建立在"模型完全听话"的假设上 | W10 前后补一次真实 3 次 |
| 多 Agent 的检索词是替它构造的 | 1.00 是能力上限，不是实测水平 | 换 `LLMReasoner` 时自然消失 |
| 任务集只用了一个工具 | "选错成另一个合法工具"测不出来 | 加带多工具的任务 |
| token 记账跨档不可比 | 成本对比没有可信数字 | W11 成本看板统一计价 |
| 并发只模拟过 | 乐观锁在真正 `asyncio.gather` 下未验证 | W10 若要并行 |
| 样本量小（3×3） | 只能支撑机制判断，不能支撑效果量级 | 扩任务集 |
