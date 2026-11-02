# 评测报告模板（W3 D5）

`run_eval.py` 会自动按这个结构产出 `evals/history/YYYY-Www-w3-eval.md`。
本文件是**人和脚本的共同契约**：四块固定结构，少一块报告就不能用来做决策。

复制下面的骨架填即可；每块后面的"读法"是提醒，写报告时删掉。

---

```markdown
# 评测报告 · <YYYY-Www>（<offline|online>）

- golden：`evals/golden/golden_v1.jsonl` sha256:`<前12位>`
- 生成时间：<ISO8601>
- 总数 <N>：passed <n> / failed <n> / interrupted <n> / skipped <n>

> 离线档声明（仅 offline）：本报告来自夹具回放，`success_rate` 只证明执行管道可用。
> 质量结论必须等在线档；`judge_score` 为 `null`（W4 才接 Judge）。

## 1. 分层指标

| 层 | 指标 | 值 | 样本数 |
|---|---|---|---|
| 任务 | success_rate | <%> | passed+failed |
| 任务 | interrupted_rate | <%> | total |
| 任务 | skip_rate | <%> | total |
| 工具 | tool_error_rate | <%> | tool_calls |
| 延迟 | latency p50 / p95 | <ms> / <ms> | latency_samples |
| 延迟 | first_token p50 / p95 | <ms> / <ms> | first_token_samples |
| 成本 | cost_per_task | <金额 币种> | 精确样本数 |
| 成本 | usage_estimated_ratio | <%> | total |
| 质量 | judge_score | null (W4) | - |

## 2. 难度与标签分布

| 维度 | 值 | 通过率 | 条数 |
|---|---|---|---|
| difficulty | 1 / 2 / 3 | <%> | <n> |
| tag | <tag> | <%> | <n> |

## 3. 趋势

<上一期数据点，例：2026-W45: 92.0% · 2026-W46: 96.0%>

## 4. 失败分类

| 分类 | 条数 | 归因性质 |
|---|---|---|
| upstream | <n> | 可靠性（不计入模型质量） |
| hallucination | <n> | 质量 |
| format | <n> | 质量 |

| case | 状态 | 分类 | 首条失败原因 |
|---|---|---|---|
| gold-xxx | failed | format | 帧序列期望 [...]，实际 [...] |
```

---

## 每块的读法（写给人看的，不写进产出）

**1. 分层指标**

- `success_rate` 的分母是 `passed + failed`。看到它涨了，先看 `total` 有没有变——
  分母变小（大量 `interrupted`）时，幸存样本偏向简单题，成功率会虚高。
- `p95` 旁边必须带 `samples`。样本 <5 时脚本直接给 `n/a`，不要手填一个数。
- `cost_per_task` 为 `n/a` 且 `usage_estimated_ratio > 0` 时，**不要引用任何成本结论**。

**2. 难度与标签分布**

- 只看总成功率会掩盖"3 档全崩"。难度 3 的通过率才是这个项目的卖点（多步任务）。
- 多 tag 的 case 会同时进各 tag 的分母，各 tag 条数之和 > 总数是特性。

**3. 趋势**

- 曲线断点比数字难看更值得追：模型快照变了、判据改过、样本换了，都会让曲线跳。
- 只有一条数据的"趋势"不叫趋势，别在第一期就下结论。

**4. 失败分类**

- `upstream` / `timeout` 是**可靠性**，不进质量结论。这两类和 `hallucination` 混在一起，
  是最典型的归因错误（去改 prompt，其实是厂商 503）。
- `hallucination` 目前只是"有引用锚点却答不出内容"的**线索**，措辞写"疑似"，
  最终判定归 W4 judge。

---

## 落盘对应的机器可读文件

同一期还有 `evals/history/YYYY-Www-eval.json`，字段与 `2026-W44.json` 的 `metrics` 对齐。
两份文件的关系：**md 给人看，json 给 W11 的曲线读**。数字不一致就是 bug，不是"视图差异"。

离线档的 json 里 `quality` 恒为 `null`、数字进 `pipeline`（ADR-W3-01 D2）——
这条约束让"夹具回放的 100%"永远没法被误当成质量结论。
