# W7 验收（Gate 7）

> 规则：**每一项都要有可执行的证据**（测试名或脚本命令），不接受"我看过代码，
> 没问题"。没做到的写"部分通过"并说明差哪一步——验收表不是给自己发奖。
>
> 实跑日期 2026-11-10；模型 `qwen3.7-plus`；pgvector `localhost:5433`。

---

## 1. 逐项结论

| # | 验收项 | 通过标准 | 证据 | 结论 |
|---|---|---|---|---|
| 1 | 预算分配 | 槽位化 + 预留输出余量，裁剪优先级确定 | `test_context_budget.py`（15 条，含 `test_exact_budget_fits` / 裁剪顺序 / `reserve_output` 不可侵占） | ✅ |
| 2 | 20+ 轮不爆 | 长会话全程在预算内，无 400 | `scripts/demo_long_session.py`（20 轮，naive 第 18 轮起爆窗、budget/summary 0 次）；`test_working_memory.py::TestTwentyRounds` | ✅ |
| 3 | 工作记忆 | 摘要标注来源、tool 截断可恢复 | `test_working_memory.py`（14 条，含 `test_refetch_recovers_full_text`、摘要带 `[会话摘要…非用户原话]` 标签） | ✅ |
| 4 | 长期记忆 | 四阶段闭环，跨会话可回忆品牌调性 | `test_long_term_memory.py`（14 条）+ `test_memory_pgvector_e2e.py`（真库 5 条）+ 实验二 `tone_recall`（off 失败 / on 答"专业克制。"） | ✅ |
| 5 | 边界清晰 | 记忆/知识库隔离，判据可操作 | `test_memory_isolation.py`（构造期拒绝 `rag_chunks`）+ `docs/w7/memory-vs-knowledge.md`（判据表 + 5 个案例） | ✅ |
| 6 | 质量不退化 | `difficulty=3` 案例过门禁 | `run_eval.py --week 2026-W50 --judge`：54 条 50 passed / 0 failed / 4 skipped，judge 均分 `1.922`（与 W49 持平）；`regression_gate.py` 门禁通过 | ✅ |
| 7 | 成本可量化 | 压缩前后 token/成本有对比数字 | `docs/w7/context-cost-ab.md` 实验一（三个 arm 的真实 usage + 摘要成本单列） | ✅ |
| 8 | Context 可解释 | 六类排除原因互斥穷尽，任一 query 能报出唯一原因 | `test_context_budget.py::TestExclusionReasons`（含 `test_upstream_reason_not_overridden`、`explain()` 四态） | ✅ |
| 9 | 记忆写入可信 | 幂等 / 冲突可追溯 / 污染默认拒绝 / 删除可传播 | `test_long_term_memory.py` + `test_memory_pgvector_e2e.py`（`test_idempotent_write_rowcount_unchanged`、`test_conflict_supersede_keeps_old_row`、`test_polluted_candidate_never_reaches_db`、`test_forget_deletes_rows_and_vectors`） | ✅ |
| 10 | Drift 可检测 | 三指标各有一个可触发构造，命中只重注入不中断 | `test_context_drift.py`（10 条，三指标各 2 条正反例） | ✅ |
| 11 | 记忆值得存在 | `off` vs `governed` 两档数字齐全，能回答"值不值" | `docs/w7/context-cost-ab.md` 实验二：通过 3/4 → 4/4（唯一变好的是跨会话召回），input tokens +89.2%、cost +22.0%（基线仅 63 token，换成 RAG 真实 payload 约 +1.5%） | ✅ |
| 12 | 合流 | history 接上，端到端仍可跑 | `evals/history/2026-W50.json`、`docs/log.md` W50 条目、`pytest` 794 passed / 30 skipped | ✅ |

---

## 2. 本次实跑记录（可复现）

```bash
# 静态检查
.venv/bin/ruff check .                                   # 全绿
.venv/bin/python -m mypy packages/agent_framework/src/agent_framework \
  apps/growth_agent/src/growth_agent evals/scripts/context_cost_ab.py \
  scripts/demo_long_session.py                           # 74 文件 0 问题
# 说明：`make type` 只查 pyproject 里配的 packages+apps（72 文件）；
# 上面把两个新脚本一起纳入是更严的口径，74 即由此而来。

# 全量离线用例
.venv/bin/python -m pytest packages/agent_framework/tests/ apps/growth_agent/tests/
# 794 passed, 30 skipped

# 真库记忆用例（需要 pgvector）
PGVECTOR_DSN=postgresql://rag:rag@localhost:5433/rag \
  .venv/bin/python -m pytest apps/growth_agent/tests/test_memory_pgvector_e2e.py -q
# 5 passed

# 评测与门禁
.venv/bin/python evals/scripts/run_eval.py --week 2026-W50 --judge
.venv/bin/python evals/scripts/regression_gate.py --current evals/history/2026-W50-eval.json

# 演示（零密钥）
.venv/bin/python scripts/demo_long_session.py

# 两档对照实验（真实调用，本机实测约 5 分钟；需密钥 + pgvector）
.venv/bin/python evals/scripts/context_cost_ab.py --json evals/history/2026-W50-ab-run.json
```

---

## 3. 与 W49 对比：用例与指标的净变化

| 指标 | W49 | W50 | 变化 |
|---|---|---|---|
| 离线用例 passed | 715 | 794 | +79（新增 84 条，含 5 条真库用例默认跳过后的净值） |
| skipped | 25 | 30 | +5（真库记忆用例 + 需要密钥的实验） |
| ruff | pass | pass | — |
| mypy 文件数 | 61 | 74 | +13（context/ 6 + memory 2 + 脚本 2 + 测试不计） |
| judge 均分（离线档） | 1.922 | 1.922 | 0（夹具档，只证管道通） |
| success_rate | 100% | 100% | 0 |

> 离线档的 judge 是 `heuristic-keyword`，同一份夹具必然同分。它能证明"没有
> 回归"，**不能**证明"质量变好了"——后者要 W4 的 live judge，本周没跑（无
> 稳定 API 配额时跑出来的 judge 分数不可比）。

---

## 4. 部分通过 / 未做

| 项 | 状态 | 差在哪 |
|---|---|---|
| Gate 7 #2 的"无 400" | 部分 | 20 轮在 8000 预算口径下未撞窗（真机 256K 窗口更不会 400），所以"不爆"的证据是**预算口径判定 + 演示脚本**，不是"真的收到过 400 又修好了"。爆窗点约在第 45 轮（按增长率外推），未实测到真 400。 |
| Gate 7 #7 的"成本下降" | 有数字但需带量纲 | 实验一 20 轮：净成本 `0.0661 → 0.0563 元`（-14.9%，已把摘要自身 0.0083 元算进去），input tokens -31.1%、p95 -37.2%。但摘要成本占 12.5%，**拐点在 12 轮附近**：更短的会话里压缩不划算。报告给的是曲线与拐点，不是单一百分比。 |
| 记忆的两档实验 | 已跑通 | 单次跑、4 个任务。样本量小，`pass +1` 的置信度有限；结论写的是"方向"而不是"幅度"。 |
| live judge 档 | 未跑 | 需要稳定配额；与 W6 同一取舍，W8 一起补。 |

---

## 5. 这一周最重要的两个判断

1. **判据决定你能看见什么**。实验一第一版没把摘要调用记进成本，"压缩"看起来
   免费——而它每次都是一次真实 LLM 调用。`SummarizerUsage` 就是为这件事加的：
   任何一个"优化"，如果它的开销不进报表，报表就会推荐它。
2. **记忆的价值必须用"关掉它"来证明**。`MEMORY=off` 那一档回答不了"品牌调性
   是什么"，`governed` 答"专业克制。"——这不是"模型更聪明了"，是**信息真的
   被记下来了**。同时 cost 涨了 44.9%，这个代价要跟着结论一起说。
