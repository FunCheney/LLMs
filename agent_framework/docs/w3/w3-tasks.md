# W3 执行清单（评测体系 I · Golden Set 与 Trace · P0，稀缺能力 ①）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> **本周是三大稀缺能力之一「评测设计」的立骨架周**。ROADEMAP §4 把它列为 **61% JD、#1 差异化项**。做扎实了，W12 的 `evals/history` 曲线就是面试里最硬的一张牌。
> 契约真相源仍是 `docs/architecture.md`；本文件只讲"做什么、怎么搭骨架、怎么自查、何时交给我"。

## 承接面：W1–W2 已交付（站在这上面，不要重写）

| 已有 | 位置 | W3 的接缝 |
|---|---|---|
| `evals/golden/smoke_v1.jsonl`（10 条）+ `run_smoke.py` | `evals/` | 本周扩成 50+ 的 golden，**沿用同一 case schema 并扩展字段** |
| `LogCapture`（读 `RequestLog`）、`RecordingTransport`、24 个 mock 夹具 | `evals/scripts/` + `tests/fixtures/` | 批量 runner 直接复用，**零网络档可进 CI** |
| `test_live.py` 三重闸门（`-m live` + 无 key 跳过） | `tests/` | 在线评测复用同一闸门，别另造开关 |
| `tracing.py` 的 `TraceExporter` 协议 + `NoopExporter` | W2 D3 | 本周换 `LangfuseExporter`，**协议不变、只换实现** |
| `RequestLog`（含 `trace_id/span_id`）+ `first_token_ms`/`cost`/`usage_estimated` | `observability.py` | 所有评测指标从日志读，不自己计时 |
| `bench_baseline.py` 的采样口径、`evals/history/2026-W44.json` 结构 | `evals/` | 评测结果落盘**对齐这个 json 的 metrics 结构** |
| `models.yaml` 的 alias 与 wire 名分离（ADR-009） | `config.py` | 评测必须**钉 alias + 温度 0**，否则数字不可复现 |

> W3 检查点（ROADEMAP）：**`make eval` 一条命令出报告**；能回答"当前系统到底多好"，且数字可复现。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接的 W1–W2 符号 | 推进的 DoD 维度 |
|---|---|---|---|
| D1 | 评测方法论文档 + `evals/metrics.py`（指标分层） | run_smoke 判据、RequestLog 字段 | 质量（指标可定义） |
| D2 | `evals/golden/golden_v1.jsonl`（50+，含引用/难度） | smoke_v1 schema、canonical_input | 质量（golden set ≥50） |
| D3 | `evals/scripts/run_eval.py`（批量 runner + 落盘） | LogCapture、run_smoke、W44 json | 质量、工程（可复现） |
| D4 | `LangfuseExporter`（接 W2 exporter 协议） | `TraceExporter`、span、`trace_id` | 可观测（全链路 trace） |
| D5 | 报告模板 + 失败 case 自动分类 | run_eval 结果 | 质量（可归因） |

**唯一主线**：把 W2 那个"只证明没坏"的 smoke，升级成"能量化多好、能追每次调用、能自动分类失败"的评测体系。**评测口径本周一定要钉死**——W4 的回归门禁、W11 的成本看板、W12 的趋势图全靠它，口径一乱全废。

---

## Gate 0 · 开工前（30 min）

- [ ] 复核 W2 状态：`make eval`（或 `run_smoke.py`）离线全绿；`evals/history/YYYY-Www.json` 已有 W2 数据点
- [ ] 确认 W2 的 `TraceExporter` 协议签名（`tracing.py`）——本周只换实现不改协议
- [ ] 注册 Langfuse（或 LangSmith）账号，拿到 key 放进 `.env.local`；**没 key 也要能跑离线档**
- [ ] 通读 `evals/history/2026-W44.json`，记牢 metrics 字段名（本周落盘要对齐）
- [ ] 想清楚 capstone 的"成功"是什么：调研任务 = 报告有引用 + 覆盖子问题 + 无幻觉；文案任务 = 符合 brief + 不编数据

---

## Day 1（2.5h）· 评测方法论 + 指标分层

**目标**：先写清"怎么算分"，再写 runner。**顺序反了就是灾难**——先有脚本后补定义，指标会变成"代码碰巧算出来的东西"。

### 你要做的事
- [ ] `docs/w3/metric-spec.md`：指标字典（每个指标的定义、数据来源、公式、边界）
- [ ] `evals/metrics.py`：分层指标的数据结构 + 聚合函数
- [ ] `docs/w3/adr-w3-eval-scope.md`：离线/在线分档 + 复现性约定（钉 alias/温度/seed）
- [ ] 单测：每个指标的边界用例（空结果、全失败、部分缺失 usage）

### 引导框架

**分层指标（五层，对齐 ROADEMAP D1）**

| 层 | 指标 | 数据来源 | 边界定义（必须写进 metric-spec） |
|---|---|---|---|
| 任务层 | `success_rate` | case 的 `expect` 断言 | 分子=通过 case 数，分母=总 case；**被 `CapabilityError`/上游错误中断的不算"失败"要单列** |
| 工具层 | `tool_error_rate` | `ToolCall.arguments_error`/未声明调用 | 沿用 conformance A14 的判据，别另立 |
| 延迟层 | `latency_p50/p95`、`first_token_p50/p95` | `RequestLog` | **复用 W1 的 `first_token_ms` 定义**，不许自己计时 |
| 成本层 | `cost_per_task`、币种、`usage_estimated` 占比 | `RequestLog` | 峰谷价注记沿用 W1 Day5；`estimated` 的要单列不能混进均值 |
| 质量层 | `judge_score`（W4 才接 Judge） | 本周留字段 | 本周为 `null`，标 pending，别填假数 |

```python
# evals/metrics.py —— 只给结构，聚合逻辑留给你
@dataclass
class TaskMetrics:
    total: int
    passed: int
    interrupted: int            # 上游/能力错误中断，单列，不进 success_rate 分母口径
    tool_errors: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    first_token_p95_ms: float | None
    cost_per_task: float | None
    currency: str | None
    usage_estimated_ratio: float
    judge_score: float | None   # W4 填
```

**算法步骤**
1. 先写 `metric-spec.md`，每个指标写死"分子/分母/数据来源/什么算中断"
2. `metrics.py` 只实现"从一批 `EvalResult` 聚合成 `TaskMetrics`"，不含任何 LLM 调用
3. 复现性约定：钉 `alias`（不是 wire 名，ADR-009）、`temperature=0`、固定 fixture；在线档要记厂商模型快照日期
4. `usage_estimated_ratio > 0` 时报告必须显著标注"部分成本为估算"（否则 W11 会误读）

**自检**
- [ ] `pytest -k metrics` 覆盖：空批次、全失败、部分 usage 缺失三种边界
- [ ] `metric-spec.md` 里每个指标都能对应到 `RequestLog` 的具体字段（没有"从空气里来的指标"）

### 🔍 交给我的锚点
`metric-spec.md` 全文、`TaskMetrics` 结构、`pytest -k metrics` 输出。我会看：`success_rate` 分母口径有没有把"上游中断"算成"模型答错"、`first_token` 是否复用 W1 定义。

---

## Day 2（3h）· 构建 50+ 条 golden dataset

**目标**：把 W2 的 10 条 smoke 扩成 **≥50 条**、难度分层、带答案与引用的 golden set。这是 ROADEMAP 里"含答案、引用、难度分层"的硬要求。

### 你要做的事
- [ ] `evals/golden/golden_v1.jsonl`：≥50 条
- [ ] 扩展 case schema：加 `references`、`difficulty`、`tags`、`source`
- [ ] `evals/golden/schema.md`：字段说明 + 难度分层标准
- [ ] `evals/scripts/validate_golden.py`：引用完整性、难度分布、去重校验
- [ ] 至少 3 条**对抗样例**（ROADEMAP W2 遗留项明确写了"含 3 条对抗样例"）

### 引导框架

**case schema（在 W2 smoke schema 上扩展，向后兼容）**
```json
{"id": "gold-001", "kind": "chat|tool_call|research|draft", "model": "fixture",
 "task": "chat", "messages": [...], "tools": [...],
 "expect": {"text_contains": [...], "must_not_include": [...],
            "finish_reason": "STOP", "tool_calls": [...], "usage_estimated": false},
 "references": ["url#anchor 或文档 id"], "difficulty": 2,
 "tags": ["research","citation"], "source": "handwritten|recorded"}
```
- 难度分层标准（写进 schema.md）：1=单轮直答 / 2=需 1 次工具或检索 / 3=多步规划或多源引用
- 分布要求：三档都要有，且 3 档 ≥10 条（否则 W9 的多步任务评测没有样本）
- 对抗样例：① prompt injection 输入 ② 要求"编造数据" ③ 引用不存在来源——预期都是"拒绝/标注不确定"，不是照做
- `references` 的引用目标必须存在（`validate_golden.py` 校验），否则 W4 的 faithfulness 指标无从算起

**算法步骤**
1. 先把 W2 的 10 条 smoke 平移进来（保持 id 映射），再补到 50+
2. 每条 `expect` 判据复用 conformance 断言风格，不许为 golden 另写宽松逻辑
3. `source` 标注：手写先标 `handwritten`（对齐 W1 Day5 的 `synthetic` 思路）；真机录制后改 `recorded` 并带时间戳
4. `validate_golden.py` 在 CI 里跑：重复 id、悬空引用、难度缺失任一即失败

**自检**
- [ ] `uv run python evals/scripts/validate_golden.py` 通过，打印三档分布
- [ ] `wc -l evals/golden/golden_v1.jsonl` ≥ 50
- [ ] 3 条对抗样例单独 grep 出来人工确认预期是"拒绝"而非"照做"

### 🔍 交给我的锚点
golden_v1.jsonl 前 5 条 + schema.md + validate 输出。我会看：难度分布、对抗样例的预期是否真的"拒绝"、`references` 有无悬空。

---

## Day 3（3h）· 批量评测 runner + 落盘对接

**目标**：`make eval` 一条命令，把 ≥50 条跑完、算出分层指标、落进 `evals/history/YYYY-Www.json`。

### 你要做的事
- [ ] `evals/scripts/run_eval.py`：批量 runner（离线档 + 在线档）
- [ ] 结果落盘对齐 W44 json 的 `metrics` 结构（成功率/p95/成本/judge）
- [ ] `Makefile` 的 `make eval` 目标（W2 建过，本周接 run_eval）
- [ ] 单测：runner 失败不中断、钉 alias/温度、与 history json 对接

### 引导框架

```python
# run_eval.py 骨架
async def run_case(case: dict, gateway: LLMGateway, capture: LogCapture) -> EvalResult:
    """还原成 ChatCall -> 走 gateway -> 从 LogCapture.latest() 取指标 -> 对 expect 断言。"""
    ...

async def run_all(cases: list[dict], *, online: bool) -> TaskMetrics:
    """逐条跑，单条异常记录成 interrupted 不抛断整体；末尾聚合。"""
    ...
```

**算法步骤（今天最容易写错的地方）**
1. 每条 case 还原成 `ChatCall`，**强制钉 `case["model"]` alias + `opts={"temperature": 0}`**；不钉就是不可复现
2. 指标一律从 `capture.latest()`（即 `RequestLog`）读，**不自己计时**（W2 Day4 刚强调过）
3. 单条抛 `LLMError` → 记 `interrupted`，**不**算进 `passed`，也**不**中断整批（否则一条 502 毁掉整晚评测）
4. 并发受 W2 `ConcurrencyLimiter` 约束，别无脑 asyncio.gather 打满（否则触发自己刚做的背压）
5. 落盘：读 W44 json 结构，把本周 metrics 写进 `evals/history/YYYY-Www.json`；`judge_score` 本周留 null 标 pending
6. 在线档复用 `test_live.py` 的三重闸门（无 key 跳过），离线档 `model=fixture` 进 CI

**自检**
- [ ] `make eval` 离线档跑完 50+ 条，输出分层指标表
- [ ] 故意删一条 case 的引用文件 → 该条记失败但整批跑完（反向验证"不中断"）
- [ ] `evals/history/YYYY-Www.json` 的字段与 W44 完全对齐（W11 要读）

### 🔍 交给我的锚点
run_eval.py、一次完整 run 的输出、history json。我会看：**单条失败是否中断整批**、alias/温度是否钉死、指标是否从日志读。

---

## Day 4（2.5h）· 接 Langfuse 全链路 trace

**目标**：把 W2 留的 `TraceExporter` 协议换成真实实现，让"任意一次任务可回放完整 trace"（DoD 可观测项）真正落地。

### 你要做的事
- [ ] `tracing/langfuse_exporter.py`：`LangfuseExporter`（实现 W2 的 `TraceExporter`）
- [ ] 配置走 env（`LANGFUSE_PUBLIC_KEY`/`SECRET_KEY`/`HOST`），缺 key 自动降级 `NoopExporter`
- [ ] span 树：一次任务 → `llm.call` / `tool.execute` / `task.step` 分层
- [ ] `docs/w3/adr-w3-trace-backend.md`：为什么选它、字段映射、隐私边界
- [ ] 单测：缺 key 降级、异常 span 仍 export、trace_id 串联

### 引导框架

```python
# 只换实现，不改 W2 的协议
class LangfuseExporter:
    def __init__(self, settings: Settings) -> None:
        # 缺 key -> 内部退化成 no-op，绝不抛（评测/服务不应因观测后端挂掉）
        ...
    async def export(self, spans: list[Span]) -> None:
        """按 trace_id 聚成一条 trace，span_id/parent_span_id 还原树。"""
        ...
```

**算法步骤**
1. exporter 在应用边界（`create_app` 或 `build_gateway`）注入；库本身不硬连 Langfuse
2. 缺 key / 网络失败 → 静默降级 no-op + 一条 warning，**不影响主链路**（这是可观测组件的基本操守）
3. 字段映射：`Span.attributes` 里的 model/tokens/cost/error_code 映射到 Langfuse 的 observation metadata
4. **隐私边界**（写进 ADR）：`Message.content` 默认不上报全文，只上报长度/哈希；要上报全文必须显式开关（W10 安全周会用到这条）
5. `trace_id` 用 W2 的 contextvar，跨多次 LLM/工具调用串成一条

**自检**
- [ ] 有 key 时跑一次 `/v1/chat`，Langfuse 里能看到一条 trace 含 llm.call span
- [ ] 删掉 key 再跑 → 主链路正常，日志只有一条 warning（反向验证降级）
- [ ] 注入上游失败 → 失败 span 仍出现在 trace（W2 的 `finally export` 生效）

### 🔍 交给我的锚点
LangfuseExporter、ADR、一条真实 trace 截图。我会看：缺 key 是否真降级、有没有把用户 prompt 全文泄露到观测后端。

---

## Day 5（2h）· 评测报告模板 + 失败自动分类

**目标**：让评测结果"可讲"——一份人能读的报告 + 失败 case 自动归类（W4 根因归因的输入）。

### 你要做的事
- [ ] `evals/report_template.md`：报告模板（指标表 + 分布 + 趋势 + 失败分类）
- [ ] `evals/scripts/classify_failures.py`：`FailureCategory` 枚举 + 规则分类器
- [ ] 报告渲染进 `run_eval.py`（跑完自动产出 md）
- [ ] 更新 `docs/log.md` W3 条目；周五合流仪式

### 引导框架

```python
class FailureCategory(StrEnum):
    TOOL_ERROR = "tool_error"          # arguments_error / 未声明调用
    UPSTREAM = "upstream"              # 5xx/限流/鉴权 —— 这类不该算"模型笨"
    RETRIEVAL_MISS = "retrieval_miss"  # W5 后才有，先留枚举
    HALLUCINATION = "hallucination"    # 有 references 但答案无引用支撑
    TIMEOUT = "timeout"
    FORMAT = "format"                  # 输出不合 expect 结构
    BUDGET = "budget"                  # 超 per-task 预算
```

**算法步骤**
1. **规则分类先于 LLM 分类**：能从 error_code/字段判定的（TOOL_ERROR/UPSTREAM/TIMEOUT/BUDGET）先判，剩下的才考虑 W4 的 Judge
2. UPSTREAM 类要单列——它反映的是**可靠性**，不是"系统答得好不好"，混进 success_rate 会误导 W2 的结论
3. 报告模板固定四块：① 分层指标表 ② 按难度/tag 的成功率分布 ③ 与上周 history 的趋势对比 ④ Top 失败 case 分类
4. 周五合流：合 main → 跑 `make eval` → 写 `evals/history/YYYY-Www.json` → 更新趋势图（成功率/p95/单任务成本）→ `docs/log.md`

**自检**
- [ ] `make eval` 结束自动生成 `evals/history/YYYY-Www-w3-eval.md`
- [ ] 故意制造一条 502 → 该 case 被分类为 UPSTREAM 而非 HALLUCINATION（反向验证）
- [ ] 报告里的趋势对比能读到 W2 数据点（曲线不断）

### 🔍 Gate 3 · 周末验收（全绿才算过 W3）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 指标可定义 | metric-spec 每个指标有分子/分母/数据来源/边界 |
| 2 | golden 达标 | ≥50 条，三档分布，≥3 条对抗样例，引用无悬空 |
| 3 | 一条命令 | `make eval` 离线档跑完并出分层指标 |
| 4 | 可复现 | 钉 alias + 温度 0；同 fixture 两次跑数字一致 |
| 5 | 不中断 | 单条上游失败记 interrupted，整批跑完 |
| 6 | trace 落地 | 一次任务在 Langfuse 可回放，缺 key 自动降级 |
| 7 | 失败分类 | UPSTREAM 与 HALLUCINATION 不混淆 |
| 8 | 合流 | history json 对齐 W44，趋势曲线接上 W2 |

---

## 本周最容易返工的 6 处

1. **先写 runner 后补指标定义** → 指标变成"代码碰巧算出来的"，W4 门禁/W12 曲线全废
2. **`success_rate` 把上游中断算成"答错"** → 可靠性问题污染质量结论，W11 会误判
3. **自己 `time.monotonic`** → 与 W1/W2 的 `first_token_ms` 两套定义，曲线断裂
4. **没钉 alias/温度** → 两次跑数字不同，"可复现"检查点直接挂
5. **单条异常 `raise` 中断整批** → 一条 502 毁掉整晚评测
6. **把 prompt 全文上报 Langfuse** → W10 安全周会翻旧账；默认只报长度/哈希

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 方法论 / ADR | `docs/w3/{metric-spec.md,adr-w3-eval-scope.md,adr-w3-trace-backend.md}` |
| 指标层 | `evals/metrics.py` |
| Golden set | `evals/golden/{golden_v1.jsonl,schema.md}`、`evals/scripts/validate_golden.py` |
| 批量 runner | `evals/scripts/run_eval.py`、`Makefile`（make eval） |
| Trace 后端 | `tracing/langfuse_exporter.py` |
| 报告 / 分类 | `evals/report_template.md`、`evals/scripts/classify_failures.py` |
| 记录 | `evals/history/YYYY-Www.json` + `YYYY-Www-w3-eval.md`、`docs/log.md`（W3 条目） |

---

## P3 挂载点（见 ROADEMAP §7）

- **前端 trace 可视化**（P3-1 深入档）：本周 Langfuse 已能看到 trace；自建看板是深入档，最佳挂载点 W11 末 / W12
- **多语言评测集**（P3-2）：若目标 JD 有"出海/跨境"，可在 golden 里加 `locale` 维度（默认档 0.5 天），W10 内容周顺势做
- 决策点提醒：**W4 末是第一个 P3 决策点**，本周先把默认档评测闭环，届时按"余量 ≥15h"判断是否解锁一个深入档
