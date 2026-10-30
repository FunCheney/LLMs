# W4 执行清单（评测体系 II · Judge、回归门禁与根因归因 · P0，稀缺能力 ①）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> 本周是「评测设计」稀缺能力的**收口周**，也是 ROADEMAP §7 的**第一个 P3 决策点（W4 末）**。周五合流后按"余量 ≥15h"判断是否解锁一个深入档。
> W3 的 metric-spec / golden / runner / 失败分类是本周的输入，别重写。

## 承接面：W1–W3 已交付

| 已有 | 位置 | W4 的接缝 |
|---|---|---|
| `TaskMetrics`（含 `judge_score: None`） | `evals/metrics.py` | 本周用 Judge/RAGAS 填充它 |
| `golden_v1.jsonl`（≥50，带 `references`/`difficulty`） | `evals/golden/` | Judge 的输入；faithfulness 靠 `references` |
| `run_eval.py` + `make eval` + history json | `evals/scripts/` | 本周加"指标劣化即 fail"的门禁 |
| `classify_failures.py` 的 `FailureCategory`（规则分类） | `evals/scripts/` | 本周的 LLM 分类接在规则分类**之后** |
| `tracing` + `trace_id` 全链路 span | W3 D4 | 根因归因靠它定位到检索/提示词/工具/模型四层 |
| `models.yaml` 的 `strong` tier（claude-sonnet-4-6） | `config.py` | Judge 用 strong 模型、走 gateway，**不自己写 HTTP** |
| 基线数据点 W2/W3 | `evals/history/` | 门禁的对比基准 |

> W4 检查点（ROADEMAP）：capstone 有了"改坏了会被拦住"的护栏；3 篇「现象 → 根因 → 量化改进」案例可 10 分钟讲清。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接的 W3 符号 | 推进的 DoD 维度 |
|---|---|---|---|
| D1 | `evals/judge/{rubric.py,llm_judge.py}` + 一致性校验 | `TaskMetrics.judge_score`、gateway | 质量（judge ≥4.0/5） |
| D2 | `evals/judge/ragas_metrics.py`（faithfulness 等） | golden `references` | 质量（可量化） |
| D3 | `evals/scripts/regression_gate.py` + 基线快照 | run_eval、history json | 工程（CI 回归门禁） |
| D4 | `docs/w4/root-cause-playbook.md` + 归因脚本 | trace span、FailureCategory | 可观测/质量（可归因） |
| D5 | 3 篇根因案例 + 合流 | 上述全部 | 证据包第 4 件 |

**唯一主线**：给 capstone 装上"改坏了会被拦住"的护栏，并把失败从"一堆红叉"变成"可归因、可量化、可讲"的案例。**Judge 不是拿 LLM 随便打分**——rubric + 人工一致性校验是本周的技术核心。

---

## Gate 0 · 开工前（30 min）

- [ ] `make eval` 离线档全绿；`evals/history` 有 W2、W3 两个数据点（曲线已连上）
- [ ] 确认 `config.py` 里 `strong` tier 可用（Judge 走它）；**别在 Judge 里硬编码模型名或自己发 HTTP**
- [ ] 准备 ≥20 条**人工标注**样本（给 Judge 做一致性校验用，ROADEMAP D1 硬要求）
- [ ] 读 W3 的 `FailureCategory` 与 `metric-spec.md`，确认 UPSTREAM 类不进 judge

---

## Day 1（3h）· LLM-as-Judge：rubric + 一致性校验

**目标**：一个可复现、与人工标注对齐的自动评分器。**没有一致性校验的 Judge 等于随机数**，这也是面试里最容易被追问的点。

### 你要做的事
- [ ] `evals/judge/rubric.py`：评分维度 + 锚定描述（每个分值配文字，不是只给 1–5）
- [ ] `evals/judge/llm_judge.py`：`Judge` 协议 + `LLMJudge` 实现（走 gateway）
- [ ] `evals/scripts/judge_agreement.py`：≥20 条抽样的人工 vs Judge 一致性
- [ ] `docs/w4/adr-w4-judge-design.md`：为什么用强模型、如何防自偏好、成本预算
- [ ] 单测：rubric 解析、Judge 异常降级、温度 0

### 引导框架

```python
# evals/judge/rubric.py —— 分值必须锚定描述，否则 LLM 每次飘
@dataclass(frozen=True)
class Rubric:
    dimensions: tuple[str, ...]        # 如 ("事实准确","引用支撑","覆盖度","简洁")
    anchors: dict[str, dict[int, str]]  # 每个维度 1/3/5 分各配一段文字锚

# evals/judge/llm_judge.py
class Judge(Protocol):
    async def score(self, case: dict, answer: str) -> JudgeResult: ...

class LLMJudge:
    def __init__(self, gateway: LLMGateway, rubric: Rubric) -> None:
        # 钉 strong alias + temperature 0；输出 JSON（json_mode 能力，ADR）
        ...
```

**算法步骤（今天最容易写错的地方）**
1. rubric 每个分值写**文字锚**（"3 分=覆盖主问题但缺 1 个次要子问题"），不给锚 LLM 打分会集中到 4 分且不稳定
2. Judge 走 `LLMGateway.chat` + `model=strong` + `temperature=0` + `json_mode`；**禁止直接 import provider**（ADR-004）
3. 输出结构化 `{dimension: score, rationale}`；解析失败降级为 `judge_score=None` 并记 warning，**不要抛断整批评测**
4. 防自偏好：Judge 模型与被评模型尽量不同源；至少记录 Judge 的 alias（写进报告）
5. 一致性校验：≥20 条人工标注，算 Cohen's kappa 或 spearman；**低于阈值就回去改 rubric，不是提高温度**

**自检**
- [ ] `pytest -k judge` 覆盖：rubric 解析、JSON 解析失败降级、温度 0 断言
- [ ] `judge_agreement.py` 输出 ≥20 条的一致性系数；记录当前版本 rubric
- [ ] Judge 调用能在 Langfuse trace 里看到（复用 W3 exporter）

### 🔍 交给我的锚点
rubric 全文、LLMJudge、agreement 报告。我会看：分值有没有文字锚、Judge 是否走 gateway 而非裸 HTTP、一致性不达标时你的处理。

---

## Day 2（2.5h）· RAGAS 指标：faithfulness / relevancy / context precision

**目标**：把评测从"主观分"补上"可复算的客观分"。本周检索还没接（W5 才做 RAG），所以先落地**指标定义与计算骨架**，用 golden 的 `references` 当 context。

### 你要做的事
- [ ] `evals/judge/ragas_metrics.py`：三个指标的计算函数
- [ ] 与 `TaskMetrics` 对接（新增字段或子结构）
- [ ] `docs/w4/metric-spec-ragas.md`：每个指标的公式与所需输入
- [ ] 单测：无引用、引用不匹配、部分支撑三种边界

### 引导框架

```python
# evals/judge/ragas_metrics.py —— 只给定义与签名
def faithfulness(answer: str, contexts: list[str]) -> float:
    """答案中可被引用的事实占比；无 context 时返回 None（不是 0）。"""
    ...
def answer_relevancy(answer: str, question: str) -> float: ...
def context_precision(question: str, contexts: list[str]) -> float: ...
```

**算法步骤**
1. 三指标的输入来自 golden 的 `references`（W5 后换成真实检索 chunk），**本周别等检索**，先用 references 打通计算链路
2. `faithfulness` 用"答案陈述 → 是否被 context 支撑"的蕴含判定（可先用 LLM 判，记录成本）；**无 context 返回 None**，不能当 0 拉低均值
3. 三个指标都标注"计算方式 = LLM 判定 / 规则"，成本与延迟记进报告（W11 成本看板要用）
4. **不要引入一个庞大的 ragas 库依赖**除非你确认它可控——先用小函数实现定义，接口稳定后再换（框架纪律：只写被用到的）

**自检**
- [ ] `pytest -k ragas` 三种边界全绿（无引用返回 None 而非抛错）
- [ ] 一条带悬空引用的 case，faithfulness 明显低于正常 case（反向验证）

### 🔍 交给我的锚点
三个函数签名、metric-spec-ragas、测试结果。我会看：无 context 是否返回 None、有没有为一个未用的指标引入重依赖。

---

## Day 3（2.5h）· CI 回归门禁 + 基线快照

**目标**：让"指标劣化"在 CI 里自动 fail——这是 DoD「CI 跑 lint + test + eval 回归门禁」的关键一环，也是 capstone"改坏了会被拦住"的具体实现。

### 你要做的事
- [ ] `evals/scripts/regression_gate.py`：对比当前 metrics 与基线，超阈值退出非 0
- [ ] `evals/history/baseline.json`：可复现的基线快照（带 commit / 日期 / 模型快照）
- [ ] 接进 `.github/workflows/ci.yml`（W2 建的 workflow 加一步）
- [ ] `docs/w4/adr-w4-gate-threshold.md`：阈值设定与"允许的合理波动"
- [ ] 单测：劣化触发、波动内不触发、基线缺失处理

### 引导框架

```python
# regression_gate.py
@dataclass
class GateRule:
    metric: str          # "success_rate" | "latency_p95_ms" | "cost_per_task" | "judge_score"
    direction: str       # "higher_is_better" | "lower_is_better"
    max_regression: float  # 相对劣化上限，如 0.05 表示不许掉 5%

def evaluate(current: TaskMetrics, baseline: TaskMetrics,
             rules: list[GateRule]) -> GateVerdict:
    """返回 (passed, violations)；UPSTREAM 类激增要单独告警但不与成功率混算。"""
    ...
```

**算法步骤（今天最容易翻车）**
1. 阈值必须是**相对劣化**而非绝对值——不同机器跑 p95 天然有波动，绝对阈值会天天误报
2. 延迟类用相对值 + 一个绝对下限（避免基线本身很小导致相对值过敏）
3. **UPSTREAM 类激增单独告警**：它涨了说明可靠性退化，但不要因此把 success_rate 判 fail（W3 口径）
4. 基线快照带 `commit` + `recorded_at` + 模型 alias 快照日期（ADR-009，模型漂移会让基线失效）
5. CI 顺序：`ruff → mypy → pytest → make eval → regression_gate`；任一非 0 阻断合并

**自检**
- [ ] 把 baseline 的 success_rate 人为调高 → gate 变红（反向验证能拦住）
- [ ] 在正常波动内（±2%）跑两次 → gate 稳定通过（不误报）
- [ ] 删掉 baseline.json → 明确报错要求先建基线，不静默放行

### 🔍 交给我的锚点
regression_gate.py、baseline.json、CI diff、ADR。我会看：相对阈值是否抗波动、UPSTREAM 有没有被误算、基线有没有带模型快照。

---

## Day 4（2.5h）· 根因归因方法（四层定位）

**目标**：把 W3 的失败分类，变成"能从 trace 定位到具体层"的可操作流程。这是 W5–W11 每次调优的通用方法，也是证据包第 4 件的前置。

### 你要做的事
- [ ] `docs/w4/root-cause-playbook.md`：四层归因决策树（检索 / 提示词 / 工具 / 模型）
- [ ] `evals/scripts/attribute_failure.py`：从一次 trace 自动给"最可能层"提示
- [ ] 与 `classify_failures.py` 对接（规则分类 → 归因）
- [ ] 单测：四类样本各命中正确层

### 引导框架

**四层归因决策树（写进 playbook）**

| 观察到的信号（来自 trace/日志） | 归因层 | 典型修法 |
|---|---|---|
| 有 `retrieval_miss` / 引用不在 context | **检索层**（W5–W6） | 混合检索、chunk、重排 |
| 同一 prompt 换模型就变好 / 输出格式稳定错 | **提示词层**（W7/W10） | rubric、few-shot、指令分层 |
| `arguments_error` / 未声明工具 / 工具 500 | **工具层**（W8） | schema 描述、参数校验、权限 |
| 上游 5xx/限流/`finish_reason` 异常 | **模型/可靠性层**（W11） | fallback、重试、降级 |

```python
# attribute_failure.py
def attribute(trace: list[Span], failure: FailureCategory) -> str:
    """结合 FailureCategory 与 span 信号，返回四层之一 + 证据字段。"""
    ...
```

**算法步骤**
1. 先用 W3 规则分类定 `FailureCategory`，再用 span 信号细化到层——**顺序不能反**（否则每个失败都靠猜）
2. 每层归因必须带"证据字段"（哪个 span、哪个日志值），playbook 里写清"凭什么判这一层"
3. 归因结果落进评测报告，W5 起每次调优都回填"归因 → 改动 → 指标变化"
4. 本周检索层还没实现，`retrieval_miss`归因先能识别"信号缺失"，不误判成模型问题

**自检**
- [ ] 四类构造样本各命中正确层（反向：把工具 500 的样本改成没有 tool span，应判"证据不足"而非硬猜）
- [ ] playbook 每层都有"证据字段"和"典型修法"两栏

### 🔍 交给我的锚点
playbook、attribute_failure.py、四类样本输出。我会看：归因是否"规则优先"、证据字段是否具体、检索层缺失时会不会误判。

---

## Day 5（2h）· 3 篇根因案例 + 合流

**目标**：产出证据包第 4 件——**3 篇「现象 → 根因 → 量化改进」案例**，每篇含前后指标对比，可 10 分钟讲清。

### 你要做的事
- [ ] `docs/w4/cases/case-{1,2,3}.md`：三篇案例（现象/归因证据/改动/前后指标）
- [ ] 每篇至少改一处真实代码并跑 `make eval` 验证改进（不能只写不练）
- [ ] 更新趋势图数据点；周五合流仪式
- [ ] **W4 末 P3 决策**：按 ROADEMAP §7，余量 ≥15h 才解锁一个深入档（按 ROI 顺序：A2A 轻量 > 多语言 > 前端极简）

### 引导框架

**案例模板（每篇固定四段）**
```
## 现象
哪条 case / 哪次线上或评测失败，复现命令，当时的指标数字。
## 根因
用 D4 的 playbook 定位到哪一层，证据字段（span/日志值）。
## 改动
改了哪个文件、为什么这样改（引用 ADR）。
## 量化改进
改动前 vs 改动后的指标（成功率/p95/judge/成本），跑 `make eval` 的两次输出。
```

**算法步骤**
1. 三篇尽量覆盖**不同层**（如一篇提示词、一篇工具、一篇可靠性），展示归因方法的通用性
2. 每篇必须有"改动前 vs 改动后"的真实数字，用 `regression_gate` 证明改进没引入劣化
3. 改进若导致另一指标劣化，如实记录取舍（这比"全绿"更可信，是面试加分点）
4. 合流：合 main → `make eval` → 写 `evals/history/YYYY-Www.json` → 更 `docs/log.md` → 判定 P3

### 🔍 Gate 4 · 周末验收（全绿才算过 W4）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | Judge 可信 | rubric 有分值锚；≥20 条人工一致性达标 |
| 2 | RAGAS 落地 | 三指标可算，无 context 返回 None 不抛错 |
| 3 | 门禁生效 | 劣化触发 CI fail，正常波动不误报（反向验证过） |
| 4 | 归因可用 | 四类样本命中正确层，证据字段具体 |
| 5 | 案例达标 | 3 篇含前后指标对比，且各改了真实代码 |
| 6 | 曲线连续 | history 接上 W2/W3，趋势图可画 |
| 7 | P3 已判定 | 按余量明确"解锁/不解锁"，写进 log |
| 8 | 合流 | 端到端仍可跑，`docs/log.md` W4 条目完整 |

---

## 本周最容易返工的 6 处

1. **Judge 没做人工一致性校验** → 打分是随机数，面试一问就穿帮
2. **rubric 只给 1–5 不给文字锚** → 分数集中到 4 且不稳定
3. **Judge 直接裸 HTTP / 硬编码模型名** → 违反 ADR-004/009，模型一换全乱
4. **门禁用绝对阈值** → 不同机器 p95 波动天天误报，最后大家忽略红叉
5. **UPSTREAM 激增被算成成功率 fail** → 可靠性问题污染质量结论（W3 已定口径）
6. **案例只写不练** → 没有前后指标对比，证据包第 4 件不成立

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| Judge | `evals/judge/{rubric.py,llm_judge.py}`、`evals/scripts/judge_agreement.py` |
| RAGAS | `evals/judge/ragas_metrics.py`、`docs/w4/metric-spec-ragas.md` |
| 门禁 | `evals/scripts/regression_gate.py`、`evals/history/baseline.json`、CI diff |
| 归因 | `docs/w4/root-cause-playbook.md`、`evals/scripts/attribute_failure.py` |
| 案例 | `docs/w4/cases/case-{1,2,3}.md` |
| ADR | `docs/w4/adr-w4-{judge-design,gate-threshold}.md` |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W4 条目 + P3 决策） |

---

## P3 挂载点（W4 末 = 第一个决策点，见 ROADEMAP §7）

- 只看一个指标：**进度余量（领先/落后天数 × 3h）**；≥15h 才解锁**一个**深入档，做完再判断下一个
- 解锁顺序按 ROI：**A2A 轻量 > 多语言 > 前端极简 > 高并发 > 向量库/微调 > Computer Use**
- 无论是否解锁，**默认档一律完成**——它们是"作品完整度"的地板
- 本周若余量不足，明确写"不解锁"，把时间留给 W5 的 RAG（P0 主线）
