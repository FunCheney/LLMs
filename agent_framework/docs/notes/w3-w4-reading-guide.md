# W3 / W4 评测体系：设计思路与阅读路线

> 面向：代码已经跑通、但打开 `evals/` 目录不知道从哪一行开始读的人。
> 结论先行：**W3、W4 几乎没改业务代码**。`packages/agent_framework/` 只多了 `exporters/langfuse.py` 和几个 span，
> 其余 2000+ 行全在 `evals/` 下。它们不是新的一层，而是 W1/W2 那套运行时数据的**消费方**。
> 契约真相源：`docs/architecture.md`；口径真相源：`docs/w3/metric-spec.md`、`docs/w4/metric-spec-ragas.md`。

## 0. 一句话定位

| 周 | 解决什么问题 | 类比 | 检查点（ROADEMAP） |
|---|---|---|---|
| **W3** | 把"系统没坏"升级为"好到什么程度" | 加装**仪表盘** | `make eval` 一条命令出报告，数字可复现 |
| **W4** | 把"好到什么程度"升级为"改坏了会被拦住 + 坏了知道为什么" | 加装**刹车 + 诊断仪** | 有护栏；3 篇「现象 → 根因 → 量化改进」案例 |

判断自己有没有读懂，只需回答一个问题：**一个 `judge_score` 从 1.92 掉到 1.61，这份体系能不能告诉你"该改哪一层"，以及"这次该不该拦下合并"？**
W3 负责让这个数字存在且可信，W4 负责让它真的产生后果。

## 1. 为什么"不知道从何看起"是正常的

因为读代码的习惯是"从入口顺着调用链往下读"，而 W3/W4 **没有自己的请求入口**。它们站在三条既有链路上：

```
/v1/draft ──┐
            ├─→ LLMGateway.chat() ─→ RequestLog ─→ span ─→ TraceExporter
run_eval ───┘       ↑                                       ├─ Noop（默认）
                    │                                       └─ Langfuse（W3 D4）
          产品与评测共用同一个入口，不是两套系统
```

所以正确的读法是**逆着数据流读**：先读"结果长什么样"（`metrics.py` 的四个 dataclass），再读"怎么算出来的"（`run_eval.py` 三个函数），最后读"算出来给谁用"（`regression_gate.py` / `attribute_failure.py`）。

## 2. 六条设计思路（每个 `if` 的来源）

### 2.1 指标只认一个来源：`RequestLog`

`CaseMetrics.from_request_log()`（`evals/metrics.py:97`）的注释写得很直白：

> 对缺失字段一律给默认值而不是抛错：日志字段随版本增删（W1 的 18 项 → W2 的 20 项），
> 读取方必须能读旧日志，否则 W11 的历史曲线一改结构就断。

这解释了两件事：
- 为什么评测代码里**搜不到 `time.monotonic()`**——自己计时就会产生第二套延迟定义，趋势曲线从那一刻起断裂；
- 为什么读旧 json 不会报错而是拿到 `None`。

**反过来验证边界**：在 `evals/` 里搜 `httpx` 或 `import openai`，应该是 0 命中。命中即说明有人绕过 gateway 自己发请求，计量和 trace 同时失效。

### 2.2 四态机决定的是**分母**，不是标签

`CaseStatus`（`metrics.py:44`）只有四个值，但关键在 `counts_toward_success`（`metrics.py:53`）：

```python
return self in (CaseStatus.PASSED, CaseStatus.FAILED)
```

`interrupted`（上游 5xx、超时）**分子分母都不进**。理由：**可靠性问题不能污染质量结论**。
如果上游抖一下就把成功率拉低到 60%，你会去查提示词，而真正该做的是看厂商状态页。

`skipped`（离线档没跑）同理——它连 `RequestLog` 都没有（`CaseMetrics.has_log`，`metrics.py:87`）。
注意 `has_log` 的判断是 `latency_ms is not None`，不是"日志对象存在"：**有日志但没延迟 = 埋点坏了**，不是"这条很快"。

### 2.3 `None` 与 `0.0` 是两个不同的结论

这条在三个地方重复出现，是整套体系里最容易写错的一处：

| 字段 | `None` 含义 | `0.0` 含义 |
|---|---|---|
| `CaseMetrics.cost` | 没有成本数据（离线夹具不计价） | 免费 / 零用量 |
| `RagasScores.*`（`metrics.py:122`） | 这条算不了（无 references、答案为空）→ **不进均值** | 算了，一条都没被支撑 → **进均值** |
| `judge_score` | 没评（没 key / 离线档） | 评了，得 0 分 |

把前者当 0，会让"数据集里一半 case 没引用"看起来像"faithfulness 掉到 0.5"。
`RagasScores` 还额外记 `method`（`lexical` / `llm`）——同一个 0.72，LLM 判定和词面匹配的可信度差一个档，报告不写这个方法，读者无法判断这个数字值多少。

### 2.4 评测是"另一个调用方"，不是"另一套系统"

`run_eval` 通过 `build_gateway_or_raise()`（`run_eval.py:661`）拿 `LLMGateway`；`LLMJudge` 也走 gateway 的 strong alias。
所以整份评测代码里没有重试、没有计量、没有 HTTP——它和 `/v1/draft` 是同一个入口的两个用户。

好处是评测顺带验证了产品链路；代价是评测必须**钉 alias + `temperature=0`**（ADR-009），否则同一个 case 两次跑出两个分，后面所有"改善"都无法归因。

### 2.5 门禁只拦"回归"，不拦"绝对水平"

`DEFAULT_RULES`（`regression_gate.py:85`）：

```python
GateRule("success_rate",    HIGHER, 0.05)                              # 必判
GateRule("latency_p95_ms",  LOWER,  0.20, min_absolute=5.0, required=False)
GateRule("cost_per_task",   LOWER,  0.15, required=False)
GateRule("judge_score",     HIGHER, 0.05, required=False)
GateRule("faithfulness",    HIGHER, 0.10, required=False)
GateRule("tool_error_rate", LOWER,  0.50, required=False)
```

三处设计都在 `GateRule` 的 docstring 里（`regression_gate.py:63`）：
- `max_regression` 是**相对**比例。0.05 = "不许比基线差 5%"——**不是**"成功率必须 ≥ 95%"。后者是 DoD，不是门禁；混用会让基线很差时的合并被无故拦下。
- `min_absolute` 只对"越低越好"的指标生效，防止小基数过敏（p95 从 40ms 到 48ms 是 +20%，但不值得拦）。
- `required=False` = "基线里没有这个指标就不判"。离线档没有 `judge_score`，不设这个就会每次都 fail。

另有 `WARNING_RULES`（`regression_gate.py:96`）单独装 `interrupted_rate`：**只告警不 fail**，与 §2.2 同一口径。

### 2.6 四层归因里 `UNKNOWN` 不是"第五层"

`Layer`（`attribute_failure.py:50`）有 5 个值，但语义只有 4 层：

```python
RETRIEVAL → 查 chunk / 召回 / 重排（W5–W6）
PROMPT    → 改指令分层、rubric、few-shot（W7/W10）
TOOL      → 查工具 schema 描述、参数校验、权限（W8）
MODEL     → 查 fallback / 重试 / 降级，先看厂商状态页（W11）
UNKNOWN   → 先补埋点，缺哪个 span 就补哪个，别急着改代码
```

`UNKNOWN` 是"**证据不足**"，它的正确处理是补埋点，不是猜一个层然后改代码。这条决定了归因的可信度：允许猜，就等于允许把模型问题归到提示词上，然后白改一周。

## 3. 阅读路线（按此顺序，别从 `run_eval.py` 第 1 行开始）

`run_eval.py` 有 1121 行，其中 `print_table` / `render_markdown` / 各种 `_fmt_*` 占了近一半——那是渲染，不是逻辑。

| 步 | 文件 : 行 | 看什么 | 为什么排在这 | 预计 |
|---|---|---|---|---|
| 1 | `evals/metrics.py:44` `:60` `:123` `:168` `:194` | `CaseStatus` / `CaseMetrics` / `RagasScores` / `EvalResult` / `TaskMetrics` | 后面所有脚本都在填这五样。先建立"字段 → 口径"的印象 | 20min |
| 2 | `docs/w3/metric-spec.md` | 每个指标的**分子、分母、数据来源** | 代码里每个 `if` 的依据都在这里；先文档后代码会快一个数量级 | 15min |
| 3 | `run_eval.py:327 evaluate` → `:253 check` → `:533 _result` | 一条 case 的完整生命周期：跑 → 判 → 装 | 这三个函数读完，1121 行就只剩壳子 | 25min |
| 4 | `run_eval.py:591 run_offline` → `:627 run_online` → `:898 snapshot` | 从 1 条扩到全量 54 条，以及落盘 | 先懂单条，再懂批次，最后懂持久化 | 15min |
| 5 | `judge/rubric.py:83` → `judge/llm_judge.py:70` `:94` `:182` `:288` → `judge/ragas_metrics.py:152` `:188` `:207` `:231` | 评分器与三指标：`Rubric` → `Judge` 协议 → `LLMJudge` / `HeuristicJudge` → 三指标 | W4 的核心。字段依赖第 1 步，读不懂就回去补 | 40min |
| 6 | `regression_gate.py:63` `:85` `:149` → `attribute_failure.py:50` `:124` | 相对阈值、四层归因 | 消费端。看不懂数据结构就看不懂策略 | 25min |
| 7 | `tracing.py:74` `:107` `:136` → `exporters/langfuse.py:49` | 协议与实现分离 | 独立模块，不影响主线，最后看 | 15min |

**如果只有半小时**：只做第 1、2 步，再跑一次 §4 的第一条命令。先有"字段 → 口径 → 落盘"的印象，再读 `run_eval.py` 就不会迷路。

## 4. 跟一条数据走（三条命令，都已实测）

```bash
# ① 单条：看它怎么从 json 变成 EvalResult（实测 0.19s，1 passed）
uv run python evals/scripts/run_eval.py --only gold-c01 --no-write

# ② 故意让这条 case 失败，验证"单条失败不中断整批"（实测跑全量 54 条，5.46s）
uv run python evals/scripts/run_eval.py --break-case gold-c01 --no-write

# ③ 对最新快照做四层归因（无参数时自动取 evals/history 最新一期）
uv run python evals/scripts/attribute_failure.py
```

第 ① 条的输出值得逐字读：`cost 0.000000`、`quality 块 null`、提示语写着"只说明执行管道通，不代表模型质量"。
**这句提示就是 W3 的立场**——离线档证明的是管线，不是能力。看到 `quality` 全 `null` 时不该怀疑代码错了，该去看是不是没配 key。

两处实测差异，提前知道能省一次自我怀疑：

- ② **会跑全批，不是只跑那一条**（`--break-case` 是"注入一条失败"，不是"只跑这条"）。预期输出 `54 条：passed 49 / failed 1 / skipped 4 → success_rate 98.0%`——那个 `failed 1` 就是你注入的，断言的是"整批仍跑完"。
- ③ 当前 `2026-W47` 快照本身 0 失败，输出是 **"本期没有失败，无需归因"**。这不是跑错了，是没东西可归因。想看真实归因，把 ② 的 `--no-write` 去掉再跑一次让它落盘，然后重跑 ③；或直接对着失败 case 调 `attribute_result()`（`attribute_failure.py:304`）。

## 5. 三处容易读错的地方

1. **`evaluate()` 返回 `tuple[EvalResult, JudgeResult | None]`**，judge 结果**不进** `EvalResult`。
   因为它聚合方式不同（`mean_overall`，`llm_judge.py:310` 只对评成的样本求均值），且是另一个维度的观测。
   想找 judge 分数却去翻 `EvalResult` 的字段，会以为没实现。

2. **`_contexts()`（`run_eval.py:393`）对两种引用分叉处理，只有 `repo` 能产出 context。**

   | `kind` | 处理 | 结果 |
   |---|---|---|
   | `repo` | 读仓库内文件正文，**截前 6000 字符** | 有 context |
   | `url` | 直接返回 `None`，丢弃 | 无 context |

   丢弃 URL 的理由写在 docstring 里：`"https://modelcontextprotocol.io/specification"` 的实词是
   `https/modelcontextprotocol/io/specification`，任何答案都不会命中，faithfulness 会恒为 0，
   而读者会以为"模型在编造"。截 6000 字符的理由同样实在：整篇进词面判定会把 faithfulness 顶满，
   因为 context 越大越容易命中——6000 字符约等于一次检索返回的前几个 chunk。

   实际数据：`golden_v1.jsonl` 54 条里 **11 条**带 `references`，共 12 个条目（`url` 3 个、`repo` 9 个）。
   9 个 `repo` 全部指向存在文件，且几乎都是 `docs/architecture.md`——**本项目的架构文档**。

   **一处交叉验证**：11 条带引用的 case 里，`gold-c01` 和 `gold-a03` 只有 `url`（被丢弃）→ context 为空
   → `RagasScores` 全 `None` → 不进均值。11 − 2 = **9**，正好等于输出里的 `RAGAS（9 条）`。
   对得上，说明 §2.3 那个 `None` 语义不是纸面约定，而是在真实数据上生效的。

3. **离线档的 RAGAS 三个数字是夹具指纹，不是质量信号。**

   这条原来容易被误读成"模型在编造"，实际因果链是这样的：

   ```
   golden 的 model="fixture"
     → run_offline 取 offline.fixture.case = "text_basic"
     → 夹具固定答复："MCP 协议最新进展：新增了工具注解与异步任务支持。"
     → 无论问什么（gold-r05 问的是"主流 LLM 评测方法"），答案都是这一句
     → 拿去比 docs/architecture.md 的正文，必然不重合
     → faithfulness 0.0、relevancy 0.0543
   ```

   `relevancy` 不是 0 而是 0.0543，只是因为"MCP 协议"和问题里的少量字有偶然的 bigram 重合。
   **所以这三个数字现在一个都不能对外引用**——它们测的是夹具和 context 的巧合，与检索准不准、模型好不好都无关。
   等 W5 把 context 换成真实 chunk、并且在线档跑一次，才有讨论价值。

   顺带这也是 **Recall@K 目前算不出来**的原因：`references` 是"这个问题出自哪份文档"（`repo`/`url` 锚点），
   不是"这个问题该命中哪个 chunk"。没有 `query → chunk_id` 的标注，Recall / MRR / nDCG 都没有分母。

## 6. 与之前代码的接缝只有三处

| 接缝 | 上游（W1/W2） | W3/W4 怎么用 |
|---|---|---|
| **模型路由** | `models.yaml` 的 alias（ADR-009） | 评测钉 `alias` + `temperature=0`，不钉 wire 名 |
| **观测数据** | `observability.RequestLog` + `tracing.span` | 所有指标和归因证据的唯一来源 |
| **历史快照** | `evals/history/*.json` | W3 写、W4 门禁读；W11 成本看板、W12 趋势图也读它 |

新增的东西只有一类：**往这三条链路上挂消费者**（judge、ragas、gate、attribution、exporter）。
理解了这点，`git log` 里 W3/W4 那两周为什么 `packages/` 几乎没动，就不奇怪了。

## 7. 读完后应该能回答的五个问题

1. `success_rate` 的分母里为什么没有 `interrupted`？如果上游抖了 20 条，你该先看什么？
2. 为什么 `CaseMetrics` 从 json 读字段时给默认值而不是抛 `KeyError`？
3. `faithfulness = None` 和 `faithfulness = 0.0` 在报告里怎么区分？为什么这个区分很重要？
4. 门禁里 `judge_score` 为什么是 `required=False`？去掉会怎样？
5. 归因结果是 `UNKNOWN` 时，下一步动作是什么？（提示：不是"改提示词试试"）

答不上来就回到 §2 对应小节；五题都通，W3/W4 的代码你已经能改了。

## 8. 相关文档

| 文档 | 内容 |
|---|---|
| `docs/w3/w3-tasks.md` §承接面 | W1–W2 交付物与 W3 的接缝表（本文件 §6 的来源） |
| `docs/w3/metric-spec.md` | 指标口径：分子 / 分母 / 数据来源 |
| `docs/w3/adr-w3-eval-scope.md` | 为什么评测只做这些，不做什么 |
| `docs/w3/adr-w3-trace-backend.md` | 为什么协议与实现分离、默认 Noop |
| `docs/w4/metric-spec-ragas.md` | RAGAS 三指标口径与 `None` 语义 |
| `docs/w4/adr-w4-judge-design.md` | Judge 为什么走 gateway、为什么钉 alias |
| `docs/w4/adr-w4-gate-threshold.md` | 相对阈值与 `min_absolute` 的取值依据 |
| `docs/w4/root-cause-playbook.md` | 四层归因的实操手册 |
| `docs/w4/cases/case-{1,2,3}.md` | 「现象 → 根因 → 量化改进」三篇案例 |
| `docs/notes/llm-gateway-chat.md` | 上游：`LLMGateway.chat()` 逐行注释（读评测前建议先看这个） |
| `ops/RUNBOOK.md` §9 | 一分钟自检：改完代码先跑这个 |
