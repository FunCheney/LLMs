# W3/W4 评测设计的理论依据与文献溯源

> 这份文档回答一个问题：**这套评测体系里的每个设计，凭什么？**  
> 每条设计给出：① 仓库里的落点；② 理论/文献来源；③ 与文献原文的差异（工程改写处）。  
> 文末附完整引用列表。文献分了四档，先看 §12 的分档表再逐条读。
>
> **引用核对说明**（2026-11-07 编写时）：大部分出处已通过检索确认卷期/DOI/arXiv 号。  
> 三篇只确认了作者与标题、**arXiv 编号未能在线核实**（写稿时网络受限），已在文中标注  
> "编号待补"——引用这三篇前请先按标题检索确认编号。

---

## 1. 口径先行：`metric-spec.md` 写在 runner 之前

**设计**：`docs/w3/metric-spec.md` 先定义每个指标的分子/分母/数据来源/边界，之后  
`evals/metrics.py` 才被允许实现聚合逻辑（W3 D1）。

**依据**：这是"评测定义"与"评测实现"分离的方法论。Gehrmann, Clark & Sellam（2023，  
TACL）对生成文本评测实践做了系统性调查，把"指标定义随实现漂移"列为整个领域反复  
出现的裂缝之一——同一指标名在不同论文里口径不同，导致数字无法横向比较。他们提出  
的修正原则正是"先固定口径、再写代码，口径变更必须显式版本化"。本仓库的  
`RUBRIC_VERSION`、`metric-spec` 先于 runner、history json 只存"真实观测或明确的 None"  
都是这条原则的落地。

- Gehrmann, S., Clark, E., & Sellam, T. (2023). *Repairing the Cracked Foundation:  
  A Survey of Obstacles in Evaluation Practices for Generated Text*. **TACL** 11, 1033–1052.

---

## 2. 四态分母：`interrupted` 不进 `success_rate`

**设计**：一条 case 只能是 `passed / failed / interrupted / skipped` 四态之一；  
`success_rate = passed / (passed + failed)`，上游 5xx/限流/鉴权不进分母  
（`evals/metrics.py::CaseStatus`；metric-spec §1）。

**依据**：这是 SRE 的错误预算（error budget）思想，不是质量度量的发明。Google SRE  
Book 把"可靠性"与"功能正确性"定义为两个独立的 SLI：可用性下降用错误预算去扣，  
功能缺陷用缺陷率去扣，两者**故意不允许互相折算**——因为把一次上游宕机算进质量分，  
会让团队去修一个没坏的东西。本仓库的分母口径是这个原则在评测数据流上的逐字映射。

**工程改写处**：SRE 的原文讲的是在线服务，不是离线评测；把它搬进评测层时增加了一条  
判据——`interrupted` 的判定只认归一化错误码（`UPSTREAM_*`/`RATE_LIMITED`/`SERVER_BUSY`），  
不认异常类型。理由：离线档与在线档的异常可能来自不同进程，日志里的错误码才是  
共同语言（W4 D5 案例 3 发现的 `SERVER_BUSY` 双份码表问题）。

- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2016). *Site Reliability  
  Engineering: How Google Runs Production Systems*. O'Reilly.（第 2 章 "Embracing Risk"、  
  第 4 章 "Service Level Objectives"）

---

## 3. 分位数与最小样本：p95 为什么需要 ≥5 个样本

**设计**：`percentile(values, q, min_samples=5)`；样本不足返回 `None` 而不是最大值  
（`evals/metrics.py::_MIN_P95_SAMPLES`）。线性插值 `p = (n-1)·q`。

**依据（两层）**：

1. **尾延迟本身就是要看的东西**。Dean & Barroso（2013）的 *The Tail at Scale* 论证：  
   在分布式系统里，p50 几乎不带信息，延迟问题全部活在 p95/p99——一个请求若在 100 条  
   里只有 5 条慢，p50 完全无感，而用户体验恰好由那 5 条决定。这是为什么 W3 把  
   `latency_p95` 与 `first_token_p95` 列进必报指标而不是可选项。
2. **分位估计的统计口径**。线性插值 `p=(n-1)q` 是 Hyndman & Fan（1996）整理的  
   九种分位数定义里的 **type 7**，也是 R 的默认实现、NumPy 的默认 `linear` 方法。  
   选它不是因为它"更好"，而是因为 W1 的 `bench_baseline.py` 已经用了它——换实现  
   就等于换口径。n<5 时报 None 不是 Hyndman & Fan 的原话，是顺序统计量的直接推论：  
   5 个样本的"p95"数学上就是 max 的别名，换一个名字不改变信息量，却会让读者以为  
   看到了一个新指标。

- Dean, J., & Barroso, L. A. (2013). The Tail at Scale. **CACM** 56(2), 74–80.
- Hyndman, R. J., & Fan, Y. (1996). Sample Quantiles in Statistical Packages.  
  **The American Statistician** 50(4), 361–365.

---

## 4. rubric 必须有文字锚：行为锚定评分法（BARS）

**设计**：`Dimension.anchors` 要求 1/3/5 分各配一段**可观察的行为描述**  
（"漏掉一个次要子问题"），不允许形容词（"较好""一般"）；2/4 分由"介于两档"隐含  
（`evals/judge/rubric.py`；ADR-W4-01 D1）。

**依据**：这是 1963 年的心理测量学方法，不是 LLM 时代的发现。Smith & Kendall（1963）  
提出 Behaviorally Anchored Rating Scales（BARS）：每个分值必须由一个具体的、  
可观察的行为实例来锚定，锚定文本还要经过"retranslation"（另一组专家在不知道原分组  
的情况下把它放回同一维度）才算有效。BARS 解决的问题与本仓库完全相同：  
没有行为锚的评分者会把所有分数堆到中间档——在 1963 年是护士绩效评定的"4 分堆"，  
在 2024 年是 LLM 对 1–5 分题一律给 4 分。ADR-W4-01 里"没有锚定描述时，54 条 case  
会有 50 条落在 4 分"正是 BARS 文献里叫 *central tendency error* 的现象。

**工程改写处**：BARS 的"retranslation"步骤在仓库里没有逐字实现（需要第二组标注员）；  
替代方案是 `judge_agreement.py` 的一致性校验——它检验的是"这个锚定集合在 LLM 上是否  
可复现"，比 BARS 的"在人类评分者间可复现"弱，但方向一致。这是已知的降级，写进了  
ADR-W4-01 §4。

- Smith, P. C., & Kendall, L. M. (1963). Retranslation of expectations: An approach to  
  the construction of unambiguous anchors for rating scales. **Journal of Applied  
  Psychology** 47(2), 149–155. DOI: 10.1037/h0047060.

---

## 5. LLM-as-a-Judge：协议、偏置与"地板"

**设计**：judge 走 gateway、钉 strong alias、`temperature=0`、解析失败降级为  
`parse_error` 而不是补中间分；judge 与被评模型**不同源**；`judge_alias` 必须落盘  
（`evals/judge/llm_judge.py`；ADR-W4-01 D2–D4）。

### 5.1 方法本身与三类偏置

**依据**：Zheng et al.（2023，NeurIPS）是 LLM-as-a-Judge 的奠基文献。他们测出 GPT-4  
与人工偏好的一致率 >80%（MT-Bench 85%，与人类之间的一致率 81% 相当），但**同时**测出  
三类系统偏置：position bias（pairwise 比较时偏好先出现的答案）、verbosity bias  
（偏好更长的答案）、self-enhancement bias（偏好自己同源的输出，GPT-4 约 +10%，  
Claude-v1 约 +25%）。本仓库的设计逐条对应：

| 文献发现的偏置               | 本仓库的对应                                                                |
| --------------------- | --------------------------------------------------------------------- |
| verbosity bias        | rubric 把 `concision` 列为独立维度并给权重，让"长"不再白送分                             |
| self-enhancement bias | judge 取 `routes["strong"]` 而**被测**取 `cheap`，默认不同源；`judge_alias` 落盘供自查 |
| position bias         | 单条评分（Likert）而非 pairwise，从根上消除这个维度                                     |
| 数值漂移                  | `temperature=0` + alias 钉死                                            |

**但 Likert 单条评分也有代价**：Zheng 原文同时指出，Likert 制下 judge 的区分度更差、  
容易堆 4 分——这正是 §4 需要 BARS 锚定的原因。两条约束是连在一起的，拆开任何一条  
都立不住。

### 5.2 防自偏好的后续证据

Panickssery et al.（2024，NeurIPS）证明 judge 能**识别**并偏好自己的生成——  
即使不告诉它答案来自谁。Wataoka et al.（2024）给出 0–1 的偏置度量  
（GPT-4 得分 0.749，八模型最高），并指出机制与 perplexity（熟悉度）相关：  
模型偏好的不只是"自己的输出"，而是"它觉得眼熟的文本"。这两条说明  
"换一个 vendor 的 judge"并不足够——所以仓库里 `judge_alias` 必须**显式落盘**，  
让读者自己判断，而不是在代码里假装问题不存在。

### 5.3 一致性校验不是可选项，是判据本身

**依据**：Krumdick et al.（2025）用 1,200 条人工标注证明：没有 ground-truth 参考  
答案时，judge 只在"它自己能答对"的题上与人类一致——聚合分数看起来很好，难子集上  
完全不可信。结论是"LLM-as-a-Judge 不能免除人工监督"。这与 ADR-W4-01 §5 是同一条：  
**一致性系数（不是 judge 的分数）才是交付物**。

**依据（更早的先例）**：Ouyang et al.（2022，InstructGPT）用人工标注训练奖励模型时，  
明确要求标注员间 agreement 达标后才允许训练——这是"先校验标注质量再谈模型输出"  
在工业界的第一次大规模落地。本仓库的 `--live` 一致性流程是它的 mini 版。

- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.  
  **NeurIPS 2023**. arXiv:2306.05685.
- Panickssery, A., Bowman, S., & Feng, S. (2024). LLM Evaluators Recognize and Favor  
  Their Own Generations. **NeurIPS 2024**（arXiv 编号待补，按标题检索）。
- Wataoka, K., et al. (2024). Self-Preference Bias in LLM-as-a-Judge  
  （arXiv 编号待补，按标题检索）。
- Krumdick, N., Lovering, C., Reddy, V., Ebner, Z., & Tanner, C. (2025).  
  *No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding*  
  （arXiv 编号待补，按标题检索）。
- Ouyang, L., et al. (2022). Training language models to follow instructions with human  
  feedback. **NeurIPS 2022**. arXiv:2203.02155.

### 5.4 "规则基线地板"：为什么要养一个 `HeuristicJudge`

**设计**：`HeuristicJudge` 只做关键词匹配，kappa 实测只有 0.343/0.200，但它的存在  
给 LLM judge 划了一条**地板**——任何 LLM judge 的一致性必须显著高于它，否则不如不用  
（ADR-W4-01 §3）。

**依据**：这是心理测量学里"指标必须证明优于平凡基线"的纪律。在 LLM 评测语境下，  
Ethayarajh & Jurafsky（2020）批评 leaderboard 上很多指标之所以看起来在涨，是因为  
没人去验证"这个指标本身是否优于一个不动脑的规则"。`HeuristicJudge` 就是把这条批评  
写成了代码：**先造一个你一定会鄙视的基线，然后要求真 judge 显著超过它**。  
0.075→0.347 的二元组→单字切换实验（ADR-W4-01 §3.2）说明这条地板不是摆设——  
第一版实现连这个基线都没达到。

- Ethayarajh, K., & Jurafsky, D. (2020). Utility Is in the Eye of the User: A Critique  
  of NLP Leaderboards. **EMNLP 2020**. arXiv:2009.13888.

---

## 6. 一致性系数的选用：为什么是加权 kappa + 名义 kappa + Spearman

**设计**：`judge_agreement.py` 输出三个数：二次加权 kappa、三档名义 kappa、Spearman ρ；  
门禁看**名义 kappa**（更严）。`Row` 里同时存 `judge`（浮点）与 `judge_rounded`（整数）。

**依据**：

1. **Cohen's kappa**（Cohen 1960）是"扣除随机一致后的观察一致率"的标准度量，  
   用于名义（无序）类别。评分是 1–5 的**有序**类别，因此用加权版——Cohen（1968）  
   提出 quadratic weights，让"差 1 档"的惩罚小于"差 3 档"。
2. **Kappa 的 prevalence paradox**：Feinstein & Cicchetti（1990）指出，当评分  
   高度集中（prevalence 失衡，比如 90% 都是 4 分）时，即使观察一致率很高，  
   kappa 也会被人为压低。Byrt et al.（1993）给出应对：kappa 必须与 **bias 与  
   prevalence 的指标一起报告**，不能只报一个 κ。这就是为什么 `judge_agreement.py`  
   除了三个系数还输出 `human_distribution` 与逐条对照——单独一个 0.34 没有解释力。
3. **Spearman 用浮点原值**：kappa 要求有序整数（人工标注只有整数），Spearman  
   用秩相关，可以拿 judge 的加权浮点总分直接算——两种量纲都留，是为了回答  
   两个不同的问题："judge 的绝对档位是否对齐"（kappa）与"judge 的相对排序是否一致"  
   （Spearman）。一个指标回答不了两个问题。

- Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. **Educational and  
  Psychological Measurement** 20(1), 37–46.
- Cohen, J. (1968). Weighted Kappa: Nominal Scale Agreement with Provision for Scaled  
  Disagreement or Partial Credit. **Psychological Bulletin** 70(4), 213–220.
- Feinstein, A. R., & Cicchetti, D. V. (1990). High agreement but low kappa: I. The  
  problems of two paradoxes. **Journal of Clinical Epidemiology** 43(6), 543–549.
- Byrt, T., Bishop, J., & Carlin, J. B. (1993). Bias, prevalence and kappa. **Journal of  
  Clinical Epidemiology** 46(5), 423–429.

---

## 7. `temperature=0` 但不承诺确定性

**设计**：评测与 judge 都强制 `temperature=0`，但 `judge_agreement.py` 用"系数是否  
达标"而不是"两次打分是否完全相同"来判定可复现性（ADR-W4-01 D3）。

**依据**：Song et al.（2024）在 7 个 benchmark 上系统测量了 LLM 的非确定性：  
**即使 temperature=0 / greedy decoding，同一输入也会得到不同输出**——源头是  
浮点非确定性、batch 归并（continuous batching）与推理后端的实现差异。他们的  
结论直接改写了"可复现"的定义：不是"两次跑数字相同"，而是"两次跑的分布相同"，  
检验工具是统计量（agreement 系数、置信区间）而不是逐位相等。仓库的  
ADR-W4-01 D3 是对这个结论的逐字翻译。更系统的机制分析见  
"Non-Determinism of 'Deterministic' LLM Settings"（arXiv:2408.04667）。

- Song, Y., Wang, G., Li, S., & Lin, B. Y. (2024). The Good, The Bad, and The Greedy:  
  Evaluation of LLMs Should Not Ignore Non-Determinism. arXiv:2407.10457.
- "Non-Determinism of 'Deterministic' LLM Settings". arXiv:2408.04667.

---

## 8. RAGAS 三指标：定义、None 语义与"已知下界"

**设计**：`faithfulness`（陈述被支撑比例）、`answer_relevancy`（问题实词覆盖）、  
`context_precision`（context 相关比例）；无 context 一律返回 `None` 不是 `0.0`；  
默认 `lexical`，`method` 字段必须落盘（`evals/judge/ragas_metrics.py`；metric-spec-ragas.md）。

**依据**：Es et al.（2024，EACL）是这三个指标的原始定义者。我们的公式与原文  
**逐项对应**：

| 指标                | RAGAS 原文定义                            | 本仓库的等价实现                        | 改写理由                                |
| ----------------- | ------------------------------------- | ------------------------------- | ----------------------------------- |
| faithfulness      | LLM 把答案拆成 claims，逐条判是否被 context 蕴含，比例 | `LexicalSupport`：二元组覆盖 ≥60% 判支撑 | LLM 判定每条 1 次调用、不可复算；lexical 可复算、零成本 |
| answer_relevancy  | 用答案反生成问题，算与原问题的 embedding 相似度         | 问题实词被答案覆盖的比例                    | 反生成多一次调用且引入新随机性；方向一致                |
| context_precision | 相关 context 条数 / 总条数（后续版本升级为 AP@K）     | 同一比例定义                          | 原文定义直接可用                            |

**关键差异（必须承认）**：`lexical` 是 RAGAS 的**下界实现**。同义改写会被判无支撑  
（单测 `test_lexical_support_misses_paraphrase` 锁住了这条），所以  
`method=lexical` 的数字只能横向比（同一批 case 之间），不能对外宣称绝对值。  
`method` 字段落盘是对 Gehrmann et al.（2023）"指标口径必须显式版本化"的直接遵守。

**None 而非 0.0 的依据**：这不是 RAGAS 的原文，而是 Gehrmann et al.（2023）调查中  
反复强调的一条——"无法计算"与"计算结果为 0"是两个不同的测量结果，把它们混在  
一个字段里是评测漂移最常见的源头。这也是为什么 `RagasScores` 的每个字段都允许  
`None`，且 `claims`/`contexts` 必须单独落盘（0.5（2 条陈述）≠ 0.5（20 条陈述））。

- Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2024). RAGAs: Automated  
  Evaluation of Retrieval Augmented Generation. **EACL 2024 (System Demonstrations)**,  
  150–158. DOI: 10.18653/v1/2024.eacl-demo.16.

---

## 9. 四层归因：先规则分类，再 LLM 归因

**设计**：`FailureCategory`（规则）→ `attribute()`（结合 span 信号定层）→  
四层（检索/提示词/工具/模型）。规则分类**先于** LLM 归因，不能反  
（`evals/scripts/classify_failures.py`、`attribute_failure.py`；root-cause-playbook.md）。

**依据（两条线）**：

1. **RAG 失败模式的经验枚举**。Barnett et al.（2024，CAIN）在三个独立领域的  
   案例研究中归纳出 RAG 系统的 7 个失败点：missing content（语料没有）、  
   missed the top ranked documents（语料有但没召回）、not in context（召回但没进  
   上下文）、not extracted（进了上下文但没被提取）、wrong format、  
   incorrect specificity、incomplete。本仓库的"检索层"分支几乎是对这张表的  
   代码化：`retrieval_spans=0` 对应 missing content，`retrieved∩cited=∅` 对应  
   missed the top ranked / not in context，`引用了但不支撑结论` 对应 not extracted。  
   Barnett 的另一个结论——"RAG 系统的鲁棒性只能在运行中逐步建立，不能一开始  
   设计进去"——正是 W5 那句"先证明每一层有收益再保留"的理论来源。


1. **行为测试优于单一总分**。Ribeiro et al.（2020，ACL）的 CheckList 证明：  
   单一 accuracy 掩盖了失败模式的多样性，应该把失败**按能力维度**分桶再谈改进。  
   本仓库的 `FailureCategory` 就是 CheckList 思想在运行期日志上的实现——  
   `HALLUCINATION`/`TOOL_ERROR`/`FORMAT`/`BUDGET` 对应不同的"能力维度"，  
   各桶有各自的修法（CheckList 里叫 MFT/INV/DIR 测试类型）。

**"先规则后 LLM"没有强文献依据**，它是工程判断：能从错误码判的（UPSTREAM/TOOL_ERROR）  
不该浪费一次 LLM 调用去判，也不该让一个不可复算的判定去复核一个可复算的判定。  
但这条与 Krumdick et al.（2025）的精神一致——LLM 判定必须用在规则判定**够不到**  
的地方，而不是所有地方。

- Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., & Abdelrazek, M. (2024).  
  Seven Failure Points When Engineering a Retrieval Augmented Generation System.  
  **CAIN 2024** (IEEE/ACM 3rd Int. Conf. on AI Engineering), 194–199. arXiv:2401.05856.
- Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond Accuracy:  
  Behavioral Testing of NLP Models with CheckList. **ACL 2020**. arXiv:2005.04118.

---

## 10. 回归门禁：相对劣化 + 绝对下限

**设计**：`GateRule(max_regression=0.05)` 是相对值；延迟类加 `min_absolute=5.0`；  
`interrupted_rate` 只告警不 fail；基线带 `commit`/`recorded_at`/`model_snapshot`  
（`evals/scripts/regression_gate.py`；ADR-W4-02）。

**依据**：这是 SRE 错误预算在 CI 上的应用，不是学术发现。SRE Book 的原则是  
"阈值必须表达为与基线的**相对**偏差，且必须能区分'信号'与'噪声'"——绝对阈值  
在负载/环境变化时必然误报。本仓库 W45 的实测（同一台机器两次跑 p95 相差 8%~25%）  
是这条原则在小样本上的直接体现。`min_absolute` 是对"小基数过敏"的补丁：  
2.3ms 的 20% 只有 0.46ms，任何进程调度抖动都会触发，所以还要绝对增量 >5ms。

**没有论文直接支持这个具体设计**——它是 SRE 思想 + 本地实测的工程结论。  
把它写进 ADR 的原因是：这类设计最怕被当成"拍脑袋"，必须把"为什么不用绝对阈值"  
的推理过程留给下一个改阈值的人。

- Beyer et al. (2016), *Site Reliability Engineering*（同上）。

---

## 11. 本仓库自己的实证发现（无论文，是本地测量）

以下结论**没有文献支撑，是本仓库跑出来的**。它们与文献同样重要——  
文献告诉你"可能会怎样"，本地测量告诉你"在你的数据上确实怎样"：

| 发现                                      | 出处                         | 一句话                              |
| --------------------------------------- | -------------------------- | -------------------------------- |
| 二元切片 kappa 只有 0.075                     | ADR-W4-01 §3.2             | 改写后的答案几乎不复用问题里连续两个字，二元覆盖率恒低      |
| `HeuristicJudge` 从不给 1–2 分              | ADR-W4-01 §3               | "事实错误"在词面上往往比正确答案更像答案，关键词匹配判不了事实 |
| `temperature=0` 的 judge 一致性需用系数而非逐位相等验证 | ADR-W4-01 D3               | 见 §7 的文献，但具体系数 0.343/0.200 是本地实测 |
| 基线 p95=2.3ms 时 20% 相对阈值会天天误报            | ADR-W4-02 D2               | 单机两期实测，样本不足，W5 起每期回看             |
| 混合检索/精排/改写在本语料上是净损失                     | `docs/w5/rag-ab-report.md` | 标准答案式 RAG 架构不总是对的，分层验证比堆组件重要     |
| 标点进分词器让 sparse R@1 从 .2500 跳到 .5357     | `docs/w5/case-1-*.md`      | 分词口径的修复收益远大于换召回算法                |

**这些是证据链的一部分，不是补充说明。** 文献给方向，本地实测给判决——  
两者一起才是"为什么要这样设计"的完整答案。

---

## 12. 文献分档表（按证据强度）

| 档             | 含义                  | 文献                                                                                                                                                                                                |
| ------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A · 经典/奠基** | 方法论的原始来源，改变过一个领域    | Smith & Kendall 1963（BARS）；Cohen 1960/1968（kappa）；Dean & Barroso 2013（尾延迟）；Hyndman & Fan 1996（分位数）；Zheng et al. 2023（LLM-as-Judge）；Es et al. 2024（RAGAS）；Ribeiro et al. 2020（CheckList）           |
| **B · 重要后续**  | 量化或修正了 A 档的结论       | Feinstein & Cicchetti 1990（kappa paradox）；Byrt et al. 1993；Panickssery et al. 2024（自偏好）；Song et al. 2024（非确定性）；Krumdick et al. 2025（judge 需要人工锚）；Barnett et al. 2024（RAG 失败点）；Wataoka et al. 2024 |
| **C · 方法论综述** | 不提出新方法，但证明"如果不做会怎样" | Gehrmann et al. 2023（评测口径漂移）；Ethayarajh & Jurafsky 2020（指标需优于基线）；Ouyang et al. 2022（标注一致性先例）                                                                                                      |
| **D · 工程文献**  | 不是论文，是工业界沉淀         | Beyer et al. 2016（SRE Book）                                                                                                                                                                       |

---

## 13. 完整引用列表

```
Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., & Abdelrazek, M. (2024).
  Seven Failure Points When Engineering a Retrieval Augmented Generation System.
  CAIN 2024, 194–199. arXiv:2401.05856.

Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2016).
  Site Reliability Engineering: How Google Runs Production Systems. O'Reilly.

Byrt, T., Bishop, J., & Carlin, J. B. (1993). Bias, prevalence and kappa.
  Journal of Clinical Epidemiology, 46(5), 423–429.

Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales.
  Educational and Psychological Measurement, 20(1), 37–46.

Cohen, J. (1968). Weighted Kappa: Nominal Scale Agreement with Provision for Scaled
  Disagreement or Partial Credit. Psychological Bulletin, 70(4), 213–220.

Dean, J., & Barroso, L. A. (2013). The Tail at Scale. CACM, 56(2), 74–80.

Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2024). RAGAs: Automated
  Evaluation of Retrieval Augmented Generation. EACL 2024 (System Demonstrations),
  150–158. DOI: 10.18653/v1/2024.eacl-demo.16.

Ethayarajh, K., & Jurafsky, D. (2020). Utility Is in the Eye of the User:
  A Critique of NLP Leaderboards. EMNLP 2020. arXiv:2009.13888.

Feinstein, A. R., & Cicchetti, D. V. (1990). High agreement but low kappa: I. The
  problems of two paradoxes. Journal of Clinical Epidemiology, 43(6), 543–549.

Gehrmann, S., Clark, E., & Sellam, T. (2023). Repairing the Cracked Foundation:
  A Survey of Obstacles in Evaluation Practices for Generated Text. TACL, 11, 1033–1052.

Hyndman, R. J., & Fan, Y. (1996). Sample Quantiles in Statistical Packages.
  The American Statistician, 50(4), 361–365.

Krumdick, N., Lovering, C., Reddy, V., Ebner, Z., & Tanner, C. (2025). No Free Labels:
  Limitations of LLM-as-a-Judge Without Human Grounding.（arXiv 编号待补）

Ouyang, L., et al. (2022). Training language models to follow instructions with human
  feedback. NeurIPS 2022. arXiv:2203.02155.

Panickssery, A., Bowman, S., & Feng, S. (2024). LLM Evaluators Recognize and Favor
  Their Own Generations. NeurIPS 2024.（arXiv 编号待补）

Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond Accuracy: Behavioral
  Testing of NLP Models with CheckList. ACL 2020. arXiv:2005.04118.

Smith, P. C., & Kendall, L. M. (1963). Retranslation of expectations: An approach to
  the construction of unambiguous anchors for rating scales. Journal of Applied
  Psychology, 47(2), 149–155. DOI: 10.1037/h0047060.

Song, Y., Wang, G., Li, S., & Lin, B. Y. (2024). The Good, The Bad, and The Greedy:
  Evaluation of LLMs Should Not Ignore Non-Determinism. arXiv:2407.10457.

Wataoka, K., et al. (2024). Self-Preference Bias in LLM-as-a-Judge.（arXiv 编号待补）

Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
  NeurIPS 2023. arXiv:2306.05685.

Non-Determinism of "Deterministic" LLM Settings. arXiv:2408.04667.
```

推 3–5 篇最值得一读的，我推荐这 5 篇（按与我们评测设计的对应关系排序）：

1. Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685

LLM-as-a-Judge 的奠基文献。它实测了 GPT-4 judge 与人类的吻合度，并系统定义了 position bias、verbosity bias、self-enhancement bias——我们 W3 里「配对打分要换序复测」「判据必须写进 rubric 而不靠 judge 自由心证」这些设计就是直接从它的偏差清单反推出来的。要理解「为什么 judge 不能裸用」，这篇是起点。
2. Es et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL 2024 (System Demonstrations)

faithfulness / answer relevance / context precision 三个指标的原始定义出处。我们 W4 的 RAGAS 公式与原文逐条对齐过，读原文能确认我们没有在转述中把指标语义改漂移（这正是第 4 篇警告的事）。
3. Barnett et al. (2024). Seven Failure Points When Engineering a Retrieval Augmented Generation System. CAIN 2024. arXiv:2401.05856

RAG 失效模式分类学。我们 attribute_failure.py 的归因类别（retrieval_miss / citation_miss / hallucinated_index 等）和 W6 引用评测的 failure 分类都是以它为骨架再按实测调整的。篇幅短、实操性强。
4. Gehrmann et al. (2023). Repairing the Cracked Foundation: A Survey of Obstacles in Evaluation Practices for Generated Text. TACL 11, 1033–1052

解释了为什么「同名指标不同实现」是行业病——这是我们坚持 --record/--replay 落盘可复现、把指标定义写死在代码里而非文档里的理论依据。想理解 W3 评测工程纪律（为什么不是跑个分数就完事），读这篇。
5. Ribeiro et al. (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL 2020. arXiv:2005.04118

「准确率之外的 behavioral testing」思想来源。我们的 golden 集按能力切片（可答/应拒答/边界），以及回归门禁「逐 case 状态对比」而非只看总分，都是 CheckList 方法论在 RAG 场景的落地。
备选一篇（如果你更关心延迟那条线）：Dean & Barroso (2013). The Tail at Scale. CACM 56(2)——为什么门禁盯 p95 而不是均值，这篇是最经典也最短的论证。
