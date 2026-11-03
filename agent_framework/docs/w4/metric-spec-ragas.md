# RAGAS 三指标口径（W4 D2）

> 代码：`evals/judge/ragas_metrics.py` · 聚合：`evals/metrics.py::RagasScores` · 报告：`make run-eval`

## 0. 为什么这三个

W3 的指标全是"跑得通不通"（成功率、延迟、成本）。它们回答不了"答得对不对"。
Judge（D1）给主观分，但主观分需要人工一致性背书，而且不能复算。
RAGAS 三指标补的是**可复算**的那一半：

| 指标 | 一句话 | 它在防什么 |
|---|---|---|
| `faithfulness` | 答案里的陈述，有多少能被 context 支撑 | 编造（幻觉） |
| `answer_relevancy` | 答案有多少落在问题关心的内容上 | 答非所问 |
| `context_precision` | 给的 context 里，有多少真相关 | 检索噪声（W5 起才有真检索） |

## 1. `faithfulness`

```
faithfulness = 被支撑的陈述数 / 陈述总数
```

- **陈述切分**：按 `。！？；` 与换行切，长度 < 8 字的片段丢弃（"是的。""如下："不构成可核对的陈述，
  留着只会稀释指标）；空答案 → 陈述数 0。
- **支撑判定**（`ClaimSupport` 协议）：
  - `lexical`（默认）：陈述的**连续二元组**有 ≥60% 出现在 context 里；
  - `llm`：把 claim 与 contexts 一起给 strong 模型，问"能否支撑"，只答 SUPPORTED/UNSUPPORTED。

  为什么是二元组而不是单字：单字口径下，context 一旦变长（例如把整篇 `docs/architecture.md`
  当 context），**任何中文句子都能凑够单字重合**，faithfulness 会恒为 1.0——这是实测到的，
  一次 `make run-eval` 就把 9 条样本全顶成了 1.0。二元组对"是否原文支撑"更精确，
  代价是改写过的正确结论会被判无支撑（见 §6），所以 `lexical` 的结果是**下界**。
- **返回 `(值, 陈述数)`**，两个值都进报告：`0.5（2 条陈述）` 与 `0.5（20 条陈述）` 不是一回事。

**无 context → `None`，不是 `0.0`。** 这是本文件最容易被改坏的一条：
把"算不了"记成 0，均值会随数据集里"有多少条带引用"漂移，读者只会看到"faithfulness 掉了"，
然后去改 prompt——实际是数据集缺引用。

## 2. `answer_relevancy`

```
answer_relevancy = |问题实词 ∩ 答案实词| / |问题实词|
```

RAGAS 原版是"用答案反生成问题，再算与原问题的相似度"。这里用词面覆盖，理由是：
反生成要多一次模型调用、且引入新的不确定性，而**方向一致**（答案有没有落在问题关心的内容上）。
真需要反生成时，把 `support` 换成 LLM 版即可，接口不变。

空答案 / 空问题 → `None`。

## 3. `context_precision`

```
context_precision = 相关 context 条数 / context 总数
```

当前 context 就是 golden 的 `references[].target`（W5 接检索后换成真实 chunk 的前 k 条）。
相关性判定与 `faithfulness` 共用阈值（问题实词覆盖 ≥60%）。

**它现在是"引用质量"的代理，不是"检索质量"。** 报告里必须写清这一点，
否则 W5 接上检索后，读者会以为这个指标"退化了"。

## 4. 聚合口径（`TaskMetrics`）

| 字段 | 分母 | 说明 |
|---|---|---|
| `faithfulness` / `answer_relevancy` / `context_precision` | `ragas_samples` | 只对**算出值**的样本求均值 |
| `ragas_samples` | — | 有 references 且至少一项有值的 case 数 |

`None` 不进分母；全 `None` 时聚合值为 `None`。均值保留 4 位小数。

## 5. 成本

| 方式 | 额外调用 | 延迟 | 适用 |
|---|---|---|---|
| `lexical`（默认） | 0 | 0 | CI、离线档、大批量趋势 |
| `llm` | 每条陈述 1 次（strong tier） | +0.5–1.5s/条 | 需要"同义改写也能判支撑"时 |

`RagasScores.method` 记录用了哪种。**同一个 0.72 由 LLM 判定还是词面匹配得来，
可信度差一个档**，报告里不带这个字段，读者无法判断这个数字值多少。

## 6. 已知局限（写在前面，别等面试被问）

1. `lexical` 对同义改写不友好：`"客户端重试需要幂等键"` 与 `"用 idempotency key 去重"` 会被判无支撑
   （单测 `test_lexical_support_misses_paraphrase` 锁住了这条）。所以 `method=lexical`
   的数字是**下界**，只能横向比（同一批 case 之间），不能对外宣称绝对值。
2. `context_precision` 在 W5 前没有"检索"可言，只是 references 的噪声度量。
3. 陈述切分按标点，中文长句里塞多个事实时会被当成一条——指标会偏乐观。
4. 没有实现 RAGAS 的 `context_recall`：它需要"标准答案里的事实清单"，golden 里没有。
   补它要先给 golden 加 `key_facts` 字段，属 W5 的事。
