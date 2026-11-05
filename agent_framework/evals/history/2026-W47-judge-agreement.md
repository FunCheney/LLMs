# Judge 一致性报告 · 2026-W47（offline）

- judge：`heuristic-keyword`；rubric：`answer-quality` vw4-v1
- 标注文件：`evals/judge/annotations_v1.jsonl`（24 条）
- 生成时间：2026-11-03T22:25:49+08:00

> **这份数字不能用来宣称 judge 可信。** `heuristic-keyword` 是关键词规则基线，
> 它给出的是**地板**：真正的 LLM judge 必须明显高于它，否则不如不用。
> 要拿到 LLM judge 的一致性，需要 `--live`（依赖 `.env.local` 的 key）。

## 系数

| 指标 | 值 | 说明 |
|---|---|---|
| 样本数 | 24 | 可用 judge 结果 |
| 不可用 | 0 | judge 报错/解析失败，不计入系数 |
| 二次加权 kappa | 0.3429 | 有序，差 1 分轻罚 |
| 三档名义 kappa | 0.2 | 门禁用这个（低/中/高） |
| Spearman | 0.4368 | 只看排序 |
| 完全一致率 | 0.1667 | 分数完全相同 |
| MAE | 1.1667 | 平均差几分 |

## 分数分布（看有没有全堆在 4 分）

- 人工：`{1: 2, 2: 7, 3: 2, 4: 3, 5: 10}`
- judge：`{2: 6, 3: 12, 4: 6}`

## 逐条

| 标注 | case | 人工 | judge | 维度分 |
|---|---|---|---|---|
| ann-01 | gold-c01 | 5 | 2.80 | fact=3; grounding=1; coverage=3; concision=5 |
| ann-02 | gold-c01 | 1 | 1.60 | fact=1; grounding=1; coverage=1; concision=5 |
| ann-03 | gold-c02 | 5 | 2.10 | fact=1; grounding=3; coverage=1; concision=5 |
| ann-04 | gold-c02 | 1 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-05 | gold-c03 | 4 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-06 | gold-c03 | 2 | 2.10 | fact=1; grounding=3; coverage=1; concision=5 |
| ann-07 | gold-c04 | 5 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-08 | gold-c04 | 2 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-09 | gold-c05 | 5 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-10 | gold-c05 | 3 | 2.10 | fact=1; grounding=3; coverage=1; concision=5 |
| ann-11 | gold-c06 | 5 | 4.50 | fact=5; grounding=3; coverage=5; concision=5 |
| ann-12 | gold-c06 | 2 | 2.10 | fact=1; grounding=3; coverage=1; concision=5 |
| ann-13 | gold-c07 | 5 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-14 | gold-c07 | 2 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-15 | gold-c08 | 5 | 4.50 | fact=5; grounding=3; coverage=5; concision=5 |
| ann-16 | gold-c08 | 3 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-17 | gold-c09 | 5 | 4.50 | fact=5; grounding=3; coverage=5; concision=5 |
| ann-18 | gold-c09 | 2 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-19 | gold-c11 | 5 | 4.50 | fact=5; grounding=3; coverage=5; concision=5 |
| ann-20 | gold-c11 | 2 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-21 | gold-c12 | 5 | 4.50 | fact=5; grounding=3; coverage=5; concision=5 |
| ann-22 | gold-c12 | 2 | 3.30 | fact=3; grounding=3; coverage=3; concision=5 |
| ann-23 | gold-c01 | 4 | 1.60 | fact=1; grounding=1; coverage=1; concision=5 |
| ann-24 | gold-c03 | 4 | 4.50 | fact=5; grounding=3; coverage=5; concision=5 |

## 判定

阈值 `min_kappa=0.4`，名义 kappa = 0.2 → **未通过**。

未通过时的处置顺序（**不是**调温度）：

1. 看分布：若 judge 分布挤在单一分值，说明锚定描述没有区分度，改锚；
2. 看逐条：若差异集中在某一维度，说明该维度问得不清楚，改 `question`；
3. 看标注：若两条人工分本身互相矛盾，先停下来复核标注，别改 rubric。
