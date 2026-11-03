# 案例 1 · RAGAS 指标结构性为 0：context 用路径字符串冒充正文

- 日期：2026-W47 · 归因层：**引用/数据层**（`docs/w4/root-cause-playbook.md` 的检索层下沿）
- 涉及文件：`evals/scripts/run_eval.py::_contexts`、`evals/judge/ragas_metrics.py`、`evals/judge/text.py`

## 1. 现象

跑 `uv run python evals/scripts/run_eval.py --judge`，RAGAS 三条指标全 0：

```
RAGAS（10 条）：faithfulness 0.0 · relevancy 0.0543 · context_precision 0.0
```

10 条样本、全部 0.0 —— 这个形态本身就是信号：**真实数据不会这么整齐**。
当时第一反应是"词面匹配阈值定高了"，但调整阈值后仍是 0.0。

## 2. 根因

`_contexts()` 直接把 `references[].target` 当 context：

```python
return [str(item.get("target", "")) for item in case.references]
```

而 golden 的引用是两种东西：

| kind | target 示例 | 当作 context 后是什么 |
|---|---|---|
| `repo` | `docs/architecture.md` | 一个**路径字符串**，不是文档正文 |
| `url` | `https://modelcontextprotocol.io/specification` | 一个 **URL 字符串** |

它们的实词是 `docs`/`architecture`/`md`、`https`/`modelcontextprotocol`/`io`。
答案里当然不会出现这些 —— `faithfulness` 的分母是全部陈述、分子恒为 0；
`context_precision` 同理。**指标结构性地为 0，与模型表现无关。**

证据：`evals/history/2026-W46-eval.json` 的 `cases[]` 里，10 条带引用的 case 全部
`ragas.faithfulness = 0.0`、`contexts = 1` 或 `2`。

## 3. 改动

1. `_reference_text()`：按 `kind` 解析
   - `repo` → 读仓库内文件正文（校验路径在仓库内，截断 6000 字符）；
   - `url` → **返回 None**，不用 URL 字符串冒充正文（离线档不联网）；
   - 显式 `text` 字段优先（为后续手写片段留口子）。
2. URL 引用因此**不贡献 context**，`ragas_samples` 从 10 变 9 —— 宁可少算，不做假算。

### 过程中发现第二个 bug：单字口径把 faithfulness 顶到 1.0

换成正文后第一次跑，`faithfulness` 变成 **1.0（9 条全满）**——同样不可信。
原因是支撑判定用"去停用字后的单字重合 ≥60%"，而 context 变成了一整篇
`docs/architecture.md` 后，**任何中文句子都能凑够单字重合**。

改成**二元组覆盖 ≥60%**（`evals/judge/text.py::bigram_coverage`）：
陈述与原文近似才算支撑。代价是改写的正确结论会被判无支撑 —— 这条写进了
`metric-spec-ragas.md §6`，并用 `test_lexical_support_misses_paraphrase` 锁住，
防止有人为了"好看"把阈值调低。

## 4. 量化改进

| 指标 | 改动前 | 改动后 | 解读 |
|---|---|---|---|
| `ragas_samples` | 10 | 9 | URL 引用无正文，不再冒充 context |
| `context_precision` | **0.00** | **1.00** | 声明的引用确实与问题topic相关（此前是结构性 0） |
| `faithfulness` | 0.00 | 0.00 | **仍然是 0，而且是真实读数**：夹具答案是固定文本，与引用无关 |
| `answer_relevancy` | 0.054 | 0.054 | 只依赖问题与答案，与 context 无关，不变（符合预期） |

`faithfulness` 保持 0 不是没修好：离线档的"答案"是 `openai/text_basic` 那条固定夹具
（`MCP 协议最新进展：新增了工具注解与异步任务支持。`），与任何引用都不相干。
**离线档不可能产出有意义的 faithfulness**——这一条比一个好看的 0.7 更值得记住。

## 5. 取舍与遗留

- 取舍：URL 引用在离线档没有正文 → 少了 1 条样本。接真实检索（W5）后，context 来自
  检索 chunk，`kind` 的区别自然消失。
- 遗留：`faithfulness` 要有意义，必须先用真实模型跑一次在线档（需要 key）。
- 遗留：词面判定认不出同义改写 → 报告里 `method=lexical` 必须随数字一起出现，
  否则读者会把下界当估计。
