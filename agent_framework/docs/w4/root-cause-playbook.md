# 根因归因手册 · 四层决策树（W4 D4）

> 代码：`evals/scripts/attribute_failure.py` · 上游分类：`evals/scripts/classify_failures.py`

## 0. 为什么需要这份手册

评测报告给出"失败 6 条，成功率 88%"，接下来的动作决定了这周的产出：

- 没有方法时：团队会挑最眼熟的那条去改 prompt，改完失败率可能不变，因为根因在检索；
- 有方法时：每条失败先落到一层，落到哪层就动哪层的文件，**改动与指标变化能对上**。

W3 的 `FailureCategory` 已经能区分"上游抖动"和"答案错"，但它还停在"像什么"。
这一层要回答的是"**下一步开哪个文件**"。

## 1. 决策树（先分类，再定层）

```
error_code 存在?
├── 是 → 模型/可靠性层（先看厂商状态页，别改代码）
└── 否 → tool_error?
     ├── 是 → trace 里有 tool span?
     │        ├── 没有 → 证据不足（补埋点，不是改工具）
     │        ├── arguments_error>0 → 工具层
     │        └── http 错误 → 工具层
     └── 否 → hallucination（疑似无引用支撑）?
          ├── 是 → 有 retrieval span?
          │        ├── 没有 → 证据不足（W5 前属正常，别判"模型编造"）
          │        ├── 有 span 但没记录 doc id → 证据不足（埋点不全）
          │        ├── 检索到了但引用 0 命中 → 检索层
          │        └── 引用了但不支撑结论 → 检索层
          └── 否 → format / budget?
               ├── 是 → 提示词层（输出结构稳定错 ≈ 指令没说清）
               └── 否 → 提示词层（conf 0.4）/ 证据不足
```

## 2. 四层的判据与修法

| 层 | 观察到什么（证据字段） | 置信 | 典型修法 | 何时开始有真输入 |
|---|---|---|---|---|
| **检索层** | `retrieval_spans>0` 且 `retrieved_ids ∩ cited_ids = ∅`；或引用了检索结果但结论仍无支撑 | 0.8 / 0.6 | 混合检索、chunk 大小、重排模型、query 改写 | W5–W6 |
| **提示词层** | `category=format`（帧序列/必填字段不符）；`finish_reason=length`；纯断言不成立且无其它信号 | 0.7 / 0.6 / 0.4 | 指令分层、rubric 写清、few-shot、约束篇幅 | W7/W10 |
| **工具层** | `tool_arguments_error>0`；工具 HTTP 错误；`tool_spans>0` 但未记录具体错误 | 0.9 / 0.8 / 0.5 | 工具 schema 描述、参数校验、权限与超时 | W8 |
| **模型/可靠性层** | `error_code ∈ {UPSTREAM_*, RATE_LIMITED, SERVER_BUSY}`；`retry_count>0` 且 `fallback_from` 非空 | 0.9 | fallback 顺序、重试与退避、降级到便宜档 | 已有（W1） |

置信度的含义是**证据强度**，不是"可能性"：0.9 = 直接信号命中，0.6 = 间接推断，0.0 = 证据不足。

## 3. 证据不足：最容易被做错的一档

两类样本必须落到 `unknown`，而不是硬猜：

1. **`tool_error` 但 `tool_spans=0`**
   分类说"工具错"，trace 里却没有任何工具调用记录。这不是"工具坏了"，是**埋点缺了**。
   判成工具层之后，团队会去改 schema 描述——改完仍然红，然后再改一次。

2. **`hallucination` 但 `retrieval_spans=0`**
   W5 之前没有检索，`classify_failures` 只能给出"疑似无引用支撑"（它是靠
   `has_references` + 文案缺失判的）。这时把结论升级成"模型编造"是越权的：
   分不清是"没检索到"还是"检索到但没用"。

处置：报 `unknown`，修法写"先补埋点：缺哪个 span 就补哪个"。

## 4. 与分类器的分工

| 层 | 负责 | 不负责 |
|---|---|---|
| `classify_failures.classify` | 从 `error_code` / `tool_errors` / `finish_reason` / 断言文案判**类别** | 不判断该改哪一层 |
| `attribute_failure.attribute` | 结合 span 信号把类别细化到**层**，给证据字段 | 不改判类别（顺序不能反） |

反过来的写法（先猜层、再找理由）会让每条失败都变成"我觉得是 prompt 问题"，
既不可复核，也无法被单测覆盖。

## 5. 用法

```bash
# 对最新一期评测快照里的失败逐条归因
uv run python evals/scripts/attribute_failure.py

# 指定快照
uv run python evals/scripts/attribute_failure.py evals/history/2026-W47-eval.json
```

输出示例（一行一条）：

```
case        层          置信  证据 / 修法
------------------------------------------------------------------
gold-t05    unknown      0.0  分类=tool_error；tool_spans=0：trace 里没有工具调用记录…
                              → 证据不足：先补埋点：缺哪个 span 就补哪个，别急着改代码
gold-c03    prompt       0.7  分类=format（帧序列或必填字段不符）；finish_reason=stop
```

## 6. 每层都要能回答的两个问题

1. **凭什么判这一层？** —— 具体的 span / 日志值（表里"证据字段"一列）。
2. **改完之后看哪个指标？** —— 检索层看 `context_precision`/`faithfulness`，
   提示词层看 `success_rate`（分难度），工具层看 `tool_error_rate`，
   可靠性层看 `interrupted_rate`。

答不出这两个问题，说明归因还停在猜测——回去补信号，不要改代码。
