# 案例 3 · 同一个错误码，两处判定相反：限流被记成"模型答错"

- 日期：2026-W47 · 归因层：**可靠性层**
- 涉及文件：`evals/scripts/classify_failures.py`（唯一码表）、`evals/scripts/run_eval.py`

## 1. 现象

W4 写归因脚本时，需要回答"这个错误码算不算上游故障"。同一份代码库里搜到**两份表**：

- `evals/scripts/classify_failures.py::_UPSTREAM_CODES`（8 个码）
- `evals/scripts/run_eval.py::_RELIABILITY_CODES`（4 个码）

于是写了个探针，把已知错误码逐个枚举、比较两处判定：

```
('UPSTREAM_AUTH',          分类器=上游   run_eval=可靠性   一致)
('RATE_LIMITED',           分类器=上游   run_eval=可靠性   一致)
('SERVER_BUSY',            分类器=上游   run_eval=不算      不一致  ← 我们自己的限流器吐的 503
('CAPABILITY_UNSUPPORTED', 分类器=上游   run_eval=不算      不一致
('BAD_REQUEST',            分类器=上游   run_eval=不算      不一致
('CONTENT_FILTERED',       分类器=上游   run_eval=不算      不一致
('CONFIG_ERROR',           分类器=上游   run_eval=不算      不一致
('TASK_NOT_FOUND',         分类器=上游   run_eval=不算      不一致
mismatch 6 / 10
```

**10 个已知错误码里 6 个判定相反（60%）。**

## 2. 根因

`run_eval.evaluate()` 用它自己那份 4 个码的表决定要不要把 case 记成 `interrupted`：

```python
elif observation.error_code in _RELIABILITY_CODES:
    status = CaseStatus.INTERRUPTED   # 不进 success_rate 分母
```

而 `SERVER_BUSY` 不在那份表里 → 走 `else` 分支 → `FAILED` → **进 quality 分母**。
`classify_failures` 却把它判成 `UPSTREAM`（"可靠性问题，不是模型笨"）。

后果可以直接算出来：`docs/w2/` 的限流压测里，100 并发下约 20% 请求拿到 `SERVER_BUSY`。
如果这批请求走的是 `run_eval`，`success_rate` 会从 1.00 掉到 0.80，
而报告的失败分类里写着"upstream"——**两个数字互相矛盾，团队会去调一个没坏的 prompt**。

根因不是某个分支写错，而是**同一条业务规则被实现了两遍**。两份表迟早会漂移，
而且漂移不会报错，只会让两个数字互相打架。

## 3. 改动

1. `classify_failures.py` 成为**唯一码表**，并显式分成两类：
   - `RELIABILITY_CODES`（厂商侧）：`UPSTREAM_AUTH/ERROR/TIMEOUT`、`RATE_LIMITED`、`SERVER_BUSY`
   - `CALLER_CODES`（我方请求/配置）：`BAD_REQUEST`、`CAPABILITY_UNSUPPORTED`、
     `CONFIG_ERROR`、`TASK_NOT_FOUND`、`CONTENT_FILTERED`
2. 新增 `is_reliability_code(code)`；未知码按可靠性处理（宁可少算一次答错，不冤枉模型）。
3. 新增 `FailureCategory.REQUEST` 与 `counts_against_model`：
   "我方错误"既不算厂商故障，也不算模型答错——报告里单列一类。
4. `run_eval` 删掉自己的表，改调 `is_reliability_code()`。
5. 加不变量测试 `test_classification_and_runner_agree_on_every_known_code`：
   遍历全部已知码，断言两处判定必须一致。**这条测试才是修复的真正交付物**——
   它保证下次有人加码时不会重新分叉。

## 4. 量化改进

| 指标 | 改动前 | 改动后 |
|---|---|---|
| 已知错误码判定不一致数 | **6 / 10**（60%） | **0 / 13**（含 3 个探测用未知码） |
| `SERVER_BUSY` 在 run_eval 里的状态 | `failed`（进 quality 分母） | `interrupted`（不进分母） |
| `CONTENT_FILTERED` 的归因 | `upstream`（算厂商故障） | `request`（我方/策略，单列） |
| 防回归手段 | 无 | 不变量单测 + 唯一码表 |

## 5. 取舍与遗留

- 取舍：新增 `REQUEST` 类后，历史快照里出现过的 `upstream` 计数不可与新数据直接相加。
  目前只有 W46/W47 两期，且两期 `failures` 都是空 —— 没有历史包袱，现在改代价最小。
- 遗留：`CONTENT_FILTERED` 归"我方"是一种判断（也可能来自模型侧的安全策略）。
  真要分清需要厂商返回更细的 `refusal` 原因；现在先按"不冤枉模型"处理，并在
  `classify_failures.py` 的注释里写明这是一次判断，不是事实。
- 遗留：`Interrupted` 本身仍会进 `interrupted_rate`，而门禁只对它告警不阻断
  （见 `adr-w4-gate-threshold.md` D3）。这是有意的：可靠性退化不该阻断合并，
  但也不能当作没发生。
