# ADR-W4-02 · 回归门禁阈值：为什么用相对劣化 + 绝对下限

- 状态：已采纳（2026-W47）
- 影响范围：`evals/scripts/regression_gate.py`、`evals/history/baseline.json`、`.github/workflows/ci.yml`

## 1. 问题

capstone 的 DoD 写着"CI 跑 lint + test + eval 回归门禁"。前三项是布尔值，好办；
**"回归"是个连续量**，"太差了"必须被翻译成一组数字。翻译错了会有两种失败：

- 阈值太松：劣化照常合并，门禁等于没有；
- 阈值太紧：同一份代码两次跑一红一绿，最后所有人学会忽略红叉——**比没有门禁更糟**。

## 2. 决策

### D1 一律用相对劣化，不用绝对阈值

```python
GateRule("success_rate", HIGHER, max_regression=0.05)   # 不许比基线差 5%
```

绝对阈值（"p95 必须 < 50ms"）在不同机器上必然误报：本仓库 W45 压测里，
同一份代码在宿主机上 p95 是 32ms，客户端观测却到 1200ms（宿主机调度停顿）。
门外汉看到 1200 会去"优化代码"，实际上什么都没变。

### D2 延迟类加**绝对下限**：两个条件同时满足才算劣化

```python
GateRule("latency_p95_ms", LOWER, 0.20, min_absolute=5.0)
```

基线 p95 = 2.3ms（离线夹具）时，"允许劣化 20%" 等于允许涨到 2.8ms——
一次进程调度抖动就能触发。所以还要求**绝对增量超过 5ms**。
反过来，基线 1000ms 涨到 1010ms 时相对只 1%，也不该红。

### D3 `interrupted_rate` 只告警，不判 fail

W3 已经定了口径（`docs/w3/metric-spec.md` §1）：上游 5xx/限流不进 success_rate 分母。
门禁必须遵守同一条：**可靠性退化要看厂商状态页，不该阻断合并**，
但也不能当没发生——所以它是 `[WARN]`，`--strict`（发布前）时才升级为失败。

### D4 基线缺失 → 退出码 2，不静默放行

没有基线就"通过"，等于门禁默认永远绿。CI 里这一步会直接失败并提示怎么建基线。

### D5 基线必须带三样东西

```json
{"commit": "afbded2", "recorded_at": "...", "model_snapshot": {"cheap": "qwen3.7-plus", "strong": "claude-sonnet-4-6", "catalog_sha256": "683bc7654b9f"}}
```

`model_snapshot` 是 ADR-009 的直接后果：**换了模型快照，旧基线就不可比**。
没有它，某天把 `qwen3.7-plus` 换成 `deepseek-flash` 后门禁报"成功率下降 6%"，
而真实原因是换了模型，不是代码退化。

### D6 离线与在线各有一份基线

`baseline.json` 是**离线**指标（success_rate 恒 100%，只代表执行管道）。
在线档接入后需要 `baseline-online.json`。`load_metrics()` 按 `mode` 取块，
**不按"字段存在与否"猜**——否则一次离线高分快照就能把门禁骗绿。

## 3. 阈值来源

| 指标 | 方向 | 阈值 | 依据 |
|---|---|---|---|
| `success_rate` | 高好 | 5% | W45/W46 两期都是 100%，无波动；5% 已是"一条 case 翻转"的量级 |
| `latency_p95_ms` | 低好 | 20% + 5ms | 同机两次跑相差 8%~25%（夹具回放，抖动来自进程调度） |
| `cost_per_task` | 低好 | 15% | 计价核对窗口是 2026-10-29，价格变动的量级通常是 10%~30% |
| `judge_score` | 高好 | 5% | W4 才有值，先保守；等 3 期数据后再收紧 |
| `faithfulness` | 高好 | 10% | 词面判定本身有噪声（见 `metric-spec-ragas.md` §6） |
| `tool_error_rate` | 低好 | 50% | 分母小（14 次工具调用），相对波动大 |

`required=False` 的规则在基线缺该指标时跳过，不判 fail——离线档没有 `judge_score`
是正常的，为此天天红一次没有意义。

## 4. 反向验证（门禁必须能红）

三条证据都在 CI 里，缺一条"全绿"就不可信：

1. `run_eval --break-case gold-c03` 必须退出非 0；
2. 把当前快照的 `success_rate` 改成 0.5，`regression_gate` 必须退出非 0；
3. 单测 `test_regression_triggers_fail` / `test_within_noise_does_not_trigger`
   分别证明"真劣化会红"和"±2% 不会红"。

## 5. 已知局限

1. 离线档的 `success_rate` 恒 100%——**门禁目前只能拦住"管道坏了"，拦不住"答得差了"**。
   真正有意义的门禁要等在线档基线（需要 key）。
2. 阈值是按单机两期数据拍的，样本不足。W5 起每期回看一次，连续 3 期无一次误报才维持，
   出现误报就先调阈值再谈加严。
3. 没有实现"多指标同时轻微劣化"的联合判定（例如成功率 -3% 且延迟 +15%）。
   单指标门槛会漏掉这种"温水"式退化，属 W11 成本/质量看板的事。
