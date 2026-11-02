# ADR-W3-01：评测分档与可复现性约定

- 状态：已采纳（2026-W46）
- 相关：`docs/w3/metric-spec.md`、`evals/metrics.py`、`evals/scripts/run_eval.py`
- 影响面：W4 回归门禁、W11 成本看板、W12 趋势图

---

## 背景

W2 的 `smoke_v1.jsonl`（10 条）只回答"没坏吗"。W3 要回答的是"多好"，随之而来三个必须
先决定的问题：

1. 离线（夹具回放）和在线（真实模型）能不能共用一套判据？
2. "可复现"具体要钉住哪几个变量？
3. 离线跑出来的数字，能不能进 `evals/history` 的质量块？

这三个不定，runner 写出来就是一堆没人敢引用的数字。

---

## 决策

### D1 · 判据分离：`expect` 与 `offline.expect` 是两套，不是一套的宽松版

```jsonc
{"id": "gold-012", "kind": "chat", "difficulty": 2,
 "expect":         {"text_contains": ["工具注解", "异步任务"], "finish_reason": "stop"},
 "offline": {"fixture": {"provider": "openai", "case": "text_basic"},
             "expect":  {"text_contains": ["MCP"], "finish_reason": "stop"}}}
```

**为什么不共用一套。** 夹具输出是固定文本（`MCP 协议最新进展：新增了工具注解与异步任务支持。`），
语义判据（"必须提到工具注解"）在离线档要么恰好成立、要么永远不成立，两者都不说明任何问题。
如果为了"让离线全绿"把判据放宽到"只要不报错"，那 CI 就装了一个永远绿的门——W2 已经踩过
这个坑（`smoke` 的判据之所以从 `conformance` 抄，就是为了避免自造宽松逻辑）。

分离之后各档位的能力边界是明确的：

| 档位 | 判据 | 证明的是 | 不能证明 |
|---|---|---|---|
| 离线 `model=fixture` | `offline.expect`（锚在夹具输出） | 断言管道、指标埋点、错误映射、SSE 组装**能工作** | 模型质量、真实延迟、成本 |
| 在线 `model=<alias>` | `expect`（语义） | 模型在 golden 上的实际表现 | 竞品对比（样本太小） |

代价：golden 里每条要写两段判据，维护成本翻倍。接受这个代价，因为"假绿"的成本更高——
一个永远绿的门禁，三个月后没人会看它。

### D2 · 离线数字不进 `quality` 块

`evals/history/*.json` 的 `quality.success_rate` 在离线档跑完后保持 `null`，
离线结果只写进新增的 `pipeline` 块：

```jsonc
"pipeline": {"cases_total": 54, "offline_run": 50, "offline_skipped": 4, "status": "pass"}
```

理由：`quality` 是给"系统好不好"用的，离线档跑通只说明代码没写崩。把两者混在一个字段里，
W12 画出来的成功率曲线会在第一次真机跑时凭空掉一大截（从 100% 掉到真实值），
而那只是第一次诚实地测量。

### D3 · 可复现：钉 alias + 温度 0 + 记录时段

| 变量 | 处理 | 不钉会怎样 |
|---|---|---|
| model | 写 `alias`（`qwen3.7-plus`），不写 wire 名（`qwen3.7-plus-2026-09`） | 厂商换 wire 名，昨天的数字对不上，且报错信息看起来像配置写错了 |
| temperature | runner 强制注入 `0`；case 显式写别的值 → 直接报错退出 | 两次跑出两个数，"可复现"这条 Gate 直接挂 |
| seed | 厂商支持时透传；不支持时忽略并记 `seed_supported=false` | — |
| 时段 | 在线档必须记 `period`（起止时刻） | DeepSeek 峰谷价差 2 倍，"成本降 40%"可能只是换了时段 |
| 模型快照 | 在线档记 `model_snapshot`（厂商版本号，没有就记首跑日期） | 静默升级导致曲线跳变，无法归因 |

`temperature=0` 强制而不是默认：默认值会被某一条 case 悄悄覆盖，而那条 case 恰好是
最常被引用的样例。

### D4 · 单条失败不中断整批

runner 逐条 `try/except` 包住，`LLMError` → `CaseStatus.INTERRUPTED`，其他异常 →
`FAILED` + `detail`。理由是运维现实：一条 502 不该毁掉整晚的评测，否则没人敢在晚上跑。

反向验证方式固定：`--break-case <id>` 注入一次失败，断言"该条失败但其余照跑、
退出码非 0"。这条在 CI 里跑，防止有人为了"让 CI 过"把 `try/except` 改成 `continue` 掩盖。

---

## 后果

**变好的**
- 离线档能在 CI 零网络跑完 50+ 条，且不会因为"夹具文本 ≠ 语义判据"而红一片；
- `quality` 块一旦有数字，就一定是真机测的，可以直接对外说；
- 钉死 alias/温度后，同一份夹具两次跑的分位数完全一致（可写进 CI 断言）。

**变差的**
- golden 每条两段判据，扩到 200 条时编辑成本明显；
- 离线档覆盖不到的语义类 case（对抗样例）永远 `skipped`，`skip_rate` 会长期在 8% 左右——
  报告里必须显式展示这个数，不能让它变成"看不见的缺口"。

**以后再改的判断依据**
如果某个语义判据在夹具上稳定成立（例如夹具本身就该包含该关键词），可以在 `offline.expect`
里复用同一串关键词——**但必须是"因为夹具确实满足"，不是为了让它绿**。
两者的区别写在 commit message 里，不做自动化检测（自动化检测不了意图）。
