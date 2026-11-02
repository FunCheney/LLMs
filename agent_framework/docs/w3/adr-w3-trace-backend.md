# ADR-W3-02：trace 后端选型与隐私边界

- 状态：已采纳（2026-W46）
- 相关：`tracing.py`（W2 的协议）、`exporters/langfuse.py`、`server/app.py` 的注入点
- 影响面：W4 的失败归因、W10 的安全审计、W11 的成本看板

---

## 决策 1 · 只实现协议，不动字段

W2 定下的 `TraceExporter.export(spans: list[Span])` 一个字不改，`Span` 的
`trace_id / span_id / parent_span_id / name / started_at / latency_ms / status / error_code /
attributes` 也不加列。

代价是 Langfuse 的某些高级特性（session、score、dataset run）用不上；收益是
**业务代码零改动**——`llm/client.py`、`middleware.py` 里一处 `span(...)` 都不用碰。
换后端时（LangSmith / Phoenix / 自建）只换 `exporters/` 下的一个文件。

## 决策 2 · 缺 key 自动降级，且降级是**返回值**不是异常

`build_exporter(settings)` 在 key 不全时返回 `NoopExporter()`，不抛 `ConfigError`。

理由很具体：`make eval` 和 `pytest` 必须能在没有 Langfuse 账号的机器上跑完。
如果缺 key 抛错，CI 里第一个红的就是"没配观测"，然后要么给 CI 塞假 key、
要么在测试里 `skipif`——两种结果都让"trace 能落地"这条 Gate 变成纸面条款。

反向验证写进了测试：`test_build_exporter_without_keys_returns_noop`、
`test_build_exporter_with_partial_keys_returns_noop`（只配 public key 这种手滑也要降级）。

## 决策 3 · 上报失败自禁用，绝不影响主链路

`tracing._safe_export` 已经兜了一层异常。这里再加一层**连续失败计数**：3 次后
`self._disabled = True`，之后直接 return。

不加这层的后果是可算的：Langfuse 挂了时，每个请求在 `finally` 里等满 3 秒超时，
p99 直接变成 3 秒——**观测组件把线上打挂了**，这是可观测性领域最经典的自我伤害。

## 决策 4 · 默认不上报 prompt / 回答全文

`_sanitize()` 会把这些键替换成 `{"len": n, "sha256": "前 8 位"}`：

| 命中条件 | 例子 |
|---|---|
| 键名含 `content` / `prompt` / `completion` / `text` / `messages` / `answer` | `prompt_content`、`messages` |
| 字符串值长度 > 200 | 有人把整段对话塞进 `note` |

要全文必须显式 `capture_content=True`，且应在部署配置里单独开关，不能跟着"打开观测"一起打开。

**为什么这条必须在第一版就写死**：观测平台是第三方。一旦上线跑过一周，用户的
prompt 就已经在对方的存储里了，事后"改配置"收不回来。W10 的安全周会回来查这条，
届时能拿出来的证据只有这个默认值。

## 决策 5 · 不引 `langfuse` SDK

只用 `httpx` POST 到 `/api/public/ingestion`。SDK 自带后台线程、全局 client、
atexit flush，在 pytest 里很难收干净（`PytestUnraisableExceptionWarning` 的常见来源），
而这里需要的只是一个带 Basic Auth 的 POST。

代价：Langfuse 改 API 版本时要自己跟。当前的 batch 结构（`trace-create` +
`span-create`）是公开文档里的稳定形态，风险可接受。

---

## 字段映射

| `Span` | Langfuse | 说明 |
|---|---|---|
| `trace_id` | `trace.id` / `span.traceId` | 由中间件生成，一次请求一条 |
| `span_id` | `span.id` | 每次 `span()` 新建 |
| `parent_span_id` | `span.parentObservationId` | `None` 即根节点 |
| `name` | `trace.name`（取最早的 span）/ `span.name` | 例如 `request`、`llm.call` |
| `started_at` | `startTime` | epoch 秒 → ISO8601 |
| `latency_ms` | `endTime = startTime + latency` | 与日志里的 `latency_ms` 同源 |
| `status` | `level`：`ok→DEFAULT` / `error→ERROR` | 失败 span 必须留下 |
| `error_code` | `statusMessage` | 与 `RequestLog.error_code` 同一个值 |
| `attributes` | `metadata`（经 `_sanitize`） | model / tokens / cost 原样透传 |

**`first_token_ms` 不在 span 里**，在 `RequestLog` 里。理由：它是客户端层的观测值，
放进 span 属性会造出第二个来源，W11 会开始问"以哪个为准"。

---

## 未做（明确记录，避免被当成遗漏）

| 项 | 为什么不做 |
|---|---|
| 采样（`trace_sample_ratio`） | 字段已进 `Settings`，但没接到 exporter 上。评测场景要求 trace 完整，先在服务端做，等真有量再谈采样 |
| session / user 维度 | 需要产品层的 `session_id` 注入点，W5 接检索时一起做 |
| OTel 兼容层 | 为还没出现的需求付复杂度（与 W2 的判断一致） |
| 自建看板 | ROADMAP 的 P3-1 深入档，挂 W11 末 / W12 |

---

## review 自查清单

- [x] 缺 key 时 `build_exporter` 返回 `Noop()`，`pytest` 与 `make eval` 不需要账号
- [x] 上报异常不冒泡（`export` 内部 try/except + 自禁用）
- [x] 默认不含 prompt/回答全文（`test_prompt_content_is_hashed_not_uploaded`）
- [x] `parent_span_id` 映射到 `parentObservationId`（`test_span_tree_keeps_parent_link`）
- [x] 失败 span 以 `ERROR` 上报（`test_error_span_is_exported_as_error_level`）
- [x] 注入点在 `create_app`，库本身不连外部服务
- [ ] 真机验证（需要账号）：跑一次 `/v1/chat`，在 Langfuse 看到含 `llm.call` 的 trace
