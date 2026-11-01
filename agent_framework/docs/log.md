# 开发日志

> 每周五合流仪式第 4 步：记一条「本周做了什么 / 指标变化 / 遗留问题」。
> 指标快照写在 `evals/history/YYYY-Www.json`，本文件只记**结论与判断**。

---

## 2026-W45（11-02 ~ 11-06）· W2：运行底座 + 可观测基线 + 评测冒烟集

> 实际落盘时间 11-01（周日）预跑；数字全部来自离线档，在线档待 key。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `session/{base,memory,redis,factory}.py` + `workflow/research_workflow.py` + `docs/w2/adr-w2-session-boundary.md` | 幂等键必须三档解析（显式 → contextvar → uuid）；少第三档，脚本与队列消费者一调就误判重复 |
| D2 | `resilience/{concurrency,errors}.py` + `queue/{base,inmemory}.py` + `server/routes_task.py` | 入口过载 `retryable=False`：让 gateway 再重试一次，100 并发立刻变 300（压测直接崩） |
| D3 | `observability.py`（+`trace_id`/`span_id`）+ `tracing.py`（Span/span()/Exporter） | `trace_id` 必须在中间件生成，放进 `span()` 会让同一请求的多个 span 各拿一个新 trace |
| D4 | `ops/{Dockerfile,docker-compose.yml,.env.example}` + `evals/scripts/locustfile.py` + 压测报告 | 客户端口径与服务端口径差一个数量级，延迟结论只能取 `RequestLog` |
| D5 | `evals/golden/smoke_v1.jsonl`（10 条）+ `run_smoke.py` + CI + `Makefile` | 门禁必须先证明"能红"：CI 里改坏一条 expect，退出码非 0 才算门禁成立 |

### 指标变化

- 离线用例：`278 → 337`（新增会话/限流/队列/trace/任务端点/smoke runner 六组），live 仍 16 个跳过
- **100 并发离线压测：1224 请求 / 0 失败 / `retry_count>0` 为 0 / 994 条日志 994 个 trace_id**
- 首 token（port 口径）：p50 `20.9 ms`、p95 `32.5 ms` —— **仅证明埋点链路正确**，fixture 回放不代表真实模型
- 成本：仍为 0（fixture 不计价）；质量：smoke `10/10`，只有"没坏"这一档
- 静态检查：`ruff` 全绿、`mypy` 43 文件无问题

### 本周暴露的四个真问题（都不是写代码时能想到的）

1. **"实现慢"可能是机器慢**：全量串跑 15 分钟、单文件跑 6 秒，同一份代码。宿主调度一停顿，
   客户端 p95 就从 35 ms 涨到 1200 ms。**结论**：延迟只认服务端 `RequestLog`，客户端分位数只能横向比。
2. **幂等的兜底键不是可选项**：没有"显式 → contextvar → 本地 uuid"第三档，`request_id=""` 会让
   第二次 `append` 被判成重复，表现为"多轮对话第二轮丢消息"——只在真实多轮里复现。
3. **入队前必须 `precheck`**：alias 打错这类注定失败的请求，占了队列槽位排 30 秒再 failed，
   等于把调用方的输入错误升级成我们的容量问题。
4. **compose 的 healthcheck 不能指向未就绪端口**：`depends_on` 只保证"容器起了"，不保证"能连"，
   `agent-server` 起得比 redis 慢时，首批请求全是 `Connection refused`。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 在线档延迟/成本未测 | 有 key 时 `MODEL=qwen3.7-plus make load`，报告记时段（峰谷差 2 倍） |
| 2 | `docker build` 未在本机执行（无 docker） | compose 仅静态检查；首次 `make up` 时补 `docker compose ps` 无 restart 的记录 |
| 3 | 队列进程内实现 | 多副本会丢任务；换 Redis Stream 时 `TaskQueue` 签名不变 |
| 4 | 满载 503 只有单测 | 把 `limits.global` 调成 1 跑 10 并发，端到端验 `Retry-After` |
| 5 | smoke 只判"全过" | W4 升级为"指标不得劣化"，需要先有 3 周基线 |

### 下周（W3）只做三件事

1. golden set 扩到 50+（每类覆盖从 2–3 条提到 8–10 条），判据从"文本包含"升级到结构化断言；
2. 接 Langfuse：只实现 `LangfuseExporter` 替换 `NoopExporter`，`RequestLog` 字段一个不改；
3. 记忆层（W7 前置）只定接口，不实现——先把"哪些事实值得记"的判据写进文档。

---

## 2026-W44（10-26 ~ 10-30）· W1：LLM 适配层 + 首个纵向切片

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | 四家 API 差异矩阵（`docs/w1/*.md`）+ uv workspace 重构 | "system 顶层 / tool_result 成组 / arguments 是字符串 / usage 挂在哪"这四件事决定归一化粒度 |
| D2 | 契约层 `types / errors / base` | 唯一一张 finish_reason 映射表；`provider_meta` 承载"框架不解释但必须回传"的字段 |
| D3 | `openai_compat` / `anthropic` / `mock` + conformance A1–A8 | 差异走 flags 表，不写 `if "deepseek" in base_url`；mock 只回放 wire，不自己造 `LLMResponse` |
| D4 | `config / observability / registry / router / gateway / server` + A9–A12 | 重试与 committed 状态机是客户端层唯一职责；`precheck` 必须在 `StreamingResponse` 之前 |
| D5 | 录制器 / 基线表 / live 用例 / A13–A14 / 产品端 `/v1/draft` | 框架"能被消费"这件事必须由产品端真跑一遍来证明，自测不算 |

### 指标变化

- 离线用例：`136 → 222 → 278`（Day3 → Day4 → Day5），live 用例 16 个默认跳过
- conformance 断言：`A1–A12 → A1–A14`（新增"录制 ⊇ 手写基线"与"工具调用落在声明 schema 内"）
- 首 token / 成本 / 质量：**仍为 null**，等真机录制与 W2 golden set
- 静态检查：`ruff` 全绿、`mypy` 26 文件无问题

### Day5 过程中暴露的四个真问题（都不是写代码时能想到的）

1. **日志静默丢弃**：`RequestLog` 打 `INFO`，而 root logger 是 `WARNING`。代码里 `emit()` 明明调了，
   `uvicorn` 起来后日志里一个字都没有——Gate 验收项「`first_token_ms` 入日志」会假通过。
   修法：`observability.configure_logging()`，在应用边界（`create_app`）调用，库本身不改宿主配置。
2. **DeepSeek 是峰谷分时计价**（工作日 09–12 / 14–18 高峰翻倍），Qwen 是**按上下文长度阶梯**
   （≤256K 为 2/8，>256K 跳到 6/24）。单一 `pricing.input/output` 表达不了，W1 取常规档并在
   `models.yaml` 里注明；基线表必须记时间戳，否则"成本降了 40%"可能只是换到了空闲时段。
3. **产品层拿不到中间件**：`RequestIDMiddleware` 原本长在 `server/app.py` 里，产品要复用它就得
   执行该模块顶层的 `app = create_app()`——为了拿一个中间件而启动一个框架应用，是边界画错的信号。
   抽到 `server/middleware.py` + `server/errors.py` 后，产品端零 `try/except` 就复用了 6 类错误码。
4. **录制器不能覆盖 `request.json`**：它是 A1 的断言锚（`build_request` 必须精确产出它）。
   覆盖了 A1 就变成"自己跟自己比对"。改为：录到不一致就硬失败，让人先决定锚该不该更新。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 真机录制未跑（手写夹具可能偏离真实协议） | 有 key 时执行 `README` 的 Day5 三连；A13 会在录制后清算 |
| 2 | `first_token_timeout_s=10s` 未经真机校准 | `uv run pytest -m live -k first_token` 取实测值后调整 |
| 3 | Anthropic 欠费仍靠文案 `credit balance` 识别 | 真机确认后收敛成固定 code 白名单 |
| 4 | 峰谷 / 阶梯计价未建模 | W11 做成本看板时改为按 (时段, 上下文长度) 取价 |
| 5 | 错误路径（`usage_missing` / `error_*`）无法安全录制 | 长期保留手写 + `error_auth` 真机探测这两条路 |

### 下周（W2）只做三件事

1. golden set 最小 10 条（含 3 条对抗样例），接进 `pytest` 做门禁；
2. 压测脚本：并发 20 × 60s，产出 p50/p95/错误率，回填 `evals/history`；
3. Temporal 长任务骨架（`legacy/temporal_demo` 的思路复用），先跑通"调研分 3 步可续跑"。
