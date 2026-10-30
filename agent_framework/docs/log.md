# 开发日志

> 每周五合流仪式第 4 步：记一条「本周做了什么 / 指标变化 / 遗留问题」。
> 指标快照写在 `evals/history/YYYY-Www.json`，本文件只记**结论与判断**。

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
