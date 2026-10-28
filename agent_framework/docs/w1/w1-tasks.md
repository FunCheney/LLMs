# W1 执行清单（LLM 适配层 + 首个纵向切片）

> 角色约定：**架构与验收由我出，代码由你写，我不代写**。每完成一个 Gate，把仓库状态给我，我按 `docs/architecture.md` 逐条 review。
> 契约细节见 `docs/architecture.md`，本文件只讲"做什么、怎么自查、什么时候交给我"。

**本周总预算**：5 天 × 2–3h ≈ 12–15h。建议 Day1 控制在 2h 内——它的产出是文档和仓库骨架，不是代码。

---

## Gate 0 · 开工前（30 分钟，必须做）

- [ ] 通读 `docs/architecture.md`，把不认同的地方**现在就提**——契约改动成本在本周之后会翻 5 倍
- [ ] 确认账号与额度：OpenAI（或兼容代理）、DeepSeek、Qwen（DashScope）、Anthropic 至少 3 个能真实调用
- [ ] 建一个 `.env.local`（**加进 .gitignore**），放 4 个 key；确认 `git status` 里看不到它
- [ ] 在 README 顶部抄入 DoD 九条 + 一句话目标（12 周后要交付什么）

---

## Day 1（1.5h）· 调研矩阵（仓库骨架已就位）

**这天的价值不在代码，在于"先把差异看清"——它决定 Week 4 的归一化写到什么粒度。**
仓库骨架已于 2026-10-25 搭好并实测通过，今天只剩调研 + 提交两件事。

### T1.1 产出 `docs/w1/llm-api-matrix.md`（60–80 分钟）

对 OpenAI / Anthropic / DeepSeek / Qwen 四家，逐项填表（**必须打开官方文档核对，不许凭记忆**）：

| 维度 | 要记录的细节 |
|---|---|
| 鉴权 | header 名、是否需额外 `anthropic-version` 之类 |
| 端点 | path、是否支持自定义 base_url |
| system 消息 | 是 messages 里一条，还是顶层独立参数 |
| 多轮中 tool 结果 | role 名、字段名、是否要求与 tool_call_id 配对 |
| tool 定义 | 字段名（`tools` vs `functions`）、schema 包裹层级 |
| tool 调用响应 | 参数是 JSON 字符串还是对象、是否返回 id、并行调用怎么表示 |
| 流式 | 传输形式、事件字段名、文本增量的路径、tool_call 增量的拼装方式、结束标记 |
| usage | 字段名、是否流式末尾才给、是否含缓存命中字段 |
| 限流/错误 | 429 的 header、错误体结构、可重试错误码 |
| 特有能力 | prompt caching / structured output / 并行工具 / 思考模式（**只记录，不实现**） |

额外要求：每格写"文档链接 + 一句话结论"，**不许只贴表格不写结论**。最后加一节「我认为最容易归一化出错的三处」。

### T1.2 仓库重构为 workspace —— ✅ 已于 2026-10-25 完成，你只需复核

已完成内容（不用重做）：

- [x] 根 `pyproject.toml` = workspace 根：`members = ["packages/*", "apps/*"]` + 根级 dev 依赖 + ruff / mypy / pytest 统一配置
- [x] `packages/agent_framework/`（可安装库，`uv_build`，入口 `agent-server`）
- [x] `apps/growth_agent/`（产品包，`[tool.uv.sources] agent-framework = { workspace = true }`，入口 `growth-agent`）
- [x] `evals/{golden,history,scripts}`、`ops/`、`docs/` 就位
- [x] 旧 Temporal demo 移入 `legacy/temporal_demo/`（含说明，不参与构建、不被 pytest 收集），`temporalio` 依赖暂时移除，W2 再加回
- [x] `.gitignore`（含 `.env*`）、`.env.example`、README 重写（DoD + 结构 + 命令 + 约定）
- [x] 骨架代码只到"能启动"为止：两个 `create_app()` + `/health`，`llm/`、`server/`、`providers/` 下只有带 TODO 的包说明文档

你只需复核这几条：

- [ ] `uv sync` 干净通过（已实测）
- [ ] `uv run python -c "import agent_framework, growth_agent"` 两个包都能 import（已实测）
- [ ] `uv run pytest` 3 条冒烟用例通过（已实测，离线）
- [ ] `uv run agent-server` → `:8000/health`；`uv run growth-agent` → `:8100/health` 且 `framework_version` 有值（已实测）
- [ ] 确认 `legacy/`、`evals/`、`ops/` 不会被 mypy / pytest 扫到（配置里已排除）
- [ ] 在父仓库里提交：**结构迁移与后续逻辑变更分开提交**，第一条 message 写"重构为 uv workspace，无逻辑变更"

> ⚠️ 注意：本项目位于父仓库 `PycharmProjects/LLMs` 内（monorepo），提交会进父仓库。做作品集前建议 W12 单独建一个干净仓库，只保留本目录历史。


### 🔍 Gate 1（交给我 review）
我会看：矩阵表的完整度与结论质量（**有没有"结论"，还是只抄了文档**）、四家是否真的都核对了流式 tool_call 拼装方式。
**你要给我**：`docs/w1/llm-api-matrix.md` 全文 + 父仓库里这次提交的 `git show --stat`。

---

## Day 2（3h）· 契约层：types + errors + 抽象

**目标：不联网就能写完、测完。这天不许调用任何真实 API。**

### T2.1 `llm/types.py`
- [ ] Message / ContentBlock（占位即可）/ ToolSpec / ToolCall / ToolResult（占位）/ Usage / LLMResponse / StreamEvent
- [ ] `finish_reason` 用枚举，含 `unknown` 兜底 + `raw_finish_reason` 字段；**枚举值按 `llm-api-matrix.md` §5 扩展为** `stop / length / tool_calls / content_filter / aborted / upstream_error`（DeepSeek 的 `insufficient_system_resource` 绝不能落进 `stop`）
- [ ] **`Message.provider_meta: dict | None`（ADR-007）**——承载 `reasoning_content` / Anthropic `thinking.signature` 这类"框架不解释、必须原样还给上游"的字段；文档字符串写清"业务层只整体传递，禁止构造与解读"
- [ ] 校验规则落地：`tool` 消息必须有 `tool_call_id`；`assistant` 有 tool_calls 时 content 可为空；`system` 只能在首条
- [ ] ToolCall 的 `arguments` 恒为 dict，`arguments_error` 记录解析失败
- [ ] **不 import 任何 provider / httpx**

### T2.2 `llm/errors.py`
- [ ] 7 个异常类 + `retryable` 属性 + 公共上下文字段（provider/model/status_code/retry_after/request_id）
- [ ] 一个"厂商状态码 → 内部异常"的映射函数（**放 errors.py，不放 adapter**，保证三家判重试逻辑一致）

### T2.3 `llm/base.py`
- [ ] `Capability` 枚举：stream / tools / parallel_tools / json_mode / vision / prompt_caching
- [ ] `LLMClient` 抽象：`name`、`capabilities`、`chat()`、`stream()`
- [ ] 定义"能力不足时怎么办"：**硬需求（如要求 tools 而模型不支持）由 client 抛 `CapabilityError`**；**软策略（`tool_choice: required`、并行）则由 adapter 降级 + 记 `*_downgraded` 日志**，不抛异常——两者不要混为一谈

### T2.4 单测（先写再实现，允许红）
- [ ] 每条校验规则 1 条用例（正/反各一）
- [ ] 状态码映射表全枚举覆盖
- [ ] **写 4 条 conformance 断言**（此时必然失败，Day3 变绿）

**自检**：`uv run pytest packages/agent_framework/tests -q` 只有 conformance 那 4 条红；`uv run ruff check .` 与 `uv run mypy packages/agent_framework/src` 干净。
> 如果 mypy 配置让你卡住 >30 分钟，降级为"只对 llm/ 目录严格"，别在工具上耗时间。

---

## Day 3（3h）· 两个真适配器 + mock

> **详细设计见 [`w1-day3.md`](./w1-day3.md)**：三段式拆分（build_request / parse_response / 组装器）、ToolCallAssembler 状态机、A1–A8 断言、夹具目录约定、3 小时时间盒、Gate 2 验收与打回清单。本节只保留任务勾选项。

### T3.1 `providers/openai_compat.py`
- [ ] 一个类吃掉 OpenAI / DeepSeek / Qwen / Ollama（靠 base_url + 差异开关）
- [ ] 差异开关用**配置里的 capability / feature flag** 表达，不许 `if "deepseek" in base_url` 这种硬编码分支（review 会卡这条）
- [ ] 双向翻译：Message↔messages、ToolSpec↔tools、响应→LLMResponse、SSE→StreamEvent
- [ ] tool_call 增量按 `index` 累加拼装（ADR-002）
- [ ] **四处三家里不一致的点必须数据驱动**（详见 `llm-api-matrix.md` §0 结论表）：① usage 一律"任何 chunk `usage` 非空就覆盖式记录"，不看 `choices`；② `parallel_tool_calls` **永远显式发送**（Qwen 默认 false、DeepSeek 无此参数）；③ `tool_choice` 按能力降级；④ 错误映射按 `code` 字符串白名单（`insufficient_quota` / `Arrearage` / `402` / `422`）
- [ ] SSE 解析要跳过三种行：空行、`:` 开头的心跳注释（**DeepSeek 会发**）、`event:` 行

### T3.2 `providers/anthropic.py`
- [ ] system 提到顶层参数；tool 结果转 user 消息内的 block；`input_schema` 映射
- [ ] 流式事件序列单独处理（`content_block_start/delta/stop`、`message_delta` 的 usage）
- [ ] **禁止复制粘贴 openai_compat 的代码**——共享逻辑抽到 `providers/_common.py`（只允许：SSE 行解析、重试相关的纯函数、错误体规范化）

### T3.3 `providers/mock.py`
- [ ] 读 `tests/fixtures/<provider>/<case>.json` 回放，支持"文本 / 带 tool_call / 流式事件序列 / 错误响应"四类夹具
- [ ] 夹具先手写（Day4–5 真机跑完再替换成真实录制）

### T3.4 测试
- [ ] conformance 4 条转绿（跑在 mock + 两家真适配器的回放上）
- [ ] 每家 1 条"tool_call 参数解析失败"用例：断言不抛异常、`arguments_error` 有值

**自检**：`pytest` 全绿且**零网络请求**（断网跑一遍验证）；日志里没有任何 key 原文。
### 🔍 Gate 2（交给我 review）
我会重点看：adapter 里有没有业务判断/重试/日志、两家是否真的零共享复制、conformance 断言是否够严格（太松的话归一化等于没做）。
**你要给我**：`pytest -q` 输出 + 三个 adapter 文件路径 + 一次断网跑的截图/输出。

---

## Day 4（3h）· 归一化收口 + SSE 端点

> **详细教程见 [`w1-day4.md`](./w1-day4.md)**：逐文件签名、`models.yaml` 全文、六条校验规则、`chat()/stream()` 算法步骤、SSE 帧格式、A9–A12 断言、8 条打回清单、3 小时时间盒。
> 两处对下方原始清单的修订：① 装饰器链改为**编排方法**（ADR-008，理由见教程 §7.1）；② 新增 `precheck()`，把可预知错误挡在 `StreamingResponse` 之前（否则一律 200）。本节保留任务勾选项。

### T4.1 `llm/client.py`（编排方法：重试 / 计量 / 日志 / fallback 唯一归属）
- [ ] 顺序固定：`trace/日志 → 计量 → 重试 → 能力校验 → adapter`
- [ ] 重试严格按 ADR-003：首 token 前才重试，`first_token_timeout` 作为判定边界
- [ ] 每次重试/fallback 记 `retry_count` / `fallback_from` / `fallback_reason`

### T4.2 `registry.py` + `router.py` + `config.py`
- [ ] registry 按 `models.yaml` 的 `type` 字段造 adapter 实例（**懒加载**，避免一次启动连 4 家）
- [ ] router 返回候选列表；fallback 全链只绕一圈
- [ ] `models.yaml` 落盘（照 architecture.md 第 4.1 节结构），**pricing 必须填**（成本计量依赖它）
- [ ] 启动时校验：路由里的模型是否都在 models 里、价格字段是否齐全——**配置错误要在启动时就炸，不要留到运行时**

### T4.3 `server/`：`/v1/chat`
- [ ] `stream=false`：返回 LLMResponse 的 JSON 投影
- [ ] `stream=true`：SSE，事件名 = StreamEvent.kind，`data` 为 JSON；**每帧带 request_id**
- [ ] 异常处理器：按 3.6 表映射为 `{code, message, request_id}`，`Retry-After` 透传
- [ ] request_id 用 contextvar 注入，贯穿到 provider 日志
- [ ] **不在此处拼任何 prompt、不做任何重试**

**自检**（自己跑，别等我）：
- [ ] `curl -N` 能看到逐帧文本；`curl` 非流式拿到规范 JSON
- [ ] 故意用错 key → 返回 `UPSTREAM_AUTH` + 502，**不是 401**
- [ ] 故意给不支持 tools 的模型发 ToolSpec → `CAPABILITY_UNSUPPORTED` + 400
- [ ] 日志里能查到 `first_token_ms`、`cost`、`retry_count`

---

## Day 5（2h）· 真机验证 + 收口

- [ ] 3 家各跑 1 次真实调用（文本 + 带 tool_call 各一），**把真实响应录成夹具**替换手写 mock
- [ ] 用录制夹具重跑 conformance：真数据下断言仍绿
- [ ] 记录一张对比表：模型 / 首 token 延迟 / 总延迟 / tokens / 单次成本（这将成为 W2 压测和 W11 成本优化的基线）
- [ ] `apps/growth_agent` 放一个真正调用框架的透传端点（**证明框架可被消费**，否则 W12 提取时会发现边界画错了）
- [ ] 补 README 的"如何切模型"一节（改 yaml 即切换，一句话可复现）
- [ ] 更新 `docs/log.md`（本周做了什么 / 遗留问题），周五合流仪式第一次执行

### 🔍 Gate 3 · 周末验收（我会逐条核对，全绿才算过 W1）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 一键切换 | 改 `models.yaml` 即可在 3 家 + mock 间切换，**代码零改动** |
| 2 | 归一化成立 | conformance 4 类断言在 3 家上全绿 |
| 3 | 流式可用 | `curl -N` 逐帧可见，`first_token_ms` 有数值入日志 |
| 4 | 错误规范 | 6 类错误码各有对应 HTTP 状态，且上游鉴权≠401 |
| 5 | 成本可查 | 每次请求日志含 tokens + cost，且数值与厂商账单同量级 |
| 6 | 离线可测 | 断网执行 `pytest` 全绿，零真实调用 |
| 7 | 边界干净 | adapter 无重试/无业务/无日志；业务层无厂商 JSON |
| 8 | 文档 | 矩阵表有结论、README 有切换说明、`docs/log.md` 有记录 |

---

## 本周最容易返工的 5 处（提前避）

1. **在 adapter 里加了重试** → 成本黑洞 + 日志重复计数，review 直接打回（ADR-003）
2. **tool_call 增量往上发** → 消费方要写两套拼装逻辑，W9 planner 会崩（ADR-002）
3. **流式已开始还重试** → 用户看到"回答半截又重来"，是线上最恶心的体验
4. **把 pricing 散在代码里** → W11 做成本看板时要重算全部历史数据
5. **`if provider == "deepseek"` 硬编码** → 第三家兼容模型接进来就要改代码，违背原则 1

---

## 本周产出与产出物位置

| 产出 | 位置 |
|---|---|
| 差异矩阵 | `docs/w1/llm-api-matrix.md` |
| 契约与抽象 | `packages/agent_framework/src/agent_framework/llm/{types,errors,base}.py` |
| 3 适配器 | `.../llm/providers/{openai_compat,anthropic,mock}.py` |
| 客户端/路由/配置 | `.../llm/{client,registry,router}.py` + `config.py` + `models.yaml` |
| HTTP + SSE | `.../server/{app,sse,routes_chat}.py` |
| 测试与夹具 | `packages/agent_framework/tests/` |
| 记录 | `docs/log.md`（W1 条目） |
