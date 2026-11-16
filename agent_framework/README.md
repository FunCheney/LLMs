# agent-framework

> 一个自研 LLM Agent 框架，和它唯一的使用者——一个增长营销 Agent。
> 12 周单仓库贯穿（W1–W12），验收标准是**可验证的证据**：1471 条测试、12 周指标曲线、
> 5 篇 ADR、5 篇线上 incident。不是功能数量。

## 一分钟

| 问题 | 答案 |
|---|---|
| 它做什么 | 主题 → 调研 → 检索+引用 → 成文 → 三平台文案 → 质检，每一步都留痕 |
| 里面有什么 | `packages/agent_framework`（框架，可独立打包安装）+ `apps/growth_agent`（产品，用框架 dogfood） |
| 凭什么说生产级 | 每个论断都能指到一个文件或一条命令——见「证据包」与「门禁」 |

三个能独立跑起来的东西，互相不依赖：

| 入口 | 是什么 | 端口 | 耗时 |
|---|---|---|---|
| `agent-server` | 框架自带服务，4 个端点 | 8000 | 秒级 |
| `growth-agent` | 产品服务，10 个业务端点 | 8100 | 秒级（检索冷启动 18–20s） |
| `agent create <name>` | 脚手架：从零到"发一条请求拿到回答" | — | 21.4s（实测） |

## 快速开始

```bash
uv sync                       # 装 workspace 全部依赖

uv run agent-server           # A. 框架服务 :8000
curl -s -X POST localhost:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"task":"chat"}'

uv run growth-agent           # B. 产品服务 :8100（需 .env 里的模型 key）
curl -s -X POST localhost:8100/v1/answer \
  -H 'content-type: application/json' \
  -d '{"session_id":"s1","question":"B2B 内容营销怎么写标题"}'

make test                     # C. 全量离线用例（零网络、零 key，这是默认门禁）
```

最后一条是重点：**门禁不需要 API key**。一条"必须先配密钥"的 target 最后一定会被绕过，
然后门禁形同虚设（`Makefile` 第 3 行）。

起新项目（第三方视角，验证"框架真能给别人用"）：

```bash
# 在仓库根目录执行；--framework-path 相对"生成出来的项目目录"解析
agent create my-bot --target . --framework-path ../packages/agent_framework

cd my-bot
export OPENAI_API_KEY=sk-...
uv sync
uv run uvicorn my_bot.app:app --reload
```

生成 8 个文件：`app.py`（装配 gateway + 一个 `/v1/ask`）、两条冒烟测试、
`pyproject.toml`、`configs/models.yaml` 等。重试、路由、熔断、并发闸门、
`request_id`、错误契约都在框架里，`app.py` 一行都不写。

"打包 → 装进干净 venv → 起服务 → 发请求"这条完整路径由
`evals/scripts/scaffold_drill.py` 实跑：21.4s（build 0.1 / install 13.9 / boot 0.6 / ask 0.08）。
本仓库未发布到 PyPI，所以演练时用 `--framework-spec "agent-framework @ file:///...whl"`
指向本地 wheel——这也是脚手架唯一没能验证到位的地方（缺"依赖从零解析"）。

## 边界：能做什么 / 不能做什么

先讲不能，否则后面每个数字都会被误读。

| 能 | 不能（及原因） |
|---|---|
| 单进程内跑完整链路，接口可注入、可测 | 多机分布式编排：`workflow/` 只有接口缝，没接 Temporal（`packages/agent_framework/src/agent_framework/workflow/`） |
| 三类故障（超时 / 5xx / 限流）自愈，长任务断点续跑 | 百万级并发：uvicorn 单进程，100 并发离线档 1224 请求 0 错误——但那是 fixture 回放，不含真实上游延迟（`evals/history/2026-W45-w2-load.md`） |
| 注入 10/10 拦截、正常 6/6 放行；工具调用全审计 + HITL 门 | 前端界面：W12 P3 判定**不解锁**（`docs/log.md`） |
| 框架零 ML 依赖，embedding / reranker / 向量库全部注入 | 开箱即用的 dense 检索：不装 `local` extra 只能跑 sparse（`ops/RUNBOOK.md` §4.5） |
| 三平台文案（微信 / LinkedIn / X） | 自动发布到平台：只产出文案，不做投放 |

`AUTH_MODE` 默认 `off`（开发/测试档）；配了 `AUTH_MODE=key` 但 `API_KEYS` 为空时
**启动直接失败**，而不是静默放行（`server/auth.py:163`）。

## 架构

五层 + 底座，只从上往下 import：

```
  Product   apps/growth_agent
            调研 → SEO → 三平台文案 / 工具 / 记忆 / 审批
  ─────────────────────────────────────────────────────
  Server    server/
            create_app / chat / tasks / SSE / auth / errors
  ─────────────────────────────────────────────────────
  Harness   planner/ multi_agent/ runtime/ skills/
            Plan-and-Execute / Workflow / Supervisor / Run+Step
  ─────────────────────────────────────────────────────
  Context   context/ retrieval/ knowledge/ guardrails/ memory/
            预算 / trust / 融合 / 引用 / 注入防护 / 记忆
  ─────────────────────────────────────────────────────
  LLM       llm/client.py
            缓存 → 路由 → 候选链 → 重试 → 熔断 → adapter
  ─────────────────────────────────────────────────────
  Base      config.py / observability.py / tracing.py
```

```
apps/growth_agent  ──>  agent_framework.*  ──>  agent_framework.llm.types
evals/scripts  ─────┘        │
                             └──> 第三方（httpx / pydantic / fastapi）

       永远不会有：agent_framework ──> apps/*
```

三处最容易被问的接线：

- **熔断只挂在 `gateway.chat()` 一处**，包住整条候选链。adapter 级细粒度熔断会让
  "候选 A 挂了换 B"和"熔断器自己重试"变成同一件事，上游调用数就不可预测了。
- **重试的唯一主人是 gateway**。adapter 明文禁止重试、打日志、做业务判断（ADR-003），
  否则一次请求会在两层各自重试：上游调用数从「3 候选 × 2 次 = 6」变成不可预测
  （W11 INC-5 实测 18，而那三条断言当时全绿）。
- **错误码单点投影**：`error_code_for()` 决定 `code`，`http_status_for()` 决定状态码。
  产品侧不写 `try/except`，否则同一个上游故障会在两侧返回不同 code。

## 六条不能让步的约束

每条都有测试守着，不是文档承诺：

| 约束 | 守它的东西 | 失守的后果 |
|---|---|---|
| 框架不 import 产品 | `test_framework_never_imports_product`（AST 扫） | 脚手架给第三方用直接不成立 |
| 公共 API 是扫出来的，不是想出来的 | `evals/scripts/api_audit.py --check` + `test_audit_script_passes` | 私有符号被外部用 → 从此不能改 |
| `import` 不付不该付的代价 | `test_importing_config_does_not_pull_httpx`（子进程跑） | 只想读一行配置却拉起 httpx，还会造循环导入 |
| 跨 provider 语义一致 | `test_conformance.py` A1–A8 / A13 / A14（工具调用 / 流式=非流式 / usage / finish_reason / 错误类 / 单次请求 / 基线超集 / 工具入参合规） | 换模型时行为漂移，排查从"改配置"变成"读两个 SDK" |
| 门禁默认离线 | `make test` / `make eval` 全用 fixture 模型 | 没 key 就跑不了的门禁 = 没门禁 |
| 默认值必须是旧行为 | 每个新机制都走 `Flag`，默认 `False`；灰度核对后再翻 | 上线新功能等于上线新故障面 |

## 框架能力地图

顶层只导出 13 个符号（`build_gateway` / `Settings` / `ChatCall` / `ChatOutcome` /
`LLMGateway` / `Message` / …），因为顶层是**起步面**不是全集索引。其余按子包访问。

| 子包 | 职责 |
|---|---|
| `llm` | 适配器 + client（路由 / 重试 / 熔断 / 缓存）+ 错误投影 |
| `context` | 上下文预算、信任级、组装 |
| `guardrails` | 注入防护（`isolate` / `unwrap`）、输出校验、HITL、行为监控 |
| `knowledge` | RAG 引用溯源、治理、增量索引 |
| `retrieval` | 召回 → 融合 → 精排 → ACL → 组装 |
| `memory` | 工作记忆 / 长期记忆 / 提取（`is_explicit_instruction`） |
| `skills` | Skill 规范 / 执行 / 沙箱 / MCP（`servers/` 独立子包） |
| `session` | 会话存储（内存 / Redis） |
| `server` | HTTP / SSE / 鉴权 / 错误投影 |
| `planner` `multi_agent` `runtime` `resilience` | W9/W11 的实验与可靠性产物，**产品没在用**，不进顶层、不承诺稳定 |

另有 `queue` / `workflow` / `exporters` 三个内部设施包，不属于公共面。

判据来自真实使用：`api_audit.py` 分五档（`product` > `product_test` > `evals` >
`framework_test` > `framework`），只有产品源码用过的抽象才够格进公共面。当前
118 个框架模块里，产品在用 54 个 / 208 个符号，私有符号泄漏 0、空壳模块 0。

## HTTP 端点

框架 `agent-server`（:8000）：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat` | 非流式或 SSE，走完整 gateway 链路 |
| POST | `/v1/tasks` | 202 + `task_id`，长任务进队列 |
| GET | `/v1/tasks/{id}` | 查状态 / 结果（含 `CIRCUIT_OPEN` 等失败形态） |
| GET | `/health` | 版本 + 组件状态 |

产品 `growth-agent`（:8100，**不含**框架的 chat/tasks——它是独立服务）：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/draft` | 单次文案草稿（输入护栏 → gateway） |
| POST | `/v1/answer` | RAG 主链路：检索 → 引用 → 记忆 → 输出校验 |
| GET | `/v1/memory` | 召回该用户长期记忆 |
| POST | `/v1/article` | 四段内容流水线（调研 → SEO → 三平台 → 质检） |
| GET/POST | `/v1/tools`、`/v1/tools/{name}`、`/v1/tools/call`、`/v1/tools/reload` | 工具发现 / 调用（审计 + HITL 判定）/ 重载 |
| GET/POST | `/v1/approvals`、`/v1/approvals/{id}` | 高风险动作审批 |
| GET | `/health` | 报 `rag` 冷/热状态——首请求 20s 是冷启动还是模型慢，靠这个字段区分 |

## 12 周数据

| 指标 | W1 | W12（收官门禁） |
|---|---|---|
| 测试 | 278 | **1471 passed / 30 skipped**（1501 collected） |
| ruff | pass | pass |
| mypy | pass | **134 files, 0 issues** |
| 覆盖率 | — | 核心 **88.6%** / 全部 85.2%（判据 70%） |
| 评测集 | 10 条 smoke | 54 golden + 30 引用样本 + 16 红队（10 攻击 / 6 误报对照）+ 6 多 Agent |
| ADR / 决策 | 4 篇 | 5 篇 ADR + 12 行 Decision Matrix |

趋势图 `docs/w12/trends.svg`（由 `evals/scripts/plot_trends.py` 从 `evals/history/`
生成，12 个点全部来自真实快照，无手工补点）。

Gate 12 收官：**10 通过 / 2 partial / 0 未做**。两处 partial 见文末「已知边界」。

## 门禁

```bash
make check     # ruff + mypy + 全量离线用例          （提交前）
make eval      # check + golden 校验 + 评测 + 回归对比（CI 里跑的那一套）
make golden    # 数据集自检：引用完整性 / 难度分布 / 去重 / 对抗规则
make gate      # 本期 vs baseline.json，指标掉了就红
make e2e       # 过载 503 / 任务端点 / trace 贯穿（进程内起服务，不占端口）
make up        # compose：agent-server / growth-agent / redis / pgvector（temporal 走 profile）
```

`make eval` 故意用 `run-eval-judge` 而不是 `run-eval`：两者写同一个快照文件，
退化成后者会把 judge/ragas 块冲成 `null`，下一期对比时 quality 指标全部变
"本期缺失"，趋势图悄悄断掉而没人发现（`Makefile` 第 52 行）。

## 证据包（8 件）

| # | 证据 | 位置 |
|---|---|---|
| 1 | 仓库本身：框架 + 产品 + 评测 + ops | `packages/` `apps/` `evals/` `ops/` |
| 2 | 12 周指标趋势图 | `docs/w12/trends.svg` |
| 3 | 12 周原始快照 | `evals/history/`（W1–W12） |
| 4 | 3 篇根因归因案例 | `docs/w4/cases/` |
| 5 | 压测 / 混沌 / 安全红队三报告 | `evals/history/2026-W45-w2-load.md`、`docs/w11/chaos-report.md`、`docs/w10/threat-model.md` |
| 6 | 单 vs 多 Agent 对比实验 | `docs/w9/single-vs-multi-ab.md` |
| 7 | 5 篇线上 incident + runbook | `docs/w11/incidents/`、`ops/RUNBOOK.md` |
| 8 | 架构图 + ADR + 可跑脚手架 | `docs/architecture-diagram.md`、`docs/adr/`、`evals/scripts/scaffold_drill.py` |

## 文档地图

按"你想知道什么"找，别从头读：

| 想知道 | 读 |
|---|---|
| 怎么跑起来、起不来先看哪 | `ops/RUNBOOK.md` |
| 框架能被 import 什么、注入点在哪 | `docs/w12/framework-api.md` |
| 一次请求具体怎么走（11 问逐题解答） | `docs/server-question-answers.md` |
| 为什么这么选、放弃了什么 | `docs/w12/decision-matrix.md`、`docs/adr/`（ADR-001…005） |
| 每周做了什么、踩了什么坑 | `docs/log.md`、`docs/w1/` … `docs/w12/` |
| 怎么对着 JD 讲这个项目 | `docs/interview-feynman.md`（费曼五层）、`docs/w12/interview-prep.md` |
| 数字怎么复现 | `evals/history/*.json` + `evals/scripts/` |
| 这东西在技术地图上的位置 | `docs/w12/frontier-notes.md` |

## 目录

```
packages/agent_framework/  框架：llm / context / retrieval / memory / skills / server …
apps/growth_agent/         产品：pipeline / routes_* / rag / memory / tools
evals/                     golden 集 + 12 周 history + 评测与审计脚本
ops/                       Dockerfile / compose / RUNBOOK / SLO / alerts
docs/                      架构 / ADR / 各周记录 / 面试材料
legacy/                    Temporal 试做（保留但不进 ruff/mypy 范围）
```

## 已知边界与遗留

诚实清单，都是已知项而非疏漏：

- **独立仓库未拆**（partial）：还在 `PycharmProjects/LLMs` monorepo 内，
  `git filter-repo` 需要外网，本机不可用。
- **demo 视频未录**（partial）：无录制环境，用 `scaffold_drill.py` 的 21.4s 实跑代替。
- **P3 全部不解锁**：前端 / 多语言 / Computer Use / LoRA。ROI 排序在最后，
  按 ROADEMAP §7 判定本周不做，不是"以后再说"。
- **`api_audit.py` 有四个盲区**：import 了没调用、字符串动态 import、`getattr` 式访问、
  不校验符号存在性。所以"审计通过"≠"绝对正确"，细节见 `docs/w12/framework-api.md` §7。
- **Decision Matrix 有三处数字与原始报告对不上**（多 Agent 收益、RRF 提升、红队类别数），
  已列进 `docs/interview-feynman.md` 的「不要说」清单。引用任何数字前，先回
  `evals/history/` 核对——包括本 README 自己。
