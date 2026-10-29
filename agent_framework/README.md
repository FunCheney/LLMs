# growth-agent

> 12 周产出的**一个**生产级 Agent 项目：从「主题」到「深度调研 → SEO 文章 → 多平台文案」，全链路可评测、可观测、可降级。
> 学习计划见 [`ROADEMAP.md`](./ROADEMAP.md)，架构契约见 [`docs/architecture.md`](./docs/architecture.md)，本周任务见 [`docs/w1/w1-tasks.md`](./docs/w1/w1-tasks.md)，各厂商 API 差异见 [`docs/w1/`](./docs/w1/)。

## 生产级 DoD（12 周后逐条自评）

| 维度 | 指标 |
|---|---|
| 可用性 | 连续 7 天可用；正常成功率 ≥ 95%；注入工具故障后 ≥ 90% 且不崩 |
| 延迟 | `/v1/chat` 首 token p95 ≤ 1.5s；单次调研任务 p95 ≤ 120s |
| 成本 | 单次完整任务 ≤ ¥2.0；有预算上限与超支熔断 |
| 质量 | golden set（≥50 条）成功率 ≥ 80%；LLM-as-Judge ≥ 4.0/5 |
| 可靠性 | 模型超时 / 工具 500 / 限流三类故障可自愈；长任务可续跑 |
| 安全 | 10 条注入用例全拦截；工具调用全审计；高风险动作 HITL 审批 |
| 工程 | 核心模块单测覆盖 ≥ 70%；CI 跑 lint + test + eval 门禁；一键部署可回滚 |
| 可观测 | 任意一次任务可回放完整 trace |
| 文档 | README + 架构图 + ADR ≥ 5 + runbook + 12 周指标趋势图 |

## 仓库结构

```text
.
├── packages/agent_framework/     # 框架库：llm / knowledge / memory / skills / planner / eval / resilience
├── apps/growth_agent/            # 产品应用（capstone），依赖框架，反向不成立
├── evals/                        # golden / history / scripts（W2 起填充）
├── ops/                          # Dockerfile / compose / runbook（W2 起）
├── docs/                         # architecture / w1-tasks / llm-api-matrix / adr
└── legacy/temporal_demo/         # 旧 Temporal demo，仅作 W2 长任务参考，不参与构建
```

依赖方向：`apps/* → packages/agent_framework`，**框架永不 import 产品**（ADR-004）。

## 快速开始

```bash
uv sync                          # 安装 workspace 全部成员 + dev 依赖
uv run pytest                    # 278 个离线用例（零网络调用）+ 16 个 live 用例默认跳过
uv run ruff check . && uv run mypy

uv run agent-server              # 框架服务  http://127.0.0.1:8000/health
uv run growth-agent              # 产品服务  http://127.0.0.1:8100/health
```

> 受限环境（容器/沙箱）里 `pytest` 建不了系统临时目录时，加 `--basetemp=.pytest_tmp`。

## 如何切模型（改 yaml 即切换，代码零改动）

模型选择全部落在 `packages/agent_framework/src/agent_framework/models.yaml`。三个层次，按需选：

**1) 换某个 task 的默认模型** —— 改 `routes` / `tasks` 两行

```yaml
routes:
  cheap:  { primary: deepseek-flash, fallback: [qwen3.7-plus, gpt-4o-mini] }
tasks:
  chat: cheap            # /v1/chat 与 /v1/draft 默认走这里
```

**2) 单次请求钉住某个模型** —— 请求里带 `model`（alias，不是厂商 model 名）

```bash
curl -s -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"claude-sonnet-4-6"}'
```

用途：调试、W3 评测集钉模型、跨模型对比。alias 打错返回 `400 BAD_REQUEST`（调用方输入错误，
不是 500）；`models.yaml` 里的路由写了不存在的 alias 才会 `500 CONFIG_ERROR`。

**3) 完全离线（不花钱、不联网）** —— 用 mock 回放夹具

```bash
curl -s -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"fixture"}'
```

**接一家新的 OpenAI 兼容厂商**（vLLM / Ollama / 第三方代理）：`providers` 加一段
（`type: openai_compat` + `base_url` + `api_key_env`），`models` 加一条 alias，
然后跑 `uv run pytest` 确认 conformance 仍是绿的。**不需要改任何 .py 文件**——
出现 `if provider == "xxx"` 就是方向错了。

## 当前状态

- [x] **W1 Day1**：uv workspace 结构就位，两个包可启停，冒烟测试与静态检查通过
- [x] **W1 Day2**：契约层（types / errors / base）+ 单测
- [x] **W1 Day3**：openai_compat / anthropic / mock 三个适配器 + conformance A1–A8 全绿
  - 详见 [`docs/w1/w1-day3.md`](./docs/w1/w1-day3.md) §11「实现状态」
- [x] **W1 Day4**：config（models.yaml + V1–V6 校验）/ observability / registry / router /
      gateway（重试 + 计量 + committed 状态机）/ `POST /v1/chat`（JSON + SSE）+ A9–A12 全绿
  - 详见 [`docs/w1/w1-day4.md`](./docs/w1/w1-day4.md) §16「实现状态」
- [x] **W1 Day5**：录制器 + 基线表 + live 用例 + A13/A14 + 产品端 `/v1/draft`
  - 代码与工具已就位，**真机那一步待执行**（需要 `.env.local`，见下）
  - 详见 [`docs/w1/w1-day5.md`](./docs/w1/w1-day5.md)

### Day5 待执行的真机三连（有 key 时按顺序跑）

```bash
# 1) 录真实夹具（默认 dry-run；--apply 才写盘）并重跑 conformance
uv run python evals/scripts/record_fixtures.py --apply
uv run pytest --basetemp=.pytest_tmp

# 2) 录真实鉴权失败体（用无效 key，不花钱），验证 401→502 的映射
uv run python evals/scripts/record_fixtures.py --bad-key openai,deepseek,qwen,anthropic --apply
uv run pytest --basetemp=.pytest_tmp -k a6

# 3) 真机用例 + 基线表
uv run pytest -m live --basetemp=.pytest_tmp -q
uv run python evals/scripts/bench_baseline.py --repeat 3
```

## 开发约定

1. **切模型只改 `models.yaml`**：业务代码零改动，这是 W1 的验收项之一。
2. **adapter 内禁止重试 / 日志 / 业务判断**（ADR-003），否则成本与日志会失真。
3. **重试只在首 token 前**；已开始输出的流不允许重放。
4. **上游鉴权失败映射 502 而不是 401**（ADR-005）：401 只表示调用方未通过我方鉴权。
5. **结构变更与逻辑变更分开提交**，便于 review 与回滚。
6. 密钥只走环境变量（见 `.env.example`），日志中一律打码。
7. **产品层只依赖框架的公开入口**（`llm.client` / `server.errors` / `server.middleware`）；
   一旦在 `apps/` 里出现 `llm.providers`，说明边界画错了。
