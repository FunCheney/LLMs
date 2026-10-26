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
uv run pytest                    # 骨架冒烟测试（离线，无网络调用）
uv run ruff check . && uv run mypy

uv run agent-server              # 框架服务  http://127.0.0.1:8000/health
uv run growth-agent              # 产品服务  http://127.0.0.1:8100/health
```

## 当前状态

- [x] **W1 Day1**：uv workspace 结构就位，两个包可启停，冒烟测试与静态检查通过
- [ ] W1 Day2：契约层（types / errors / base）+ 单测
- [ ] W1 Day3：openai_compat / anthropic / mock 三个适配器 + conformance 套件
- [ ] W1 Day4：client / registry / router / config + `POST /v1/chat`（SSE）
- [ ] W1 Day5：真机验证、录制夹具、基线对比表

## 开发约定

1. **切模型只改 `models.yaml`**：业务代码零改动，这是 W1 的验收项之一。
2. **adapter 内禁止重试 / 日志 / 业务判断**（ADR-003），否则成本与日志会失真。
3. **重试只在首 token 前**；已开始输出的流不允许重放。
4. **上游鉴权失败映射 502 而不是 401**（ADR-005）：401 只表示调用方未通过我方鉴权。
5. **结构变更与逻辑变更分开提交**，便于 review 与回滚。
6. 密钥只走环境变量（见 `.env.example`），日志中一律打码。
