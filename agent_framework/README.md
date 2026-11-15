# Growth Agent

12 周从零到生产级的 LLM Agent 项目。不堆功能，只做**可验证证据**。

## 一句话

一个从调研到发布的增长营销 Agent，用自研框架 `agent_framework` 做编排、检索、护栏、记忆、技能。

## 12 周指标

| 指标 | W1 | W12 |
|---|---|---|
| 测试数 | 278 | 1471 |
| ruff | pass | pass |
| mypy | pass | 134 files 0 issues |
| 覆盖率(核心) | - | 88.6% |

趋势图：`docs/w12/trends.svg`（由 `evals/scripts/plot_trends.py` 从 `evals/history/` 生成）。

## 快速开始

```bash
# 装依赖
uv sync

# 起服务
uv run agent-server

# 发一条 chat
curl -s -X POST localhost:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"task":"chat"}'
```

## 用脚手架起新项目

```bash
pip install agent-framework
agent create my-bot --target .
cd my-bot
export OPENAI_API_KEY=sk-...
uv sync
uv run uvicorn my_bot.app:app --reload
```

## 证据包

| # | 证据 | 位置 |
|---|---|---|
| 1 | 仓库 | 本仓库 |
| 2 | 趋势图 | `docs/w12/trends.svg` |
| 3 | 12 周原始数据 | `evals/history/` |
| 4 | 根因案例 | `docs/w4/cases/` |
| 5 | 压测/混沌/安全报告 | `docs/w2/`, `docs/w11/`, `docs/w10/` |
| 6 | 单/多 Agent 对比 | `docs/w9/single-vs-multi-ab.md` |
| 7 | 上线 incident + runbook | `docs/w1/incidents/`, `ops/RUNBOOK.md` |
| 8 | 架构图 + ADR + Decision Matrix | `docs/architecture-diagram.md`, `docs/adr/`, `docs/w12/decision-matrix.md` |

## 目录

```
packages/agent_framework/  LLM Agent 框架
apps/growth_agent/         增长营销产品
evals/                     评测 + 12 周 history
ops/                       Dockerfile / runbook / SLO
docs/                      架构 / ADR / 各周执行记录
```

## 核心设计决策

见 `docs/w12/decision-matrix.md`（12 行取舍汇总）和 `docs/adr/` (5 篇独立 ADR)。

## ADR 索引

| 编号 | 标题 |
|---|---|
| ADR-001 | 分层架构：单向依赖 |
| ADR-002 | Mock 做回放引擎，不做假 LLM |
| ADR-003 | Adapter 禁止重试、日志、业务判断 |
| ADR-004 | 产品只依赖公共 API |
| ADR-005 | pgvector over Qdrant / FAISS |

## 覆盖率

核心模块 88.6%（判据 70%），全部模块 85.2%。详细：`evals/coverage-report.json`。
