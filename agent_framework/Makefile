# 常用命令入口（`make` 不带参数会打印帮助）
#
# 设计原则：**每条命令都必须能在没有 API key 的机器上跑**（除 load 的在线档）。
# 一条"必须先配密钥"的 make target，最后一定会被绕过，然后门禁形同虚设。

.DEFAULT_GOAL := help
SHELL := /bin/bash

# 报告期号。默认按 ISO 周，需要写"下周"的报告时 `make run-eval WEEK=2026-W47`。
WEEK ?= $(shell date +%G-W%V)

.PHONY: help sync test lint type check check-all golden run-eval run-eval-judge gate baseline-snapshot judge-agreement smoke eval baseline up down ps logs load e2e clean

help: ## 列出所有目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

sync: ## 安装依赖（含 dev）
	uv sync

test: ## 全量离线用例（零网络）
	uv run pytest --basetemp=.pytest_tmp

lint: ## ruff
	uv run ruff check .

type: ## mypy
	uv run mypy

check: lint type test ## 本地门禁：静态检查 + 全量用例

smoke: ## 10 条 smoke eval（离线档，model=fixture）
	uv run python evals/scripts/run_smoke.py

golden: ## 校验 golden set（引用完整性 / 难度分布 / 去重 / 对抗样例规则）
	uv run python evals/scripts/validate_golden.py

run-eval: ## 54 条 golden 离线评测（分层指标 + 落盘报告）
	uv run python evals/scripts/run_eval.py --week $(WEEK)

run-eval-judge: ## 同上，但顺带跑 judge（离线档=规则基线）
	uv run python evals/scripts/run_eval.py --week $(WEEK) --judge

gate: ## 回归门禁：当前一期 vs baseline.json
	uv run python evals/scripts/regression_gate.py --current evals/history/$(WEEK)-eval.json

baseline-snapshot: ## 用当前一期重建基线（慎用：会覆盖历史基准）
	uv run python evals/scripts/regression_gate.py --current evals/history/$(WEEK)-eval.json --write-baseline

judge-agreement: ## Judge 与人工标注的一致性（离线档=规则基线）
	uv run python evals/scripts/judge_agreement.py --week $(WEEK)

# 注意：门禁这条链故意用 `run-eval-judge` 而不是 `run-eval`。
# 两者写的是同一个文件 `evals/history/<WEEK>-eval.json`，谁后跑谁说了算；
# 若这里退化成 `run-eval`，每周快照就会把 judge/ragas 块冲成 null,
# 下一期对比时 quality 指标全部变成"本期缺失 → skipped"，趋势图悄悄断掉。
# 离线档 judge 是规则基线、零成本，没有理由不带。
eval: check golden run-eval-judge gate ## CI 里跑的那一套：静态 + 用例 + golden + 评测（含质量）+ 回归门禁

baseline: ## W1 基线表（dry-run，不落盘）
	uv run python evals/scripts/bench_baseline.py --dry-run --models fixture

up: ## 一键拉起 app + redis（需要 docker）
	cd ops && docker compose up -d --build

down: ## 停掉并清理
	cd ops && docker compose down

ps: ## 容器状态
	cd ops && docker compose ps

logs: ## 跟踪 agent-server 日志（grep first_token_ms / trace_id 用）
	cd ops && docker compose logs -f agent-server

load: ## 离线档压测（20 并发 30s，需先 make up 或 uv run agent-server）
	uv run --with locust locust -f evals/scripts/locustfile.py \
		--headless -u 20 -r 5 -t 30s --host http://127.0.0.1:8000

e2e: ## 端到端自检：过载 503 / 任务端点 / trace 贯穿（进程内起服务，不占端口）
	uv run python evals/scripts/w2_e2e_check.py

clean: ## 清掉本地缓存（不碰 .venv 与 .env*）
	rm -rf .pytest_tmp .pytest_cache .ruff_cache .mypy_cache
