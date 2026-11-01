# 常用命令入口（`make` 不带参数会打印帮助）
#
# 设计原则：**每条命令都必须能在没有 API key 的机器上跑**（除 load 的在线档）。
# 一条"必须先配密钥"的 make target，最后一定会被绕过，然后门禁形同虚设。

.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help sync test lint type check smoke baseline up down ps logs load clean

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

eval: check smoke ## CI 里跑的那一套

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

clean: ## 清掉本地缓存（不碰 .venv 与 .env*）
	rm -rf .pytest_tmp .pytest_cache .ruff_cache .mypy_cache
