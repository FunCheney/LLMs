# 多阶段构建：builder 里装依赖，运行镜像只带 venv + 源码。
# 单阶段构建的镜像会带着 uv 缓存、编译器、.git，动辄 1GB+——压测时镜像拉取时间
# 会盖过被测指标。

FROM ghcr.io/astral-sh/uv:0.9-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# rag-mvp 是 workspace 之外的路径依赖：apps/growth_agent 里写的是
# `path = "../../../06_rag/rag_mvp"`。三件事必须同时成立，否则构建会挂在
# "Distribution not found" 或 "failed to open file .../README.md"：
#
#   1. 构建上下文在 agent_framework 内，够不到 06_rag → 由 compose 的
#      `additional_contexts: ragmvp` 单独传进来；
#   2. 落点必须是 /06_rag/rag_mvp —— 这个绝对路径是从容器内
#      /app/apps/growth_agent 往上三级算出来的，换个位置就对不上；
#   3. README.md 不能漏 —— uv_build 要读它，缺了报的错跟依赖毫无关系。
COPY --from=ragmvp pyproject.toml README.md /06_rag/rag_mvp/
COPY --from=ragmvp src /06_rag/rag_mvp/src

# 先只拷依赖描述：源码变了不必重装依赖，构建缓存才有意义。
COPY pyproject.toml uv.lock ./
COPY packages/agent_framework/pyproject.toml packages/agent_framework/pyproject.toml
COPY apps/growth_agent/pyproject.toml apps/growth_agent/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY packages ./packages
COPY apps ./apps

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# curl 只用于 healthcheck：compose 的 depends_on 需要它判断"真的就绪了"，
# 否则压测会打在还没起完的容器上，报一堆 connection refused。
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/packages /app/packages
COPY --from=builder /app/apps /app/apps
COPY --from=builder /app/pyproject.toml /app/uv.lock /app/
# editable 安装会把 /06_rag/rag_mvp/src 写进 .pth，运行镜像不拷这份源码
# 就会在 import 时报 "No module named 'rag_mvp'"，而构建阶段完全正常。
COPY --from=builder /06_rag /06_rag

# 非 root 运行：容器里跑 root，一旦有目录穿越就是一个完整的写权限。
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8100

CMD ["uvicorn", "agent_framework.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
