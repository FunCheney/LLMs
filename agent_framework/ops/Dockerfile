# 多阶段构建：builder 里装依赖，运行镜像只带 venv + 源码。
# 单阶段构建的镜像会带着 uv 缓存、编译器、.git，动辄 1GB+——压测时镜像拉取时间
# 会盖过被测指标。

FROM ghcr.io/astral-sh/uv:0.9-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

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

# 非 root 运行：容器里跑 root，一旦有目录穿越就是一个完整的写权限。
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8100

CMD ["uvicorn", "agent_framework.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
