# agent-framework

FastAPI + Temporal demo.

## Start (local)

1. Start Temporal: `docker compose up -d`
2. Worker: `uv run python -m agent_framework.worker`
3. API: `uv run agent-server`

Health: `GET http://localhost:8000/health`  
Start workflow: `POST http://localhost:8000/v1/run`

Optional env: `TEMPORAL_ADDRESS` (default `localhost:7233`), `TEMPORAL_TASK_QUEUE` (default `my-task-queue`).
