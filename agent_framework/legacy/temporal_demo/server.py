import uvicorn


def main() -> None:
    uvicorn.run(
        "agent_framework.web:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
