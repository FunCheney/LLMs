"""骨架冒烟测试：保证重构后项目仍可导入、可启动。

W1 Day2 起新增契约测试与 conformance 套件（见 docs/w1-tasks.md）。
"""

import agent_framework
from agent_framework.server.app import app
from fastapi.testclient import TestClient


def test_package_importable() -> None:
    assert agent_framework.__version__


def test_health_ok() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
