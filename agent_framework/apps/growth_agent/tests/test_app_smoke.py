"""骨架冒烟测试：验证产品包能消费框架包（依赖方向）。

这条用例是 ADR-004 的最小保障：一旦有人把方向写反（框架 import 产品），
workspace 内会出现循环依赖，这里或 agent_framework 的测试会先炸。
"""

import agent_framework
from fastapi.testclient import TestClient
from growth_agent.app import app


def test_framework_dependency_wired() -> None:
    body = TestClient(app).get("/health").json()
    assert body["framework_version"] == agent_framework.__version__
