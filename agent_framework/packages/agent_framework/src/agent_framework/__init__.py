"""agent-framework —— 可复用的 Agent 框架。

目录规划（随周次逐步填充，禁止提前实现）：
- llm/          W1  适配层、客户端、路由
- knowledge/    W5–W6  RAG
- memory/       W7   工作 / 长期记忆
- skills/       W8   Skill 与 MCP
- planner/      W9   规划与多智能体
- eval/         W3–W4  评测
- resilience/   W11  容灾与安全中间件
- server/       W1  HTTP / SSE 传输层

依赖方向：apps/* -> agent_framework，框架永不 import 产品。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
