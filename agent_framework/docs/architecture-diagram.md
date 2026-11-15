# 架构图

> 五层 + 依赖方向。上层数 import 下层，下层不知道上层存在（ADR-001/004）。

```
┌─────────────────────────────────────────────────────────────────┐
│  Product: growth_agent                                          │
│  (apps/growth_agent)                                             │
│  调研 → SEO → 3 平台文案                                       │
├─────────────────────────────────────────────────────────────────┤
│  Server Layer                                                    │
│  (server/)                                                       │
│  create_app() / /v1/chat / /v1/tasks / SSE                     │
│  auth / middleware / errors → HTTP 影射                         │
├─────────────────────────────────────────────────────────────────┤
│  Harness / Orchestration                                          │
│  (planner/ multi_agent/ runtime/ skills/)                    │
│  Plan-and-Execute / Workflow / Supervisor / Run+Step          │
│  Skill 治理 / sandbox / visibility / audit                     │
├─────────────────────────────────────────────────────────────────┤
│  Context / Knowledge / Guardrails                                │
│  (context/ retrieval/ knowledge/ guardrails/ memory/)        │
│  预算 / trust / 检索融合 / 引用溯源 / 注入防护 / 记忆         │
├─────────────────────────────────────────────────────────────────┤
│  LLM Adapter + Gateway                                             │
│  (llm/)                                                           │
│  client.py: 路由→重试→熔断→并发→adapter                          │
│  providers/: openai_compat / anthropic / mock                   │
├─────────────────────────────────────────────────────────────────┤
│  Config / Observability                                             │
│  (config.py / observability.py / tracing.py)                    │
│  models.yaml / Settings / RequestLog / TraceExporter           |
└─────────────────────────────────────────────────────────────────┘
        ↑ 单向依赖：只从上往下 import
```

## 依赖方向

- `apps/` → `agent_framework.*`（公共面）
- `agent_framework.server` → `agent_framework.llm/context/skills/...`
- `agent_framework.llm.client` → `agent_framework.llm.{base,types,errors,router,registry}`
- `agent_framework.llm.providers.*` → `agent_framework.llm.{base,types,_common}`（不 import client）
- `agent_framework.config` → `pydantic` only（最底层，不被别人 import 的实现细节）

## 公共面

| 入口 | 覆盖 |
|---|---|
| `agent_framework` (顶层) | 13 个符号：`build_gateway`, `Settings`, `ChatCall`, ... |
| `agent_framework.llm` | ChatCall/ChatOutcome/LLMGateway + 错误类 + 类型 |
| `agent_framework.server` | `create_app` + 鉴权 + 中间件 + 错误投影 |
| `agent_framework.guardrails` | isolate/unwrap + 注入检测 |
| `agent_framework.retrieval` | 召回→融合→精排→ACL→组装 |
| `agent_framework.memory` | Working + LongTerm + Extract |
| `agent_framework.skills` | Skill spec + executor + sandbox + MCP servers |
| `agent_framework.runtime` | RunState + flags + assets + resume |
| `agent_framework.resilience` | circuit_breaker + degradation + concurrency + semantic_cache |

## 关键约束

1. 产品不穿实现模块（ADR-004，`api_audit.py --check` 守着）
2. Adapter 不重试/不日志（ADR-003）
3. 熔断包整条候选链（ADR-w11-001）
4. 解封皮对称（isolate → 必须 unwrap）
