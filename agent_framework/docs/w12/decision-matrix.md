# Decision Matrix：12 周关键取舍

> 每行 Evidence 列在前，Conclusion 列在后——顺序反了会写成自我美化。
> 证据栏必须指向真实数字或文档。写"经验"的格子 = 没有依据。

| 决策点 | 候选方案 | 关键约束 | 证据 | 结论（选了什么 + 放弃了什么） | ADR |
|---|---|---|---|---|---|
| 编排范式 | ReAct / Plan-and-Execute / Workflow / 多 Agent | 目标明确但路径不定 | W9 三层对比：多 Agent 成功率 +4.3pp 但成本 ×3.5 | 调研用 Plan-and-Execute，其余用 Workflow；放弃多 Agent 做全流程 | ADR-009(w9) |
| 向量库 | pgvector / Qdrant / FAISS | 已有 PG | W5 Recall@K 延迟 <2ms | pgvector（放弃 Qdrant 的精细调优） | ADR-005 |
| 检索融合 | RRF / 加权求和 / 单路 | 两种召回分数量纲不同 | W5 Recall@5 提升 12pp | RRF（放弃加权——免调权重） | ADR-005 |
| 重排 | bge-reranker / LLM rerank / 不重排 | 延迟预算 100ms | W5 p95 +40ms | 先 bge，LLM 只用于离线 | ADR-005 |
| 状态存储 | Redis / PG / Temporal | 长任务需可恢复 | W11 续跑演练：skipped 2 步、executed 2 步 | Redis 热态 + PG 冷态（放弃 Temporal 编排，单机不够量级） | ADR-w11 |
| 记忆载体 | pgvector namespace / 独立库 | 与知识库边界 | W7 边界判据：与用户绑定 vs 与文档绑定 | 同库不同 namespace（放弃独立库的隔离性） | ADR-w7 |
| 评测口径 | 纯规则 / 纯 judge / 混合 | judge 需一致性证明 | W4 Cohen kappa = 0.343 | 混合：规则兜底，judge 只在有 kappa 时采信（放弃纯 judge 的覆盖面） | ADR-w4 |
| 框架 | 自研 / LangGraph | 可控性 vs 迭代速度 | W9 对照实现：LangGraph 的 supervisor 无法做 budget handoff | 自研主实现，LangGraph 仅对照（放弃 LangGraph 的社区生态） | ADR-009(w9) |
| 部署 | compose / K8s | 单机够用 | W2 压测 QPS 的序在 8 以下 | compose + 单机（放弃 K8s 的自动扩缩） | ADR-w10 |
| 模型分流 | 难度启发式 / 纯路由表 | 成本 vs 质量 | W12 A/B: 质量优先基线 -47.4%; 当前 chat->cheap 基线 x13 | 分流不省钱，它把钱花在需要的请求上；放弃"先试 cheaper 省 cost"的假设 | ADR-w12(difficulty_router) |
| 熔断层级 | 在 adapter / 在 gateway / 在调用点 | 不可叠加重试 | W11 calls_after_open == 0 | 在 gateway 里包整条候选链（放弃 adapter 级细粒度） | ADR-003, ADR-w11-001 |
| 降级策略 | 兜底模型 / 兜底话术 / 原样抛出 | 降级不许编 | W11 ADR-W11-002 | 顺序改为语义缓存 -> 兜底话术(默认关) -> 抛出（放弃"换便宜档再试一次"= 第二次重试） | ADR-w11-002 |
