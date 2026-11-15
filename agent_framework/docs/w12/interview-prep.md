# 模拟面试准备（W12 D5）

> 按两份 JD 十条职责逐条讲。每条挂一个数字 + 一个取舍 + 一个最难的问题。

## 职责 1：设计并实现 LLM Agent 框架

- **我的对应实现**：W1-W12 的 `agent_framework` 包（llm/context/skills/memory/retrieval/resilience/runtime/server）
- **量化数字**：1471 passed / 30 skipped, mypy 134 files 0 issues, ruff 0 errors
- **架构取舍**：自研而非用 LangGraph（ADR-009），因为 budget handoff 搜不到现成实现
- **最难的问题**：W11 的解封皮 incident——guardrails.isolate 加了锁但unwrap forgot to开锁导致封皮流进标题。根因：isolate/unwrap 不在同一个调用栈里，所以测试隔离独立绿
- **可演示**：`evals/scripts/api_audit.py --check`

## 职责 2：多模型适配与路由

- **对应**：`llm/client.py` + `llm/router.py` + `llm/difficulty_router.py`
- **数字**：5 家厂商（OpenAI/DeepSeek/Qwen/Anthropic/Mock），按 task -> tier -> 候选链 fallback
- **取舍**：路由只改顺序不缩短候选链（ADR-w12）。难度分流不省钱——它把钱花在需要的请求上
- **最难**：CapabilityError 在重试循环之外抛，带 tools 的请求路由到不支持 tools 的便宜档会直接失败不 fallback

## 职责 3：生产可靠性（熔断/重试/降级）

- **对应**：`resilience/` 包（circuit_breaker/degradation/concurrency/errors）
- **数字**：W11 混沌 7 场景全受控, calls_after_open == 0, Kill Switch 0.5s
- **取舍**：熔断在 gateway 不在调用点（ADR-w11-001），降级去掉兜底模型 = 第二次重试（ADR-w11-002）
- **最难**：HALF_OPEN 只放行一个探测——同步 MockTransport 让探测瞬间完成，用 asyncio.Event 把它停在半空才测出来

## 职责 4：评测体系设计

- **对应**：`evals/` 目录 + 12 周 history
- **数字**：W4 Cohen kappa = 0.343 -> 混合口径（规则兜底 + judge 只在有 kappa 时采信）
- **取舍**：golden 集只做回归不做准线（不拿golden分数当质量及格线）
- **最难**：judge 一致性低时怎么用——纯弃了可惜，纯信了会假绿。解法是只在有 kappa 时采信

## 职责 5：检索增强（RAG）

- **对应**：`retrieval/` + `knowledge/` 包
- **数字**：W5 Recall@5 +12pp（RRF 融合 vs 单路），p95 +40ms（bge rerank）
- **取舍**：pgvector over Qdrant（ADR-005），RRF over 加权（免调权重）
- **最难**：引用溯源需要从 raw chunk 到最终输出的全链路标注，不只是 Retriever 返回的 chunk ID

## 职责 6：安全护栏（注入防护）

- **对应**：`guardrails/` 包
- **数字**：W10 红队 12 类攻击全部拦截，0 误报
- **取舍**：isolate/unwrap 对称化（封皮上加锁，出来时必须开锁），而不是只做检测
- **最难**：W11 INC-2 封皮流进标题——攻击不在输入里，在结构化包装里

## 职责 7：内容生成 Pipeline（增长营销 Agent）

- **对应**：`apps/growth_agent`（调研 -> SEO -> 3 平台文案）
- **数字**：smoke eval 10/10 PASS, 3 平台文案风格差异可调
- **取舍**：平台文案分别生成（W8 实测合并改写会互相污染）
- **最难**：风格隔离 vs 成本——3 个独立 LLM 调用 vs 1 次生成 3 段。我们选了隔离。

## 职责 8：上线与运维

- **对应**：`ops/` 目录
- **数字**：5 篇 incident + runbook + Kill Switch 0.5s
- **取舍**：compose + 单机（放弃 K8s 自动扩缩，ADR-w10）
- **最难**：W11 INC-5 演练假绿——混沌演练"成功"但成功率没标定，等于没测

## 职责 9：多 Agent 协调

- **对应**：`multi_agent/` 包
- **数字**：W9 单 vs 多 A/B：成功率 +4.3pp, 成本 x3.5
- **取舍**：调研用多 Agent，其余用 Workflow（成本不对称）
- **最难**：预算每跳重置——LangGraph 的 supervisor 是全局预算，调研 Agent 能把全部预算花光

## 职责 10：框架提取与文档化

- **对应**：W12 D1-D2
- **数字**：顶层 13 个公共符号, 54 个模块被产品使用, 审计 0 泄漏/0 空壳
- **取舍**：顶层不做全集索引（179 -> 13）; 能力子包各管自己的 `__all__`
- **最难**：产品用了私有符号 `_EXPLICIT_MARKERS`——正确的公共面是判断函数（is_explicit_memory）不是正则

## 自测要点

随机抽一条，10 分钟讲清：
1. 我的对应实现（哪个模块/哪周）
2. 一个量化数字
3. 架构取舍（为什么这样做，替代方案）
4. 最难的问题（一个 incident 或反直觉发现）
5. 可演示：demo/命令
