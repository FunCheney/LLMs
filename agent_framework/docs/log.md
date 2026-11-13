# 开发日志

> 每周五合流仪式第 4 步：记一条「本周做了什么 / 指标变化 / 遗留问题」。
> 指标快照写在 `evals/history/YYYY-Www.json`，本文件只记**结论与判断**。

---

## 2027-W01（01-04 ~ 01-08）· W11：生产硬化

> 这周的主线是**"敢给别人用"**，落到四件事：能看到问题（SLO）、不会被一个问题拖垮
> （熔断 + 降级 + 续跑）、出事能讲清（Run Timeline + incident）、改坏了能退回
> （Flag + 资产版本 + Kill Switch）。
> 一句话结论：**这周所有真 bug 都不在"新写的代码"里，而在"我以为已经生效的代码"里**——
> 熔断器差点造了第二次重试、缓存因为一个 `opts` 判断在真实路径上从未命中、
> Skill 漂移检查把正确做法也报成异常。三次都是"测试全绿 + 机制不存在"。
> Gate 11 判 **10 通过 / 3 partial**，DoD 九条判 **2 达标 / 7 部分达标 / 0 未做**。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `ops/slo.py` + `alerts.yml` + `slo_report.py` + `slo-spec.md` | 延迟用 `first_token_ms` 的 p95；成本必须分 `estimated` / `cache_hit` |
| D2 | `resilience/{circuit_breaker,degradation}.py` + `chaos.py`（7 场景） | 熔断挂在 `LLMGateway` 里（ADR-W11-001）；**上游调用数里熔断的贡献恰好为 0** |
| D3 | `runtime/{model,timeline,resume}.py` + `error_codes.py` + `resume_drill.py` | 九态执行模型；`run.state` 是算出来的；续跑不越过审批 |
| D4 | `docs/w11/incidents/` 5 篇 + `w11_fork_ablation.py` | 5/5 都不在 W4 的四层归因里；解封皮消融 A2 正常放行率 **0%** |
| D5 | `runtime/{flags,assets}.py` + `resilience/semantic_cache.py` + `llm/difficulty_router.py` | Flag 默认=旧行为；Kill Switch **0.5s**；三手段单变量 A/B |

### 指标变化

- 全量：**1446 passed / 30 skipped**（W10 末 1250 passed），ruff 全绿，mypy **134 文件 0 问题**
- 新增用例 **195 条**（逐个文件核对过：slo 16 / breaker 22 / degradation 9 / runtime_model 26 /
  idempotency 14 / resume_timeline 12 / difficulty_router 18 / semantic_cache 26 /
  flags 25 / assets 17 / bootstrap 10）
- 混沌：7 场景全部"受控"，`calls_after_open == 0`，共 82 次上游调用
- 续跑：`skipped=('step-1','step-2')`、实际只跑 `step-3/4`，终态 `completed`
- Kill Switch：**0.5s**（判据 ≤300s），回退后 `run_smoke` **10/10**
- 成本 A/B：缓存 **−37.6%**（16→10 次调用）、分流 **−47.4%**（质量优先基线）、并发 **2.99×** 墙钟

### 本周最重要的一条：ADR-003 不是"别重试"，是"重试只能有一个主人"

熔断器的失败计数单位是"一次 `gateway.chat()`"，而那次内部可能已经重试 3 次、
换过 2 个候选。所以上游调用数 = `放行次数 × 候选链长度 × max_attempts`，
这个乘积归 `LLMGateway`，**熔断器的贡献必须恰好为 0**。

`chaos.py` 的断言就是照这句话写的：`upstream_calls == threshold * chain * max_attempts`
且 `calls_after_open == 0`。第一版漏了候选链这个乘数（实测 18 写成 6），
断言红得莫名其妙——**先把"应该等于几"算清楚，再写断言**，否则调的是断言不是代码。

顺带在降级策略里删掉了"兜底模型（强→便宜）"：那就是第二次重试，
换了个名字躲在 `degradation.py` 里。顺序改成 语义缓存 → 兜底话术（默认关）→ 原样抛出。

### 三个"测试全绿但机制不存在"的缺陷

1. **语义缓存从未生效。** `_cacheable` 里写了"`opts` 里有采样参数就不缓存"，
   而产品端每个调用都带 `temperature`。于是缓存对所有真实路径都判 miss，
   而单测全绿——因为单测的 `ChatCall` 不带 `opts`。
   修法：`opts` **进缓存键**而不是排除，不同温度 = 不同键。这是"精确"与"一刀切"的区别。
2. **`skill_drift` 把正确做法报成异常。** 判据写成"键变了就报"，
   于是"改了内容也改了版本号"（正确做法）同样被列出来。一份整天误报的清单没人看，
   而没人看的清单等于没有。
3. **过期条目被记了两次。** `expired` 记一笔、`misses` 再记一笔，
   `lookups` 把一次查询算两次，命中率分母直接翻倍。改成"清掉过条目就不再记 miss"。

三次的共同形状：**判断逻辑写在一处、断言写在另一处，而两处对"什么算命中"理解不同。**

### 被实验逼出来的设计缺陷（难度分流）

第一版 `difficulty_signals` 里 `hard_intent` 与"文本长度"同权（都是 1），阈值 2。
实测 6 条硬样本有 **4 条只拿 1 分**，落回便宜档——"规划一下灰度方案"被判成简单请求。

提权到 2 的理由不是"感觉"，是**两个方向的误判代价不对称**：
硬请求误走便宜档 = 省了钱、可能答错、而答错当天没人发现；
简单请求误走强档 = 多花 20 倍单价，但看得见、可回滚、答案仍是对的。
没有真实质量数据能标定权重之前，宁可偏向强档。

还有一个反直觉结论写进了报告：**分流不"省钱"，它把钱花在需要的请求上。**
当前 `models.yaml` 是 `chat → cheap`，在这个基线上分流让成本 **×13**；
只有基线本来全走强档时它才省 **47.4%**。只报后一半就是挑数字。

### 遗留问题（按"要不要现在处理"排）

- **成本没有事中闸门**：全是"事前的比例"（便宜档 + 缓存），没有"钱花超了就停"。
  一个反复重试 + 长上下文的请求能安静花掉一天预算，看板第二天才显示。**本该本周做，没做**
- **7 天连续可用 / 首 token p95 / judge ≥4.0 全是空的**：没有真实流量，机制有、数字没有
- **覆盖率未测**：用"1446 条测试"代替"覆盖率 ≥70%"是偷换，测试集中在一个模块时覆盖率可以很低
- **镜像 tag 回退未实跑**：本机没有既有镜像，构建需要仓库外 `06_rag/rag_mvp`
- **`skill` 这一类资产没进快照**（`missing_kinds=('skill',)`），六类缺一类就少一条回滚路径
- **语义缓存是进程内缓存**：多副本不共享，集群命中率被稀释，重启即清空
- **远端 trace 后端从未在真实 key 下跑过**（缺 key → 强制 Noop，这是对的，但等于没验证）

### P3 决策：不解锁深入档

余量不足以同时保 W12 主线（框架提取 + 脚手架 + 文档 + 模拟面试）。
前端极简（P3-1）与多语言（P3-2）都**不做**，把时间留给"能被第三方使用"这件事——
一个只有自己能跑的系统，前端再好看也不构成交付物。
按 ROADEMAP §7 的规则，这条不是"以后再说"，是**本周明确判定不解锁**。

---

## 2026-W53（12-28 ~ 01-01）· W10：内容生成 + 安全护栏 + 首次上线

> 实际落盘 11-13 实跑。这周的主线是**"把四道门装在一个入口上"**，
> 而不是"装四道门"。三个缺陷全是同一类：每一层单独测都绿，串起来才炸——
> 护栏把自己的封皮当成攻击、封皮流进了标题、`AUTH_MODE=key` 静默没生效。
> 所以这周的标题是「**防线不是层数，是必经点**」。
> Gate 10 判 7 项通过、2 项 partial（端到端真实性 / 真实上线），
> 而这两项差的是**凭证与域名**，不是代码。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `pipeline/seo.py`（关键词→大纲→成文→meta）+ `POST /v1/article` | 四段各自可单测；零证据直接拒写，不降级成"凭常识写一篇" |
| D2 | `pipeline/style.py` + `PlatformSpec` | 约束数据驱动；确定性校验优先于再调一次 LLM；三平台顺序稳定 |
| D3 | `guardrails/injection.py` + 10 条红队 + `threat-model.md` | 10/10 拦截、6/6 放行；规则命中散在 10 条规则上（无单点依赖） |
| D4 | `guardrails/{input,output,behavior,hitl}.py` + `pipeline.py` | 四层收在一个 `run()`；未审批时 `execute` 计数为 0（红线断言） |
| D5 | `server/auth.py` + prod compose + Caddyfile + `deploy_check.sh` + RUNBOOK §10–12 | 401/502 严格分流；本机同构验证 5/5 |

### 指标变化

- 离线用例：`1126 → 1280`（新增 154，逐文件核对过），ruff 全绿，mypy **122 文件 0 问题**
- 全量：**1250 passed / 30 skipped**
- 评测（W53 离线档）：54 条 50 passed / 0 failed / 4 skipped，judge 1.922，`regression_gate` pass
- 红队：注入 **10/10 拦截**，正常用例 **6/6 放行**
- 部署自检：`deploy_check.sh` **5/5**（本机 127.0.0.1，真 HTTP，真 401）

### 本周最重要的一条：三道门，不如一个必经点

`input.py` / `output.py` / `behavior.py` / `hitl.py` 各自都能单测，但它们
**不构成一道门**——链路上每多一个调用点，就多一次"这个新接口忘了过某层"的机会，
而漏掉的那层不会报错，它只是**不存在**。

所以这周真正写下的不是"第四层"，是 `GuardrailPipeline.run`：顺序硬编码，
审批在执行**之前**（不是"执行前检查一下"），`skipped` 如实报告哪层没生效。
`/v1/article` 接上去之后，注入在花掉四次模型调用之前被拦——
测试里用空脚本网关证明这一点：如果先调模型再检查，那里会抛 `IndexError` 而不是干净的 400。

### 三个"只在组合时才出现"的缺陷

1. **护栏与自己的封皮互相指认**。外层隔离成 `<untrusted>`，内层再检一遍，
   `fake_delimiter` 看到 `</untrusted>` 就判越界——那是我们自己写的。第一次跑就 500。
   修法不是给规则豁免，是让隔离**可逆且幂等**（`unwrap` + 幂等 `isolate`）。
2. **封皮流进了标题**：`<untrusted source="user">内容营销怎么做</untrusted>：为什么…`。
   红队不会发现（不是攻击），误报率也不会（不影响拦截），只有人打开页面才看得到。
3. **`AUTH_MODE=key` 其实没生效**。pydantic-settings 的 env 名取自字段名，
   字段叫 `auth` 时读的是 `AUTH`。写错名字不报错，只是静默用默认值——而默认是 `off`。

第 3 条是 `deploy_check.sh` 抓到的，脚本第一版跑出来 **3/5**。
如果那个脚本只测"带 key 能通"，两档都会绿。**它写下来的价值不在绿的那天，在第一次红的那天。**

### 遗留问题（按"要不要现在处理"排）

- **真实模型一次没跑**：没有凭证。所有安全结论建立在规则层，模型自身被 jailbreak 的路径零数据
- **文章质量零证据**：没真实生成过一次，可读性/引用贴合度/平台风格全是未知
- **公网/证书/镜像构建未验证**：W11 要的真实反馈因此没有输入来源
- **语义级注入、多轮渐进不防**：`threat-model.md` §6 已列，10/10 不等于"注入已解决"
- **误报率只有 6 条样本**：不足以支撑"误报可接受"这句话
- **`/v1/approvals` 无角色校验**：持有效 key 的人都能批，包括提交人自己

---

## 2026-W52（12-21 ~ 12-25）· W9：规划与多智能体

> 实际落盘 11-12 实跑。W1 那个"只做归一化"的 `ToolCall` 在 W8 学会了守边界，
> 这周轮到**模型自己决定下一步**。落点不是"Agent 能自己跑"，而是
> **"自主性有上限、中止有原因"**。Gate 9 判 10 项通过、2 项 partial——
> 两项都不是"没做"，而是"做了，但它证明得没那么多"。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `planner/{base,controls,plan_execute,react,reasoners}.py` + ADR-W9-001 | 四道上限 + 控制权矩阵；模型只拿 3 个决定 |
| D2 | `planner/research_planner.py` | Agentic Search：四个停止原因都能单独断言；`SearchOutcome` 让后端成为唯一执行点 |
| D3 | `multi_agent/{roles,handoff,shared_state,supervisor}.py` + ADR-W9-002 | 三道守卫在编排层最外层；预算递减、环路指纹、乐观锁 |
| D4 | `multi_agent/langgraph_variant.py` + ADR-W9-003 + 两份笔记 | 8 条 parity 逐项对齐；主实现不换，理由是"守卫需要一个必经点" |
| D5 | `evals/scripts/{trajectory_eval,agent_paradigm_ab}.py` + 对比报告 | 过程层补上；七档 × 三层实跑 |

### 指标变化

- 离线用例：`1024 → 1126`（新增 102），ruff 全绿，mypy **110 文件 0 问题**
- 全量：**1096 passed / 30 skipped**
- 评测（W52 离线档）：54 条 50 passed / 0 failed / 4 skipped，回归门禁通过
- 对照实验：复杂档 5 步，`multi_agent` 1.00 / `react_w3` 0.80 / `react_w8` 1.00 /
  `plan_execute_w3` 0.80 / `plan_execute_w8` 1.00；极差全 0.00（= 可复现，不是稳定）

### 本周最重要的一件事：一个数字好看，可能只是没测到

D5 的对比实验我跑了三遍才跑出信息量。

**第一遍，七档全部 1.00。** 我先怀疑"是不是实现都对"，然后去看任务集：
回忆距离设了 2 步，而 ReAct 的默认窗口是 3。**机关根本没触发。**
一次没有区分度的实验，比没做实验更危险——它让人以为"验过了"。

**第二遍，把距离拉到 4 步，`react_w1` 掉了，其他还是 1.00。** 这次是判据错了：
我用"答案里有没有每一步的真值"当覆盖，而 `PolicyReasoner.compose` 是**按任务书
合成答案**的，答案天生含真值。测的是一个恒等式。

**第三遍，把判据改成"证据有没有真的取回来"**，曲线才出来：

| 配置 | 窗口 | 覆盖 |
|---|---|---|
| `react_w1` / `react_w3` | 1 / 3 | 0.80 |
| `react_w8` | 8 | 1.00 |
| `plan_execute_w3` | 3 | **0.80** |
| `plan_execute_w8` | 8 | 1.00 |
| `multi_agent` | 无（共享状态） | 1.00 |

这张表推翻了我在 D1 写下的一个说法。我在 ADR 里写"P&E 更适合长任务"，
但 `plan_execute_w3` 说明：**小窗口下 P&E 和 ReAct 一样掉。** 它赢在默认窗口 8，
不赢在"有计划"。同理，多 Agent 到 1.00 也不是因为"分工"——是因为共享状态
不受窗口限制，等价于窗口无限大。**"多 Agent 更好"在这里等价于
"记性更好更好"，而记性只要把窗口开大就行，不必拆。**

所以 Gate 9 #2 判了 partial。数字上 1.00 > 0.80 是成立的，
但"显著高于朴素 ReAct"这句话背后的因果，被自己的对照实验否掉了。

### 另外三件

1. **控制权矩阵第一版是装饰品。** `test_control_matrix_has_no_dead_entries` 红了，
   而它是对的——矩阵当时没有任何运行时查询。修法不是加豁免，是把
   `assert_code_owned` / `owner_of` / `model_may_decide` 接进 `StepRunner` 与 `PlannerBase`。
   这和 W8 那次"两份 `_env` 装配"是同一类：**测试红了先怀疑被测对象，别先怀疑测试。**
2. **步数如果按工具调用计数，守卫就永远不会触发。** 一个只会说"我没有新问题可查"的
   角色不产出任何动作，循环转得很欢而步数停在 0。改成数"角色轮数"之后，
   `no_progress` 与 `max_steps` 才真的会响。
3. **`LoopDetected` 一开始是个没人抛的类。** 和装饰品矩阵同一个味道——导出面好看，
   运行时没有路径到达它。改成 `HandoffLedger.record` 直接抛，调用方只要写了
   `record` 就没法忘了检查。

### 遗留

- **在线档一次没跑**：所有模型侧结论都建立在"模型完全听话"的假设上。
  真实模型会在工具选择、参数、停止时机上引入新的失败面，而那是本次实验的盲区。
- **多 Agent 档的检索词是替它构造的**（"有完整共享状态时本应提出的词"），
  1.00 是它的能力上限，不是实测水平。
- **token 列不可跨档比较**：多 Agent 只记角色申报的成本，supervisor 与 validator
  的推理没计。跨档成本对比现在没有可信数字，等 W11 统一计价。
- **任务集只有一个工具**："选错成另一个合法工具"这类失败测不出来，因为它不存在。
- **共享状态只在单测里模拟过并发**（在 `act` 里抢先 commit），没有真正跑过
  `asyncio.gather`。
- **LangGraph 是对照工件不是产品代码**：`langgraph` v1→v2 若改签名，它会先烂；
  没装 extra 的 CI 跑不到那 8 条 parity。
- 极差 0.00 在报告里必须配一句解释，否则读者会读成"系统非常稳定"。

---

## 2026-W51（12-14 ~ 12-18）· W8：Skills 系统与 MCP

> 实际落盘 11-11 实跑。W1 那个只做归一化、从不执行的 `ToolCall`，这周终于产生副作用——
> 但本周的重点不是"能调工具"，而是**"调工具这件事有边界"**。Gate 8 证据脚本 23/23 全绿，
> 零密钥零联网。有一个洞没补上（工具未进主链路），验收表里如实标了 partial。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `skills/mcp/{types,client,inmemory}.py` + ADR-001 | MCP 只做传输与翻译；业务层 AST 扫不到 `import mcp` |
| D2 | `servers/{search,fetch}_server.py` + `replay.py` + `redact.py` + ADR-002 | 未命中抛 `ReplayMissError` 绝不联网；回放标记与成本走 `_meta` |
| D3 | `skills/{spec,validator}.py` + `skill-spec.md` | 六层结构；第 6 层只允许四种机器可判的完成标准 |
| D4 | `skills/{sandbox,visibility,loader,executor}.py` | 五道门顺序固定、沙箱在副作用之前；可见面最大 50%、渐进式 19.1% |
| D5 | `skills/{guard,action,audit,library}.py` + ADR-003 | `ToolCall` 是意图不是执行；三个 badcase 全拦住 |
| 接线 | `apps/growth_agent/{tools,routes_tools}.py` + `runtime.ensure_tools` | `/v1/tools` 四个端点；`TOOLS=off` 直接拒绝启动 |

### 指标变化

- 离线用例：`794 → 994`（新增 200），ruff 全绿，mypy **97 文件 0 问题**
- Gate 8 证据脚本：**23/23**（`evals/scripts/tool_boundary_check.py`，默认离线档）
- 评测（W51 离线档）：54 条 50 passed / 0 failed / 4 skipped，`success_rate 100%`；
  回归门禁通过（只比了 success_rate / latency_p95 两项，其余按设计跳过）
- 可见面矩阵：6 个 Skill 全集下最大 **50%**（判据 <60%）；渐进式加载 **19.1%**（判据 <20%）

### 本周最重要的一件事：验证通过了，有两种

自检脚本第一次跑是 20/23。三条红里，只有一条是真的功能问题（`content` 只有 9 字符，
不满足 schema 的 `minLength: 20`），另外两条是**验错了对象**：

- 「参数篡改」那条，脚本里存在两份 `_env` 装配，其中 `check_action_runtime` 手拼的那份
  **没装业务规则**。于是 `NumericCapRule` 一次都没跑过，红得莫名其妙——不是规则写错，
  是被验的 runtime 里根本没有规则。修法不是改断言，是只留一份装配、差异用参数表达。
- 「mock 成功」那条的 `replayed=False`，是因为夹具还没录。这类"红"看起来像噪声，
  很容易被一句"重跑一下"糊过去。

这两条合起来是一句话：**"验证通过了"有两种，一种是功能对，一种是被验对象选错了。
后者更危险，因为它的绿看起来完全正常。**

同一类病这周还犯了三次，症状各异：

| 症状 | 本质 |
|---|---|
| `VerifyOutcome.ok` 把 `skipped` 算失败 | 每次搜索都变 `unverified`——信号处处出现等于没有信号 |
| 死循环预警放在幂等检查之后 | 「连发 3 次」永远先被拦成 `duplicate`，预警代码一次都没执行 |
| 沙箱相对路径按进程 cwd 解析 | 本地跑得通，容器里写到根目录 |

第三个尤其典型：它不会报错，只会在换环境的那天突然生效。

### 五个判断

1. **业务上限不能写进 schema 的 `maximum`**：schema 是给模型看的契约（它会照着改），
   业务上限是我们的规矩。混在一起，改上限就得动 schema——等于把说明书当授权系统。
2. **`tool_not_visible` 与 `permission_denied` 必须分开**：前者是安全事件（模型知道了
   它不该知道的东西），后者是运营事件（权限配置要调）。混了，安全信号会被当配置问题处理。
3. **可见面的主体维度只用粗粒度权限**：把 `read`/`net` 也纳进去，匿名用户就看不到任何
   检索工具——而检索恰恰是匿名该能做的。收敛靠阶段裁剪，危险能力才靠主体。
4. **幂等键带 `intent_id` 而不是 `ToolCall.id`**：后者每次重试都会变，用它做键等于
   "重试就是新动作"，幂等在最需要它的场景里失效。
5. **死循环只报警不阻断**：一次误报被升级成任务失败，比多花一点 token 贵得多。
   终止策略属于 W9 多 Agent 的职责。

### P3 决策：不解锁

按 ROADMAP §7（余量 ≥15h 才解锁一个深入档）：W8 预算 12–15h，实际用满并溢出
（15 个模块 + 200 条用例 + 产品层 4 端点 + 4 份文档 + 收尾修 3 处自检失败），
余量是负的。且 `skill-spec.md` §5 还挂着 3 个已知缺口待收。

A2A 的默认档（读规范 + 一次跨进程 handoff demo）留到 W9——那才是它的自然挂载点，
这周硬塞进来既挤掉 W8 的边界工作，又因为没有多 Agent 场景而退化成纯读书。

### 遗留

- **工具未进 `/v1/draft` 主链路**：Gate 8 #11 只能判部分通过。交付的是"能力 + 约束"，
  端到端价值未证明。这是 W9 的第一件事。
- 真 API 一次没跑过（只有桩上游）：`HttpSearchUpstream` 写好了但没配凭证，
  "MCP 接得上真 server"缺一次实测。
- `tool_error_rate` 口径只统计参数解析失败，不含执行结果——因为执行层还没接。
- 幂等缓存是进程内的（24h TTL），多副本会重复执行。
- `sandbox=container` 未实现，只有本地三道白名单；契约已立，容器化留 W11。
- `contains` 会被否定句误命中（"本文没有结论"能通过"必须包含结论"），
  已写进 `skill-spec.md` §5.1 并留了一条显式用例——不粉饰，也不用假的精确去修它。

---

## 2026-W50（12-07 ~ 12-11）· W7：上下文工程与记忆

> 实际落盘 11-10 实跑。"喂给模型什么"这件事第一次变成**可预算、可裁剪、
> 可解释**的流水线；记忆作为可开关的能力接进 `/v1/answer`，并跑了 off/governed
> 两档对照。实验暴露的两个测量问题（摘要成本、思考 token）比代码本身更值钱。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `context/{schema,budget,builder}.py` + Debug Report | 预算变成数据结构的属性：五槽上限 + 六类排除原因都能报出唯一原因 |
| D2 | `memory/working.py` | 滚动摘要按阈值触发（不是每轮）、tool result 截断留可重取句柄 |
| D3 | `memory/{long_term,extract}.py` | 四阶段闭环 + 写入五闸（污染/置信度/幂等/顶替/原子提交） |
| D4 | `context/{degradation,drift}.py` + 边界文档 | U 形排布升级到整条消息序列；Drift 三指标纯规则、命中只重注入 |
| D5 | `context_cost_ab.py` + `demo_long_session.py` + 五个文档 | 两档实验跑通；20 轮演示脚本零密钥可跑 |
| 接线 | `answer.py`（记忆槽位 + session_id）/ `routes_memory.py` / `runtime.py` | 记忆失败只降级、不 500；`memory_keys` 让"有没有真进 prompt"可核对 |

### 指标变化

- 离线用例：`715 → 794`（新增 84，含 5 条真库用例默认跳过），ruff 全绿，mypy 74 文件 0 问题
- 评测（W50 离线档）：54 条 50 passed / 0 failed / 4 skipped，judge 均分 `1.922`（与 W49 持平）
  回归门禁通过（success_rate / latency_p95 两项可比）
- 真库记忆（pgvector）：写入-召回往返、幂等行数不变、冲突顶替保留旧行、污染候选不入库、
  `forget` 连向量一起删 —— 5/5 通过
- 两档实验（详见 `docs/w7/context-cost-ab.md`）：`MEMORY=off` 3/4 → `governed` 4/4 通过，
  跨会话"品牌调性"召回从"答不出"变成"专业克制。"

### 本周最重要的一件事：优化不记账，报表就会推荐它

实验一第一版跑完，"压缩"看起来是**免费**的：`summary` arm 的 token 明显更低、成本更低。
但摘要是一次真实的 LLM 调用，而 `CompressResult.input_tokens` 从来没有被填过——
字段存在、恒为 0、没人发现。

补上 `SummarizerUsage`（摘要器与工作记忆共用同一个累计器）之后，结论反过来了：
20 轮这个长度上，压缩**还没回本**（省下的输入 token 抵不过 3 次摘要调用）。
真正的收益不是"更便宜"，而是"增长有上限"——naive 的曲线一路线性上升，summary 的曲线
在每个压缩点回落，稳定在一个平台。

第二个测量问题是 Qwen 默认开思考：同一条 26 字答案，`output_tokens` 实测 876（默认）
vs 14（`thinking=False`）。不钉住这个开关，输出侧噪声会把输入侧的差异完全淹没，
实验会得出"压缩没用"——而真相是"测量被另一种开销盖住了"。

### 四个判断

1. **六个排除原因必须由一个闸门记录**：两个原因并存就是两份互相矛盾的 Debug Report，
   而排障的人不知道该信哪份。
2. **槽位间不挪用预算**：借预算会让"第 6 条为什么没进来"依赖当轮别的槽位的占用，
   报告从"配置问题"退化成"运行时问题"，不可复算。
3. **记忆不进 system 消息**：记忆是会话沉淀的用户偏好，混进 system 会被当指令——
   这既是语义问题，也是 W10 注入防护的同一条原则提前落地。
4. **记忆的判据是"关掉它"**：`off` 档答不出品牌调性，`governed` 答得出；同时成本上涨，
   两件事必须一起说，否则"加了记忆效果变好"是一句无法被否决的话。

### 遗留

- Gate 7 #2 的"无 400"证据是预算口径判定 + 演示脚本，不是真的撞过厂商 400（20 轮撞不穿）
- 记忆两档实验样本量小（4 个任务、单次跑）：结论是方向，不是幅度
- live judge 未跑（与 W6 同一取舍）；`/v1/draft` 仍未接预算与记忆
- 工作记忆压缩状态在进程内，多副本不同步（W11）
- W6 `assemble` 的裁剪原因未并入 Debug Report，查"第 6 块为什么没进"要翻两块报告

---

## 2026-W49（11-30 ~ 12-04）· W6：生产 RAG II（溯源 / 增量 / ACL / 上下文 / 治理）

> 实际落盘 11-07 实跑。检索链路这次**真的接进服务了**（`/v1/answer`），
> 所以引用抽样是"答案里的 `[n]` 指哪一段"的实测，不是脚本自测。
> `/v1/draft` 仍未接检索。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `knowledge/citation.py` + `evals/scripts/citation_accuracy.py` + 30 条抽样集 | 溯源贯通；**但这周最值钱的产出是判据返工，不是代码** |
| D2 | `knowledge/incremental.py` + `PgVectorStore.replace_document` + 文档注册表 | 真库验过：新增不动旧文档、内容未变零 embedding、更新同事务无残留 |
| D3 | `retrieval/acl.py` + pipeline 接入 | 默认拒绝未标注；过滤挪到精排之前（侧信道） |
| D4 | `retrieval/context_assemble.py` | 去冗余 → 预算裁剪 → 首尾放置 → 才编号，顺序固定 |
| D5 | `knowledge/governance.py` + 调优清单 + ADR + 验收 | 降权不丢弃、标注不裁决；清单 11 节可照着排查 |
| 接线 | `answer.py` / `runtime.py` / `routes_answer.py` / `app.py` | 惰性装配（首请求 18–20s），`/health` 报 `cold/ready` |

### 指标变化

- 离线用例：`602 → 715`（新增 113；live 25 跳过），ruff 全绿，mypy 61 文件 0 问题
- 引用抽样（28 可答 / 2 应拒答，chunk 级 gold）：
  - 准确率 `1.0000`（28/28，门槛 0.90）；应拒答 `2/2` 未编引用；全量引用 `0` 条
  - **精确率 micro `0.4286`（36/84）/ macro `0.4855`** —— 未命中 48 条中 11 条同页、37 条异页
  - 延迟 p50 `14.7s` / p95 `22.2s`（含 LLM；检索段 p95 仍约 160ms）
  - `--replay` 逐位复现（准确率、精确率、tokens 全同）
- 增量索引（真 pgvector）：新增 A 不动 B；SKIPPED 时 embedder 零调用；更新写 4 删 3 无残留；删除幂等
- 端到端：54 条 50 passed / 0 failed / 4 skipped，success_rate `100%`（离线夹具档）
  → 回归门禁通过；`latency_p95_ms` `2.48ms`（本期无 socket broker 污染，可作基线）
- RAGAS faithfulness `0.0` 与 W48 一致 —— 夹具属性，不是本周劣化

### 本周最重要的一件事：判据错了，指标就成了假的

引用抽样第一版跑出 **0/2**。两条答案其实都引到了标注段（`golden-001` 引 5 条，
2 条命中，另 3 条是同页相邻块），是判据把"混入非标注段"判成了失败。

**根因是标注只覆盖 69/425 个块**：按"条条命中"判，等于用"标注是否穷尽"衡量
"引用是否正确"。模型引对了没标的地方反而扣分，而这类失败会被归因到生成层——
于是人去改 prompt，改完指标不动，再改一轮。

改法是把一件事拆成两个数：`verdict` 只回答"有没有引对"（至少一条命中），
`precision` 回答"引的东西有多干净"。再补一条反制：context ≥4 块却全引一遍，
判 `blanket_citation`，不算通过——放宽口径必须配反制，否则"每段挂个编号"能稳过。

### 三个判断

1. **"精确率不设门槛"是有意的**：它的上限由标注工作量决定（69/425），
   拿它当门禁等于让"标了多少"决定"验收过不过"。但它必须被看见。
2. **ACL 过滤必须在精排之前**：功能上放后面也"对"，但 4.7s 精排会花在调用方
   看不到的块上，且 p95 随"库里无权限文档数量"变化——那是可观测的侧信道。
3. **过期降权、冲突不裁决**：丢弃过期会让"最新政策是什么"直接无答案，
   比给一条旧政策更糟；裁决需要业务权威，模型猜错比暴露冲突贵。

### 遗留

- Gate 6 #8 治理三条只有单测：语料里没有过期/矛盾/低质量文档，端到端未跑真样本
- `/v1/draft` 未接检索；`normalize_query` 未接入；ACL 无真鉴权（W10）
- 准确率在本语料已饱和（28/28 全过）：下一期提高区分度要靠"相关段"标注或逐句 judge
- 镜像缺 `--extra pgvector`，构建仍卡 docker.io 502

---

## 2026-W48（11-23 ~ 11-27）· W5：生产 RAG I（pgvector / 混合检索 / 精排 / 改写）

> 实际落盘 11-06 预跑；六档检索已量化，但**检索还没接进 `/v1/draft`**，
> 所以本期没有任何一次「线上请求含检索」的延迟读数。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `knowledge/base.py` + `PgVectorStore` + `ops/pgvector/init.sql` + 十项决策 ADR | 协议先行落地：pgvector 只实现协议，业务层 grep `psycopg\|pgvector` = **0 处** |
| D2 | `retrieval/fusion.py`(RRF) + `pipeline.py` + `evals/scripts/retrieval_metrics.py` | **实测混合不如单路稠密**，默认档从 `hybrid` 改 `dense` |
| D3 | 精排装配 + 三条降级路径 + `rerank_preflight()` | 精排是负收益，但先查清了一个更糟的事实：默认超时让它在 CPU 上**永远不生效** |
| D4 | `query_rewrite.py`（multi-query / HyDE）+ 接进装配 | 改写也是负收益；顺手修掉一个死开关 |
| D5 | `rag-ab-report.md` + `case-1` + ADR 十项决策 | 「全负」不是本周失败，是本周最硬的结论 |

### 指标变化

- 离线用例：`496 → 602`（框架侧 +106），rag_mvp 侧另 +34；live 22 跳过
- 检索（28 正例 / 425 chunks / bge-m3 1024 维，`retrieval_metrics.py` 生产口径）：
  - 纯稠密：R@1 `.7857` / R@10 `1.0000` / MRR `.8628` / nDCG `.7416`，p95 `175ms`
  - 纯 BM25：R@1 `.2500` / R@10 `.7143` / MRR `.4122`，p95 `0.9ms`
  - 混合 RRF：R@1 `.5714` / R@10 `.9286` / MRR `.6921` —— **三项全低于纯稠密**
  - +精排（20 候选）：R@1 `.7143` / MRR `.8146`，p95 `5381ms`
  - +multi-query：R@1 `.7143~.7857`（两轮不同）/ MRR `.8089~.8482`，p95 `15.2s`，¥0.0888 / 25 次
- 后端等价：local 与 pgvector `5/5` 同 chunk、分数小数点后 6 位一致
- 静态检查：`ruff` 全绿、`mypy` 53 文件无问题
- 全量测试：`602 passed / 22 skipped`（59s）

> **门禁这次是红的，但红得不是地方。** `latency_p95_ms` 2.29ms → 30.81ms（+1244%）。
> 判为环境伪影，不是代码劣化，三条判据：①30ms 对每条 case 恒定、方差 0.6ms，
> 连 error case 也 ≥30ms，不符合真实链路抖动；②检索未接入 `/v1/draft`，
> 本期跑的路径与 W47 逐字节一致；③cProfile 把 3.44s 定位到
> `_socket.socket.recv` + `sitecustomize.py:361(_broker_readline)` 109 次——
> 记录这个数字的 python 进程被 CLI shim 注入了 socket broker。
> 处理：本期延迟**不进基线**，裸终端重跑后替换。同类现象 W4 已记过一次（`tmp_path` 无写权限）。

### 本周暴露的五个真问题

1. **「混合检索更好」是教科书假设，不是本地事实**：sparse 路 R@1 只有 `.25`，
   而 RRF 的 `1/(k+rank)` 把「dense 排 1、sparse 排 30」的正解算成 `0.0275`，
   「两路都排 5」的邻居页得 `0.0308`。**等权融合在惩罚单路强信号**，这是算法性质。
2. **分词器把纯标点当 token**：`tokenize("如何配置 MCP 服务器？")` 产出 `'？'`。
   golden-018 的 sparse top1 是一个含 12 个「？」的列表页。去纯标点后 sparse R@1
   `.2500 → .5357`（+114%）——**修好了分词，混合仍然不如 dense**，两件事要分开说。
3. **默认超时值可以让组件「永远不生效」**：`rerank_timeout_ms=3000`，本机 CPU 20 条要
   4.7s，于是 28/28 全降级、指标等于没开。这类默认值比「功能缺失」更危险，
   它让人以为「我试过了，没用」。
4. **死开关比没实现更危险**：`retrieval_rewrite` 字段存在、pipeline 支持 `rewriter=` 注入，
   但 `rag.py` 从没读过它。设了没有任何效果，也不报错。
5. **重排是在删除信息，不是重排**：`hybrid+rerank` 与 `dense+rerank` 五个指标**完全相同**；
   golden-022 的 gold chunk 从第 8 位掉出 top10，而同一页的邻居被顶上来。
   精排偏好「主题邻近段落」，题目要的是「恰好回答的那一段」。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 检索未接 `/v1/draft` | 唯一功能缺口；接上之前 Gate 5 #5 只能是 partial，p95 门禁也覆盖不到检索 |
| 2 | 28 条正例不足以支撑细粒度结论 | R@1 差 6 条、MRR 差 15.7pp 算系统性；R@10 差 1 条不算 |
| 3 | 桥接层不转发 `filters` | W6 的 ACL 若直接依赖会静默越权，先修桥接 |
| 4 | 镜像缺 psycopg | Dockerfile 补 `--extra pgvector`；完整构建仍卡 docker.io 502 |
| 5 | 改写两轮结果不一致 | `temperature=0.3` 有随机性；任何改写结论需三次取中位数 |

### 验收口径的修订（需确认）

Gate 5 第 3 条原写「Recall@K 相对 baseline 提升 ≥10%」。**这条无法验收**——
纯稠密 R@10 已经是 `1.0000`，没有 10% 的空间。

改为：**在 R@10 触顶的前提下以 R@1 / MRR 为主指标；混合与精排为负收益，如实记录并默认关闭。**
这不是降标准，是把「提升」从「多叠加组件」换成「选对单路 + 证明其余无效」——
后者更难，因为它要求每个被关掉的组件都有反向证据。

### 下周（W6）只做三件事

1. `/v1/draft` 接检索，把 `context_precision` 从「引用质量代理」换成真实 chunk 前 k 条；
2. ACL 走 `filters`（桥接层当前不转发，等于越权），同时做 `delete_document` 的增量更新；
3. 裸终端重跑一次 `run_eval` 换掉本期被污染的延迟块，恢复门禁可比性。

---

## 2026-W47（11-16 ~ 11-20）· W4：评测体系 II（Judge / RAGAS / 回归门禁 / 四层归因）

> 实际落盘 11-03 预跑；judge 一致性来自**规则基线**（无 key），LLM judge 与在线档待凭证。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `evals/judge/{rubric,llm_judge,stats}.py` + `judge_agreement.py` + `adr-w4-judge-design.md` | **一致性系数才是交付物**：没有 kappa 的 Judge 等于随机数，"分数涨了"就无法被追问 |
| D2 | `evals/judge/{ragas_metrics,text}.py` + `metric-spec-ragas.md` + `metrics.RagasScores` | `None ≠ 0`：无 context 返回 None，否则 faithfulness 随数据集构成漂移 |
| D3 | `regression_gate.py` + `baseline.json` + CI 两步 + `adr-w4-gate-threshold.md` | 相对劣化 + 延迟绝对下限；**基线缺失退出 2，不静默放行** |
| D4 | `root-cause-playbook.md` + `attribute_failure.py` | 证据不足判 `unknown`：把"埋点缺失"判成"某层坏了"，团队会去改一个没坏的模块 |
| D5 | `docs/w4/cases/case-{1,2,3}.md` + 3 处真实代码改动 | 每篇都有前后数字，且都不是"全绿"——真实性来自取舍被写出来 |

### 指标变化

- 离线用例：`416 → 496`（judge 24 / ragas 20 / gate 17 / 归因 15 / 分类不变量 4），live 仍 16 跳过
- **`make run-eval-judge`：50 passed / 0 failed / 4 skipped，2.12s**
- Judge：`judge_score 1.92`（50 条，alias `heuristic-keyword`）、rubric `answer-quality@w4-v1`
- 一致性（24 条人工标注）：加权 kappa `0.343`、名义 kappa `0.200`、spearman `0.437`、MAE `1.167`
  → **阈值 0.40 未通过**（结论见下）
- RAGAS（9 条带引用）：`faithfulness 0.0`、`answer_relevancy 0.054`、`context_precision 1.00`
- 延迟：混合 p95 `2.45ms`，但 chat p95 `0.47ms`、tool_call `0.74ms`、error p50 `3.95ms`
- 回归门禁：对比 W46 基线，`2 项可比 / 4 项跳过 / 0 违规`，通过
- 静态检查：`ruff` 全绿、`mypy` 45 文件无问题
- 全链路 `make eval WEEK=2026-W47` 一次通过：`check → golden → run-eval-judge → gate`
  （`judge_score` 判跳过是因为基线是 W46、那时还没有这个指标，不是本期缺数据）

> 排障记录：若在**只读沙箱**里跑 `make test`，会看到 20+ 个
> `ERROR ...::test_x - assert not self._finalizers`，散布在 `test_config` / `test_regression_gate`
> 等互不相关的模块。这不是代码问题：`tmp_path` 无写权限建不出来，teardown 被打断后
> finalizer 留在 fixturedef 上，之后每个用 `tmp_path` 的用例都在同一行炸。判据是
> "换到可写环境立刻全绿"——别去改 conftest。

### 本周暴露的五个真问题

1. **references 只有路径时，RAGAS 结构性为 0**：context 是 `docs/architecture.md` 这样的字符串，
   实词是 `docs/architecture/md`，任何答案都命中不了。指标恒 0 不代表模型差，代表我们把路径当成了正文。
2. **单字口径会把 faithfulness 顶到 1.0**：context 换成整篇文档后，任何中文句子都能凑够单字重合。
   换成二元组后回落——**同一份数据、两种分词，结论相反**，指标口径必须写进文档。
3. **一个 p95 混四种调用形态**：混合 2.45ms 是 chat p95 的 5.2 倍，误差量级 420%，
   抬它起来的是 4 条故意重试的 error case。读者会去查一个没变慢的链路。
4. **错误码表写两遍必然漂移**：10 个已知码里 6 个判定相反，`SERVER_BUSY`（自家限流器）在
   `run_eval` 里是"质量失败"、在分类器里是"上游问题"。漂移不报错，只让两个数字互相矛盾。
5. **规则基线判不了事实**：它从不给 1–2 分，因为"事实错误"在词面上往往比正确答案更像个答案。
   kappa 0.20 就是这件事的数字，也是 LLM judge 必须存在的理由。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | LLM judge 一致性未测（当前 0.20 是规则基线） | 填 key → `judge_agreement.py --live`；低于 0.40 先改锚，不改温度 |
| 2 | 24 条标注 `labeler=impl-draft` | 负责人逐条复核改成 `human`；自评自标只能证明流程通 |
| 3 | 门禁拦不住"答得差" | 离线 success_rate 恒 100%；建 `baseline-online.json` 后才真正有质量门禁 |
| 4 | `faithfulness` 离线恒 0 | 夹具答案是固定文本，与引用无关；W5 接检索后才有读数 |
| 5 | 字面判定认不出同义改写 | 需要 `LLMSupport`（走 strong 模型）；成本约 0.4 美分/条 |

### 收尾时补的一个坑

`run-eval` 与 `run-eval-judge` 写的是**同一个** `evals/history/<WEEK>-eval.json`，谁后跑谁说了算。
`make eval` 原来走 `run-eval`，于是本地一跑就把刚产出的 `judge`/`ragas` 块冲成 `null`；
下一期对比时质量指标会"本期缺失 → skipped"，趋势图悄悄断掉，而且**不会报错**。
已把 `eval` 改成依赖 `run-eval-judge`（离线档是规则基线、零成本）。同类隐患记一条原则：
同一份产物有两个写入者时，默认路径必须选"信息更全"的那个，否则隐性丢数据。

### P3 决策（W4 末 = ROADEMAP §7 第一个决策点）

**不解锁深入档。** 本周预算 12–15h 全部用于默认档，进度余量 < 15h；
按 ROI 顺序（A2A 轻量 > 多语言 > 前端极简 > 高并发 > 向量库/微调 > Computer Use），
解锁任何一个都会挤压 W5 的 RAG（P0 主线）。余量补上后再判。

### 下周（W5）只做三件事

1. 接检索（chunk + 混合召回），把 `context_precision` 从"引用质量代理"换成真实检索前 k 条；
2. 用 D3 的门禁 + D4 的归因跑第一轮真实调优，产出第 4 篇案例；
3. 首次在线档，回填 judge kappa 与 faithfulness 的真实读数。

## 2026-W46（11-09 ~ 11-13）· W3：评测体系 I（golden set / 分层指标 / trace 落地）

> 实际落盘 11-02 预跑；数字全部来自离线档，在线档与 Langfuse 真机验证待凭证。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `docs/w3/metric-spec.md` + `evals/metrics.py` + `docs/w3/adr-w3-eval-scope.md` | **口径必须先于 runner**：四态机（passed/failed/interrupted/skipped）定完，`success_rate` 的分母才有意义 |
| D2 | `evals/golden/{golden_v1.jsonl,schema.md}` + `validate_golden.py` | 54 条、难度 12/26/16、4 条对抗；校验器对待"重复输入/悬空引用/拼错判据键"零容忍 |
| D3 | `evals/scripts/{run_eval,offline_fixtures}.py` + `Makefile` 的 `run-eval` | 一条命令出分层指标 + md + json；`--break-case` 让"单条失败不中断"可被反向证明 |
| D4 | `exporters/langfuse.py` + `server/app.py` 注入点 + `adr-w3-trace-backend.md` | **缺 key 返回 Noop 而不是抛错**：否则没有账号的机器连测试都跑不了 |
| D5 | `classify_failures.py` + `report_template.md` + 报告自动渲染 | 规则先行：`UPSTREAM` 与 `HALLUCINATION` 混淆是最贵的归因错误 |

### 指标变化

- 离线用例：`337 → 416`（新增指标层 24 / golden 校验 13 / runner 11 / 分类器 13 / exporter 18），live 仍 16 跳过
- golden：`10 → 54` 条，难度 `12 / 26 / 16`，对抗样例 4 条，引用 20 处全部可达（anchor 逐条命中）
- **`make run-eval`：50 passed / 0 failed / 0 interrupted / 4 skipped，2.17s，零网络**
- 分位：`latency p50 0.17ms / p95 1.80ms`（50 样本）；首 token 仅 4 个流式样本，**p95 按规则返回 null**
- 成本：`0.0 CNY`（fixture 不计价）；`usage_estimated_ratio 0`
- **质量：`quality` 块恒 `null`** —— 离线跑通的 100% 不是质量结论
- 静态检查：`ruff` 全绿、`mypy` 45 文件无问题

### 本周暴露的六个真问题

1. **口径先于代码**：先写 runner 后补定义，指标会变成"代码碰巧算出来的东西"。
   `success_rate` 的分母一旦把上游中断算进去，一次厂商发版就能把质量分打掉 20 个点，
   然后团队去调 prompt——查三天发现是对方 503。
2. **`usage_estimated` 混进均值 = 无意义的数**：既不是实测也不是上界。宁可 `cost_per_task=null`
   并让 `usage_estimated_ratio` 说明原因，也不给一个能画在图上但解释不了的数。
3. **"多币种"报警其实是错误路径的默认值**：错误日志没有价目，`currency` 落的是 dataclass 默认的 USD。
   把带 `error_code` 的样本排除出币种判定即可，不必回头改 W1 的 emit。
4. **对抗样例不能有离线判据**：补一个 `offline.expect` 让 CI 变绿，等于"用夹具回放证明模型会拒绝"，
   比没有测试更糟。宁可让它长期 skipped，并在报告里显式展示 7.4% 的覆盖率缺口。
5. **离线判据抄在线判据必然二选一**：恰好成立（假绿）或永远不成立（红一片后被人放宽）。
   离线只断言 `require_log` + `finish_reason`，在线才断言语义。
6. **`ruff` 的 line-length 按显示宽度算**：85 个字符的中文注释被判成 105。按 `len()` 折行写了等于没写。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 在线档质量分仍为 null | 填 `.env.local` 后 `make run-eval WEEK=2026-W47 --live`，回填 quality 与模型快照日期 |
| 2 | Langfuse 只做过单测与降级验证 | 注册账号 → 配 `LANGFUSE_*` → 跑一次 `/v1/chat` → 确认 `llm.call` span 与 trace 树 |
| 3 | 4 条对抗样例长期 skipped | W4 接 Judge 后改为在线档评估；每期报告保留 skip_rate |
| 4 | `judge_score` 字段恒 null | W4 接 LLM-as-judge，在此之前"质量趋势"只有成功率一条线 |

### 下周（W4）只做三件事

1. 接 LLM-as-judge，把 `judge_score` 从 null 变成数字（含 judge 与被评模型不同的证据）；
2. 把 `make eval` 升级成"指标不得劣化"的对比门禁（对上一期 history 做差）；
3. 首次在线档跑通，产出第一批真实延迟/成本/质量数据点。

---

## 2026-W45（11-02 ~ 11-06）· W2：运行底座 + 可观测基线 + 评测冒烟集

> 实际落盘时间 11-01（周日）预跑；数字全部来自离线档，在线档待 key。

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | `session/{base,memory,redis,factory}.py` + `workflow/research_workflow.py` + `docs/w2/adr-w2-session-boundary.md` | 幂等键必须三档解析（显式 → contextvar → uuid）；少第三档，脚本与队列消费者一调就误判重复 |
| D2 | `resilience/{concurrency,errors}.py` + `queue/{base,inmemory}.py` + `server/routes_task.py` | 入口过载 `retryable=False`：让 gateway 再重试一次，100 并发立刻变 300（压测直接崩） |
| D3 | `observability.py`（+`trace_id`/`span_id`）+ `tracing.py`（Span/span()/Exporter） | `trace_id` 必须在中间件生成，放进 `span()` 会让同一请求的多个 span 各拿一个新 trace |
| D4 | `ops/{Dockerfile,docker-compose.yml,.env.example}` + `evals/scripts/locustfile.py` + 压测报告 | 客户端口径与服务端口径差一个数量级，延迟结论只能取 `RequestLog` |
| D5 | `evals/golden/smoke_v1.jsonl`（10 条）+ `run_smoke.py` + CI + `Makefile` | 门禁必须先证明"能红"：CI 里改坏一条 expect，退出码非 0 才算门禁成立 |

### 指标变化

- 离线用例：`278 → 337`（新增会话/限流/队列/trace/任务端点/smoke runner 六组），live 仍 16 个跳过
- **100 并发离线压测：1224 请求 / 0 失败 / `retry_count>0` 为 0 / 994 条日志 994 个 trace_id**
- **端到端自检（`make e2e`）**：限额 1 + 20 并发（每请求 200ms）→ `200×1 / 503×19`、`Retry-After: 1`、`code=SERVER_BUSY`；`/v1/tasks` 走到 `succeeded`；响应头 `X-Trace-ID` 与日志 `trace_id` 一致
- 首 token（port 口径）：p50 `20.9 ms`、p95 `32.5 ms` —— **仅证明埋点链路正确**，fixture 回放不代表真实模型
- 成本：仍为 0（fixture 不计价）；质量：smoke `10/10`，只有"没坏"这一档
- 静态检查：`ruff` 全绿、`mypy` 43 文件无问题

### 本周暴露的五个真问题（都不是写代码时能想到的）

1. **"实现慢"可能是机器慢**：全量串跑 15 分钟、单文件跑 6 秒，同一份代码。宿主调度一停顿，
   客户端 p95 就从 35 ms 涨到 1200 ms。**结论**：延迟只认服务端 `RequestLog`，客户端分位数只能横向比。
2. **幂等的兜底键不是可选项**：没有"显式 → contextvar → 本地 uuid"第三档，`request_id=""` 会让
   第二次 `append` 被判成重复，表现为"多轮对话第二轮丢消息"——只在真实多轮里复现。
3. **入队前必须 `precheck`**：alias 打错这类注定失败的请求，占了队列槽位排 30 秒再 failed，
   等于把调用方的输入错误升级成我们的容量问题。
4. **compose 的 healthcheck 不能指向未就绪端口**：`depends_on` 只保证"容器起了"，不保证"能连"，
   `agent-server` 起得比 redis 慢时，首批请求全是 `Connection refused`。
5. **`mock` 压不出并发**：回放是同步读文件、全程不 await，事件循环里天然一次只有一个在飞。
   用它验证"限额 1 会不会 503"会得到 30 并发全 200 的假绿——压测对象先要能并发，结论才有意义。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 在线档延迟/成本未测 | 有 key 时 `MODEL=qwen3.7-plus make load`，报告记时段（峰谷差 2 倍） |
| 2 | `docker build` 未在本机执行（无 docker） | compose 仅静态检查；首次 `make up` 时补 `docker compose ps` 无 restart 的记录 |
| 3 | 队列进程内实现 | 多副本会丢任务；换 Redis Stream 时 `TaskQueue` 签名不变 |
| 4 | 满载 503 已由 `make e2e` 覆盖，但只在进程内验证 | 真机/容器内复跑一次 `make e2e`，确认反向代理层没有改写 `Retry-After` |
| 5 | smoke 只判"全过" | W4 升级为"指标不得劣化"，需要先有 3 周基线 |

### 下周（W3）只做三件事

1. golden set 扩到 50+（每类覆盖从 2–3 条提到 8–10 条），判据从"文本包含"升级到结构化断言；
2. 接 Langfuse：只实现 `LangfuseExporter` 替换 `NoopExporter`，`RequestLog` 字段一个不改；
3. 记忆层（W7 前置）只定接口，不实现——先把"哪些事实值得记"的判据写进文档。

---

## 2026-W44（10-26 ~ 10-30）· W1：LLM 适配层 + 首个纵向切片

### 本周做了什么

| Day | 产出 | 结论 |
|---|---|---|
| D1 | 四家 API 差异矩阵（`docs/w1/*.md`）+ uv workspace 重构 | "system 顶层 / tool_result 成组 / arguments 是字符串 / usage 挂在哪"这四件事决定归一化粒度 |
| D2 | 契约层 `types / errors / base` | 唯一一张 finish_reason 映射表；`provider_meta` 承载"框架不解释但必须回传"的字段 |
| D3 | `openai_compat` / `anthropic` / `mock` + conformance A1–A8 | 差异走 flags 表，不写 `if "deepseek" in base_url`；mock 只回放 wire，不自己造 `LLMResponse` |
| D4 | `config / observability / registry / router / gateway / server` + A9–A12 | 重试与 committed 状态机是客户端层唯一职责；`precheck` 必须在 `StreamingResponse` 之前 |
| D5 | 录制器 / 基线表 / live 用例 / A13–A14 / 产品端 `/v1/draft` | 框架"能被消费"这件事必须由产品端真跑一遍来证明，自测不算 |

### 指标变化

- 离线用例：`136 → 222 → 278`（Day3 → Day4 → Day5），live 用例 16 个默认跳过
- conformance 断言：`A1–A12 → A1–A14`（新增"录制 ⊇ 手写基线"与"工具调用落在声明 schema 内"）
- 首 token / 成本 / 质量：**仍为 null**，等真机录制与 W2 golden set
- 静态检查：`ruff` 全绿、`mypy` 26 文件无问题

### Day5 过程中暴露的四个真问题（都不是写代码时能想到的）

1. **日志静默丢弃**：`RequestLog` 打 `INFO`，而 root logger 是 `WARNING`。代码里 `emit()` 明明调了，
   `uvicorn` 起来后日志里一个字都没有——Gate 验收项「`first_token_ms` 入日志」会假通过。
   修法：`observability.configure_logging()`，在应用边界（`create_app`）调用，库本身不改宿主配置。
2. **DeepSeek 是峰谷分时计价**（工作日 09–12 / 14–18 高峰翻倍），Qwen 是**按上下文长度阶梯**
   （≤256K 为 2/8，>256K 跳到 6/24）。单一 `pricing.input/output` 表达不了，W1 取常规档并在
   `models.yaml` 里注明；基线表必须记时间戳，否则"成本降了 40%"可能只是换到了空闲时段。
3. **产品层拿不到中间件**：`RequestIDMiddleware` 原本长在 `server/app.py` 里，产品要复用它就得
   执行该模块顶层的 `app = create_app()`——为了拿一个中间件而启动一个框架应用，是边界画错的信号。
   抽到 `server/middleware.py` + `server/errors.py` 后，产品端零 `try/except` 就复用了 6 类错误码。
4. **录制器不能覆盖 `request.json`**：它是 A1 的断言锚（`build_request` 必须精确产出它）。
   覆盖了 A1 就变成"自己跟自己比对"。改为：录到不一致就硬失败，让人先决定锚该不该更新。

### 遗留问题（带责任人与时限）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 真机录制未跑（手写夹具可能偏离真实协议） | 有 key 时执行 `README` 的 Day5 三连；A13 会在录制后清算 |
| 2 | `first_token_timeout_s=10s` 未经真机校准 | `uv run pytest -m live -k first_token` 取实测值后调整 |
| 3 | Anthropic 欠费仍靠文案 `credit balance` 识别 | 真机确认后收敛成固定 code 白名单 |
| 4 | 峰谷 / 阶梯计价未建模 | W11 做成本看板时改为按 (时段, 上下文长度) 取价 |
| 5 | 错误路径（`usage_missing` / `error_*`）无法安全录制 | 长期保留手写 + `error_auth` 真机探测这两条路 |

### 下周（W2）只做三件事

1. golden set 最小 10 条（含 3 条对抗样例），接进 `pytest` 做门禁；
2. 压测脚本：并发 20 × 60s，产出 p50/p95/错误率，回填 `evals/history`；
3. Temporal 长任务骨架（`legacy/temporal_demo` 的思路复用），先跑通"调研分 3 步可续跑"。
