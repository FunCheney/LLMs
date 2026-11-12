# W8 验收（Gate 8）

> 规则同 W7：**每一项都要有可执行的证据**，不接受「我看过代码，没问题」。
> 没做到的写「部分通过」并说明差哪一步——验收表不是给自己发奖。
>
> 实跑日期 2026-11-11；Python `.venv` 3.12；全部离线（零密钥、零联网）。

---

## 1. 逐项结论

| # | 验收项 | 通过标准 | 证据 | 结论 |
|---|---|---|---|---|
| 1 | MCP 边界 | 业务只认内部 `ToolSpec`，不 import MCP 库 | 自检 #1 `tools=['search__search_web']`；`test_mcp_boundary.py`（8 条，AST 扫全部 `src/` 断言无 `import mcp`） | ✅ |
| 2 | 动态发现 | server 加工具不改主流程即可调用 | 自检 #2 `['search__search_web'] → ['search__search_web','search__trend_chart']`；`test_visibility.py` 的注册表 reload 用例（新文件落盘 → `reload()` → 可 `get`，主流程代码未改） | ✅ |
| 3 | 录制回放 | 断网全绿；未命中明确报错不联网 | 自检 #3 两条：`replayed=True cost=0.0` / `ReplayMissError`；`test_tool_replay.py`（13 条）；自检脚本本身零联网（桩上游） | ✅ |
| 4 | 脱敏 | 录制响应里的 key 被打码 | 自检 #4 扫 `fixtures/tools/**/*.json` 无 `sk-live` / `supersecret`；`test_tool_replay.py` 打码用例 | ✅ |
| 5 | 执行闭环 | 校验→可见面→权限→沙箱→执行→回注 顺序固定 | 自检 #5 四条：`tool_not_visible` / `invalid_arguments`（先于权限）/ `permission_denied` / `ok`+`role=tool`；`test_skill_executor.py`（14 条） | ✅ |
| 6 | 全审计 | 每次调用可查主体/工具/风险级/结果/replay | 自检 #6 两条：Gate 要求字段齐全（缺失 `无`）+ 参数值不落盘（只留 `args_digest`）；`test_action_runtime.py` | ✅ |
| 7 | Skill 可验证 | `done_criteria` 机器可判，无证据的「已完成」被拦 | 自检 #7：`未满足 2 条：['结果里必须给出 post_id…', '必须明确写出发布状态…']`；`test_skill_validator.py`（12 条）；`test_skill_spec.py`（25 条） | ✅ |
| 8 | 可见面收敛 | 任一 `(principal, stage)` < 全集 60%；渐进式 < 20% | 自检 #8：矩阵最大 **50%**；`describe` 531 vs `instructions` 2778 = **19.1%**；`test_visibility.py`（40 条） | ✅ |
| 9 | 沙箱边界 | 路径/命令/网络三类越界各被拦，且在副作用之前 | 自检 #9 三条全 `sandbox_denied`，`escaped=False`；`test_sandbox.py`（39 条） | ✅ |
| 10 | Action Runtime | 重复/篡改/假成功 三个 badcase 全拦住 | 自检 #10 四条：`['verified','duplicate','duplicate']` 副作用 1 次 / `numeric_cap` 拒 / `unverified` + 0 posts / 无 HITL 则 `pending`；`test_action_runtime.py`（20 条） | ✅ |
| 11 | 质量守门 | 接真实工具后 `success_rate` 过门禁 | `run_eval --week 2026-W51`：54 条 50 passed / 0 failed / 4 skipped，`success_rate 100%`；`regression_gate` 门禁通过 | ⚠️ 部分通过 |
| 12 | P3 已判定 | 按余量明确解锁/不解锁，写进 log | 见 §4：**不解锁** | ✅ |

**11/12 全过，1 项部分通过。** #11 差哪一步写在 §3.1，不掩饰。

---

## 2. 本次实跑记录（可复现）

```bash
# 静态检查
.venv/bin/python -m ruff check .                       # All checks passed!
.venv/bin/python -m mypy packages/agent_framework/src apps/growth_agent/src
# Success: no issues found in 97 source files

# 全量用例
.venv/bin/python -m pytest -p no:warnings
# 994 passed, 30 skipped in 12.69s

# Gate 8 证据脚本（零密钥零联网，默认离线档）
.venv/bin/python evals/scripts/tool_boundary_check.py
# 23/23 通过

# 需要更新夹具时才加 --record（会用桩上游重录，仍不联网）
.venv/bin/python evals/scripts/tool_boundary_check.py --record

# 评测与门禁
.venv/bin/python evals/scripts/run_eval.py --week 2026-W51
# 54 条：passed 50 / failed 0 / skipped 4 → success_rate 100.0%（mode=offline）
.venv/bin/python evals/scripts/regression_gate.py --current evals/history/2026-W51-eval.json
# 门禁通过
```

测试增量：W7 末 794 → W8 末 **994**（+200）。

新增用例分布：

| 文件 | 条数 | 主题 |
|---|---|---|
| `test_sandbox.py` | 39 | 路径/命令/网络三类越界 + 超时 + 进程组 |
| `test_visibility.py` | 40 | 裁剪矩阵 + 注册表 + 渐进式加载 |
| `test_skill_spec.py` | 25 | 六层校验 + A14 |
| `test_action_runtime.py` | 20 | 三个 badcase + 幂等/并发/审批 |
| `test_skill_executor.py` | 14 | 五道门顺序 + 截断句柄 |
| `test_mcp_client.py` | 13 | 发现/映射/超时/错误规范化 |
| `test_tool_replay.py` | 13 | 命中/未命中/脱敏/成本 |
| `test_skill_validator.py` | 12 | 无证据的「已完成」+ 一次重试 |
| `test_mcp_boundary.py` | 8 | AST 扫 `import mcp` |
| `test_tools_routes.py` | 16 | HTTP 面（`/v1/tools` 四个端点） |

---

## 3. 关键证据细节

### 3.1 #11 为什么是「部分通过」

`success_rate 100%` 是**离线夹具回放**的结果，`run_eval` 自己就打了这句提醒：

> 离线档是夹具回放，success_rate 只说明**执行管道**通，不代表模型质量；quality 块保持 null。

两条硬事实：

1. **真 API 没接。** `servers/upstream.py` 的 `HttpSearchUpstream` / `HttpFetchUpstream`
   写好了（密钥进 header 不进 URL），但本周没配凭证跑过。桩上游是确定性的，
   所以「接真实工具后」这个前提**尚未成立**。
2. **工具没进主生成链路。** `/v1/tools` 是独立面（可见面 / 详情 / 调用 / 重载），
   `/v1/draft` 与 `/v1/chat` 仍未消费工具结果——执行循环按 `__init__.py` 的约定
   留给产品层或 W9，本周只把「能力」和「约束」做出来。

差的那一步是明确的：**把 `SkillExecutor` 挂进 `/v1/draft` 的多轮循环，
并接一个真 provider 复跑评测**。这属于 W9「多智能体 / 端到端」的范围，
不在 W8 的 5 天预算里。

顺带一个被门禁暴露的口径问题，值得写清楚（第一眼容易误读成"缺指标"）：

```
跳过（基线或当前期缺该指标）：cost_per_task, judge_score, faithfulness, tool_error_rate
```

`regression_gate` 的规则是「**基线值为 0 或无值 → 不判**」（`base == 0` 意味着当时没测到，
从 0 到有既不是劣化也不是改进）。所以这 4 个是被设计跳过的，不是 bug。但它们
当前**没有趋势保护**——要等下一期成为基线后才生效。

另一个更容易误读的点：评测报告里 `tool_calls: 14 / tool_errors: 0` 看着像「工具跑通了」。
实际口径是：

| 字段 | 真实含义 |
|---|---|
| `tool_calls` | 模型**提议**了几次工具调用（`response.tool_calls` 的长度） |
| `tool_errors` | 其中**参数 JSON 没解析开**的条数（`arguments_error`） |

它**不统计执行结果**——因为执行层还没接进主链路，这 14 次提议没有任何一次
到达 `SkillExecutor`。所以「工具执行失败率」这个 W8 之后最该盯的数，目前不存在。
接主链路时一起接上，而不是现在造一个恒为 0 的指标。

### 3.2 可见面矩阵（全集 6 个内置 Skill）

| stage | anonymous | viewer | editor | publisher | ops |
|---|---|---|---|---|---|
| research | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) |
| draft | 3/6 (50%) | 3/6 (50%) | 3/6 (50%) | 3/6 (50%) | 3/6 (50%) |
| publish | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 3/6 (50%) | 2/6 (33%) |
| maintenance | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 3/6 (50%) |

最大 50% < 60%。规则表与「为什么主体维度只用粗粒度」写在 `docs/w8/skill-spec.md` §4。

### 3.3 三个 badcase 的原始输出

```
[PASS] 10 badcase-重复请求：只产生 1 次副作用
       statuses=['verified', 'duplicate', 'duplicate']，副作用 1 次
[PASS] 10 badcase-参数篡改：业务规则拒绝（不是权限拒绝）
       业务规则 numeric_cap 拒绝：budget=1000000 超过业务上限 200000（schema 允许不等于业务批准）
[PASS] 10 badcase-接口假成功：unverified 而非已发布
       status=unverified
[PASS] 10 审批门：无 HITL 时 irreversible 动作 pending
       status=pending
```

第二条的措辞是刻意的：「schema 允许不等于业务批准」。schema 的 `maximum` 是给模型看的
契约，业务上限是我们的规矩，两者分开的理由见 `adr-w8-action-runtime.md` §4。

### 3.4 审计样本（Gate 要求的字段）

字段齐全：`action_id` / `tool` / `skill_name` / `status` / `risk` / `principal` /
`args_digest` / `arg_keys` / `latency_ms` / `replayed` / `cost` / `approval` / `stage`。

参数值不落盘：断言「正文前 20 字符」不出现在审计 dump 里。审计只留 `args_digest`
（12 位）与 `arg_keys`——**键名要知道，值不要**。键名用于排查「模型是不是漏了必填参数」，
值属于用户数据。

---

## 4. #12 · P3 决策（W8 末 = 第二个决策点）

**结论：不解锁。**

依据（ROADEMAP §7：领先/落后天数 × 3h，余量 ≥15h 才解锁**一个**深入档）：

- W8 预算 12–15h（5 天 × 2–3h），实际用满并溢出：15 个新模块、
  200 条新用例、产品层 4 个端点、1 个自检脚本、4 份文档。
- 余量不仅没到 15h，而且是负的（收尾修了 3 处自检失败 + 1 处文档数字与实测不符）。
- 校验口径也支持这个判断：`docs/w8/skill-spec.md` §5 记了 3 个已知缺口
  （`contains` 会被否定句误命中 / `regex` 判不了语义 / `min_items` 对散文低估），
  这些是「该做但本周没做」的东西，先于任何 P3 深入档。

**默认档照做**：A2A 协议读规范留在 W9（多智能体周）——那才是它的自然挂载点，
本周硬塞进来既挤掉 W8 的边界工作，又因为没有多 Agent 场景而变成纯读书。

---

## 5. 已知缺口（进 W9 的待办）

| 缺口 | 影响 | 计划 |
|---|---|---|
| 工具未进 `/v1/draft` 主链路 | #11 只能算部分通过；工具能力尚无端到端价值证明 | W9 |
| 真 API 未跑（只有桩上游） | 「MCP 接得上真 server」缺一次实测 | W9（配凭证跑 `-m live`） |
| `tool_error_rate` 口径不含执行结果 | 只看参数解析失败率（0/14），真正的工具执行失败率没有 | W9，随主链路一起接 |
| 幂等缓存是进程内 | 多副本部署会重复执行 | W9/W11 换 Redis |
| `sandbox=container` 未实现 | 只有本地三道白名单 | W11 |
| `contains` 会被否定句误命中 | 空话答案可能过 `done_criteria` | 见 `skill-spec.md` §5.1，暂用 `cites`/`min_items` 兜 |
| `LocalWorkspace` 未做配额 | 大文件可能写满磁盘 | W11 |

---

## 6. 本周最容易返工的 8 处 · 实际踩到几个

任务清单列了 8 处。复盘结果：**8 处都没踩到**，但踩了 4 个清单没列、同类的坑——
都是「验的东西不是要验的东西」，比清单里那些更难自查。

| # | 风险 | 是否踩到 | 依据 |
|---|---|---|---|
| 1 | 先接真 API 再做回放 | 未踩 | 顺序一开始就是录制优先，`upstream.py` 的桩在真实现之前写 |
| 2 | 回放未命中偷偷联网 | 未踩 | 直接抛 `ReplayMissError`，且有用例反向验证 |
| 3 | 回放成本与真调用不分 | 未踩 | `_meta.replayed` + `cost=0`，`summary()` 分开统计 |
| 4 | 不声明 permissions/dangerous | 未踩 | `validate()` 强制交叉一致性（有 `entrypoint` 必须有 `permissions`） |
| 5 | 执行循环写进 `LLMGateway` | 未踩 | 执行在 `skills/`，`llm/` 无引用（`test_mcp_boundary.py` 扫得到） |
| 6 | MCP 原始错误泄漏到上层 | 未踩 | 错误码全是自有字符串常量（`tool_not_found` 等） |
| 7 | `ToolCall` 直接当执行指令 | 未踩 | D5 的 `ActionRuntime` 就是为这条建的 |
| 8 | 沙箱检查放在副作用之后 | 未踩 | `executor.py:263` 是第 5 步，`_invoke` 在其后；`escaped=False` 断言守着 |

踩到的 4 个坑（清单没列，都是同一类错误）：

| 坑 | 症状 | 修法 |
|---|---|---|
| 死循环预警放在幂等检查之后 | 「同一动作连发 3 次」永远先被拦成 `duplicate`，预警永不触发 | 移到 `run()` 开头 |
| `VerifyOutcome.ok` 把 `skipped` 算失败 | 每次搜索都变 `unverified`，信号处处出现等于没有信号 | 改 `failed` 属性，只 `unverified` 算失败 |
| 自检脚本有两份 `_env` 装配 | badcase-参数篡改被一个**没装业务规则**的 runtime 验，红得莫名其妙 | 只留一份装配，差异用参数表达 |
| A14 校验在测试里另写一份 | 纸面断言与线上不同源，测试绿而线上可能已经变了 | 复用 `validate_arguments` |

另有两处「数字与实测不符」，属于同一类病的轻症：

- `visibility.py` 的 docstring 写着「16% / ~2.4k 字符」，实测是 **19.1% / 531 字符**——
  写给读者看的数字没人验，就不会准。已按实测改。
- 自检脚本成功路径的 `content` 只有 9 字符（schema 要求 ≥20），
  失败却只打 `invalid_arguments`，看不出是哪个参数。已改成打 `detail`。

**共性**：这 6 条都指向同一件事——「验证通过了」有两种，一种是功能对，
一种是被验对象选错了。后者更危险，因为它的绿看起来完全正常。
