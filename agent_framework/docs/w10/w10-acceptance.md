# Gate 10 验收：内容生成 + 安全护栏 + 首次上线

> 结论：**7 项通过，2 项判 partial**（#1 端到端真实性 / #7 真实上线）。
>
> 两项 partial 的性质与 W9 那次不同。W9 的两项是"做了，但证明得没那么多"；
> 这两项是"**代码到这里了，但差一样东西：凭证与域名**"。
> 差的东西不在我能单方面补上的范围内，所以不写成 pass——写 pass 等于把
> "没跑过真实模型"这件事藏起来，而它正是 W10 唯一真正没被验证的面积。

## 验收明细

### #1 端到端 · **partial**

**通过的部分**：`POST /v1/article` 主题进 → 关键词 → 大纲 → 成文 → meta →（可选）三平台扇出，
全程 `stages[]` 带 `ms / prompt_tokens / completion_tokens / alias / fallback_from`。
端点级端到端有 12 条用例（`apps/growth_agent/tests/test_article_route.py`），
覆盖投影干净、错误分级、部分失败收口、注入拦截、隔离封皮不外泄。

**没通过的部分**：**没有用真实模型跑过一次**。没有厂商凭证，所有"文章质量"的
判断都不存在。端点级用的是 `ScriptedGateway`（按脚本返回固定文本），它证明的是
**管道接线**，不是**生成质量**。

这条差距的性质要说清楚：它不是"再写点代码"，而是"需要一把 key"。

### #2 引用溯源 · pass

- 成文 prompt 明确"每个事实性陈述后跟引用编号，无出处不写"，参考资料用 `build_context` 的编号；
- 零证据**直接失败**，不降级成"凭常识写一篇"（`SEOError(stage="evidence")`）——这是最容易被绕过的一条；
- 引用体检（`coverage`）算引了几个、有没有悬空编号、几句事实没出处；
- 悬空编号是**硬错**，与 policy 无关。

判据有一条已知的漏：`_CLAIM_SIGNALS` 只对"带事实信号"的句子要求标号
（数字/百分比/年份、据…统计、研究显示）。"内容营销的投资回报很高"这种没有
信号词但有事实性的句子会漏。**这是刻意的**：全句要求标号会把过渡句也判成违规，
运营会直接把这条检查关掉——那时防护水平是 0，比现在低。

### #3 品牌调性 · pass

- `PlatformSpec` 数据驱动（`max_chars / tone / required_elements / forbidden`），不是散在 prompt 字符串里；
- 品牌禁用词从 `LongTermMemory` 取，与平台 spec **合并**；
- `validate` 是确定性校验（长度/禁用词/必需元素），**规则校验优先于再调一次 LLM**；
- 三平台改写并发，返回顺序与请求顺序一致（不按完成顺序，否则前端没法做 diff）。

15 条用例含反向验证：注入一个含禁用词的改写 → `validate` 拦下并返回违规项。

### #4 注入全拦截 · pass

```
注入用例：10/10 拦截
正常用例：6/6 放行

按攻击形态覆盖：
  boundary_escape        1/1
  data_exfiltration      2/2
  instruction_override   2/2
  obfuscation            2/2
  prompt_leak            1/1
  role_switch            1/1
  tool_hijack            1/1

规则命中分布：
  override_zh 3 · prompt_leak 2 · role_switch_zh 2 · secret_exfil 2 · fake_delimiter 2
  override_en 1 · shell_exec 1 · base64_blob 1 · decoded:override_en 1 · invisible_chars 1
```

**间接注入覆盖**（本周头号风险）：`rt-005` 藏在检索内容里、`rt-006` 藏在 tool result 里
（伪造系统消息头）、`rt-007` 试图闭合隔离标签。

**为什么要打命中分布这张表**：10 条用例全挂在同一条规则上时，那条规则一改，
整体仍然显示 10/10——因为它测的还是同一件事。分布散在 10 条规则上，
才说明"每条用例在被自己的防线拦"。

**为什么要报误报率**：只报召回的红队报告是不完整的。6 条正常用例里有 3 条是
关键误报用例（"把密码强度检查加到注册流程"、"提到 shell 但不是执行"、
"system prompt 在输出之前"）。误报的代价不是"多拦一条"，是"这个功能被关掉"。

### #5 HITL 红线 · pass

红线只有一句话：**未审批的 dangerous 技能绝不执行**。所以断言不是"抛了异常"，
而是"`execute` 一次都没被调用"：

```python
with pytest.raises(ApprovalRequired):
    await pipe.run("发一篇文章", execute=execute, approval_for=(publish_skill(), args))
assert execute.calls == []          # 副作用从未发生
```

状态机 `pending → approved / rejected / expired`；**过期默认拒绝**；
`guard_execute` 没有任何参数能关掉这条（没有 `skip_approval`、没有 `force`）；
每次拦断都写一条 `blocked` 审计。

反向用例同样重要：**批过的必须放行**，否则这道门就是墙（8 条 `test_hitl_flow` + 13 条编排器用例）。

### #6 鉴权不混 · pass

| 谁错了 | 状态码 | code |
|---|---|---|
| 调用方没带/带错我方 key | **401** | `AUTH_REQUIRED` / `AUTH_INVALID` |
| 上游厂商 key 失效/欠费 | **502** | `UPSTREAM_AUTH` |

测试 `test_our_401_is_not_upstream_502` 以 `llm/errors.py` 的 `ERROR_HTTP` 表为参照，
不自己再写一份期望值——参照物写死在测试里，等于给同一份契约造第二份定义。

本机进程实测（`AUTH_MODE=key`，真 HTTP）：

```json
{"code":"AUTH_REQUIRED","message":"缺少凭据：请在 X-API-Key 或 Authorization 中提供",
 "request_id":"req_71f1570dc97b"}
```

`request_id` 有值说明中间件顺序对（RequestID 在鉴权外层）——
顺序反了的表现是 401 响应里 `request_id == "-"`，而排障时最需要它的正是被拒的请求。

### #7 真实上线 · **partial**

**通过的部分**：`ops/deploy_check.sh` 五项，在本机用生产同构参数起进程，**5/5 通过**：

```
✓ /health 200        {"status":"ok","service":"growth-agent",...,"rag":"cold"}
✓ 无 key → 401        code=AUTH_REQUIRED
✓ 错 key → 401        code=AUTH_INVALID
✓ 带 key → 200        request_id=req_826f6bb204a3
✓ X-Request-Id 存在
```

**没通过的部分**：公网不可访问，没有域名、没有证书、没有真实用户。
`docker-compose.prod.yml`、`Caddyfile` 都没跑过。

这里有个自我检查值得记下来：`deploy_check.sh` 第一版跑出来是 **3/5**——
第 2、3 项红，因为 `AUTH_MODE=key` 根本没生效（pydantic-settings 的 env 名取自
字段名，字段叫 `auth` 时读的是 `AUTH` 不是 `AUTH_MODE`）。**如果这个脚本只测
"带 key 能通"，两档都会绿**，而"以为开了其实没开"会一路带到线上。

写脚本的价值不在它绿的那天，在它第一次红的那天。

### #8 runbook · pass

`ops/RUNBOOK.md` 新增三节：

- **§10 上线**：三档环境（开发/预演/生产）、生产档两个硬约束（只有 443 对外、
  框架服务不对外）、上线前三件事、五项自检表（含"失败的含义"）、真实的首次上线记录、回滚；
- **§11 密钥轮换**：我方 key 双写过渡（**不要**直接删旧 key）、上游 key 的顺序不能反
  （先作废旧 key 再换新 key，中间那段时间全是 502 `UPSTREAM_AUTH`，而值班会去查上游）；
- **§12 排障**：状态码先分流再看 code 的表、`CONTENT_BLOCKED` 突增的两种相反处理、
  审批积压不是安全事件而是流程卡住。

回滚那条特意写了"不重建镜像，用 `--no-build`"：重建会带着当前（可能有问题的）源码，
那不是回滚，是"重启了另一个 bug"。

### #9 合流 · pass

```
make eval WEEK=2026-W53
54 条：passed 50 / failed 0 / interrupted 0 / skipped 4 → success_rate 100.0%（mode=offline）
judge 均分 1.922（50 条）
基线 2026-W46 · 对比 success_rate, latency_p95_ms · 门禁通过
```

`evals/history/2026-W53.json` 与本文件、`docs/log.md` 同批落盘。

---

## 门禁四件套

| 项 | 结果 |
|---|---|
| `ruff` | All checks passed |
| `mypy` | 122 files, 0 issues |
| `pytest` | **1250 passed / 30 skipped**（收集 1280；W9 末 1096，+154） |
| `make eval` | 50 passed / 0 failed / 4 skipped，judge 1.922，`regression_gate` pass |

本周新增 154 条用例，逐文件核对过：框架侧 `test_injection`(45) / `test_guardrails`(31) /
`test_guardrail_pipeline`(13)；产品侧 `test_seo_pipeline`(18) / `test_style_rewriter`(15) /
`test_article_route`(12) / `test_auth`(12) / `test_hitl_flow`(8)。

---

## 本周真正的收获：三个"只在组合时才出现"的缺陷

这三个都不是"写错一行"，而是**各层单独测都绿、串起来才炸**：

### 1. 护栏与自己的封皮互相指认

`/v1/article` 第一次跑就 500：

```
{"code":"ARTICLE_STAGE_FAILED","stage":"input",
 "message":"主题疑似提示注入：boundary_escape(fake_delimiter)"}
```

外层 `GuardrailPipeline` 把主题隔离成 `<untrusted source="user">…</untrusted>`，
内层 `SEOPipeline` 又检一遍，`fake_delimiter` 看到 `</untrusted>` —— 那是我们自己写的。

修法不是给规则豁免，是让隔离**可逆且幂等**：`unwrap()` 剥封皮（只在声明的来源
与期望一致时剥）、`isolate()` 幂等、检测前先 unwrap。

### 2. 封皮流进了标题

修完 500 之后标题变成 `<untrusted source="user">内容营销怎么做</untrusted>：为什么…`。

红队不会发现它（不是攻击），误报率也不会（不影响拦截），只有人打开页面才看得到。
现在有一条例行断言守着。

### 3. `AUTH_MODE=key` 其实没生效

pydantic-settings 的 env 名取自字段名：字段叫 `auth` 时读的是 `AUTH`，不是 `AUTH_MODE`。
写错名字不报错，**只是静默用默认值**——而默认值是 `off`。

这三条的共同点：**单层测试的盲区是系统性的**。W11 要做故障注入，理由就在这里。

---

## 诚实的缺口清单

1. **真实模型一次没跑**。所有安全结论都建立在规则层上；模型自身被 jailbreak 的路径没有数据。
2. **文章质量零证据**。没有真实生成过一次，可读性、引用贴合度、三平台风格是否到位，全是未知。
3. **公网、证书、DNS、镜像构建全部未验证**。W11 要的"真实反馈修 bug"目前没有输入来源。
4. **语义级注入与多轮渐进完全没防**（`threat-model.md` §6 已列）。10/10 是"10 条我们知道怎么打"。
5. **误报率只有 6 条手工样本**。不足以支撑"误报可接受"这句话。
6. **`/v1/approvals` 没有角色校验**。任何持有效 key 的人都能批，包括提交人自己——
   这在"审批是为了防误操作"的语义下是成立的，在"审批是为了防内部越权"的语义下不成立。
   本周是前者，但这条差别该写下来。
