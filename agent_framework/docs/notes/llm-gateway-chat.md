# `LLMGateway.chat()` 逐行注释与解析

> 源码：`packages/agent_framework/src/agent_framework/llm/client.py:144-263`
> 定位：**编排层**。`LLMClient`（adapter）只负责 wire ↔ internal 的协议翻译；`LLMGateway` 负责路由、能力校验、重试、退避、计量、成本、日志、fallback。

## 一句话概括

`chat()` = **两层循环 + 一次真实 HTTP + 一条日志出口**：

```
_plan(call)                      # 1. 定候选链：[primary, fb1, fb2...]
for alias in aliases:            # 2. 候选维度（fallback 链，只走一遍不回头）
    adapter = 校验能力(alias)     #    硬需求不满足 → CapabilityError 直接上抛
    for attempt in 1..max_attempts:   # 3. 同一候选的重试维度
        async with span(...):     #    每次真实 HTTP 一次 span（重试 3 次 = 3 个 span）
            try: adapter.chat()   # 4. 只调 adapter，不碰厂商字段
            except: 可重试?睡一会重试 : 记日志+break 换下一个候选
        成功 → 计费/打点/emit RequestLog → return   # 5. 有且只有一条日志
全链失败 → emit 一条 error 日志 → raise           # 6. 不返回 None，不吞异常
```

## 逐行注释版

```python
async def chat(self, call: ChatCall) -> ChatOutcome:
    # ── 阶段 1：路由。_plan 内部已经做完结构校验：
    #    call.model 拼错（调用方输入问题）→ BadRequestError(400)
    #    路由里的 alias 在 models.yaml 里不存在（我方配置问题）→ ConfigError(500)
    aliases = self._plan(call)
    tier = self._tier(call)              # 只用于日志：task 所属的 tier（cheap/strong/...）
    fallback_from: str | None = None     # 记录"我是从哪个 alias 掉下来的"，写进日志
    last_error: LLMError | None = None   # 保留最后一次真实错误，全链失败时抛它（不要造新错）
    started_all = self._clock()          # 整条链的起点，只用于「全链失败」那条日志的耗时

    # ── 阶段 2：候选维度（fallback 链）
    for alias in aliases:
        # 能力校验放在每次尝试之外：能力是静态属性，重试 3 次不该校验 3 次。
        # 注意：这里抛的 CapabilityError 不进入 except（在 try 外），
        #      即"候选不支持 tools"会直接终止整个调用，而不是换下一个候选。
        adapter = self._ensure_capabilities(alias, call, stream=False)
        model = self._registry.get_model(alias)   # 取价格表；放在 attempt 循环外避免重复查
        started = self._clock()                   # 该候选的起点（注意：包含后续退避等待）

        # ── 阶段 3：重试维度
        for attempt in range(1, self._settings.max_attempts + 1):
            # span 只包「一次真实 HTTP」，不是包整个 chat()：
            # 重试 3 次就该有 3 个 span，trace 里才看得出是第几次才通。
            async with span(
                "llm.call",
                task=call.task,
                alias=alias,
                provider=adapter.name,
                attempt=attempt,
                stream=False,
            ) as current_span:
                try:
                    # adapter.chat 只做协议翻译（OpenAI/Anthropic/Qwen 各家格式 → LLMResponse）
                    # **call.opts 透传**：temperature / max_tokens 等厂商中立参数，网关不认识具体字段
                    response = await adapter.chat(
                        call.messages,
                        call.tools,
                        tool_choice=call.tool_choice,
                        **call.opts,
                    )
                except (LLMError, httpx.HTTPError) as exc:
                    # 统一异常口径：adapter 内部的 httpx 传输错误在这里翻译成 UpstreamTimeout/UpstreamServer，
                    # 否则 ConnectError 会以原始形态冒到 HTTP 层，变成没法归因的 500。
                    error = self._as_llm_error(exc, adapter)
                    self._mark_span_error(current_span, error)   # 失败 span 也带 error_code，便于按码归因

                    # 是否重试只看 error.retryable（由 errors.py 按 429/402/5xx/code 白名单判定），
                    # 这里不再写 if status == 429 —— 同一份逻辑两处实现，必然改一处漏一处。
                    if not self._should_retry(error, attempt):
                        last_error = error
                        # 候选"最终失败"单独打 WARNING，不进 RequestLog：
                        # 一个请求可能试 3 个候选，但只允许产出一条 RequestLog，
                        # 否则 W11 统计请求量和成本时会翻倍。
                        emit_candidate_failure(
                            alias, attempt, error_code_for(error), str(error)
                        )
                        break      # 跳出重试循环 → 进入下一个候选（fallback）

                    # 可重试：指数退避 + 抖动（限流时听上游 Retry-After）
                    _LOG.info(
                        "retry alias=%s attempt=%d code=%s",
                        alias, attempt, error_code_for(error),
                    )
                    await self._sleep(self._delay(attempt - 1, error.retry_after))
                    continue

                # ── 阶段 4：成功路径（以下代码只在没抛异常时执行）
                latency_ms = (self._clock() - started) * 1000
                response.provider = response.provider or adapter.name   # 兜底：adapter 没填就用适配器名
                response.latency_ms = latency_ms                        # 延迟由网关测，不信上游
                cost = compute_cost(model.pricing, response.usage)      # 计费只写一份（observability.py）

                # 先把这次调用的运营指标挂到 span 上（token/成本/延迟/finish_reason/第几次重试）
                current_span.attributes.update({
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "cost": cost,
                    "currency": model.pricing.currency,
                    "latency_ms": round(latency_ms, 3),
                    "finish_reason": str(response.finish_reason),
                    "retry_count": attempt - 1,
                })

                # 每个请求恰好一条结构化 JSON 日志（字段固定，W11 看板直接读它）
                emit(RequestLog(
                    request_id=current_request_id(),
                    task=call.task,
                    tier=tier,
                    provider=response.provider,
                    model=alias,                 # 注意：日志里 model 记的是 alias（我方别名），provider 才是厂商
                    fallback_from=fallback_from, # 从谁掉下来的，一眼看出降级发生了
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    cost=cost,
                    currency=model.pricing.currency,
                    latency_ms=round(latency_ms, 3),
                    first_token_ms=None,         # 非流式没有首 token 概念
                    retry_count=attempt - 1,
                    finish_reason=str(response.finish_reason),
                    error_code=None,
                    stream=False,
                    usage_estimated=response.usage.estimated,   # 厂商没给 usage 时由 adapter 估算，必须标记
                ))

                # 成功即返回，不再尝试后续候选；ChatOutcome 携带路由/重试元信息，
                # 这些是"客户端层运营信息"，不塞进契约层 LLMResponse。
                return ChatOutcome(
                    response=response,
                    alias=alias,
                    retry_count=attempt - 1,
                    fallback_from=fallback_from,
                )

        # ── 阶段 5：该候选彻底失败（重试用尽或不可重试）→ 换下一个候选
        # 单向：fallback 链只绕一圈，绝不回头（避免 primary/fb 互相踢皮球形成活锁）
        fallback_from = alias

    # ── 阶段 6：全链失败
    # 没有 last_error 说明 aliases 为空之类的情况，给一个明确的 BadRequest 而不是 assert
    error = last_error or BadRequestError(
        f"task '{call.task}' 没有可用候选",
        provider="gateway",
    )

    # 失败也要有且只有一条 RequestLog（与成功路径对称）：
    # 少了它，"请求量 / 失败率"只能从 access log 猜。
    emit(RequestLog(
        request_id=current_request_id(),
        task=call.task,
        tier=tier,
        provider=error.provider,
        model=fallback_from or "",          # 最后一个试过的候选
        fallback_from=fallback_from,
        latency_ms=round((self._clock() - started_all) * 1000, 3),  # 这里才用整链耗时
        finish_reason="error",
        error_code=error_code_for(error),
        stream=False,
    ))

    raise error    # 保持异常语义：不返回 None、不吞错，让上层（server 层）统一映射 HTTP 状态码
```

## 五个关键设计点

| 设计点 | 代码位置 | 为什么这么做 |
|---|---|---|
| **失败也要留一条日志** | 全链失败分支先 `emit` 再 `raise` | 成功/失败两条路径都恰好一条 `RequestLog`，成功率与成本才有唯一口径 |
| **不可重试错误直接换候选** | `if not self._should_retry(...): break` | 400/内容拦截重试无意义，重试只会拖长尾延迟并放大计费 |
| **候选失败与请求失败分开记** | `emit_candidate_failure` + 仅一次 `emit` | 试 3 个候选不等于 3 次用户请求，混在一起统计会翻倍 |
| **span 粒度 = 一次尝试** | `async with span(...)` 在 attempt 循环内 | trace 里能直接看出"第几次才通"，而不是只有一个 3 秒的黑盒 |
| **错误归属分层** | `_plan()` | 调用方拼错 alias → 400；我方 `models.yaml` 坏了 → 500。混一起会让值班同学被误告警 |

## 与 `stream()` 的差异（同一套骨架，一处关键状态）

`stream()` 复用完全相同的「候选 → 重试 → 日志」结构，唯一新增的是运行时状态 `committed`：

- 首个事件之前失败 → 用户什么都没看到 → **允许重试 / fallback**（`_first_event` + `first_token_timeout_s`，10s）
- 已发出 `TEXT_DELTA / REASONING_DELTA / TOOL_CALL` 之后失败 → **绝不能重放**，只能以 `StreamEventType.ERROR` 帧终止

这也是 ADR-008 拒绝装饰器链的原因：装饰器把 `stream()` 当黑盒，拿不到 `committed`；而在 async generator 边界上，`yield` 之后抛出的异常外层也接不住。

## 两处值得留意的细节

1. **`started` 在 attempt 循环外**（第 154 行）→ `latency_ms` 实际是"该候选从第一次尝试到成功"的累计耗时，**包含退避等待**；而 `stream()` 里 `started` 在循环内，是单次尝试耗时。两者口径不一致，若看板把 chat/stream 的 p95 放一起比，需要留意。
2. **`CapabilityError` 在 try 外抛出**（第 152 行）→ 不满足能力会终止整个调用，而不是降级到下一个候选。如果希望"候选不支持 tools 就换下一个"，需要把这个校验挪进循环并当成 `retryable=False` 的候选失败处理。
