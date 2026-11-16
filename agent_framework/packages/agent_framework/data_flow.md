# 数据流：一次 `/v1/chat` 的往返

> 这份文档只回答一个问题：**打开 `agent_framework`，先读哪五个文件、按什么顺序读、怎么确认自己真读懂了。**
>
> 范围限定在 `server/` + `llm/` 这一条竖切。`retrieval/` `memory/` `skills/` `guardrails/`
> 是另外四条链路，各有各的入口，不在本文。行号以当前 HEAD 为准——**函数名比行号可靠**，
> 移动代码后行号会漂，函数名不会。

---

## 0. 八步，一张图

```
POST /v1/chat
  │
  ├─ 1  server/middleware.py        RequestIDMiddleware
  │        request_id 写进 contextvar；后面每条日志、每个 span、每个错误体都用它
  │
  ├─ 2  server/routes_chat.py:47    chat()
  │        ChatRequest → ChatCall（投影）。不拼 prompt、不判厂商、不重试
  │
  ├─ 3  llm/client.py:153           precheck()
  │        _plan() 选出候选链 + 能力校验 → 返回 alias[0]
  │        **必须在构造 StreamingResponse 之前调用**
  │
  ├─ 4  resilience/concurrency.py   limiter.acquire(alias)
  │        拿不到槽位 → 503 + Retry-After（不能是 200 + error 帧）
  │
  ├─ 5  llm/client.py:168           chat()      ← 熔断器唯一挂载点
  │
  ├─ 6  llm/client.py:183           _chat_impl()   六个阶段
  │        ① 语义缓存命中 → 直接返回（不路由、不花钱、不打上游）
  │        ② _plan() 路由 → 候选链
  │        ③ 每个候选：能力校验 → 取价格表（放在 attempt 循环外）
  │        ④ 每次尝试：adapter.chat() → 按 error.retryable 决定重试还是换候选
  │        ⑤ 成功：emit 一条 RequestLog → cache_store → 返回 ChatOutcome
  │        ⑥ 全链失败：emit 一条失败 RequestLog → 抛最后一次真实错误
  │
  ├─ 7  llm/providers/openai_compat.py
  │        build_request(:436) → httpx → parse_response(:616)
  │        流式：stream(:712) + OpenAICompatAssembler(:201) 按 index 累加增量
  │
  └─ 8  回程
           非流式：ChatResponse.from_outcome → JSON + X-Model-Alias
           流式：  gateway.stream() → server/sse.py:30 encode_event() → SSE 帧
```

一次请求的**上游调用上限 = 候选数 × max_attempts**（默认 3 × 3 = 9）。
知道这个数，W11 INC-5 那条"期望 6、实测 18"才解释得通：断言漏乘了候选链长度。

---

## 1. 五个文件，按顺序读

| # | 位置 | 它只负责 | 读完要能回答 |
|---|---|---|---|
| 1 | `server/routes_chat.py:47` `chat()` | 校验 body、投影成 `ChatCall`、过并发闸门、把结果编码成 JSON 或 SSE 帧 | 为什么业务判断不许下沉到这一层 |
| 2 | `llm/client.py:153` `precheck()` / `:168` `chat()` / `:183` `_chat_impl()` | 路由、候选链、重试、熔断挂载、计量与日志 | 重试为什么只能在首 token 之前；熔断为什么包整条候选链 |
| 3 | `llm/registry.py:78` `_build()` | alias → adapter 的懒加载；`flags` 在哪合并 | 换模型为什么"改 yaml 即可、代码零改动" |
| 4 | `llm/providers/openai_compat.py:436/616/712` + `:201` 组装器 | 三段式：`build_request` → 发 → `parse_response`；流式增量在 assembler 里拼 | 四家的 usage / finish_reason / 错误怎么归一（A4/A5/A6） |
| 5 | `llm/types.py:35` `StreamEventType` + `server/sse.py:30` `encode_event()` | 内部契约长什么样；SSE 帧怎么拼（单行 JSON、`\n\n` 分帧、不带 `provider_meta`） | 为什么只有 `TOOL_CALL` 没有 `tool_call_delta` |

读法建议：**先读 1 再看 2，然后跳到 7 看最脏的一层（协议翻译），最后回头读 3 和 5**。
先读 registry 会陷入"配置长什么样"，而配置的意义只有被链路消费时才显现。

---

## 2. 三处"为什么在这个位置"

这三处是最容易背下来却答错的地方——它们都不是风格偏好，每一处都能指到一个失败形态。

### 2.1 并发闸门在 gateway 之前、`StreamingResponse` 之前

- **在 gateway 之前**：gateway 的重试只对**上游错误**生效。入口过载若进了 gateway，
  重试会把过载放大成"再打一遍"。
- **在 `StreamingResponse` 之前**：`StreamingResponse` 先发 header 再迭代 body，
  一旦进了生成器，任何错误都只能是 `200 + event: error`——过载会伪装成"请求成功但没内容"。
- 于是过载的形态是确定的：`ServerBusyError` → **503 + `Retry-After`**，且不会被 retry 放大。

代码在 `routes_chat.py` 的 `chat()`：非流式用 `async with limiter.slot(alias)`；
流式先 `await limiter.acquire(alias)` 再返回 `StreamingResponse`，并在 `finally` 里 release
（客户端断开也算结束，漏还一次就永久少一个并发额度）。

### 2.2 重试只能发生在首 token 之前（`committed` 状态机）

`stream()` 里有一个运行时变量 `committed`，由三种事件置位：
`TEXT_DELTA` / `REASONING_DELTA` / `TOOL_CALL`（`client.py:68`）。

- `committed` 还是 `False` 时失败（连第一个事件都没拿到）→ 允许重试、允许换候选。
- `committed` 变成 `True` 之后再出错 → **只发一个 `error` 帧然后结束，绝不重放**。
  重放会让用户看到两段开头，而且前一段的 token 已经计费了。

注意两个集合不是一回事：`first_token_ms` 只认 `TEXT_DELTA` / `TOOL_CALL`（`:77`），
`reasoning_delta` 不算——否则思考型模型的 p95 会虚高到 5s 以上，而用户其实什么都没看到。

### 2.3 熔断包整条候选链，语义缓存放在熔断**之内**

```
chat()                       ← breaker.call(lambda: _chat_impl(call))
  └── _chat_impl()
        ├── ① cache_lookup   ← 缓存在熔断器里面
        └── ②③④ 路由 / 候选 / 重试
```

- **失败计数的单位是"一次 `chat()`"**，不是"一次 HTTP"。那一次内部可能已经重试 3 次、
  换过 2 个候选——把计数单位搞错，熔断阈值就永远调不准。
- **缓存在熔断之内**：命中缓存不消耗熔断配额。一个已经被上游打成 OPEN 的系统，
  仍然应该能靠缓存返回老答案。
- 打开期间抛 `CircuitOpenError` → **503 `CIRCUIT_OPEN`**，不进候选链。
  对已经全挂的上游再打 9 次请求没有任何收益。

---

## 3. 验证阶梯

四层，每层都标了"需要什么"。**任何一层没跑过，就不要说这条路径验证过。**

> 下文的数字是本文档写时在本机实跑的结果（`1501 collected / 1471 passed / 30 skipped`），
> 与 `evals/history/2027-W02.json` 的快照一致。数字会随提交漂，红不红不会。

### L0 · 离线单测（零网络、零 key，最先跑）

```bash
uv run pytest --basetemp=.pytest_tmp -q            # 全量：1471 passed / 30 skipped
uv run pytest -k a3 --basetemp=.pytest_tmp -q      # 只跑某条 conformance 断言
uv run pytest packages/agent_framework/tests/test_gateway_retry.py -q
```

- `--basetemp=.pytest_tmp`：把临时目录落在仓库内。受限环境（沙箱 / CI）写不进系统 temp 时，
  不带这个参数会在一堆用例上炸出 `PermissionError`，看起来像代码坏了。
- 30 个 skip 是 `-m live`（要真 key），**不是失败**。跳过数变了不要紧，红才是要紧的。
- 上面那个 1471 会漂。以 `evals/history/<最近一期>.json` 的 `metrics.tests` 为准。

### L1 · 本地服务 + curl（不联网、不花钱，用 `fixture` 别名回放夹具）

```bash
uv run agent-server                     # 另开终端，:8000
curl -s localhost:8000/health

# 非流式：钉 fixture 别名 → mock provider 回放录制体
curl -s -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"fixture"}'

# 流式：逐帧可见
curl -sN -X POST localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"fixture","stream":true}'

# 调用方把模型名写错 → 400 BAD_REQUEST（BadRequestError，不是 500）
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"x"}],"model":"ghost"}'
```

三种错误必须区分开，混在一起就说明没读懂路由层：

| 情况 | 谁的问题 | 状态码 / code |
|---|---|---|
| `model` 字段里的名字不存在 | 调用方 | **400** `BAD_REQUEST` |
| `models.yaml` 的路由里引用了不存在的 alias | 我方配置 | **500** `CONFIG_ERROR` |
| 上游 401/403/402（凭证失效 / 欠费） | 上游 | **502** `UPSTREAM_AUTH`——不是 401，别把上游的凭证问题甩给调用方 |
| 上游 429 | 上游 | **429** `RATE_LIMITED` + 透传 `Retry-After` |
| 上游 5xx（含 Anthropic 529） | 上游 | **502** `UPSTREAM_ERROR` |
| 首 token / 整体超时 | 上游 | **504** `UPSTREAM_TIMEOUT` |
| 入口过载 / 熔断打开 | 我方 | **503** `SERVER_BUSY` / `CIRCUIT_OPEN` |

同一张表就是 `llm/errors.py:236` 的 `ERROR_HTTP`，未登记的异常一律按 **502** 处理：
宁可当成可重试的上游故障，也不要把未知错误伪装成 400 让调用方以为自己参数写错。

没有配 key 又不想改代码？把 `models.yaml` 的 `routes.default` 指向 `fixture` 即可：

```yaml
routes:
  default: { primary: fixture, fallback: [fixture] }
```

### L2 · 反向验证（改坏一个假设，看它是否红）

正向跑绿只证明"现在没坏"。要证明**这条断言真的在守东西**，得先把它弄红。

两条都只改夹具、不改源码，还原一律用 `git checkout --`：

```bash
# 1) 往手写基线里塞一个真实协议不存在的字段 → A13「超集」断言应红
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("packages/agent_framework/tests/fixtures/_synthetic/openai/text_basic/response.json")
d = json.loads(p.read_text())
d["hallucinated"] = {"x": 1}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
PY
uv run pytest -k a13 -q          # 应 FAIL，报 $.hallucinated.x
git checkout -- packages/agent_framework/tests/fixtures/_synthetic/openai/text_basic/response.json
uv run pytest -k a13 -q          # 应绿

# 2) 让流式与非流式对不上 → A3 应红
python3 - <<'PY'
import pathlib
p = pathlib.Path("packages/agent_framework/tests/fixtures/openai/text_basic/stream.txt")
p.write_text(p.read_text().replace("MCP 协议最新进展", "MCP 协议最新动态", 1))
PY
uv run pytest -k "a3 and openai" -q    # 应 FAIL：accumulator.text != non_stream.text
git checkout -- packages/agent_framework/tests/fixtures/openai/text_basic/stream.txt
```

> 这两条都实跑过，红的时候长这样：
> `test_a13_recorded_body_is_superset_of_handwritten[text_basic-openai]` 报
> `Extra items in the left set: '$.hallucinated.x'`；
> `test_a3_stream_equals_non_stream[text_basic-openai]` 在 `test_conformance.py:190` 报文本不一致。
> 用 python 改文件而不是 `sed -i ''`：BSD sed 在非 UTF-8 locale 下处理中文会直接报
> "No such file or directory"，而这个错误看起来像路径写错了。

> ⚠️ 还原**不要**用录制体覆盖手写基线（`cp fixtures/openai/... _synthetic/openai/...`）。
> 那样 A13 会永远绿——而它守的恰恰是"手写基线里有没有幻觉字段"。这类"还原动作把断言杀死"
> 的操作，比忘记还原更危险：它留下一条永远通过的断言，还让人以为有保护。

### L3 · 真机（要 key，花钱）

```bash
cp .env.example .env.local          # 填 ≥1 家（已存在就别覆盖）；确认 git status 里看不到它
uv run pytest -m live -q            # 只跑已配 key 的那几家，其余自动 skip

# 录制：先 dry-run 看会写哪些文件，确认后加 --apply
uv run python evals/scripts/record_fixtures.py --providers deepseek
uv run python evals/scripts/record_fixtures.py --providers deepseek --apply

# 用无效 key 录一次真实鉴权失败体（零成本），验证 401 → 502 的映射
uv run python evals/scripts/record_fixtures.py --bad-key deepseek --apply

# 延迟 / 成本基线
uv run python evals/scripts/bench_baseline.py --models deepseek-flash --repeat 3
```

录制这一步的意义不是"多一份夹具"，而是**用真实响应给手写基线结账**：
A13 只允许真实协议比手写基线**多**字段，不允许**少**——少字段意味着 adapter 里可能为
一个幻觉字段写了永远不会执行的代码。

---

## 4. 读完之后的自检

能不看文档答出这五条，这条链路就算读完了：

1. 为什么 `precheck()` 必须在构造 `StreamingResponse` **之前**调用？
2. 一次 `/v1/chat` 最多打几次上游？这个数由哪两个变量相乘决定？
3. 熔断器的失败计数单位是什么？（提示：不是一次 HTTP）
4. 缓存命中会不会消耗熔断配额？为什么？
5. 流式已经吐出 3 帧后上游断开，客户端会看到什么？RequestLog 里 `error_code` 是什么？

只要有一条要靠翻代码才能答，就还没读懂——**能背流程图不等于能解释位置**。

---

## 5. 三句不要相信的话

| 会听到的说法 | 实际 |
|---|---|
| "adapter 也会重试，双层重试更稳" | adapter 禁止重试（ADR-003）。重试只此一处，双层重试等于把上游调用数乘以候选链长度 |
| "降级就是换个便宜模型再试一次" | 那是第二次重试。W11 已否决：降级顺序是 语义缓存 → 兜底话术（默认关）→ 抛原错 |
| "错误码在路由层定义" | 唯一真相源是 `llm/errors.py` 的 `error_code_for` / `http_status_for`；`server/errors.py` 只做翻译，产品侧不写 `try/except` |
