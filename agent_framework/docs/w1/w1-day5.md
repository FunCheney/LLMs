# W1 Day5 · 真机校准：价格核对 + 录制夹具 + 基线表（3h 可照做教程）

> 前置：Day1–Day4 已完成，`uv run pytest` 全绿（222 passed）、ruff / mypy 全过。
> 今天的代码量在 1200 行左右，但**真正的产出不是代码，是三个"可复现的测量口径"**。

---

## 0. 今天结束时的可验收物

| 产出 | 验收命令 | 通过判据 |
|---|---|---|
| 真实价格进 `models.yaml` | `uv run python -c "from agent_framework.config import load_catalog; print(load_catalog().get_model('deepseek-flash').pricing)"` | 不再是占位值；峰谷 / 阶梯有注释说明口径 |
| 录制器（含 tee 流） | `uv run pytest -k record_fixtures` | 8 例全绿，含"未消费流逐字节抄录"一例 |
| A13 / A14 | `uv run pytest -k "a13 or a14"` | 全绿；**故意在手写夹具里加一个幻觉字段后 A13 必须变红** |
| 基线对比表脚本 | `uv run python evals/scripts/bench_baseline.py --dry-run --models fixture` | 打印表格；**dry-run 不落盘** |
| live 用例 | `uv run pytest` → live 全跳过；`uv run pytest -m live`（无 key）→ 仍跳过 | 两种状态下都不会真调 API |
| 产品端 `/v1/draft` | `curl -X POST localhost:8100/v1/draft -d '{"brief":"...","model":"fixture"}' -H 'content-type: application/json'` | 返回文案；`apps/` 里 grep 不到 `llm.providers` |
| 结构化请求日志真的出现 | 起服务打一发 → `grep '"first_token_ms"' /tmp/ga.log` | **有行**。这一步今天会暴露一个真 bug，见 §8.4 |
| 合流记录 | `cat evals/history/2026-W44.json` + `docs/log.md` | 指标可空，但**空的原因必须写清** |

真机那三连（`--apply` 录制 / `-m live` / 真机基线）需要 `.env.local` 里有 key。
没有 key 不是跳过今天的理由：**所有工具都要能离线自测**，真机只是最后按一下按钮。

---

## 1. 今天回答的唯一问题：adapter 是真的，还是"我们以为的"

Day3 的 136 个用例证明的是：

> **adapter 与我理解的协议一致。**

它没有证明：

> **adapter 与厂商真实返回一致。**

手写夹具是你自己写的，断言也是你自己写的——两边共享同一个错误理解时，测试会**稳定地、漂亮地全绿**。
这类假绿在生产上的表现是：上线第三天，DeepSeek 的 `usage` 挂在了内容块上，你的成本日志全是 0，
而监控没有任何报警，因为"没有 usage"被你自己定义成了合法状态。

Day5 的全部动作都围绕一件事：**引入一个不共享你错误理解的外部证据源。**

| 证据源 | 能抓什么 | 抓不到什么 |
|---|---|---|
| 真机录制夹具 | 手写夹具里的幻觉字段、漏掉的字段、SSE 分帧差异 | 错误路径（不能故意制造限流） |
| A13（录制 ⊇ 手写） | 手写基线多出来的字段 | 录制后新加的字段（那本来就是允许的） |
| A14（tool_call 落在 schema 内） | 真实模型乱调工具、凭空加参数 | prompt 写得不好但没越界 |
| 基线表（从 RequestLog 读） | 首 token 真实分布、成本口径错误 | 质量好坏（那是 W2 的 golden set） |
| `--bad-key` 探测 | 真实的鉴权失败体长什么样 | 欠费 / 限流的真实文案 |

**不要指望一条断言抓住所有问题。** 五条证据源各有盲区，盲区的交集才是剩余风险——
剩下的写进 `docs/log.md` 的遗留问题表，别假装闭环了。

---

## 2. 前置：两分钟的准备

```bash
cd /Users/fanchen/PycharmProjects/LLMs/agent_framework

# 1) 确认基线是绿的，别把旧问题算到今天头上
uv run pytest --basetemp=.pytest_tmp | tail -2     # 222 passed
uv run ruff check . && uv run mypy | tail -1

# 2) 密钥只走环境变量；.env.local 不进 git
ls -a | grep env
```

`.env.local` 的四个变量名必须与 `models.yaml` 的 `api_key_env` 一致：
`OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `ANTHROPIC_API_KEY`。
**今天所有脚本都必须在没有 key 时给出可操作提示并退出，而不是抛栈。**

---

## 3. Step 1（20 min）· 核对真实价格：单值字段表达不了的两种计价

Day4 留的尾巴：`models.yaml` 里除 Qwen 外都是占位值。今天去官网核对，但**核对完会发现一个建模问题**。

### 3.1 两种"一个字段写不下"的计价

**DeepSeek 是峰谷分时计价。** 空闲时段与高峰时段（工作日 09–12、14–18 北京时间）单价差 2 倍。
`pricing: {input: x, output: y}` 只有一个值，你选哪个？

**Qwen 是按上下文长度阶梯计价。** ≤256K 与 >256K 两档单价不同。
同样一个字段，选哪档？

错误做法有三个，都别写：

1. **取平均** —— 平均值不代表任何一个真实时段，成本既不高也不低，是最没用的数字。
2. **取最贵的** —— 看着保守，但会让"成本优化 30%"的评估失去意义（你的分母是假的）。
3. **现在就建模成 `pricing_by_window` / `pricing_by_context`** —— 今天没有消费者，
   为一个 W11 才用的看板提前加结构，是 W1 最典型的时间黑洞。

**正确做法：取常规档 + 在 yaml 里写明口径 + 把"基线必须记时间戳"变成硬约束。**

```yaml
  deepseek-flash:
    provider: deepseek
    model: deepseek-flash
    tier: cheap
    capabilities: [stream, tools]
    # 峰谷分时：空闲时段 1.5/4.5，高峰（工作日 09–12、14–18 北京时间）翻倍为 3/9。
    # 此处取空闲档；若基线在高峰时段测，成本会偏低 2 倍——对比表里必须记时间戳。
    pricing: { input: 1.5, output: 4.5, cache_read: 0.05, currency: CNY, per: 1000000 }
```

在 `models:` 上方再写一段全局口径注释，四条：
① 一律每百万 token（`per` 必须是 1000000，config 的 V5 强制）；
② `cache_read` 是输入缓存命中价，`compute_cost` 暂不使用，W11 接入；
③ 单值字段无法表达峰谷 / 阶梯，一律取最保守的常规档并注明；
④ W11 做成本看板时改为按 `(上下文长度, 时段)` 取价。

### 3.2 顺手补上 `cache_read`

四家都有缓存命中价，且差异很大（OpenAI 是 5 折、Anthropic 是 1 折、DeepSeek 是 1/30）。
现在不接，但**字段要先有**——否则 W11 接 prompt caching 时会发现配置模型里没地方放。

### 3.3 自检

```bash
uv run python -c "
from agent_framework.config import load_catalog
c = load_catalog()
for a in ('qwen3.7-plus','deepseek-flash','gpt-4o-mini','claude-sonnet-4-6'):
    p = c.get_model(a).pricing
    print(a, p.input, p.output, p.cache_read, p.currency)
"
```

四种币种混排是允许的（ADR-009 附注），但 `currency` 必须一路进日志。

---

## 4. Step 2（40 min）· 录制器：`RecordingTransport`

文件：`evals/scripts/record_fixtures.py`。这是今天的主菜。

### 4.1 为什么不能另写一个 HTTP 客户端

最直觉的写法是：在脚本里用 `httpx` 直接 POST 四家，把返回存成夹具。
**这个写法录下来的东西没有价值。**

因为它录的是"录制脚本以为的 body"，不是"生产路径真正会发出的 body"。
两者的差别恰好是 A1 想守的东西：`build_request` 有没有漏 `stream_options`、
`tool_choice` 降级后长什么样、Anthropic 的 `system` 有没有被正确提到顶层。

所以必须**复用真 adapter**，只把它的网络出口接管：

```python
recorder = RecordingTransport(transport)
registry = Registry(catalog, settings, transport=recorder)
adapter = registry.get(alias)           # 真 adapter，真的 build_request
response = await adapter.chat(canonical_messages(), canonical_tools(), max_tokens=MAX_TOKENS)
```

`Registry` 因此要加一个可选 `transport` 参数，透传给两个 adapter 的构造函数。
**注意这是给测试/录制用的注入点，不要为它加任何业务分支。**

### 4.2 tee 的两个分支（今天最容易写错的地方）

transport 拿到 response 后要抄下原始 bytes。问题：`body` 此时可能还没读。

```python
async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
    response = await self._inner.handle_async_request(request)
    capture = Capture(url=str(request.url), request_body=..., status_code=..., content_type=...)
    self.captures.append(capture)

    if response.is_stream_consumed:
        # 替身（MockTransport / 已 aread 的响应）：body 已在内存里，没有 stream 可 tee
        capture.sink.extend(response.content)
    else:
        # 真实网络路径：换成 tee，之后的 aread()/aiter_lines() 都会经过它
        response.stream = _TeeStream(response.stream, capture.sink)

    return response
```

两个分支都必须有，且**必须分开测**：

- 只写 `response.content`：真流式下会抛 `ResponseNotRead`，或逼你先 `aread()` ——
  后者把整段 SSE 缓存进内存，W2 压测长流直接 OOM。
- 只写 tee：你的离线单测全用 `MockTransport`，它返回的响应是"已读"的，
  `response.stream` 上没有东西可 tee，抄下来是空的 → 单测假绿、真机录出空文件。

tee 本体只有十几行，注意 `aclose` 要透传：

```python
class _TeeStream(httpx.AsyncByteStream):
    def __init__(self, inner: httpx.AsyncByteStream, sink: bytearray) -> None:
        self._inner, self._sink = inner, sink

    async def __aiter__(self):
        async for chunk in self._inner:
            self._sink.extend(chunk)      # 边透传边抄，不改动任何字节
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()
```

**为什么要包在 transport 层而不是在 adapter 里截流**：adapter 的薄壳用 `aiter_lines()` 消费，
按行截流会丢掉 `event:` 行、冒号心跳、空行的分帧细节，而夹具要的是逐字节原文。

### 4.3 `request.json` 绝不覆盖（这条今天救了一次 A1）

`request.json` 是 A1 的**断言锚**：A1 断言 `build_request` 精确产出它。
如果录制器顺手把它也覆盖成"真实发出的 body"，A1 就变成自己跟自己比对，永远绿。

```python
drift = diff_request(provider, case, chat_capture.request_body, root=root)
if drift and not allow_request_drift:
    out("\n".join(drift))
    raise SystemExit("将要发出的 body 与 request.json 不一致。先确认 A1 的锚该不该更新。")
```

**漂移 = 硬失败 + 打印 unified diff**，由人决定改锚还是改代码。
只有显式 `--allow-request-drift` 才放行，且这个开关要能被测试覆盖到。

### 4.4 脱敏

错误体里厂商经常回显 key 片段。落盘前把真实 key 替换掉：

```python
for secret in secrets:
    if secret and len(secret) >= 8:      # 把空串或 "sk" 当秘密替换，整个 body 会变马赛克
        text = text.replace(secret, _REDACTION)
```

### 4.5 `--bad-key`：用无效 key 录真实鉴权失败体

`error_auth` 这类 case 无法通过正常调用录到，但可以用一个**故意无效的 key** 打一次：
成本 0、无副作用、能拿到真实错误体。这正是 ADR-005（上游鉴权失败 → 502 而非 401）需要的证据。

### 4.6 自检

```bash
uv run python evals/scripts/record_fixtures.py --providers openai     # dry-run：不发请求、不写盘
uv run pytest packages/agent_framework/tests/test_record_fixtures.py --basetemp=.pytest_tmp
```

单测里必须包含一条 `_ChunkStream`（未消费的 `AsyncByteStream`）用例，
断言 `recorder.captures[0].body == b"".join(chunks)`。没有它，tee 分支等于没测。

---

## 5. Step 3（25 min）· A13 / A14：让手写夹具被清算

在补断言之前，先做一件事：**把现有手写夹具快照到 `fixtures/_synthetic/`。**

```bash
mkdir -p packages/agent_framework/tests/fixtures/_synthetic
for p in openai deepseek qwen anthropic; do cp -R "$p" "_synthetic/$p"; done
```

不快照的话，真录制一覆盖，"手写时我以为协议长什么样"就永远消失了——
而那个差异恰恰是本周最有价值的学习材料。遍历夹具的代码记得**跳过下划线目录**。

### 5.1 A13：录制体必须是手写基线的超集

真录制可以多字段（厂商加字段是常态），绝不能少字段。少的含义只有一个：
**手写基线里存在一个真实协议里不存在的字段**，而 adapter 里可能为它写了永远不会执行的代码。

比较要用**键路径**，不要用键名集合：

```python
def key_paths(value: Any, prefix: str = "$") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            paths.add(path)
            paths |= key_paths(item, path)
    elif isinstance(value, list):
        for item in value:
            paths |= key_paths(item, f"{prefix}[]")
    return paths
```

为什么：`id` / `type` / `index` 四家到处都是，键名集合会让"位置放错但字段存在"的 bug 蒙混过关。
路径能区分 `$.usage.prompt_tokens` 与 `$.choices[].usage.prompt_tokens`——
**后者正是 DeepSeek 的真实形态**，也是本周成本恒为 0 的根因。

```python
missing = key_paths(baseline) - key_paths(recorded)
assert missing == set(), f"手写夹具里有真实协议不存在的字段路径：{sorted(missing)}"
```

再补一条 meta 一致性：`synthetic is False` 必须有 `recorded_at`，手写体必须没有。
否则"哪些是编的"查不清。

### 5.2 A14：工具调用必须落在自己声明的 schema 内

手写夹具永远是"照 schema 编的"，所以这条断言**只有在真录制替换后才有杀伤力**。
真实模型会跑偏：调用未声明的工具名、凭空加参数、漏必填参数。

```python
assert call.name in schemas, f"调用了未声明的工具 '{call.name}'"
assert set(call.arguments) <= allowed, f"收到 schema 外的参数 {sorted(...)}"
assert required <= set(call.arguments), f"缺少必填参数 {sorted(...)}"
```

从 `request.json` 抽 schema 时两种结构都要认：OpenAI 系是 `tools[].function.parameters`，
Anthropic 是 `tools[].input_schema`。

**跑偏时改 prompt / 改工具描述，不要放宽断言。** 放宽等于把"模型乱调工具"放进生产。

---

## 6. Step 4（35 min）· 基线对比表：指标只从 `RequestLog` 读

文件：`evals/scripts/bench_baseline.py`。

### 6.1 唯一口径：不自己计时

这张表是 W2 压测、W11 成本优化、W12 作品集里"改动前"那一列。
**它最容易被写坏的地方，是顺手 `time.monotonic()` 包一层。**

自己计时会造出第二套定义：`first_token_ms` 在日志里是"首个 `text_delta` / `tool_call` 事件的时间"
（`reasoning_delta` 不算），自己计时很容易把首字节、首 reasoning、首 content 混成一个数——
三个数差好几倍，结论会完全反过来。

```python
class LogCapture(logging.Handler):
    """把 agent_framework.request 的 JSON 行收进列表。

    刻意不用 dataclass：logging.Handler.__init__ 会建锁与级别，
    dataclass 生成的 __init__ 不调父类构造，emit 里用 self.lock 直接炸。
    """
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[dict[str, Any]] = []
```

挂到 `get_request_logger()` 上，每次调用前 `capture.records.clear()`，调完读 `latest()`。
`usage_estimated` / `retry_count` / `finish_reason` 一起进样本——
**没有 `usage_estimated`，你分不清"成本低"和"厂商没给 usage"。**

### 6.2 warmup 必须独立且不计入

建连 + TLS 握手 + 厂商侧冷启动都算在第一次请求里。不预热的话第一个样本稳定偏高，p50 直接被拉歪。
`--warmup 1` 默认，`--repeat 3` 默认，表格里写清这两个数。

### 6.3 时间戳必须进表

Step 1 已经说明 DeepSeek 峰谷差 2 倍。表里不带时间戳，"成本降了 40%"可能只是换到了空闲时段。

### 6.4 dry-run 绝不落进 `evals/history/`

那里是"真机基线"的档案库。混进一份 mock 延迟的表，W11 做成本曲线时会把它当真数据读。

```python
if args.dry_run and not args.out:
    print("\n" + report)
    print("dry-run 不写盘（要落盘请显式给 --out）")
    return 0
```

### 6.5 自检

```bash
uv run python evals/scripts/bench_baseline.py --dry-run --models fixture --repeat 2 --warmup 0
uv run pytest -k bench_baseline --basetemp=.pytest_tmp
ls evals/history/        # dry-run 之后这里不该多文件
```

---

## 7. Step 5（20 min）· live 用例：三重闸门

文件：`packages/agent_framework/tests/test_live.py`，16 例（4 家 × 4 条）。

**为什么默认不跑**：一次全量 `pytest` 就真调四家 API，CI 里没 key 会红、有 key 会烧钱，
最后大家都学会忽略红叉。

闸门一，注册 marker（`pyproject.toml` 已配 `--strict-markers`，不注册直接报错）：

```toml
markers = ["live: 需要真实 API 凭证的用例，默认跳过；显式执行 `uv run pytest -m live`"]
```

闸门二，session 级 autouse fixture，只认显式 `-m live`：

```python
@pytest.fixture(scope="session", autouse=True)
def require_explicit_marker(pytestconfig: pytest.Config) -> None:
    if "live" not in (pytestconfig.getoption("-m") or ""):
        pytest.skip("真机用例需显式执行：uv run pytest -m live")
```

闸门三，adapter fixture 里取不到 key 就 `pytest.skip`，不要抛栈。

四条用例各守一件"手写夹具永远测不出来"的事：
① `usage` 真的回来了（`stream_options.include_usage` 不是摆设）；
② 首 token 在 `first_token_timeout_s` 之内（**10s 这个默认值必须被真机校准过**）；
③ 真实模型的 tool_call 落在声明 schema 内（A14 的真机版）；
④ 文本路径真的能通。

```bash
uv run pytest -q                                          # live 全跳过
uv run pytest -m live -q --basetemp=.pytest_tmp           # 无 key 时仍跳过
uv run pytest -m live -k deepseek -s                      # 有 key 后取首 token 实测值
```

---

## 8. Step 6（40 min）· 产品端 `/v1/draft`：证明边界画对了

W1 的 Gate 里有一条"框架可被消费"。**自测不算，必须由产品端真跑一遍。**

### 8.1 端点本身很简单，重要的是它不能 import 什么

`POST /v1/draft`：给一个 brief，返回一版中文短文案。产品层只准碰三个公开入口：
`llm.client`（ChatCall / ChatOutcome / LLMGateway）、`server.errors`、`server.middleware`。

```python
DRAFT_SYSTEM_PROMPT = (
    "你是增长营销团队的文案助手。根据用户给出的 brief 写一版中文短文案："
    "控制在 80 字以内，不使用夸张承诺，不编造数据，不复述 brief。"
)
```

**prompt 属于产品层**：框架不许知道"增长营销"这四个字。

响应 DTO 手写字段，不带 `cost`。理由：cost 依赖 `models.yaml` 的 pricing，是**运营口径**；
放进对外响应等于把内部计价表变成对外契约，改一次价格就要发一次版本。成本留在结构化日志里。

### 8.2 一写就发现：产品层拿不到中间件

`RequestIDMiddleware` 原本长在 `server/app.py` 里。产品要复用它就得
`from agent_framework.server.app import RequestIDMiddleware` ——
而这会顺带执行该模块顶层的 `app = create_app()`。

**为了拿一个中间件而启动一个框架应用，是边界画错的信号。**

抽成两个文件，各自只做一件事：

- `server/middleware.py`：纯 ASGI 的 `RequestIDMiddleware`（`x-request-id` 校验
  `^[A-Za-z0-9_-]{1,64}$`，不校验等于给日志注入开口子；在 `http.response.start` 时 append header）。
- `server/errors.py`：`error_body()` + `install_error_handlers(app)`。
  **只做翻译，不重复定义状态码**——`http_status_for` / `error_code_for` 的唯一真相源仍是 `llm/errors.py`。

抽完之后产品端零 `try/except` 就复用了 6 类错误码。
否则 W12 提取框架时会发现"同一个上游故障，两个服务返回不同 code"。

### 8.3 生命周期别忘

产品端自己建 gateway 就要自己关：`lifespan` 里 `await gateway.aclose()`。
测试里注入假 gateway 时也要验证关闭被调用，否则连接泄漏在 W2 压测才爆。

### 8.4 今天最值钱的发现：日志静默丢弃

起服务打一发，然后：

```bash
grep '"first_token_ms"' /tmp/ga.log        # 一行都没有
```

`RequestLog` 打的是 `INFO`，而 root logger 默认 `WARNING`。代码里 `emit()` 明明调了，
uvicorn 起来后日志里一个字都没有——**Gate 验收项「`first_token_ms` 入日志」会假通过**，
因为测试里用的是 caplog，caplog 自己会改 level。

修法是在**应用边界**配置日志，库本身不擅自改宿主的 logging 配置：

```python
def create_app(gateway: LLMGateway | None = None) -> FastAPI:
    configure_logging()          # 幂等：重复调用不能重复挂 handler
    app = FastAPI(...)
```

框架与产品的 `create_app` 都要调。配完再打一发，必须能 grep 到带 `request_id` 的 JSON 行。
补一条幂等测试：连调两次后 owned handler 数量 ≤ 1（测试里多次建 app 是常态）。

---

## 9. 验收清单（自己跑完再交给我）

```bash
uv run pytest --basetemp=.pytest_tmp                       # 278 passed, 16 skipped
uv run ruff check . && uv run mypy | tail -1               # 全绿
uv run pytest -k "a13 or a14" --basetemp=.pytest_tmp       # 新断言全绿
uv run python evals/scripts/record_fixtures.py             # dry-run 正常
uv run python evals/scripts/bench_baseline.py --dry-run --models fixture
grep -rn "llm.providers" apps/ || echo "产品层边界干净"
grep -rn "time.monotonic\|perf_counter" evals/scripts/bench_baseline.py || echo "基线口径唯一"
```

再手工做一次 A13 的**反向验证**：往 `fixtures/_synthetic/openai/text_basic/response.json`
里塞一个不存在的字段（比如 `"hallucinated": {"x": 1}`），A13 必须变红；还原后全绿。
**不变红就说明这条断言是摆设。**

端到端（mock 回放，不花钱）：

```bash
uv run growth-agent &     # 8100
uv run agent-server &     # 8000
curl -s localhost:8100/health
curl -s -X POST localhost:8100/v1/draft -H 'content-type: application/json' \
     -d '{"brief":"给新用户写一条首购优惠推送","model":"fixture"}'
curl -s -X POST localhost:8100/v1/draft -H 'content-type: application/json' \
     -d '{"brief":"x","model":"ghost"}'          # 400 BAD_REQUEST，不是 500
```

---

## 10. 我会打回的 7 种写法

1. **录制器另写 HTTP 客户端**，不复用真 adapter —— 录的是你以为的 body。
2. **`request.json` 被录制覆盖** —— A1 失去断言锚。
3. **基线脚本自己 `time.monotonic()`** —— 出现第二套 `first_token_ms` 定义。
4. **dry-run 往 `evals/history/` 写文件** —— 污染基线档案。
5. **A13 用键名集合比对** —— 位置放错的字段测不出来。
6. **live 用例没有显式 `-m live` 闸门** —— CI 有 key 就每次提交烧钱。
7. **产品端自己写一遍 `try/except` 映射错误码** —— 错误契约出现第二份定义。

---

## 11. 时间盒与降级顺序

| 时间 | 做什么 |
|---|---|
| 0:00–0:20 | 价格核对 + `models.yaml` 口径注释 |
| 0:20–1:00 | `RecordingTransport` + tee + 脱敏 + `--bad-key` |
| 1:00–1:25 | `_synthetic` 快照 + A13 / A14 |
| 1:25–2:00 | `bench_baseline.py`（LogCapture / warmup / dry-run 不落盘） |
| 2:00–2:20 | live 用例三重闸门 |
| 2:20–3:00 | `/v1/draft` + 抽 middleware/errors + configure_logging + 合流记录 |

超时按此顺序砍：
1. 砍 `--bad-key` 探测（留 TODO，ADR-005 的证据仍缺）
2. 砍基线表的 `blocking` 模式（只留 stream，首 token 才是重点）
3. 砍 A14 的 expect 校验（保留真响应校验）
4. **绝不砍**：tee 双分支、`request.json` 不覆盖、live 三重闸门、configure_logging

---

## 12. 今天的 ADR 记录

**ADR-010：夹具分两级，手写基线永久保留在 `_synthetic/`，真录制带 `synthetic=false + recorded_at`。**
理由：手写基线是"我的理解"的快照，真录制是"事实"。两者都要留，差异才是学习材料；
只用真录制会失去对照，只用手写会假绿。代价：夹具目录翻倍，A13 需要维护键路径比对。

**ADR-011：录制走 transport 层 tee，不另写 HTTP 客户端，且不覆盖 `request.json`。**
理由见 §4.1 / §4.3。代价：`Registry` 多一个 transport 注入点；漂移需要人工裁决。

**ADR-012：日志在应用边界配置（`configure_logging()`），库不擅自改宿主配置；指标读取方挂 handler 消费同一条 `RequestLog`。**
理由：`emit()` 被调用 ≠ 日志可见；读取方另起计时会产生第二套指标口径。
代价：每个可执行入口都要记得调一次，需要幂等。

---

## 13. 可选增强（有余量再做）

1. **SSE 心跳**：DeepSeek 长思考可能 30s 无输出，每 15s 发 `: ping\n\n` 穿透代理超时。
2. **`PreparedRequest.notes` 贯通**：让"降级率"可统计（Day4 已列为 W2 项）。
3. **基线表加 p95 列**：现在只有 p50 / min / max，样本量 ≥10 后 p95 才有意义。
4. **`/v1/models` 端点**：暴露 catalog，W11 看板顺手就有。

---

## 14. 交给我 review 时给这三样

```bash
uv run pytest --basetemp=.pytest_tmp | tail -3
grep -rn "llm.providers" apps/ ; grep -rn "is_stream_consumed" evals/scripts/
ls evals/history/ && tail -20 docs/log.md
```

我重点看四件事：
① tee 是否有"已读 / 未读"两个分支且各有测试；
② `request.json` 是否只读不写；
③ 基线表的指标是否全部来自 `RequestLog`；
④ `evals/history/*.json` 里为 null 的指标是否都写了原因。

---

## 15. 实现状态（Day5 收尾实录）

> 以下全部由我（AI）实现并验证。**真机那一步未执行**（无 `.env.local`），
> 所有工具都已具备离线自测能力，真机只是按按钮。

### 15.1 产出清单

| 产出 | 状态 | 文件 | 用例数 |
|---|---|---|---|
| 真实价格 + 计价口径注释 | ✅ | `src/agent_framework/models.yaml` | — |
| 录制器（tee / 脱敏 / 漂移检查 / bad-key） | ✅ | `evals/scripts/record_fixtures.py`（567 行） | 8 |
| 手写基线快照 | ✅ | `tests/fixtures/_synthetic/`（4 家 × 6 case） | — |
| A13 / A14 | ✅ | `tests/test_conformance.py` | conformance 共 104 |
| 基线对比表 | ✅ | `evals/scripts/bench_baseline.py`（500 行） | 8 |
| live 用例（三重闸门） | ✅ 默认跳过 | `tests/test_live.py` | 16 |
| 中间件 / 错误契约抽离 | ✅ | `server/middleware.py` / `server/errors.py` | — |
| 产品端 `/v1/draft` | ✅ | `apps/growth_agent/.../routes_draft.py` + `app.py` | 6 |
| 日志边界配置 | ✅ | `observability.configure_logging()` | 1（幂等） |
| 合流记录 | ✅ | `evals/history/2026-W44.json` + `docs/log.md` | — |
| 切模型说明 | ✅ | `README.md`「如何切模型」 | — |

### 15.2 与设计的三处偏差

| # | 偏差 | 原因 |
|---|---|---|
| 1 | 录制器 / 基线脚本都支持 `fixtures_root` / `--out` 注入 | 测试必须写 `tmp_path`，否则跑一次测试就把真机基线冲掉 |
| 2 | dry-run 默认**不写盘**，只有显式 `--out` 才写 | `evals/history/` 是真机基线档案库，混进 mock 延迟会污染 W11 曲线 |
| 3 | `LogCapture` 用普通类而非 dataclass | dataclass 生成的 `__init__` 不调 `logging.Handler.__init__`，`self.lock` 不存在 |

### 15.3 验收结果（可复现）

```text
uv run pytest            → 278 passed, 16 skipped（222 → 278，新增 72 个收集项）
uv run ruff check .      → All checks passed
uv run mypy              → Success: no issues found in 26 source files
uv run pytest -m live    → 无 key 时 16 个全部 skip（不花钱、不报错）
grep -rn "llm.providers" apps/                → 空
grep -rn "time.monotonic" evals/scripts/      → 空
uv run pytest -k "a13 or a14"                 → 33 passed
uv run python evals/scripts/record_fixtures.py（无 key）
    → 干净退出并提示「缺少凭据 OPENAI_API_KEY：请在 .env.local 中设置 ...」，不抛栈
uv run python evals/scripts/bench_baseline.py --dry-run --models fixture
    → 打印表格 + 峰谷/阶梯提醒；`evals/history/` 不新增文件
```

**A13 反向验证**：往 `_synthetic/openai/text_basic/response.json` 注入 `"hallucinated": {"x": 1}`
→ `test_a13_recorded_body_is_superset_of_handwritten[text_basic-openai]` **变红**
（`Extra items in the left set: '$.hallucinated.x', '$.hallucinated'`）；还原后 33 个全绿，
`_synthetic/` 与现夹具 `diff -rq` 无差异。

端到端（真起两个服务 + curl，走 `fixture` 回放）：

```text
GET  /health（8000 / 8100）        → 均 ok
POST /v1/draft  model=fixture      → 200，文案 + request_id
POST /v1/draft  model=ghost        → 400 BAD_REQUEST（调用方输入错误）
POST /v1/chat   stream=true        → text_delta ×2 → message_end
grep '"request_id"' /tmp/ga.log    → 有结构化 JSON 行（configure_logging 修复后）
```

### 15.4 仍未闭环（全部需要真机，已写进 `docs/log.md`）

| # | 事项 | 何时 |
|---|---|---|
| 1 | 真机录制未跑，A13 的清算能力未兑现 | 有 key 时跑 README 的 Day5 三连 |
| 2 | `first_token_timeout_s = 10s` 未校准 | `uv run pytest -m live -k first_token` 取实测 |
| 3 | Anthropic 欠费仍靠文案 `credit balance` 识别 | 真机确认后收敛成固定 code 白名单 |
| 4 | 峰谷 / 阶梯计价未建模 | W11 成本看板 |
| 5 | `usage_missing` / `error_*` 无法安全录制 | 长期保留手写 + `error_auth` 真机探测 |
