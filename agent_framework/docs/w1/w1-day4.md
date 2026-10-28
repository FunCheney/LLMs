# W1 Day4 · 归一化收口 + SSE 端点（3h 可照做教程）

> 定位：Day3 结束时你有 **4 个同签名的 adapter**，但没有"谁来选模型、谁来重试、谁来计算成本、谁来对外服务"。
> 今天的产出是 **把 adapter 装进一条可运营的链路**，并开出第一个对外端点。
>
> 契约唯一依据：`docs/architecture.md` §3.6 / §3.7 / §4 / §5。本文只讲"怎么写、写到什么程度、怎么自查"。
> 代码由你写。本文给签名、算法步骤、关键片段形状，**不给完整实现**。

---

## 0. 今天结束时的可验收物

| 文件 | 一句话职责 | 预估 |
|---|---|---|
| `agent_framework/config.py` | 读 `models.yaml` + 环境变量，**启动即校验** | 20 min |
| `agent_framework/models.yaml` | 别名 / wire 模型名 / tier / capabilities / pricing | 含在上项 |
| `agent_framework/observability.py` | request_id 上下文 + 17 字段结构化日志 + 成本计算 | 15 min |
| `agent_framework/llm/registry.py` | alias → adapter 实例（**懒加载 + 缓存**） | 25 min |
| `agent_framework/llm/router.py` | `route(task) -> list[alias]`（+ 第二实现） | 15 min |
| `agent_framework/llm/client.py` | **重试 / 计量 / 日志 / fallback 的唯一归属** | 50 min |
| `agent_framework/server/schemas.py` | 请求/响应投影（**绝不直接返回 Message**） | 含在下面 |
| `agent_framework/server/sse.py` | `StreamEvent` → SSE 帧 | 10 min |
| `agent_framework/server/routes_chat.py` | `POST /v1/chat` | 15 min |
| `agent_framework/server/app.py` | 挂载 + 异常处理器 + request_id 中间件 | 含在上项 |
| `packages/agent_framework/tests/test_{config,router,cost,gateway_retry,gateway_stream,server_chat}.py` | 离线测试 + A9–A12 | 15 min |

**今天不做**：工具执行循环、真实鉴权、语义缓存、熔断、限流、Temporal、LangChain（architecture §7）。

---

## 1. 调用链总览

```text
HTTP 请求
  │
  ├─ middleware: 生成 request_id → contextvar → 响应头 X-Request-ID
  │
  ├─ routes_chat: 校验 body → 投影成内部 Message/ToolSpec/ChatCall
  │
  ├─ gateway.precheck(call) ──► router.route(task) → [alias...]
  │                             registry.get(alias) → adapter
  │                             能力校验（不满足 → CapabilityError → 400，此时还没开始流）
  │
  ├─ stream=False ──► gateway.chat(call) ──┐
  └─ stream=True  ──► gateway.stream(call) ┤
                                            │
              ┌─────────────────────────────┘
              │  对每个 alias（全链只绕一圈）
              │    重试循环（只在 committed=False 时）
              │      adapter.chat / adapter.stream
              │      ↑ 只做翻译，无重试无日志
              │    计量（pricing）→ 记录 RequestLog（17 字段）
              ▼
         LLMResponse / AsyncIterator[StreamEvent]
```

**一次流式请求要经过的 12 个动作**（照着这个顺序写，不会漏）：

1. 生成 `request_id`，写入 contextvar
2. 解析 body → `ChatCall`（task 默认 `chat`）
3. `gateway.precheck(call)`：路由 + 取 adapter + 能力校验（**必须在返回 StreamingResponse 之前**）
4. `router.route(task)` → `["qwen3.7-plus", "deepseek-flash"]`
5. 取第 1 个 alias → `registry.get(alias)`（首次调用才建 httpx client）
6. 记录 `t0`（`perf_counter`）
7. `adapter.stream(...)` 拿到 async generator
8. **首次 `__anext__()` 用 `first_token_timeout` 包住**，超时 → `UpstreamTimeoutError`（此时 `committed=False`，可重试）
9. 首个 `text_delta` / `tool_call` 到达 → 记 `first_token_ms`，**置 `committed=True`**
10. 透传事件；`message_end` 里的 `usage` 用于计量
11. 计算 `cost` → 发一条 `RequestLog`
12. 结束（`finally` 里 `await agen.aclose()`）

---

## 2. 前置：两分钟的准备

```bash

# 1) 依赖：yaml 解析 + 类型存根（mypy strict 下没有存根会报 import-untyped）
uv add pyyaml --package agent-framework
uv add --dev types-pyyaml

# 2) 确认当前基线是绿的，避免把已有问题算到今天头上
uv run pytest -q | tail -2
uv run ruff check . | tail -2
uv run mypy | tail -2
```

预期：`136 passed` / `All checks passed` / `Success: no issues found in 15 source files`。

> 如果你不想加 `pyyaml`，也可以写 `models.py`（Python 字面量）——但那样"改配置切模型"就变成"改代码"，Gate3 第 1 条会挂。**加 yaml，别省这一步。**

---

## 3. Step 1（20 min）· `config.py` + `models.yaml`

### 3.1 五个 pydantic 模型

文件：`packages/agent_framework/src/agent_framework/config.py`

```python
class Pricing(BaseModel):
    input: float                    # 每 per 个输入 token 的价格
    output: float
    currency: Literal["USD", "CNY"]
    per: int = 1_000_000            # 必须显式写，见下方校验规则
    cache_read: float | None = None # 预留，W1 不用

class ProviderConfig(BaseModel):
    type: Literal["openai_compat", "anthropic", "mock"]
    base_url: str = ""
    api_key_env: str | None = None
    timeout_s: float = 60.0
    flags: dict[str, Any] = Field(default_factory=dict)   # 仅 openai_compat 消费

class ModelConfig(BaseModel):
    provider: str                   # 指向 providers 的 key
    model: str                      # **wire 上的真实模型名**
    tier: str                       # 必须能在 routes 里找到
    capabilities: list[Capability]
    pricing: Pricing
    max_tokens: int | None = None           # Anthropic 必填，registry 传成 default_max_tokens
    reasoning_replay: bool | None = None    # Qwen 3.8 必须回传 / 3.7 不能回传

class RouteConfig(BaseModel):
    primary: str
    fallback: list[str] = Field(default_factory=list)

class Catalog(BaseModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    routes: dict[str, RouteConfig]
    tasks: dict[str, str]

    def resolve(self, task: str) -> list[str]:
        """task → tier → [primary, *fallback]（去重、保序）。"""
```

### 3.2 六条校验规则（`model_validator(mode="after")`）

| # | 规则 | 不校验的后果 |
|---|---|---|
| V1 | `models[*].provider` ∈ `providers` | 运行时 `KeyError`，且发生在第一次真实请求上 |
| V2 | `routes[*].primary` 与 `fallback` ∈ `models` | 路由出空候选 → 报错信息完全对不上根因 |
| V3 | `tasks` 的值 ∈ `routes` | `task="research"` 静默走 default，评测数据全是假的 |
| V4 | 每个 model 的 `tier` ∈ `routes` | 同上 |
| V5 | `pricing.per == 1_000_000` | **成本差 1000 倍**（Qwen 文档常写"元/千 token"） |
| V6 | `capabilities` ⊆ `Capability` 枚举值 | 拼错 `"tool"`（少个 s）不报错，能力校验永远放行 |

V5 值得单独说：这是最容易被忽略、后果最贵的一条。**在 validator 里直接拒绝 `per != 1_000_000`，比在注释里写"注意单位"有效一百倍。**

### 3.3 `Settings`

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local", env_file_encoding="utf-8", extra="ignore"
    )

    models_yaml: Path = Path("models.yaml")
    first_token_timeout_s: float = 10.0
    request_timeout_s: float = 60.0
    max_attempts: int = 3
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0
    log_prompts: bool = False           # 默认关闭，默认不记 prompt 原文

    openai_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    def api_key(self, env_name: str | None) -> str:
        """按 models.yaml 的 api_key_env 取密钥；缺了给可执行报错。"""
```

`api_key()` 用一张映射表把 `OPENAI_API_KEY → openai_api_key` 对应起来（4 条，别用 `getattr` 拼字符串——拼错了会得到 AttributeError 而不是配置错误）。

### 3.4 `models.yaml`（别名与 wire 名分离，这是今天的 ADR-009）

```yaml
providers:
  openai:
    type: openai_compat
    base_url: "https://api.openai.com/v1"
    api_key_env: OPENAI_API_KEY
  deepseek:
    type: openai_compat
    base_url: "https://api.deepseek.com/v1"
    api_key_env: DEEPSEEK_API_KEY
  qwen:
    type: openai_compat
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: DASHSCOPE_API_KEY
    flags:
      send_parallel_tool_calls: true      # Qwen 默认 false，不显式开就静默串行
  anthropic:
    type: anthropic
    base_url: "https://api.anthropic.com"
    api_key_env: ANTHROPIC_API_KEY

models:
  # alias（左）≠ wire 名（model 字段）：Qwen 必须钉日期快照，否则 12 周指标曲线不可比
  qwen3.7-plus:
    provider: qwen
    model: qwen3.7-plus-2026-05-26
    tier: cheap
    capabilities: [stream, tools, parallel_tools]
    pricing: { input: 2.0, output: 8.0, currency: CNY, per: 1000000 }

  deepseek-flash:
    provider: deepseek
    model: deepseek-flash
    tier: cheap
    capabilities: [stream, tools]
    pricing: { input: 0.5, output: 2.0, currency: CNY, per: 1000000 }   # 占位，按官网核对

  gpt-4o-mini:
    provider: openai
    model: gpt-4o-mini
    tier: cheap
    capabilities: [stream, tools, json_mode, parallel_tools]
    pricing: { input: 0.15, output: 0.60, currency: USD, per: 1000000 }

  claude-sonnet-4-6:
    provider: anthropic
    model: claude-sonnet-4-6
    tier: strong
    max_tokens: 4096                    # Anthropic 必填
    capabilities: [stream, tools, parallel_tools, prompt_caching]
    pricing: { input: 3.0, output: 15.0, currency: USD, per: 1000000 }

routes:
  cheap:  { primary: qwen3.7-plus, fallback: [deepseek-flash, gpt-4o-mini] }
  strong: { primary: claude-sonnet-4-6, fallback: [gpt-4o-mini] }

tasks:
  chat: cheap
  research: strong
```

> **价格必须你自己去官网核。** 上面除 Qwen 外的数字都是占位示例——价格填错的代价不是"报错"，而是 W11 成本看板全线偏差，且你无法举证 DoD 里"单任务 ≤ ¥2"。

**混币种是允许的**（DeepSeek/Qwen 只按 CNY 计价），因此不要写"全表同币"的校验；但 `currency` 必须进日志，否则 W11 汇总时无法折算。这条记为 **ADR-009 附注**。

### 3.5 两种"配置错误"要分时炸

| 类型 | 例子 | 何时炸 |
|---|---|---|
| **结构性错误** | 路由引用了不存在的 alias、缺 pricing、`per` 写错 | **启动时**（`create_app` 里加载 catalog 即校验） |
| **凭据缺失** | 某个 provider 的 API key 没配 | **首次使用该 provider 时**（`registry.get`） |

理由：如果把密钥校验放进启动流程，你的离线测试就永远启动不了——这会逼你在 CI 里塞假 key，然后**真的 key 泄露到日志**。

### 3.6 自检

```bash
uv run python -c "
from agent_framework.config import load_catalog
c = load_catalog('packages/agent_framework/src/agent_framework/models.yaml')
print(c.resolve('chat'), c.resolve('research'))
"
```
预期：`['qwen3.7-plus', 'deepseek-flash', 'gpt-4o-mini'] ['claude-sonnet-4-6', 'gpt-4o-mini']`

再故意把某个 `per` 改成 `1000`，确认**加载即报错**，然后改回来。

---

## 4. Step 2（15 min）· `observability.py`

文件：`packages/agent_framework/src/agent_framework/observability.py`

### 4.1 request_id 用 contextvar，不许层层传参

```python
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

def new_request_id() -> str: ...          # "req_" + uuid4().hex[:12]
def current_request_id() -> str: ...      # 读不到就返回 "-"? 不。见下

@contextmanager
def request_scope(request_id: str | None = None) -> Iterator[str]:
    """设置并复位 token；异常也要复位，否则线程池复用会串号。"""
```

`current_request_id()` 读不到时**返回空串而不是抛异常**——日志缺一个字段可以接受，但"打日志时抛异常"会让一个正常请求变成 500。

### 4.2 `RequestLog`：字段固定 17 个（architecture §5）

```python
@dataclass(slots=True)
class RequestLog:
    request_id: str
    task: str
    tier: str
    provider: str
    model: str                 # 记 **alias**，不是 wire 名（见 ADR-009）
    fallback_from: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: float
    currency: str
    latency_ms: float
    first_token_ms: float | None
    retry_count: int
    finish_reason: str
    error_code: str | None
    stream: bool
```

`emit(record)` 就一行：`logging.getLogger("agent_framework.request").info(json.dumps(payload, ensure_ascii=False, default=str))`。
不要现在引入 structlog / opentelemetry，W3 接 Langfuse 时只换 exporter，字段不动。

### 4.3 三个必须精确的定义

| 字段 | 精确定义 | 为什么必须写清 |
|---|---|---|
| `first_token_ms` | **首个 `text_delta` 到达时刻 − 请求开始**；若整轮无 `text_delta`，取首个 `tool_call` | DoD 的 p95 ≤1.5s 全靠它，口径漂移就没法比 |
| `latency_ms` | 请求开始 → 最后一个事件 / 响应返回 | 与 `first_token_ms` 是两个指标，别混用 |
| `retry_count` | **同一 alias 内**的重试次数（不含 fallback 换模型） | 与"降级率"是两个指标，混在一起就无法判断是模型不稳还是链路不稳 |

对思考型模型（DeepSeek / Qwen thinking），`reasoning_delta` **不计入** `first_token_ms`，否则首 token 会虚高到 5s+，指标失真。

### 4.4 成本计算

```python
def compute_cost(pricing: Pricing, usage: Usage) -> float:
    return (usage.input_tokens * pricing.input
            + usage.output_tokens * pricing.output) / pricing.per
```

三条约束：
1. `usage.estimated is True` 时照算，但日志里要能看出来（`prompt_tokens` 旁边没有额外字段，就在 `error_code` 里放 `usage_estimated` 不合适——**改成在 `RequestLog` 后面追加 `usage_estimated: bool = False`**，字段从 17 变 18 是允许的，漏掉才不允许）。
2. 不要用 `total_tokens` 算钱（缓存命中/未命中价格不同，W1 只做 input/output 两档）。
3. 不要四舍五入到 2 位再存——`round(cost, 6)` 起步，成本小到 0.0001 时会被抹成 0。

### 4.5 打码

```python
def mask_secret(value: str | None) -> str:
    """sk-abcdef...wxyz → sk-***wxyz；长度 <8 一律 ***。"""
```

---

## 5. Step 3（25 min）· `llm/registry.py`

```python
class Registry:
    def __init__(self, catalog: Catalog, settings: Settings) -> None: ...

    def get(self, alias: str) -> LLMClient:
        """alias → adapter 实例；首次调用才建，之后复用。"""

    def model_config(self, alias: str) -> ModelConfig: ...

    async def aclose(self) -> None: ...
```

### 5.1 工厂用 `match provider.type`，不用 `if alias == "xxx"`

| `type` | 构造 |
|---|---|
| `openai_compat` | `OpenAICompatClient(name=provider_name, model=cfg.model, base_url=..., api_key=..., flags=..., timeout=...)` |
| `anthropic` | `AnthropicClient(name=..., model=cfg.model, base_url=..., api_key=..., default_max_tokens=cfg.max_tokens, timeout=...)` |
| `mock` | `MockLLMClient(adapter, fixtures_root=..., case=...)` |

`flags` 的合并顺序（后者覆盖前者）：
`PROVIDER_FLAGS[provider_name]` 默认值 → `provider.flags`（yaml）→ 若 `cfg.reasoning_replay is not None` 则覆盖 `reasoning_replay_override`。

**为什么要第三步**：Qwen 同一个 provider 下 3.7 和 3.8 的回传规则相反，只能按模型覆盖——Day3 留的 `reasoning_replay_override` 就是给这里用的。

### 5.2 三个必须注意的点

1. **懒加载**：`get()` 内部才建 httpx client。写在 `__init__` 里会导致"启动时连 4 家"，离线测试和 CI 直接挂。
2. **`name` 用 provider 名，不用 alias**：`adapter.name` 进日志是 `qwen`，而 `model` 记 alias `qwen3.7-plus`——两者都有信息量，混用会丢掉"这条日志是哪个配置项"。
3. **`aclose()` 要遍历所有已创建的实例**：只关 gateway 不管 registry，W2 压测时会出现 `Too many open files`。

### 5.3 自检

```bash
uv run python -c "
from agent_framework.config import load_catalog, Settings
from agent_framework.llm.registry import Registry
r = Registry(load_catalog('.../models.yaml'), Settings(_env_file=None))
print(r.model_config('claude-sonnet-4-6').model)   # wire 名
print(r.model_config('qwen3.7-plus').model)        # 快照名
"
```
**这一步不建实例**（没 key 也能跑），正好验证懒加载写对了。

---

## 6. Step 4（15 min）· `llm/router.py`

```python
class Router(Protocol):
    @property
    def name(self) -> str: ...
    def route(self, task: str) -> list[str]: ...
    """返回候选 alias 有序列表；首个为主，其余为 fallback 链。"""

class TaskRouter:
    """catalog 驱动：task → tier → [primary, *fallback]。"""

class StaticRouter:
    """第二实现：固定候选列表，测试与调试用。"""
```

**第二实现不是凑数**：没有 `StaticRouter`，你就得为测试构造整份 `models.yaml`；而且"抽象只有一个实现"违反架构原则 3。

### 6.1 `route()` 的四条规则

1. task 不在 `tasks` 里 → **回落到 `default`**，并**记一条 warning 日志**（静默回落会让你以为评测跑的是 strong）
2. 候选去重保序（`dict.fromkeys`）
3. 候选为空 → 构造期就该炸（V2 已保证），运行期抛 `BadRequestError`
4. **不要在 router 里做 fallback**——router 只回答"候选是谁"，"绕不绕第二圈"由 gateway 决定

### 6.2 tier 与 alias 的关系

`tasks: chat: cheap` 里的 `cheap` 是 **routes 的 key，不是 model 的 tier 字段**。两个地方都叫 tier 很容易混：model 的 `tier` 字段目前只用于 W11 的成本报表分组，路由只看 `tasks → routes`。**在 docstring 里写清这一点**，否则你三个月后会来改这里。

---

## 7. Step 5（50 min）· `llm/client.py` —— 今天的主菜

### 7.1 ADR-008：这里不用装饰器链，用编排方法

`w1-tasks.md` 原写的是"装饰器链：trace → 计量 → 重试 → 能力校验 → adapter"。**我建议改掉，理由是可执行的技术判断，不是风格偏好**：

- 重试的正确性依赖一个**运行时状态**：`committed`（是否已发出首个事件）。装饰器把 `stream()` 当黑盒，拿不到这个状态；
- 你要在装饰器之间传递 `retry_count` / `first_token_ms` / `fallback_from`，只能靠在 `StreamEvent` 里塞私有字段或搞 contextvar 隐式通道——**这比不拆更难读**；
- async generator 的边界上，`yield` 之后的异常无法被外层装饰器捕获重放（这正是 §3.7 禁止的行为）。

所以：**一个编排方法 + 若干职责单一的私有方法**，并把这条决策写进 `docs/adr/ADR-008.md`（或 architecture §9 补一行）。

### 7.2 数据结构与签名

```python
@dataclass(slots=True)
class ChatCall:
    messages: list[Message]
    task: str = "chat"
    tools: list[ToolSpec] | None = None
    tool_choice: ToolChoice | None = None
    model: str | None = None              # 覆盖路由，仅调试用
    opts: dict[str, Any] = field(default_factory=dict)   # max_tokens / temperature


class LLMGateway:
    def __init__(
        self,
        *,
        registry: Registry,
        router: Router,
        catalog: Catalog,
        settings: Settings,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,   # 测试注入
        clock: Callable[[], float] = time.perf_counter,              # 测试注入
    ) -> None: ...

    def precheck(self, call: ChatCall) -> str:
        """路由 + 取 adapter + 能力校验；返回选中的 alias。

        必须在构造 StreamingResponse **之前**调用，否则能力错误会变成 200 + error 帧。
        """

    async def chat(self, call: ChatCall) -> LLMResponse: ...

    def stream(self, call: ChatCall) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...
```

`sleep` / `clock` 注入不是洁癖：**没有它，你的重试测试会真的睡 0.5 + 1 + 2 秒**，三个用例就 10 秒，W2 压测时会失控。

### 7.3 `chat()` 算法（12 步，按序写）

```text
1. candidates = [call.model] if call.model else router.route(call.task)
2. tier = catalog.tasks.get(call.task, "default")
3. fallback_from = None
4. for alias in candidates:                       # 全链只绕一圈，绝不回头
5.     adapter = registry.get(alias)
6.     self._check_capabilities(alias, adapter, call)   # 硬需求 → CapabilityError
7.     retry_count = 0
8.     for attempt in range(1, settings.max_attempts + 1):
9.         t0 = clock()
10.        try: response = await adapter.chat(...)      # 只调一次，adapter 不重试
11.        except LLMError as exc:
12.            retry_count += 1(仅当真的要重试)
                 if not self._should_retry(exc, attempt): → 记录 error 日志；break 到下一个候选
                 await sleep(self._delay(attempt - 1, exc.retry_after))
                 continue
13.        填充 response.provider / latency_ms
14.        usage = response.usage（estimated 已由 adapter 标记）
15.        cost = compute_cost(pricing, usage)
16.        emit(RequestLog(...fallback_from=fallback_from...))
17.        return response
18.    fallback_from = alias（进入下一轮前设置）
19. 全部候选失败 → raise 最后一个异常（保留 context，别包装成新异常）
```

### 7.4 重试谓词（这段必须自己写对）

```python
def _should_retry(self, exc: LLMError, attempt: int) -> bool:
    return exc.retryable and attempt < self._settings.max_attempts
```

**只依赖 `exc.retryable` 一个字段**。`retryable` 的判定已经在 `errors.py` 里做完了（含 429/402/422/code 白名单），client 里再写一遍 `if exc.status_code == 429` 就是把同一份逻辑放两个地方——W11 你会改一处漏一处。

退避：

```python
def _delay(self, retry_index: int, retry_after: float | None) -> float:
    """retry_after 优先（限流要听上游的），否则指数退避 + 抖动，封顶 backoff_cap_s。"""
    if retry_after is not None:
        return min(retry_after, self._settings.backoff_cap_s) * (1 + random.random() * 0.1)
    base = self._settings.backoff_base_s * (2 ** retry_index)
    return min(base, self._settings.backoff_cap_s) * (0.5 + random.random())
```

抖动乘 `0.5 + rand` 而不是 `1 + rand` 是刻意的：**多个实例同时被限流时，只有向下抖动才能把重试打散**，向上抖动会让尖峰更高。

### 7.5 能力校验：硬需求抛，软策略降级

| 调用方要求 | adapter 声明 | 处理 |
|---|---|---|
| 传了 `tools` | 无 `TOOLS` | **抛 `CapabilityError`**（400） |
| `stream=True` | 无 `STREAM` | **抛 `CapabilityError`** |
| `tool_choice=REQUIRED` 或带 name | 该厂商不支持 | **adapter 已降级为 auto，不抛**（Day3 的 flags 就是干这个的） |
| `tool_choice=REQUIRED, name=x` | 支持 required 但不支持 name | 降级为 required（不指定 name） |

第三、四行写完要提醒自己：**`tool_choice` 是策略意图，不是硬保证**，静默降级会让调用方以为"必定调工具"而写出依赖它的循环——所以降级必须有日志。这就是 Day3 里 `PreparedRequest.notes` 存在的原因（见 §14 可选增强）。

### 7.6 `stream()` 算法（今天最容易写错的地方）

```python
async def stream(self, call: ChatCall) -> AsyncIterator[StreamEvent]:
    for alias in candidates:
        adapter = registry.get(alias)
        for attempt in range(1, max_attempts + 1):
            committed = False
            agen = adapter.stream(...)          # 只是拿到 generator，尚未发请求
            try:
                # ① 首事件超时：只作用在第一次 __anext__
                event = await asyncio.wait_for(agen.__anext__(), first_token_timeout_s)
                committed = True                 # 从这里开始绝不重放
                first_token_ms = (clock() - t0) * 1000
                yield event
                # ② 之后不再有 per-chunk 超时（W1），但保留整体超时
                async for event in agen:
                    yield event
                emit(日志, success); return
            except StopAsyncIteration:
                # 空流：视为上游异常（可重试），不要当成正常结束
                exc = UpstreamServerError("empty stream", provider=adapter.name)
            except asyncio.TimeoutError:
                exc = UpstreamTimeoutError("first token timeout", provider=adapter.name)
            except LLMError as exc:
                pass
            finally:
                await agen.aclose()              # ← 关键，见下

            if committed:
                # 已经发过事件：只能以 error 帧终止，绝不重试（§3.7 红线）
                yield StreamEvent(type=StreamEventType.ERROR, data={...})
                emit(日志, error); return
            if not self._should_retry(exc, attempt):
                break
            await sleep(self._delay(attempt - 1, exc.retry_after))
```

三个必须写对的点：

**① `committed` 的置位时机**
必须在**第一个事件被 `yield` 出去之前**置位（`yield` 之后置位也可以，但 `yield` 是挂起点，中间可能被取消）。判定"是否已提交"用**是否已经 `yield` 过**，不是"是否已收到事件"——收到但还没吐给用户时重试是安全的，吐出去之后就不安全了。

**② `finally: await agen.aclose()`**
放弃一个半开的 httpx 流而不关闭它，连接会留在池里。**一天测不出来，W2 一压测就是 `Too many open files`**。这行是今天最容易漏、后果最延迟的一行。

**③ 空流不等于正常结束**
`StopAsyncIteration` 在第一次 `__anext__` 就抛出，意味着上游 200 但没数据。当正常结束处理，你会给用户返回一个空字符串且 `finish_reason=stop`——**这是最难排查的一类事故**（"模型偶尔不回答"）。

### 7.7 事件透传时不要加工

`stream()` 里**不要**做这些事：
- 累积文本（adapter 已组装，gateway 再攒一份就是第二份真相）
- 给 `tool_call` 补 `id`（adapter 已做，`_common.make_tool_call`）
- 改写 `finish_reason`（`types.normalize_finish_reason` 是唯一映射点）

gateway 在流式路径上只做四件事：**首 token 计时、committed 判定、usage→cost、日志**。

### 7.8 自检

```bash
uv run python -c "
import asyncio
from agent_framework.llm.client import LLMGateway
print([m for m in dir(LLMGateway) if not m.startswith('__')])
"
```
预期能看到 `precheck / chat / stream / aclose`。

---

## 8. Step 6（25 min）· `server/`：`POST /v1/chat`

### 8.1 `schemas.py`：请求进、响应出，都要显式投影

```python
class MessageIn(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None
    name: str | None = None

class ToolSpecIn(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    task: str = "chat"
    tools: list[ToolSpecIn] | None = None
    tool_choice: Literal["auto", "none", "required"] | None = None
    stream: bool = False
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None

class ChatResponse(BaseModel):
    request_id: str
    id: str | None
    provider: str | None
    model: str | None
    text: str | None
    tool_calls: list[ToolCallOut]
    finish_reason: str
    raw_finish_reason: str | None
    usage: UsageOut
    latency_ms: float | None
    retry_count: int = 0
    fallback_from: str | None = None
```

**红线：不许 `return response.model_dump()`。** `LLMResponse.message` 里有 `provider_meta`，里面装着 `reasoning_content` 和 `thinking.signature`——直接 dump 会把模型的思考内容吐给调用方，还顺带暴露内部契约。响应必须是**手写投影**。

一句话记住：**adapter 往内翻译，schema 往外投影，中间任何一层都不许把内部对象整体序列化。**

### 8.2 `sse.py`：帧格式逐字节确定

```python
def encode(event: StreamEvent, *, request_id: str) -> str:
    payload = {
        "type": event.type.value,
        "request_id": request_id,
        **event.model_dump(exclude_none=True, exclude={"type"}),
    }
    return f"event: {event.type.value}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
```

四条规定：

1. 事件名 = `StreamEvent.type`（`text_delta` / `tool_call` / `message_end` / `error` / `reasoning_delta` / `message_start`）
2. `data` 是**单行** JSON——`json.dumps` 默认不换行，安全；**永远不要用 f-string 拼原始文本**（文本里的 `\n` 会直接断帧）
3. 每帧带 `request_id`
4. 帧间空行是 `\n\n`，少一个 `\n` 浏览器会合并两帧

`StreamingResponse` 的三个 header：

```python
headers={
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",   # 以后上 nginx 不加这个，流式会被整段缓冲
}
```

### 8.3 为什么流式错误经常变成 200

`StreamingResponse` **先发 header 再迭代 body**。所以 generator 里抛异常时，状态码 200 已经发出去了，客户端只能收到一个 `event: error` 帧。

→ 因此 `precheck()` 必须在构造 `StreamingResponse` **之前**调用，把"路由失败 / 能力不支持 / 模型不存在"这些**可预知**错误挡在 200 之前：

```python
@router.post("/v1/chat")
async def chat(body: ChatRequest, request: Request) -> Response:
    call = to_call(body)
    gateway: LLMGateway = request.app.state.gateway

    alias = gateway.precheck(call)        # ← 可能抛 CapabilityError / BadRequestError → 400

    if not body.stream:
        resp = await gateway.chat(call)
        return JSONResponse(ChatResponse.from_response(resp, alias).model_dump())

    async def frames() -> AsyncIterator[str]:
        async for event in gateway.stream(call):
            yield encode(event, request_id=current_request_id())

    return StreamingResponse(frames(), media_type="text/event-stream", headers=...)
```

**只有"上游在流中途挂了"才是 200 + error 帧**，这才是流式协议的正确语义。

### 8.4 异常处理器：一张表，别写 if/elif 树

| 异常 | HTTP | code |
|---|---|---|
| `UpstreamAuthError` | 502 | `UPSTREAM_AUTH` |
| `RateLimitError` | 429 | `RATE_LIMITED` |
| `UpstreamTimeoutError` | 504 | `UPSTREAM_TIMEOUT` |
| `BadRequestError` | 400 | `BAD_REQUEST` |
| `ContentFilteredError` | 422 | `CONTENT_FILTERED` |
| `UpstreamServerError` | 502 | `UPSTREAM_ERROR` |
| `CapabilityError` | 400 | `CAPABILITY_UNSUPPORTED` |

```python
_HTTP = {UpstreamAuthError: (502, "UPSTREAM_AUTH"), ...}   # 7 行

@app.exception_handler(LLMError)
async def on_llm_error(request: Request, exc: LLMError) -> JSONResponse:
    status, code = _HTTP[type(exc)]                       # 用精确类型查，别用 isinstance 链
    headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else {}
    return JSONResponse(
        {"code": code, "message": str(exc), "request_id": current_request_id()},
        status_code=status, headers=headers,
    )
```

`type(exc)` 精确查表而不是 `isinstance` 链：`isinstance` 链的顺序会决定行为，而顺序是**隐式契约**——哪天有人给 `RateLimitError` 加个子类，行为就静默变了。

### 8.5 中间件注入 request_id

```python
@app.middleware("http")
async def with_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or new_request_id()
    with request_scope(rid):
        response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response
```

两个细节：
1. **接受调用方传入的 `X-Request-ID`**（便于跨服务串联），但要校验长度与字符集，防止日志注入
2. `request_scope` 要在 `call_next` 外层——放在里层，`StreamingResponse` 的 body 迭代发生在 `call_next` 返回之后，contextvar 已经复位了

### 8.6 `app.py` 的依赖注入

```python
def create_app(gateway: LLMGateway | None = None) -> FastAPI:
    app = FastAPI(...)
    app.state.gateway = gateway or build_gateway(Settings())   # 生产路径
    ...
```

`build_gateway(settings)` 里做：`load_catalog` → `Registry` → `Router` → `LLMGateway`。**启动时只校验结构，不建 adapter**（懒加载，§3.5）。

测试里直接 `create_app(gateway=fake_gateway)`，不需要 monkeypatch 任何模块级单例。

### 8.7 自检

```bash
# 非流式（真机，需 key）
curl -s -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话解释 SSE"}]}' | python3 -m json.tool

# 流式：应看到逐帧 event:/data:
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"从1数到5"}],"stream":true}'

# 错误：故意错 key（用 env 覆盖），应得 502 + UPSTREAM_AUTH，且**不是 401**
OPENAI_API_KEY=sk-invalid uv run agent-server &
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"task":"chat"}'
```

---

## 9. Step 7（15 min）· 测试：8 个用例形状 + A9–A12

现有 136 个用例已覆盖 adapter，今天补的是**链路层**。全部离线，零真实调用。

| 文件 | 必有用例 | 断言要点 |
|---|---|---|
| `test_config.py` | 好 yaml 能加载 / 缺 pricing 报错 / `per=1000` 报错 / 路由引用不存在 alias 报错 | 报错信息里要含**字段路径**，否则排查全靠猜 |
| `test_router.py` | task→候选取序去重 / 未知 task 回落 default / `StaticRouter` 一致 | 回落要有 warning |
| `test_cost.py` | 两档价格算对 / `estimated=True` 照算 / 极小成本不被抹零 | `round(cost,6)` |
| `test_gateway_retry.py` | 429→429→成功 ⇒ 3 次调用、`retry_count=2`、只 1 条 success 日志；`UpstreamAuthError` ⇒ 不重试、直接切 fallback | 用注入的 `sleep` 记录延时序列，断言 [0.5, 1.0] 级别而非真睡 |
| `test_gateway_stream.py` | 首 chunk 前失败 ⇒ 重试；**已有 text_delta 后失败 ⇒ 不重试，只发 error 帧**；空流 ⇒ 当错误；`aclose` 被调用 | 这条是 A10，今天最值钱的测试 |
| `test_server_chat.py` | 非流式 200 + 投影字段；流式帧能 `event/data` 解析回 `StreamEvent`；`CapabilityError` → 400（**不是 200**）；错 key → 502 + `UPSTREAM_AUTH`；响应头有 `X-Request-ID` | 用 `httpx.ASGITransport`，不起服务 |

### 9.1 新增四条 conformance 断言（凑齐 A1–A12）

| # | 断言 | 为什么值得 |
|---|---|---|
| **A9** | 重试归属：`chat()` 遇 2 次 429 后成功 ⇒ **adapter 的 HTTP 请求次数 == 3**，且 adapter 内部无 sleep | 把 ADR-003 从"文档写了"变成"断言守着" |
| **A10** | 已提交后不重试：夹具第 2 个 chunk 后抛错 ⇒ 输出的事件序列里**没有重复文本**、HTTP 只 1 次、最后是 `error` 帧 | 半截回答重放是线上最恶心的 bug，只有这条能防 |
| **A11** | 计量一致：同一夹具流式与非流式的 `cost` 相等；`estimated` 缺失场景两者都标 `True` | 成本曲线要能跨流式/非流式汇总 |
| **A12** | 帧可逆：`encode(event)` 的输出能被自己解析回等价 `StreamEvent` | W3 批量跑评测要单测 SSE 编解码，不能只靠 curl |

A9 建议做**反向验证**：在 adapter 里插一个 `asyncio.sleep(0)` 形状的自重试，看 A9 是否变红。**不会红的断言等于没写**（Day3 的 A8 已用同样方法验过）。

---

## 10. 验收清单（自己跑完再交给我）

```bash
uv run pytest -q | tail -3          # 期望：新增 25+ 用例，全绿，且总耗时 < 10s（证明没真睡）
uv run ruff check . | tail -2
uv run mypy | tail -2
```

| # | 项 | 通过标准 |
|---|---|---|
| 1 | 一键切模型 | 只改 `models.yaml` 的 `routes`，`task=chat` 换到另一家，代码零改动 |
| 2 | 重试归属 | `test_gateway_retry.py` + A9 绿；`grep -rn "sleep\|retry" providers/` 仍为空 |
| 3 | 流式可用 | `curl -N` 逐帧可见；日志里有 `first_token_ms` 数值 |
| 4 | 错误规范 | 7 类错误各返回对应 HTTP+code；上游鉴权 = **502** 不是 401 |
| 5 | 成本可查 | 日志行含 `prompt_tokens / completion_tokens / cost / currency` |
| 6 | 离线可测 | 断网 `pytest` 全绿 |
| 7 | 边界干净 | `server/` 里无 prompt 拼接、无重试；`client.py` 里无厂商字段名 |
| 8 | request_id | 响应头有 `X-Request-ID`，且每条请求日志同一 id |

第 7 项自查命令：

```bash
grep -rn "reasoning_content\|input_schema\|enable_thinking" \
  packages/agent_framework/src/agent_framework/llm/client.py \
  packages/agent_framework/src/agent_framework/server/ | grep -v "^$" || echo "干净"
```

---

## 11. 我会打回的 8 种写法

| # | 写法 | 为什么打回 |
|---|---|---|
| 1 | `client.py` 里 `if exc.status_code == 429` | 重试判定与 `errors.py` 双份真相，W11 必改漏 |
| 2 | `asyncio.sleep` 直接调用，不可注入 | 测试真睡 3 秒；CI 从 3s 变 60s |
| 3 | 流式重试不判 `committed` | 用户看到"半截回答 + 从头重来"（§3.7 红线） |
| 4 | 放弃半开 generator 不 `aclose()` | 连接泄漏，W2 压测才炸 |
| 5 | `return response.model_dump()` | 把 `provider_meta`（含 reasoning_content）漏给调用方 |
| 6 | 日志里打 key 原文 或 默认记 prompt 全文 | 安全事故 + 合规问题；`log_prompts` 必须默认 False |
| 7 | pricing 按 per-token 记 | 成本差 1000 倍，DoD 的"单任务 ≤¥2"直接失效 |
| 8 | `client.py` 里出现 `"deepseek"` / `"qwen"` 字面量 | 厂商差异只允许存在于 `models.yaml` 与 adapter flags |

---

## 12. 时间盒与降级顺序

| 时段 | 任务 |
|---|---|
| 0:00–0:20 | `config.py` + `models.yaml` + V1–V6 校验 |
| 0:20–0:35 | `observability.py`（contextvar + RequestLog + cost） |
| 0:35–1:00 | `registry.py` + `router.py` |
| 1:00–1:40 | `client.py` 的 `chat()`（重试/计量/日志/fallback） |
| 1:40–2:15 | `client.py` 的 `stream()`（**committed + first_token_timeout + aclose**） |
| 2:15–2:40 | `schemas / sse / routes_chat / app` |
| 2:40–3:00 | 测试 + curl 自检 |

超时按此顺序砍：
1. 砍 `StaticRouter`（第二实现）→ 但 **PR 里要留 TODO，别静默省**
2. 砍 fallback（只留 primary）→ 接口保留，`fallback_from` 恒 None
3. 砍 `reasoning_delta` 的日志埋点
4. **绝不砍**：`committed` 判定、`first_token_ms`、`aclose()`、`precheck` 前置

---

## 13. 今天的 ADR 记录（写进 `docs/adr/`）

**ADR-008：客户端层用编排方法，不用装饰器链。**
理由见 §7.1。代价：单个方法较长（约 80–120 行），需要靠私有方法 + 命名控制可读性。

**ADR-009：模型别名（alias）与 wire 模型名分离。**
`models.yaml` 的 key 是 alias（路由、日志、评测集都用它），`model` 字段是真正发给厂商的名字。理由：Qwen 必须钉日期快照、DeepSeek 旧名会静默路由到新模型——别名层让"换模型"不动评测集，让"钉快照"不动业务代码。代价：多了一层映射，排查时要同时看两个名字。

**ADR-009 附注：允许混币种，但 `currency` 必须进日志。** W11 汇总成本时才引入汇率折算，W1 不做。

---

## 14. 可选增强（有余量再做，不做也行）

1. **`PreparedRequest.notes` 贯通**：现在 adapter 算了降级说明（`tool_choice_downgraded` 等）但丢掉了。要贯通需给 `LLMResponse` 加 `notes` 字段、给 `message_end` 事件的 `data` 塞 notes——收益是"降级率"可统计，成本约 40 分钟 + 改 4 个 adapter。**建议 W2 做**。
2. **SSE 心跳**：长思考（DeepSeek 可能 30s 无输出）时每 15s 发 `: ping\n\n`。收益是可穿透代理超时，成本 10 行。
3. **`X-Request-ID` 校验**：长度 ≤64 且仅 `[A-Za-z0-9_-]`。
4. **`/v1/models` 端点**：把 catalog 暴露成列表，W11 做看板时顺手就有了。

---

## 15. 交给我 review 时给这三样

```bash
uv run pytest -q | tail -5
find packages/agent_framework/src/agent_framework -name "*.py" | sort | xargs wc -l | tail -20
grep -rn "sleep\|retry" packages/agent_framework/src/agent_framework/llm/providers/ || echo "adapter 干净"
```

我重点看四件事：
1. `stream()` 的 `committed` 与 `finally: aclose()` 是否都在
2. `chat()` 的 fallback 是否真的只绕一圈（有没有 `while` 回环）
3. `schemas.py` 是否是手写投影（有没有 `model_dump` 直出）
4. `routes_chat.py` 有没有在流式分支里做业务判断
