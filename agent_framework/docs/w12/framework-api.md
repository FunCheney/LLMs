# 框架公共接口与依赖注入（W12 D1）

> 一句话：**公共面是扫出来的，不是想出来的**。`evals/scripts/api_audit.py` 扫产品真实
> import，顶层只导出「起一个服务、发一条 chat」所需的 13 个符号，其余按能力子包访问；
> `apps/` 已无一处穿透到实现模块。门禁：`uv run python evals/scripts/api_audit.py --check`。

## 1. 判据：什么叫「真实使用过」

ROADEMAP §8 的反过度设计纪律是「只提取被真实使用过的抽象」。这句话要能验证，得先定义口径。
`api_audit.py` 把使用方分成五档：

| 域 | 路径 | 用途 |
|---|---|---|
| `product` | `apps/growth_agent/src` | **只有这一档够格进公共 API** |
| `product_test` | `apps/growth_agent/tests` | 产品测试在用——导不出来会让测试在 collection 阶段炸 |
| `evals` | `evals/scripts` | 评测脚本（内部探针，不承诺稳定） |
| `framework` | `packages/agent_framework/src` | 框架内部互相引用（**只用来判定死代码**） |
| `framework_test` | `packages/agent_framework/tests` | 框架自测 |

分五档而不是两档，是因为三种「在用」的性质完全不同：

- **产品源码在用** → 这是契约。改了就是 breaking change，要有 ADR。
- **产品测试在用** → 也是契约，但它的失败形态更隐蔽：不是运行时报错，而是
  `ImportError: cannot import name 'AUTH_INVALID'` 让整个模块 collection 失败，看起来像
  "测试文件写错了"。第一版审计只查 `product` 域，就漏了 `server` 的 5 个鉴权符号。
- **框架内部在用** → 不代表它是公共面。`server/routes_chat.py` 只被 `server/app.py` 用，
  它是内部实现；但它**也不是死代码**。没有这一档，审计会建议删掉 `/v1/chat` 的实现。

## 2. 当前数字（可复现）

```
uv run python evals/scripts/api_audit.py
```

| 指标 | 值 |
|---|---|
| 框架模块总数 | **118** |
| 产品在用（product + product_test） | **54 个模块 / 208 个符号** |
| 私有符号泄漏 | **0** |
| 任何使用域都不引用的模块（空壳） | **0** |
| 子包未导出产品在用的符号 | **0** |
| `__init__.py` re-export 映射条数 | 426 |

> 口径说明：208 含 **re-export 双视角**——`ContextBuilder` 既记在定义处
> `context.builder`，也记在公共入口 `context`。两者都要知道：前者决定改代码影响谁，
> 后者决定第三方从哪 import。

## 3. 公共面

### 3.1 顶层 13 个（`from agent_framework import ...`）

顶层是**起步面**，不是全集索引。它只回答一件事：起一个服务、发一条 chat 需要什么。

| 符号 | 为什么在顶层 |
|---|---|
| `__version__` | `/health` 要报版本 |
| `build_gateway` | 造链路（读 `models.yaml` + 装配 router/breaker/cache/limiter） |
| `Settings` / `ConfigError` | 配置是每个服务的第一行 |
| `ChatCall` / `ChatOutcome` / `LLMGateway` | 发一条 chat |
| `Message` / `MessageRole` / `ToolCall` / `Usage` | 构造消息与读用量 |
| `configure_logging` / `current_request_id` | 上线的第一件事 |

**不在顶层的**：`retrieval` / `memory` / `skills` / `planner` / `multi_agent` / `runtime` /
`context` / `guardrails` / `knowledge` 九个能力包。测试 `test_subpackages_are_not_re_exported_at_top_level`
守着这条，`test_top_level_is_deliberately_small` 守着规模（≤20）。

为什么不做全集索引：179→208 个符号全塞进来，等于承诺一个谁也不会去读的契约，
真正的兼容性风险反而被稀释。被问"你的公共 API 是什么"，答案不该是"全部"。

### 3.2 能力子包（9 个）

```
agent_framework
├── llm           LLM 适配 / client / 错误投影 / 路由
├── context       Context 预算 / 信任级 / 组装
├── guardrails    注入防护 / 输出校验 / HITL / 行为监控
├── knowledge     RAG 引用 / 治理 / 增量索引
├── retrieval     召回 → 融合 → 精排 → ACL → 组装
├── memory        工作记忆 / 长期记忆 / 提取
├── skills        Skill 规范 / 执行 / 沙箱 / MCP（servers/ 是独立子包）
├── session       会话存储（内存 / Redis）
├── server        HTTP / SSE / 鉴权 / 错误投影
├── planner       规划（W9，实验性）
├── multi_agent   多 Agent（W9，实验性）
├── runtime       执行模型 / 续跑 / Flag / 资产版本（W11，实验性）
└── resilience    熔断 / 降级 / 语义缓存 / 幂等（W11，实验性）
```

后四者（planner / multi_agent / runtime / resilience）**产品没在用**，是 W9/W11 的实验与
可靠性产物。它们不进顶层、不承诺稳定，但在仓库里保留——它们是稀缺能力①②③的证据。

## 4. 依赖方向

```
apps/growth_agent  ──┐
                     ├──>  agent_framework.<子包>  ──>  agent_framework.llm.types（叶子）
evals/scripts  ──────┘             │
                                   └──> 第三方（httpx / pydantic / fastapi）

        永远不会有：agent_framework ──> apps/*
```

`test_framework_never_imports_product`（AST 扫 `packages/` 下有无 `import growth_agent`）
守着 ADR-004。这条失守意味着框架绑死在产品上，"脚手架给第三方用"直接不成立。

`llm/providers/*` 是唯一允许 import httpx 的地方；产品侧 `grep -rn "llm.providers" apps/`
为空，说明适配层没有泄漏到业务。

## 5. D1 改了什么（8 处，全部有测试守着）

这些不是"整理代码"，每一处都对应一个具体的失败模式。

| # | 问题 | 动作 | 失败模式（不改会怎样） |
|---|---|---|---|
| 1 | 产品 `from agent_framework.memory.extract import _EXPLICIT_MARKERS` | 新增公开 `is_explicit_instruction(text)`，产品改用它 | 框架想改正则就得先跟产品商量；私有符号一旦被外部用，它就再也不能改 |
| 2 | `memory` 未导出 `LongTermMemory`，产品只能穿深层路径 | 补 `__all__` | 第三方照文档写 `from agent_framework.memory import LongTermMemory` 直接 ImportError |
| 3 | `guardrails` 导出了 `isolate`（上锁）却没导出 `unwrap`（开锁） | 补 `unwrap` | **W11 INC-2 的根因**：接口面不对称会放大"隔离了却忘了解封皮"这个错 |
| 4 | `llm` 未导出 `ChatCall/ChatOutcome/LLMGateway` | 惰性导出（`__getattr__`） | `client.py` 的内部重构变成没人守着的 breaking change |
| 5 | `llm` 未导出 `error_code_for/http_status_for` | 补 `__all__` | 写错误处理的产品只能穿深层路径 |
| 6 | `server` 未导出鉴权 5 个符号 | 补 `__all__` | 产品测试 collection 阶段炸 |
| 7 | `memory_demo.py` 躺在 `src/` 下（谁都不用） | 移到 `examples/memory_policy_demo.py` | 被打进 wheel，成为公共面上一个没人用的文件 |
| 8 | 产品 24 个文件穿透实现模块 import | 归一化到公共入口 | 实现模块无法重构 |

### 5.1 两个坑

**坑一：re-export 穿透不做，会误删产品在用的代码。**
产品写 `from agent_framework.context import ContextBuilder`，而它定义在 `context/builder.py`。
不做穿透，审计会报 `context/builder.py` 没人用——而 D1 的动作恰恰是"删掉没人用的抽象"。
`build_reexport_map()` 迭代到不动点（`agent_framework` → `llm` → `llm.client` 这种两级
re-export 也要追到），把使用归因到**定义模块**。

**坑二：`__init__` 的导入代价 = 它导入的所有东西。**
直接 `from .client import ChatCall` 会造出真环：

```
config.py ── from .llm.base import Capability
             └─ 触发 llm/__init__.py
                └─ from .client import ...
                   └─ client.py: from ..config import Catalog   ← config 正卡在半路
```

根因不是 import 顺序，而是"`import llm.base` 顺手把整个 client 拉起来了"：`config` 只需要
一个枚举，却背上了 client 的全部依赖。用 PEP 562 的模块级 `__getattr__` 解决——
属性第一次被访问才加载。同理顶层也惰性，于是：

```
import agent_framework.config   →  不加载 llm.client，不加载 httpx，不加载 bootstrap
from agent_framework import ChatCall  →  照常可用
```

`test_importing_config_does_not_pull_httpx` 和 `test_importing_llm_base_does_not_pull_client`
在**子进程**里跑，避免测试进程的 `sys.modules` 污染结论。这不是洁癖：脚手架生成的工具里
"读配置校验一下"是最常见的第一个动作。

## 6. 依赖注入现状

### 6.1 注入点（全部是构造参数，无模块级全局单例）

| 入口 | 可注入项 | 默认行为 |
|---|---|---|
| `create_app(gateway=...)` | `gateway` / `session_store` / `limiter` / `task_queue` / `exporter` | 自建 |
| `build_gateway(settings=..., catalog=...)` | `settings` / `catalog` | 读 `models.yaml` |
| `LLMGateway(registry, router, catalog, settings, *, breaker, cache, sleep, clock)` | 8 个 | 空 / 真实时钟 |
| `RetrievalPipeline(retriever, reranker, ...)` | 后端实现 | 无（必须传） |
| `LongTermMemory(store, embedder, ...)` | 存储与 embedding | 无（必须传） |
| `SemanticCache(embedder, clock, ...)` | embedding / 时钟 | 哈希 embedding |

框架**零 ML 依赖**：embedding、reranker、向量库都以协议/回调注入，实现由 `apps/` 装配。
这是"框架包不被 torch 绑死"的前提。

### 6.2 唯一的模块级全局槽：`tracing._exporter`

```
tracing._exporter: TraceExporter        ← 模块级可变状态
   ↑ 唯一写入点：create_app() 里的 configure_exporter(resolved_exporter)
   ↑ 唯一读取点：span() 内的 _safe_export()
```

**为什么允许它存在**：`span()` 是自由函数，调用点分布在 `llm/client.py`、
`server/middleware.py`、`retrieval/pipeline.py` 等十几处（`async with span(...)`）。
改成注入就得给每个调用点传 exporter，等于让 trace 变成业务代码的负担——
而 trace 是观测手段，不该出现在业务签名里。

**它为什么不算"没管住的全局"**：注入点唯一（`create_app`）、在应用边界、且
谁建谁关（`lifespan` 里 `aclose`）。测试用 `configure_exporter(dummy)` 注入替身。

这条判断可以推广：**自由函数式的横切关注点允许一个全局槽，前提是注入点唯一且在边界**。
`gateway` / `store` / `retriever` 不满足这个前提（它们有明确的调用方），所以必须注入。

## 7. 已知边界

静态扫描看不见的四件事，别把"审计通过"读成"绝对正确"：

1. **import 了但没调用**。`from agent_framework.retrieval import Retriever` 只为了写类型注解，
   审计当成"在用"。它确实是公共契约（类型标注也是契约），但别据此推断这条路径被测过。
2. **字符串动态 import**。`importlib.import_module("agent_framework.llm.providers.openai_compat")`
   （`registry.py` 就这么干）扫不到。所以 providers 那层的"产品未使用"是**假**的。
3. **`getattr` 式符号访问**。`getattr(router, "route_for", None)`（`LLMGateway._plan`）扫不到。
4. **符号存在性不校验**。审计只读 import 语句，不看符号是否真的存在——
   产品 import 一个不存在的名字，它会安静通过，只有 mypy / collection 会炸。
   这次就漏过一次（`from agent_framework.skills import FetchServer`），是 mypy 抓到的。

## 8. 怎么用

```bash
# 全量报告
uv run python evals/scripts/api_audit.py

# CI 断言（无泄漏 / 无空壳 / 子包导出完整）
uv run python evals/scripts/api_audit.py --check

# 机器可读
uv run python evals/scripts/api_audit.py --json
```

契约测试：`packages/agent_framework/tests/test_public_api.py`（11 条），
其中 `test_audit_script_passes` 直接跑 `--check`——让审计脚本和 CI 用**同一份规则**，
而不是脚本报一套、测试守另一套。
