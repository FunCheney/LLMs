# W8 执行清单（Skills 系统与 MCP · P0，市场 54% 点名）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> ROADEMAP §4：**MCP 54% JD 点名**。本周把 W1 只做"归一化、不执行"的 `ToolCall` 真正跑起来（architecture §7：工具执行留给 W8/W9）。
> 本周也是 **W8 末 P3 决策点**（ROADEMAP §7）——按余量 ≥15h 判断是否解锁一个深入档。

## 承接面：W1–W7 已交付

| 已有 | 位置 | W8 的接缝 |
|---|---|---|
| `ToolCall`（`arguments` 恒 dict，`arguments_error`）+ ADR-002 单次发出 | `llm/types.py` | 本周终于要**消费**它，执行循环从这里开始 |
| conformance A14（调用落在声明 schema 内） | `tests/test_conformance.py` | 工具执行前的校验直接复用，别另写 |
| `Message.provider_meta`（ADR-007） | `llm/types.py` | tool result 回注走正常消息结构 |
| `/v1/draft`、`/v1/chat` | `server/`、`apps/` | 工具能力挂到产品链路，不改框架 provider |
| `SessionStore`、`ContextBuilder` | `session/`、`context/`（W2/W7） | 工具执行结果进 context 受预算约束 |
| `RequestLog` trace + `attribute_failure` | W2/W4 | 工具调用全审计、失败归因"工具层" |
| `make eval` + 门禁 | `evals/` | 接真实工具后仍需过门禁（用录制回放，见下） |

> ⚠️ **ROADEMAP §8 风险项**：搜索/抓取 API 不稳定且有成本 → **W8 起就用录制回放（record/replay）做离线可复现，评测不受外部波动影响**。本周必须落地，否则 CI 天天被外部 API 波动搞红。

> W8 检查点（ROADEMAP）：**新增工具/Skill 不改主流程即可被调用；调用全审计**；capstone 具备真实外部检索能力。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接符号 | 推进的 DoD |
|---|---|---|---|
| D1 | `skills/mcp/`（MCP Client + 协议边界） | `ToolSpec`/`ToolCall` | 能力（外部工具） |
| D2 | 搜索/抓取 MCP Server + 录制回放 | W1 `RecordingTransport` | 可靠性（离线可复现） |
| D3 | `skills/{spec,validator}.py`（Skill **六层**规范 + 完成标准校验） | `ToolSpec` | 工程（可验证） |
| D4 | `skills/{loader,executor,sandbox,visibility}.py`（加载 + 执行边界 + 最小可见面） | Skill 规范 | 安全（越界拦截）、工程（可扩展） |
| D5 | `skills/{guard,action}.py`（**Action Runtime**：提议→校验→授权→审批→执行→验真）+ 合流 | `RequestLog`、A14 | 安全（受控执行） |

**唯一主线**：建立"工具声明 → 受控执行 → 回注 → 审计"闭环，并用 MCP 把外部检索接进来。**先录制回放，再接真 API**——顺序反了评测会被外部拖垮。

> 本周 3 天按《课程大纲对齐评审》§3 抬高验收标准（Skill 六层 + Validator / 沙箱 + 最小可见面 / Action Runtime），
> 均为替换实现方式，仅 D4 净增 0.5h。见 `docs/plan-review-course-outline-gap.md`。

---

## Gate 0 · 开工前（30 min）

- [ ] 重读 `architecture.md` §7（W1 明确不做"工具执行循环"）与 ADR-002（tool_call 单次发出）
- [ ] 复核 conformance A14（工具调用 schema 校验）——本周执行前直接复用
- [ ] 确认一个可抓取目标与一个搜索 provider；**先准备录制样本，别先花真钱**
- [ ] `make eval` 全绿、门禁生效
- [ ] 通读 W1 `tests/fixtures/` 的 mock 回放机制（本周录制回放复用它）

---

## Day 1（2.5h）· MCP 协议深入 + Client 骨架

**目标**：搞清 MCP 的 Server/Client、工具发现、协议边界，落一个最小 Client。**先把边界画对**，别急着封装工具。

### 你要做的事
- [ ] `skills/mcp/client.py`：`MCPClient`（连接 server、列出工具、调用工具）
- [ ] `skills/mcp/types.py`：MCP tool 描述 ↔ 内部 `ToolSpec` 的映射
- [ ] `docs/w8/adr-w8-mcp-boundary.md`：协议边界 + 为什么 MCP 与 Skill 分层
- [ ] 单测：工具发现、映射到 ToolSpec、调用超时处理

### 引导框架

```python
# skills/mcp/client.py —— 协议边界先行
class MCPClient:
    async def connect(self, server: "MCPServer") -> None: ...
    async def list_tools(self) -> list[ToolSpec]:
        """MCP tool 描述 -> 内部 ToolSpec；只做翻译，不执行业务。"""
        ...
    async def call_tool(self, name: str, arguments: dict) -> "ToolResult":
        """带超时；工具错误规范化成内部结构，别把 MCP 错误体直接抛给上层。"""
        ...
```

**算法步骤（今天最容易写错的地方）**
1. `list_tools` 把 MCP 的 tool schema 转成**内部 `ToolSpec`**（W1 类型），上层只认内部类型，**不 import MCP 库**
2. 协议边界写进 ADR：MCP = 外部工具传输协议；Skill = 内部能力封装（指令+脚本+参数契约），两者不是一回事
3. `call_tool` 必须带超时；超时/错误规范化，别让 MCP 的原始错误体泄漏到 `LLMGateway`/对外响应
4. 工具发现是**动态**的：server 加了工具，client 重启后就能列出（对齐检查点"新增工具不改主流程"）

**自检**
- [ ] `pytest -k mcp_client` 覆盖：发现、映射、超时
- [ ] 用一个 fake MCP server 加一个新工具 → `list_tools` 能看到（反向验证动态发现）
- [ ] 业务层 grep 不到 `import mcp`（边界干净）

### 🔍 交给我的锚点
MCPClient、ADR、映射测试。我会看：是否转成内部 ToolSpec、协议边界是否清楚、超时有没有做。

---

## Day 2（3h）· 搜索/抓取 MCP Server + 录制回放

**目标**：把"搜索/抓取"封装成 MCP Server，并落地**录制回放**——本周可靠性的关键。

### 你要做的事
- [ ] `skills/servers/search_server.py`：搜索 MCP Server（工具：search_web）
- [ ] `skills/servers/fetch_server.py`：抓取 MCP Server（工具：fetch_url）
- [ ] `skills/replay.py`：录制回放（复用 W1 `RecordingTransport` 思路，落 `tests/fixtures/tools/`）
- [ ] `docs/w8/adr-w8-record-replay.md`：回放策略 + 数据脱敏
- [ ] 单测：回放命中、录制写入、脱敏

### 引导框架

```python
# skills/replay.py —— 关键：离线档永不打真 API
class RecordedToolTransport:
    def __init__(self, *, fixtures_root: Path, record: bool) -> None: ...
    async def call(self, tool: str, args: dict) -> ToolResult:
        # record=False: 命中 fixture 就回放，未命中 -> 明确报错（不许偷偷打真 API）
        # record=True : 打真 API -> 脱敏 -> 落 fixtures_root
        ...
```

**算法步骤（今天最容易写错的地方）**
1. 回放命中判据用 `(tool, 规范化后的 args)` 的哈希；**args 顺序不同要算同一条**（排序后哈希）
2. `record=False` 时未命中必须**明确报错**，绝不静默打真 API——否则 CI 会偶发联网、指标飘
3. 录制落盘前**脱敏**（URL 里的 key、响应里的敏感字段），复用 W1 Day5 的 `sanitize`
4. fixture 带 `meta.synthetic` 标记（手写 vs 真录制），对齐 W1 ADR-010
5. 真 API 调用成本进 `RequestLog`；回放调用标记 `replayed=True`，成本记 0 但**要在报告里区分**（否则 W11 成本看板失真）

**自检**
- [ ] 断网跑 `make eval` 全绿，工具调用全部走回放
- [ ] 删掉一条 fixture → 明确报错而非联网（反向验证）
- [ ] 录制一条含 key 的响应 → 落盘文件里 key 被打码

### 🔍 交给我的锚点
replay 实现、ADR、断网跑 eval 输出。我会看：未命中是否真报错、脱敏是否到位、回放成本是否与真调用区分。

---

## Day 3（2.5h）· Skill 规范设计

**目标**：定义 Skill = 元数据 + 指令 + 脚本 + 参数契约 + 沙箱执行。**规范先行**，加载器明天才写。

### 你要做的事
- [ ] `skills/spec.py`：`SkillSpec` 数据结构 + 校验（**六层结构**，见下）
- [ ] `docs/w8/skill-spec.md`：Skill 文件布局与字段说明
- [ ] `skills/spec` 校验器：参数契约、必需字段、危险操作声明
- [ ] `skills/validator.py`：**SkillValidator**（"声称完成"必须给证据）
- [ ] 把 2–3 个工具（search/fetch/…）描述成 Skill 草案（先不加载）
- [ ] 单测：规范校验（缺字段、参数不匹配、危险声明）、Validator 拦下无证据的"已完成"

### 引导框架

```python
# skills/spec.py
@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str          # 给 LLM 看，决定何时调用
    instructions: str         # 执行时的系统提示片段
    parameters: dict          # JSON Schema，与 ToolSpec 对齐（复用 A14 校验）
    entrypoint: str           # 脚本/工具入口
    permissions: frozenset[str]   # 需要的权限（W8 D5 白名单用）
    sandbox: str              # 执行沙箱级别（none | subprocess | container）
    dangerous: bool           # 是否高风险动作（W10 HITL 门要用）
    # ---- 六层结构的后三层：只有步骤没有这三层的 Skill，模型会"做完了但其实没做" ----
    mission: str              # 使命边界：这个 Skill 负责什么、明确不负责什么
    principles: tuple[str, ...]   # 领域原则：做这行当的硬规矩（如"引用必须给原文"）
    done_criteria: tuple[Check,]  # 完成标准：可机器判定的证据要求
```

**Skill 六层结构（为什么不能只写"操作步骤"）**

只写步骤的 Skill 会稳定地产出一种失败：模型按步骤跑完，然后宣布"已完成"，而结果里缺少最关键的东西
（没给引用、没覆盖某个维度、数字对不上）。**这不是模型不听话，是 Skill 没定义"什么算完成"。**

| 层 | 字段 | 缺了会怎样 |
|---|---|---|
| 1 身份 | `name` / `description` | 模型不知道何时该用它 |
| 2 使命边界 | `mission` | 拿它去干不该干的活（万能 Skill） |
| 3 领域原则 | `principles` | 步骤对但违反行业硬规矩 |
| 4 执行步骤 | `instructions` | — （唯一大家都会写的一层） |
| 5 工具依赖 | `parameters` / `entrypoint` / `permissions` | 调不动或越权 |
| 6 完成标准 | `done_criteria` | **"已完成"无法被验证** |

```python
# skills/validator.py —— 第 6 层的执行者
@dataclass(frozen=True)
class Check:
    kind: Literal["contains", "cites", "min_items", "regex"]
    target: str          # 在输出里找什么
    hint: str            # 不满足时回喂给模型的话

class SkillValidator:
    def verify(self, skill: SkillSpec, output: str) -> list[str]:
        """返回未满足的标准；非空 -> 不承认完成，把 hint 回喂重试一次。"""
        ...
```

**算法步骤**
1. `parameters` 直接复用 `ToolSpec` 的 JSON Schema 形态，让 A14 校验能覆盖 Skill
2. `permissions` 与 `dangerous` 今天就要声明——**W8 D5 白名单、W10 HITL 都依赖它**，别事后补
3. `sandbox` 分级：只读检索可 `none`，执行脚本至少 `subprocess` 隔离（本周先把契约定好，沙箱实现可最小化）
4. Skill 描述与内部 `ToolSpec` 双向可转（MCP 工具也能包成 Skill）
5. `done_criteria` 只允许**机器可判**的四类（`contains / cites / min_items / regex`）。
   写"质量要高"这种标准等于没写——校验器判不了，就又回到"模型自己说好"
6. Validator 失败**只重试一次**，第二次仍不满足就如实报"未达标 + 缺什么"，**不给通过**。
   这和 W4 judge"解析失败给 None 不补分"是同一条原则：不确定就暴露，不粉饰

**自检**
- [ ] `pytest -k skill_spec` 覆盖：缺字段、参数不匹配、危险声明
- [ ] 2–3 个 Skill 草案通过校验
- [ ] Skill 的 `parameters` 能被 conformance A14 的校验逻辑接受
- [ ] 构造一个"输出里没有引用但自称完成"的结果 → Validator 拦下并回喂 hint
- [ ] 每个 Skill 草案的 `mission` 都写了"不负责什么"（只写"负责什么"等于没边界）

### 🔍 交给我的锚点
SkillSpec 六层、skill-spec.md、Validator、校验测试。我会看：permissions/dangerous 是否今天声明、parameters 是否与 A14 对齐、
`done_criteria` 是否机器可判、mission 有没有写清不负责的部分。

---

## Day 4（2.5h）· Skill 加载器 + 重构工具为 Skill

**目标**：注册/发现/动态加载，并把 D2 的搜索/抓取工具重构成 Skill（对齐检查点"新增不改主流程"）。

### 你要做的事
- [ ] `skills/loader.py`：`SkillRegistry`（发现、加载、热重载、**渐进式加载**）
- [ ] `skills/executor.py`：`SkillExecutor`（校验 → 权限 → 执行 → 回注）
- [ ] `skills/sandbox.py`：**执行边界**（路径 / 命令 / 网络三道白名单 + 超时）
- [ ] `skills/visibility.py`：**最小模型可见面**（按 principal + 任务阶段裁剪工具集）
- [ ] 把 search/fetch 重构成 Skill，主流程只调 `registry` 不硬编码
- [ ] 单测：动态加载新 Skill、执行校验、结果回注、三类越界各被拦一次

### 引导框架

```python
# skills/loader.py
class SkillRegistry:
    def discover(self, paths: list[Path]) -> list[SkillSpec]: ...   # 扫描目录
    def get(self, name: str) -> SkillSpec: ...
    async def reload(self) -> None: ...                              # 热重载
    def describe_all(self) -> list[str]:                             # 只给 description，省 token
    ...

# skills/executor.py
class SkillExecutor:
    async def execute(self, call: ToolCall, principal) -> ToolResult:
        # 1 A14 schema 校验 2 权限白名单 3 沙箱边界 4 执行(带超时/回放) 5 回注为 Message
        ...
```

**最小模型可见面：工具越多，选错越多**

把所有 Skill 一次性塞进 system prompt，会有三个后果：token 涨、模型选错、越权面变大。
`visibility.py` 按两个维度裁剪，规则表写进 `skill-spec.md`：

| 维度 | 裁剪依据 | 例子 |
|---|---|---|
| 主体 | `principal.groups` ∩ `skill.permissions` | 只读访客看不到 `publish_post` |
| 阶段 | 当前任务阶段 | 调研阶段不需要 `publish_post`，写作阶段不需要 `search_web` |

**判据**：任一 `(principal, stage)` 组合下可见工具数 **< 全集的 60%**；且被裁剪掉的工具若被强行调用，
必须返回 `tool_not_visible` 而不是 `permission_denied`——两者的日志含义完全不同，混了就没法归因。

**渐进式加载同理**：`describe_all()` 只返回 `name + description`（约 20 token/个），
模型选中后才把 `instructions`（可能几百 token）注入。Skill 从 3 个涨到 30 个时，这是唯一的省钱办法。

**Sandbox：本地级不是容器，但边界仍然要真**

我们有 `apply_patch`/`run_command` 这类工具，就必须回答"它能碰什么"。完整 Docker 隔离需要容器权限，
CI 里跑不了，所以本周做**本地级三道白名单 + 超时**，把契约立住，容器化留到 W11：

| 边界 | 规则 | 违反时 |
|---|---|---|
| 路径 | 只允许 `repo_root` 子树写入；`~/.ssh`、`.env*`、`.git/config` 硬拒 | `path_denied` + 审计 |
| 命令 | 白名单前缀（`pytest`/`ruff`/`git status`…）；`rm -rf`、`curl | sh`、`sudo` 硬拒 | `command_denied` + 审计 |
| 网络 | 默认拒绝出网；白名单域名放行 | `network_denied` + 审计 |
| 超时 | 硬超时 + 输出上限（防 `yes` 刷爆日志） | kill 进程树，返回 `timeout` |

**关键**：三类拒绝都要进审计日志并且**可被 eval 断言**——否则"边界"只是代码里一段没人测过的 if。

**算法步骤（今天最容易写错的地方）**
1. `execute` 顺序固定：**A14 schema 校验 → 权限 → 沙箱边界 → 执行 → 回注**，任何一步失败都要有确定结果，不能静默
2. 工具结果回注成 `Message`（tool 角色 + `tool_call_id`），复用 W1 消息结构；**大结果走 W7 截断+句柄**
3. 动态加载：新增一个 Skill 文件 → `reload()` 后可被 `list` 到并调用，**主流程零改动**（检查点）
4. 执行走 D2 的回放传输（离线）/真 API（在线），成本与 replay 标记进日志
5. 别把执行循环写进 `LLMGateway`（ADR-003 重试只在 client；执行编排是产品/技能层的事）
6. 沙箱检查**必须在副作用发生之前**（写文件前、起进程前），不能"先跑再判断"
7. 命令白名单用**前缀匹配 + 参数级黑名单**双保险：光前缀匹配挡不住 `git push --force`

**自检**
- [ ] 新放一个 Skill 文件 → reload 后可调用，主流程代码未改（反向验证检查点）
- [ ] 参数不合 schema → 执行前被 A14 拦下，返回确定错误
- [ ] 大结果被截断且留句柄
- [ ] 三个越界构造各被拦一次：写 `../outside.txt`、`rm -rf /tmp/x`、访问白名单外域名
- [ ] 可见工具数 < 全集的 60%，且被裁剪工具调用返回 `tool_not_visible`
- [ ] 30 个 Skill 时 `describe_all()` 的 token 总量 < 全量 `instructions` 的 20%

### 🔍 交给我的锚点
SkillRegistry/Executor、sandbox/visibility、检查点验证、日志。我会看：执行顺序是否固定、动态加载是否真零改动、
三类边界是否在副作用之前、可见面裁剪日志是否区分 `not_visible` 与 `denied`。

---

## Day 5（2.5h）· Action Runtime：把高风险动作收进受控闭环 + 合流

**目标**：从"调工具时顺手判个权限"升级为**受控执行闭环**；工具调用全审计，满足 DoD 安全项；
周五合流 + W8 末 P3 决策。

### 你要做的事
- [ ] `skills/guard.py`：`PermissionGuard`（白名单 + 风险分级 + 二次确认钩子）
- [ ] `skills/action.py`：**`ActionRuntime`**（提议 → 校验 → 授权 → 审批 → 执行 → 验真）
- [ ] 审计日志：每次调用的工具/参数摘要/主体/结果/耗时进 `RequestLog`
- [ ] 与 W10 HITL 预留缝：`dangerous` 动作走审批钩子（本周占位 `pending`）
- [ ] 三个 badcase 全部拦住：重复请求 / 参数篡改 / 接口假成功
- [ ] 更新 `evals/history/YYYY-Www.json`、`docs/log.md`；周五合流仪式
- [ ] **W8 末 P3 决策**：余量 ≥15h 才解锁一个深入档（ROI 顺序，见 ROADEMAP §7）

### 引导框架

**为什么 `ToolCall` 不能直接等于"执行"**

`ToolCall` 是**模型的意图**，不是事实。模型可能在同一个任务里重复发起同一笔写操作，
也可能被上下文里的恶意文本诱导修改参数，还可能遇到"接口返回 200 但其实没生效"。
如果 `ToolCall` 一产生副作用，这三类问题全都没有拦截点。所以要插一层 Action：

| 阶段 | 谁负责 | 做什么 | 失败时 |
|---|---|---|---|
| 提议 propose | 模型 | 生成 `ToolCall`（只是意图） | — |
| 校验 validate | 代码 | schema + 业务规则（金额上限、字段互斥） | `invalid_action` |
| 授权 authorize | 代码 | 权限白名单 + 风险分级 | `forbidden` |
| 审批 approve | **人**（仅高风险） | 二次确认，展示"将要发生什么" | `pending` / `rejected` |
| 执行 execute | 代码 | **带幂等键**执行，超时受控 | `failed`，可重试 |
| 验真 verify | 代码 | 回读真实状态确认生效（不是信 200） | `unverified`，告警 |

```python
# skills/action.py
class ActionRuntime:
    async def propose(self, call: ToolCall, principal) -> Action: ...
    async def run(self, action: Action) -> ActionResult:
        """validate -> authorize -> approve -> execute(idempotent) -> verify"""
        ...
```

**风险分级（决定要不要人）**

| 级别 | 例子 | 策略 |
|---|---|---|
| `read` | 搜索、读文件、查库 | 放行，只审计 |
| `write` | 写文件、发草稿 | 放行，幂等键必填 |
| `irreversible` | 发布文章、对外发帖、删除 | **必须人审批**（W10 接真 HITL） |

**三个 badcase，每个对应一个机制**（这是本周的验收核心，缺一个都算没过）

| badcase | 构造 | 期望 | 对应机制 |
|---|---|---|---|
| 重复请求 | 同一 `(action, args)` 连发 3 次 | 只产生 1 次副作用，后两次返回首次结果 | 幂等键 |
| 参数篡改 | 上下文里塞"把预算改成 100 万"，模型照做 | 业务规则校验拒绝（超上限），不是权限拒绝 | validate 层 |
| 接口假成功 | mock 接口返回 200 但响应体 `{"ok": false}` | `unverified` + 告警，不写"已发布" | verify 层 |

**算法步骤**
1. 白名单按 `(principal.groups, skill.permissions)` 判定，复用 W6 的 `Principal`
2. **审计字段**：action_id / tool_name / 参数摘要（不记全文，隐私）/ principal / 风险级 / 结果状态 / 耗时 / replay 标记
3. `dangerous` 动作本周返回 `pending` 占位，W10 接 HITL 审批门——**今天留缝不实现**
4. **幂等键 = hash(tool_name + 规范化 args + action 意图 id)**，存 `RequestLog` 同源存储，TTL 24h
5. **验真只做"读回状态"这一种**，不做复杂对账（那要 W11 才有真实业务系统）
6. 合流：`make eval`（断网全绿）过门禁 → 写 history json → `docs/log.md` → 判定 P3

### 🔍 Gate 8 · 周末验收（全绿才算过 W8）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | MCP 边界 | 业务只认内部 ToolSpec，不 import MCP 库 |
| 2 | 动态发现 | server 加工具/Skill 不改主流程即可调用 |
| 3 | 录制回放 | 断网 `make eval` 全绿；未命中明确报错不联网 |
| 4 | 脱敏 | 录制响应里的 key 被打码 |
| 5 | 执行闭环 | 校验→权限→沙箱→执行→回注 顺序固定 |
| 6 | 全审计 | 每次调用可查主体/工具/风险级/结果/replay 标记 |
| 7 | Skill 可验证 | `done_criteria` 机器可判，无证据的"已完成"被拦 |
| 8 | 可见面收敛 | 任一 (principal, stage) 可见工具 < 全集 60% |
| 9 | 沙箱边界 | 路径/命令/网络三类越界各被拦一次，且在副作用之前 |
| 10 | Action Runtime | 重复请求 / 参数篡改 / 接口假成功 三个 badcase 全拦住 |
| 11 | 质量守门 | 接真实工具后 success_rate 过门禁 |
| 12 | P3 已判定 | 按余量明确解锁/不解锁，写进 log |

---

## 本周最容易返工的 7 处

1. **先接真 API 再做回放** → CI 被外部波动拖垮，评测不可复现（ROADEMAP §8 风险项）
2. **回放未命中偷偷联网** → 指标偶发飘，排查极难
3. **回放成本与真调用不分** → W11 成本看板失真
4. **Skill 今天不声明 permissions/dangerous** → W8 D5 白名单、W10 HITL 都要返工
5. **执行循环写进 `LLMGateway`** → 违反 ADR-003/职责分层
6. **MCP 原始错误/工具描述泄漏到上层** → 边界破，对外响应带厂商细节
7. **`ToolCall` 直接当执行指令** → 幂等/篡改/假成功三件事全都没有拦截点；中间必须隔一层 Action
8. **沙箱检查放在副作用之后** → 文件已经写了、进程已经起了才判断越界，等于只有日志没有边界

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| MCP | `skills/mcp/{client,types}.py`、`skills/servers/{search,fetch}_server.py` |
| 回放 | `skills/replay.py`、`tests/fixtures/tools/` |
| Skill 规范/加载 | `skills/{spec,validator,loader,executor}.py`、`docs/w8/skill-spec.md` |
| 执行边界 | `skills/{sandbox,visibility}.py` |
| 受控执行 | `skills/{guard,action}.py`（Action Runtime + 三个 badcase 用例） |
| ADR | `docs/w8/adr-w8-{mcp-boundary,record-replay,action-runtime}.md` |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W8 条目 + P3 决策） |

---

## P3 挂载点（W8 末 = 第二个决策点，见 ROADEMAP §7）

- **A2A 协议**（P3-3）：默认档 = 读规范 + 一次跨进程 handoff demo（0.5 天），最佳挂载点就是本周/W9 多智能体周
- 只看余量（领先/落后天数 × 3h）：≥15h 才解锁**一个**深入档，顺序 A2A 轻量 > 多语言 > 前端极简
- 无论是否解锁，**默认档一律完成**；本周先把录制回放 + 权限审计做扎实
