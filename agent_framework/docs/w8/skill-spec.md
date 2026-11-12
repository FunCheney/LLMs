# Skill 规范（六层结构 + 可见面规则）

> 实现：`packages/agent_framework/src/agent_framework/skills/spec.py`
> 唯一实现处：`validate()`（结构硬错误）、`lint()`（可疑但不报错）、
> `validate_arguments()`（A14 参数校验，工具与 Skill 共用）。
> 本文只描述**契约**；数字全部来自 `evals/scripts/tool_boundary_check.py` 实跑。

---

## 1. 为什么是六层

只写「操作步骤」的 Skill 会稳定地产出一种失败：模型按步骤跑完，然后宣布「已完成」，
而结果里缺了最关键的东西——没给引用、没覆盖某个维度、数字对不上。

**这不是模型不听话，是 Skill 没定义「什么算完成」。**

| 层 | 字段 | 缺了会怎样 | 校验方式 |
|---|---|---|---|
| 1 身份 | `name` / `description` | 模型不知道何时该用它 | `validate()` 硬错误 |
| 2 使命边界 | `mission` | 拿它去干不该干的活（万能 Skill） | `validate()` 非空 + `lint()` 要求写「不负责」 |
| 3 领域原则 | `principles` | 步骤对但违反行业硬规矩 | `lint()` |
| 4 执行步骤 | `instructions` | — （唯一大家都会写的一层） | `validate()` 非空 |
| 5 工具依赖 | `parameters` / `entrypoint` / `permissions` / `sandbox` | 调不动或越权 | `validate()` 交叉一致性 |
| 6 完成标准 | `done_criteria` / `verify` | **「已完成」无法被验证** | 运行期 `SkillValidator` / `ActionRuntime` |

第 2 层为什么必须写「不负责什么」：只写「负责检索行业数据」，
模型会把「顺便判断资料可不可信」也当成它的活。边界要靠否定句划，不是靠肯定句暗示。

第 6 层为什么不能让模型自评：W4 的 judge 已经证明过——让模型判自己，
它给的分数和实际质量的相关性很低。所以 `done_criteria` **只允许四种机器可判的类型**。

---

## 2. 字段契约

### 2.1 身份与边界

```python
name: str          # ^[a-zA-Z0-9_-]{1,64}$，同时也是合法工具名（直接喂给模型）
description: str   # ≤ 300 字符；渐进式加载只注入这一段，太贵
mission: str       # 必含「不负责/不用于/不做/不包含」之一（lint 检查）
principles: tuple[str, ...]
instructions: str  # 执行步骤，模型选中后才注入
```

`description` 硬限 300 字符的理由：渐进式加载把 `name + description` 注入给模型，
其余部分等模型选中后再注入。这一段的长度 × Skill 数量 = 每轮固定开销。
实测 6 个内置 Skill：`describe` 531 字符 vs 全量 `instructions` 2778 字符 = **19.1%**。

`to_tool_spec()` 会把 `mission` 里的否定半句拼进工具描述：

```
检索公开网页，返回标题/链接/摘要。（不负责：不负责抓取正文（用 web_fetch）、不负责判断资料可信度）
```

因为模型判断「该不该用这个工具」时，最需要的就是这一句。

### 2.2 工具依赖

```python
parameters: dict[str, Any]        # JSON Schema 子集，与 ToolSpec 同形态
entrypoint: str                   # "server.tool" 形式，如 "workspace.write_file"
permissions: frozenset[str]       # read | net | write | shell | publish
sandbox: str                      # none | subprocess | container
stages: frozenset[str]            # research | draft | publish | maintenance
dangerous: bool                   # 高风险动作（W10 HITL 门的输入）
```

**交叉一致性全是硬错误**，因为一个字段写错的 Skill 一旦加载成功，
它会以一个看起来正常的方式失败：

| 规则 | 为什么 |
|---|---|
| 有 `entrypoint` 必须声明 `permissions` | 否则绕过白名单 |
| 有 `permissions` 必须有 `entrypoint` | 声明了能力却没有落地处 |
| `write`/`shell` 权限不允许 `sandbox=none` | 有副作用就必须有隔离 |
| 有 `entrypoint` 必须声明 `parameters` | 空 schema 在严格模式下会拒掉一切参数 |
| `dangerous=True` 必须有 `done_criteria` | 没有验真标准的「已发布」没法验 |
| `verify` 只允许 `write`/`publish` 类 Skill | 只读动作没有「生效」可验 |
| 未知字段直接报错 | 拼错的键（`permisions`）被静默忽略 = 给了一个没有权限限制的 Skill |

### 2.3 完成标准

```python
CHECK_KINDS = ("contains", "cites", "min_items", "regex")

@dataclass(frozen=True)
class Check:
    kind: Literal["contains", "cites", "min_items", "regex"]
    target: str = ""    # 在输出里找什么
    hint: str = ""      # 不满足时回喂给模型的话
    count: int = 1      # 仅 min_items 用
```

`verify`（D5 的验真层）：

```python
entrypoint: str                          # "workspace.get_post"
id_field: str = "post_id"
status_field: str = "status"
ok_status: tuple[str, ...] = ("published", "ok", "done")
```

`verify` 与 `done_criteria` 的区别——这是本周最容易混的一对：

| | 判什么 | 在哪一层 | 失败后果 |
|---|---|---|---|
| `done_criteria` | **模型输出的文本**里有没有证据 | `SkillValidator`（D3） | 不承认完成，回喂 hint 重试一次 |
| `verify` | **外部系统的真实状态**有没有变 | `ActionRuntime`（D5） | `unverified` + 告警 |

前者防「模型自称完成」，后者防「接口返回 200 但没生效」。两个都要有，
只有前者挡不住假成功，只有后者挡不住空话。

---

## 3. 参数校验：A14 只有一个实现

`validate_arguments(parameters, arguments) -> list[str]` 是工具调用与 Skill 调用
**共用的唯一实现**（`tests/test_conformance.py` 的 A14 也改成了复用它的）。

刻意只支持子集：`type` / `properties` / `required` / `additionalProperties` /
`enum` / `minimum` / `maximum` / `minLength` / `maxLength`。

不引 `jsonschema` 的理由与 `redact` 不做全文扫描一样：Skill 的参数契约是我们自己写的，
用到的模式就是这些；为了没用到的一半规范引一个依赖，换来的是「校验行为和线上不一致」的风险。

两个刻意的语义选择：

- **`additionalProperties` 缺省视为允许**（JSON Schema 默认语义）。想严格必须显式写
  `false`——这条默认值不能替调用方改，改了会让所有手写 schema 的 Skill
  突然全部拒绝合法参数。
- **违反项全部返回**而不是抛第一个。A14 的价值是「一次看清模型怎么跑偏的」，
  逐个报错会让调用方重试三次才凑齐一份完整报告。

---

## 4. 最小模型可见面（规则表）

工具越多，选错越多。`visibility.py` 按两个维度裁剪，纯函数、无状态。

| 维度 | 裁剪依据 | 例子 | 隐藏原因码 |
|---|---|---|---|
| 主体 | 粗粒度权限（只有 `publish` / `shell` 参与） | 只读访客看不到 `publish_post` | `principal` |
| 阶段 | `skill.stages` 是否含当前阶段 | 调研阶段看不到 `publish_post` | `stage` |

**为什么主体维度用粗粒度**：如果把 `read`/`net` 也纳入裁剪，
`anonymous` 会看不到任何检索工具，而检索恰恰是匿名也能做的。
把 `write` 也纳入，会让「写作阶段只给编辑器」变成日常配置负担。
所以只有两项真正危险的能力（对外发布、跑命令）进主体维度，其余靠阶段收敛。

**为什么 `tool_not_visible` 与 `permission_denied` 必须分开**：

| 码 | 含义 | 值班含义 |
|---|---|---|
| `tool_not_visible` | 这个工具本来就不该出现在该主体的视野里 | **安全事件**：模型/注入知道了它不该知道的东西 |
| `permission_denied` | 工具可见，但该主体无权调用 | **运营事件**：权限配置需要调整 |

混在一起，安全事件会被当成配置问题处理——这正是可见面收敛想做的那件事被抵消的方式。

### 4.1 实跑矩阵（6 个内置 Skill 全集）

| stage | anonymous | viewer | editor | publisher | ops |
|---|---|---|---|---|---|
| research | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) |
| draft | 3/6 (50%) | 3/6 (50%) | 3/6 (50%) | 3/6 (50%) | 3/6 (50%) |
| publish | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 3/6 (50%) | 2/6 (33%) |
| maintenance | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 3/6 (50%) |

判据「任一 `(principal, stage)` 可见数 < 全集的 60%」：实测最大 **50%**，成立。

收敛靠两件事配合：`stages` 把 Skill 钉在它该出现的阶段，主体维度再裁掉
发布与命令。单靠任一维度都不够——只靠阶段，`editor` 在 `publish` 阶段
仍看得见 `publish_post`；只靠主体，`publisher` 在调研阶段仍看得见发布工具。

### 4.2 渐进式加载

`describe_all()` 只返回 `name + description`，`instructions` 等模型选中后再注入。

实测：531 / 2778 = **19.1%**（判据 < 20%）。Skill 从 3 个涨到 30 个时，
这是唯一还能把固定开销压住的办法。

---

## 5. 已知缺口（不粉饰）

### 5.1 `contains` 会被否定句误命中

`Check(kind="contains", target="结论")` 在输出「本文**没有**结论」上会通过。
`tests/test_skill_validator.py::test_contains_can_be_fooled_by_negation` 显式记录了这个行为。

为什么先不修：

- 想做对得引入句法分析或让模型再判一次，前者是假精确，后者是让被验的人给自己打分；
- 真正需要「必须有结论」的场景，`min_items`（至少 3 条支撑数据）或 `cites`
  （必须有可点击来源）更难被空话满足，实际效果比 `contains` 好；
- 所以 `contains` 的定位是**廉价的存在性提示**，不是质量判据。这条写在这里，
  是为了在它某天漏掉一个空话答案时，没人会说「没想到」。

### 5.2 `regex` 只能判形态，判不了语义

`post_id` 用 `regex: ^[0-9a-f]{8}$` 能挡「没给 ID」，挡不住「ID 是编的」。
编造的 ID 由 `verify` 层回读拦下（读不到 → `unverified`），两层各管一半。

### 5.3 `min_items` 的计数方式对列表型输出更准

`min_items` 依赖输出里的结构性标记（列表项、URL），对纯散文输出会低估。
这不是 bug，是「用散文交付的 Skill 本来就该换一种 `done_criteria`」。

---

## 6. 内置库（6 个，`library.py`）

| Skill | 权限 | 阶段 | 风险 | 存在理由 |
|---|---|---|---|---|
| `web_search` | read, net | research, draft | read | 检索样本 |
| `web_fetch` | read, net | research | read | 抓取样本（正文不可信） |
| `write_brief` | read, write | draft | write | 有副作用但可重做（幂等键的价值） |
| `publish_post` | read, write, publish | publish | irreversible | 审批门 + 回读验真 |
| `run_checks` | read, shell | maintenance | write | 沙箱三道边界的样本 |
| `archive_assets` | read, write | maintenance | write | 与 `write_brief` 同权限、不同阶段（可见面收敛的样本） |

`builtin_skills()` 在**构造时**就逐个 `validate()`：内置库自己先过一遍规范，
不然「规范」只是写给外部作者看的。`lint()` 的结果在 `/v1/tools/{name}` 里可查，
不进启动期错误——它们可疑但不该拦住服务启动。
