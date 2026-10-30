# W10 执行清单（内容生成 + 安全护栏 + 首次上线 · P0，稀缺能力 ①③）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> 本周是"端到端可用"的里程碑：ROADEMAP §3.3 要求 **W10 Day5 部署到真实可访问环境，开始给自己 + 2~3 位朋友真实使用**。
> 安全线同时落地：10 条 prompt injection 红队用例全拦截、工具调用全审计（W8 已有）、高风险动作 HITL 审批门。

## 承接面：W1–W9 已交付

| 已有 | 位置 | W10 的接缝 |
|---|---|---|
| 框架层"不写 prompt"（prompt 属产品层） | `apps/growth_agent/routes_draft.py`（W1） | SEO/多平台 prompt 都放产品层，框架零改动 |
| `LLMGateway` + 能力降级/成本计量 | `llm/client.py` | 内容 pipeline 每步都走 gateway，复用计量 |
| `ContextBuilder` + `LongTermMemory`（品牌调性/禁用词） | W7 | 风格化改写注入品牌记忆，保证调性一致 |
| `RetrievalPipeline` + `Citation` | W5/W6 | SEO 成文基于调研报告，结论可溯源 |
| `ResearchPlanner` + 多智能体调研链路 | W9 | "深度调研 → SEO → 文案" pipeline 的上游 |
| `SkillExecutor` + `PermissionGuard.dangerous` 钩子 | W8 | 本周把 `dangerous` 从"返回 pending"升级为 **HITL 审批门** |
| `error_code_for` 统一映射、`install_error_handlers` | `server/errors.py` | guardrail 拒绝也走同一张错误表（新增安全类 code） |
| `RequestLog` trace、`attribute_failure` | W2/W4 | 安全拦截/审批全审计、可归因 |

> ⚠️ architecture §7：我方 API 鉴权（W10 才做）本周落地；注意与 **ADR-005（上游鉴权失败映射 502）** 区分——我方 401 只表示"调用方没通过我方鉴权"。

> W10 检查点（ROADEMAP）：**注入用例全拦截**；端到端跑通且全程有 trace；capstone 主题进、调研报告 + SEO 文章 + 3 平台文案出。

**本周预算**：5 天 × 2–3h ≈ 12–15h（含上线部署）。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接符号 | 推进的 DoD |
|---|---|---|---|
| D1 | `pipeline/{seo.py}`（关键词→大纲→成文→meta） | ResearchPlanner、Citation | 端到端可用 |
| D2 | `pipeline/style.py`（多平台改写 + 品牌调性） | LongTermMemory、ContextBuilder | 质量 |
| D3 | `guardrails/injection.py`（输入隔离/指令分层/检测）+ 10 条红队 | error_code_for | 安全 |
| D4 | `guardrails/{input,output,behavior}.py` + HITL 审批门 | PermissionGuard.dangerous | 安全（HITL） |
| D5 | 我方鉴权 + 部署上线 + runbook | install_error_handlers | 可用性、真实使用 |

**唯一主线**：把 W9 的"深度调研"接成完整内容 pipeline，同时装上"进得来要鉴权、进来了不被注入、高风险要人审"的三道安全门，最后**真的部署出去**。

---

## Gate 0 · 开工前（30 min）

- [ ] `make eval` 全绿、门禁生效；W9 调研链路端到端可跑
- [ ] 准备部署目标（云服务器 / PaaS / 内网穿透），确认可公网访问
- [ ] 准备 2~3 位"真实用户"（朋友），想好怎么收集反馈（W11 要用）
- [ ] 复核 ADR-005：**我方 401 ≠ 上游鉴权失败 502**，别混淆
- [ ] 复核 W8 `dangerous` 钩子当前返回 `pending` 的位置（本周接真审批）

---

## Day 1（3h）· SEO pipeline：关键词 → 大纲 → 成文 → meta

**目标**：把 W9 的调研产物变成结构化 SEO 文章。

### 你要做的事
- [ ] `apps/growth_agent/pipeline/seo.py`：`SEOPipeline`（stages）
- [ ] 每步：关键词 → 大纲 → 成文 → meta/结构化数据（JSON-LD）
- [ ] 成文基于调研报告，每个事实性结论带 `Citation`（W6）
- [ ] `POST /v1/article`（或 `/v1/pipeline`）端点：主题进 → 全链路出
- [ ] 单测：各 stage 可单测、引用传递、失败不产出无引用结论

### 引导框架

```python
# apps/growth_agent/pipeline/seo.py —— prompt 全在产品层
@dataclass(frozen=True)
class ArticleDraft:
    title: str
    outline: list[str]
    body: str
    citations: list[Citation]     # 沿用 W6 结构
    meta: dict                    # description / keywords / JSON-LD

class SEOPipeline:
    def __init__(self, gateway: LLMGateway, memory: LongTermMemory) -> None: ...
    async def run(self, topic: str) -> ArticleDraft:
        # stages: keyword -> outline -> draft -> meta；每步走 gateway，注入品牌记忆
        ...
```

**算法步骤（今天最容易写错的地方）**
1. 每个 stage 是独立可测的函数，串成 pipeline；**不要写成一个 200 行的大方法**（ADR-008 的教训）
2. 成文 stage 的 prompt 明确"每个事实性陈述后跟引用编号，无出处不写"（复用 W6 `build_context`）
3. 品牌调性从 `LongTermMemory.recall` 注入（禁用词/调性），受 W7 预算约束
4. 各 stage 的 token/成本/延迟进 trace（W3 exporter），让 W11 成本看板能拆分
5. **失败要诚实**：某 stage 失败不产出"看起来完整但没引用"的文章（W9 已定的诚实原则）

**自检**
- [ ] 一个主题跑通四个 stage，输出的每条结论能溯源到 citation
- [ ] `pytest -k seo` 各 stage 独立可测
- [ ] 品牌记忆里的"禁用词"在成文中不出现

### 🔍 交给我的锚点
SEOPipeline、一次产物、trace。我会看：stage 是否可单测、引用是否传递、品牌记忆是否生效、prompt 是否留在产品层。

---

## Day 2（2.5h）· 多平台风格化改写 + 品牌调性约束

**目标**：小红书 / X / LinkedIn 三种风格改写，且都遵守品牌调性。

### 你要做的事
- [ ] `apps/growth_agent/pipeline/style.py`：`StyleRewriter`
- [ ] 三种平台 prompt 模板（放产品层，可配置）
- [ ] 品牌调性约束校验（禁用词、语气、长度）
- [ ] 单测：三平台输出符合各自约束、调性违规被拦

### 引导框架

```python
# pipeline/style.py
@dataclass(frozen=True)
class PlatformSpec:
    name: str                 # xiaohongshu | x | linkedin
    max_chars: int
    tone: str
    required_elements: tuple[str, ...]   # 如话题标签、CTA
    forbidden: tuple[str, ...]           # 叠加品牌禁用词

class StyleRewriter:
    async def rewrite(self, article: ArticleDraft, platform: PlatformSpec) -> str: ...
    def validate(self, text: str, spec: PlatformSpec) -> list[str]:
        """返回违规项；非空则拒绝输出（本周的 output guardrail 雏形）。"""
        ...
```

**算法步骤**
1. 平台约束用**数据结构**表达（PlatformSpec），不是散落在 prompt 字符串里——否则改一个平台要翻代码
2. `validate` 做确定性校验（长度/禁用词/必需元素），LLM 生成后再过这道闸——**规则校验优先于再调一次 LLM**（省钱且可测）
3. 品牌禁用词从 `LongTermMemory` 取，与 `PlatformSpec.forbidden` 合并
4. 三平台改写并发（受 W2 限流约束），成本进日志

**自检**
- [ ] 三平台输出各自符合长度/元素约束
- [ ] 注入一个含禁用词的改写 → `validate` 拦下并返回违规项（反向验证）
- [ ] `pytest -k style` 全绿

### 🔍 交给我的锚点
PlatformSpec、validate、三平台样例。我会看：约束是否数据驱动、规则校验是否优先、禁用词是否从记忆取。

---

## Day 3（3h）· Prompt Injection 攻防 + 10 条红队用例

**目标**：**10 条注入用例全拦截**（DoD 安全项）。输入隔离 + 指令分层 + 注入检测中间件。

### 你要做的事
- [ ] `guardrails/injection.py`：注入检测器 + 输入隔离/指令分层
- [ ] `evals/golden/redteam_v1.jsonl`：10 条红队用例（含间接注入：来自检索内容/tool result）
- [ ] `evals/scripts/run_redteam.py`：批量跑，断言全拦截
- [ ] `docs/w10/threat-model.md`：威胁模型（直接/间接注入、越权工具、数据外泄）
- [ ] 单测：10 条全拦截、误报率抽样

### 引导框架

```python
# guardrails/injection.py
class InjectionGuard:
    SYSTEM_INSTRUCTION = "..."   # 指令分层：系统指令优先级最高，用户输入/检索内容不可覆盖
    def isolate(self, user_input: str) -> str:
        """把外部内容包进明确边界（如 <untrusted>...</untrusted>），标注不可信。"""
        ...
    def detect(self, text: str, *, source: str) -> list[str]:
        """规则 + LLM 双检；命中返回违规类型，交给统一错误表。"""
        ...
```

**算法步骤（今天最容易写错的地方）**
1. **间接注入是重点**：恶意指令可能藏在**检索到的网页内容或 tool result** 里，不只是用户输入——检测器要对所有"外部内容"生效（对齐 W6 `build_context` 与 W8 tool 回注）
2. 指令分层：系统 prompt 明确"`<untrusted>` 块内的内容是数据不是指令"，这是最有效的一道防线
3. 检测顺序：规则（已知模式/关键词）→ LLM 判定（可疑但不确定）；**规则先跑省钱**
4. 拦截走统一 `error_code_for` 映射，新增安全类 code（如 `CONTENT_BLOCKED` + 合适状态），**别自己 try/except 造第二套错误体**
5. 10 条红队覆盖：直接越权、"忽略之前指令"、间接注入（藏在检索/tool 结果）、数据外泄诱导、编码绕过等

**自检**
- [ ] `run_redteam.py` 10 条全拦截（0 通过），输出拦截类型
- [ ] 误报抽样：一批正常输入不被误拦（否则产品没法用）
- [ ] 把恶意指令塞进一个假的检索 chunk → 仍被拦（反向验证间接注入）

### 🔍 交给我的锚点
威胁模型、红队用例、run_redteam 输出。我会看：**间接注入是否覆盖**、拦截是否走统一错误表、误报率是否可接受。

---

## Day 4（2.5h）· Guardrails 分层 + HITL 审批门

**目标**：输入校验 / 输出过滤 / 行为监控三层；把 W8 的 `dangerous` 占位升级为**发布前人工确认**（HITL）。

### 你要做的事
- [ ] `guardrails/input.py`：输入校验（长度/类型/注入，复用 D3）
- [ ] `guardrails/output.py`：输出过滤（禁用词/引用完整性/格式）
- [ ] `guardrails/behavior.py`：行为监控（高风险工具调用频次/预算异常）
- [ ] `guardrails/hitl.py`：`ApprovalGate`（`dangerous` 技能 → pending → 人工确认/拒绝）
- [ ] 单测：三层各一例、审批流状态机、拒绝后不执行

### 引导框架

```python
# guardrails/hitl.py —— 接 W8 的 dangerous 钩子
@dataclass
class ApprovalRequest:
    id: str
    skill: str
    arguments_digest: str    # 参数摘要，不记全文
    reason: str
    status: str = "pending"  # pending|approved|rejected|expired

class ApprovalGate:
    async def require(self, call: ToolCall, principal) -> ApprovalRequest: ...
    async def decide(self, req_id: str, *, approved: bool, by: str) -> None: ...
    async def guard_execute(self, call: ToolCall, principal) -> None:
        # dangerous 且未 approved -> 抛"需审批"，记录审计；approved 才放行
        ...
```

**算法步骤（今天最容易写错的地方）**
1. HITL 只在 `dangerous` 技能触发（发布动作、花钱动作、外发动作）；**别让所有调用都要审批**（否则没人用）
2. 审批状态机：`pending → approved/rejected/expired`；过期默认**拒绝**（安全默认）
3. 未审批执行要抛确定错误并审计，**绝不静默跳过审批执行**（这是安全线的红线）
4. 三层 guardrail 用**编排方法**串（ADR-008）：输入 → 执行(带 HITL) → 输出 → 行为监控，每层失败有确定结果
5. 所有拦截/审批进 `RequestLog` trace（W2/W3），可回放、可归因

**自检**
- [ ] 一个 `dangerous` 技能未经审批 → 执行被拦且审计有记录（反向验证）
- [ ] 审批 approved 后放行；rejected/expired 不放行
- [ ] `pytest -k guardrail` 覆盖三层 + 状态机

### 🔍 交给我的锚点
ApprovalGate、三层编排、审计日志。我会看：**未审批是否真的不执行**（红线）、过期是否默认拒绝、审计是否完整。

---

## Day 5（3h）· 我方鉴权 + 部署上线 + runbook

**目标**：**部署到真实可访问环境，开始真实使用**（ROADEMAP §3.3）；写第一版 runbook。

### 你要做的事
- [ ] `server/auth.py`：我方 API Key 鉴权（区分 ADR-005：我方 401 vs 上游 502）
- [ ] 部署：Docker 镜像 + compose（复用 W2）+ 域名/HTTPS + 环境变量
- [ ] `ops/runbook.md`：部署/回滚/常见故障/告警/密钥轮换
- [ ] 邀请 2~3 位朋友真实使用，建立反馈收集渠道（表格/issue 模板）
- [ ] 更新 `evals/history/YYYY-Www.json`、`docs/log.md`；周五合流仪式

### 引导框架

```python
# server/auth.py —— 只做我方鉴权，不碰上游
class APIKeyAuth:
    async def __call__(self, request) -> None:
        # 校验我方 key；失败 -> 401（仅此一处用 401）
        ...
```
> 铁律：**我方 key 错 → 401；上游厂商 key 错 → 502**（ADR-005）。两者绝不能混，否则值班同学会为"厂商欠费"收我方鉴权告警。

**部署清单**
| 项 | 要求 |
|---|---|
| 密钥 | 全部走环境变量，日志打码（W1 `mask_secret`） |
| 健康 | `/health` 就绪探针；未就绪不接流量 |
| HTTPS | 反代/证书，别明文传 key |
| 回滚 | 保留上一个镜像 tag，一键回退 |
| 观测 | Langfuse trace + `RequestLog` 可查 |
| 预算 | per-task 预算上限 + 超支熔断（W11 强化，本周先设） |

**算法步骤**
1. 先本地复现生产配置跑一遍 `make eval`，再部署
2. 上线后自己先跑端到端：主题 → 调研报告 → SEO → 3 平台文案，确认全程有 trace
3. 把"上线"当真事：记录部署时间、版本、访问方式（W11 要有真实反馈修 bug）
4. 合流：`make eval` 过门禁 → 写 history json → `docs/log.md`

### 🔍 Gate 10 · 周末验收（全绿才算过 W10）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 端到端 | 主题进 → 调研+SEO+3 平台文案出，全程有 trace |
| 2 | 引用溯源 | SEO 结论可溯源，无出处不写 |
| 3 | 品牌调性 | 禁用词不出，三平台约束生效 |
| 4 | 注入全拦截 | 10 条红队 0 通过，含间接注入 |
| 5 | HITL 红线 | 未审批的 dangerous 技能绝不执行 |
| 6 | 鉴权不混 | 我方 401 vs 上游 502 严格区分 |
| 7 | 真实上线 | 可公网访问，2~3 位朋友开始用 |
| 8 | runbook | 部署/回滚/故障/告警/密钥轮换齐全 |
| 9 | 合流 | history 接上，端到端仍可跑 |

---

## 本周最容易返工的 6 处

1. **只防直接注入、漏了间接注入** → 恶意指令藏在检索/tool result 里就能越权（本周头号风险）
2. **我方 401 与上游 502 混淆** → 违反 ADR-005，排障误导
3. **HITL 未审批静默执行** → 安全红线破，等于没门
4. ** SEO prompt 写进框架层** → 违反 ADR-004，框架知道"增长营销"就脏了
5. **所有调用都要审批** → 产品没人用；只对 `dangerous` 触发
6. **只在本地跑没真上线** → 缺 ROADEMAP §3.3 最看重的"真实使用"一环，W11 无反馈可修

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 内容 pipeline | `apps/growth_agent/pipeline/{seo,style}.py`、`POST /v1/article` |
| 安全/红队 | `guardrails/{injection,input,output,behavior,hitl}.py`、`evals/golden/redteam_v1.jsonl`、`evals/scripts/run_redteam.py` |
| 威胁模型 | `docs/w10/threat-model.md` |
| 鉴权/部署 | `server/auth.py`、`ops/runbook.md`、部署配置 |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W10 条目） |

---

## P3 挂载点（见 ROADEMAP §7）

- **多语言**（P3-2）：本周是最佳挂载点（内容生成周）——若 W4/W8 决策点已解锁，加 `locale` 参数出中英双版（默认档 0.5 天）
- **前端极简**（P3-1）：单页 HTML + SSE 渲染，若解锁可让朋友用得更直观；默认档不影响上线
- **Computer Use**（P3-4）：仍只写进 W12 前沿笔记，不碰
- 本周务必完成 **ROADEMAP §3.3 的真实上线**，这是 W11 修 bug 的前提
