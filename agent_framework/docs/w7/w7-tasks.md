# W7 执行清单（上下文工程与记忆 · P0，市场趋势项）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> ROADEMAP §4：**Context Engineering 招聘量 +7711%**，是 2026 最热的趋势项。本周把"每次调用喂给模型什么"从随手拼装，变成有预算、有记忆、可量化的工程。
> W7 检查点：**20+ 轮不爆上下文；跨会话可回忆；token 成本下降可量化**。

## 承接面：W1–W6 已交付

| 已有 | 位置 | W7 的接缝 |
|---|---|---|
| `SessionStore`（多轮消息） | `session/`（W2） | 工作记忆建立在它之上，别另建会话存储 |
| `Message.provider_meta`（ADR-007） | `llm/types.py` | 记忆提取/回注走正常内容，不碰 provider_meta |
| `RetrievalPipeline` + `context_assemble`（裁剪/预算局部用） | `retrieval/`（W5/W6） | 本周把"局部预算"升级为全局 token 预算分配器 |
| `RequestLog` 的 tokens/cost/trace | `observability.py` | 成本优化实验的度量来源 |
| `make eval` + `TaskMetrics` + 门禁 | `evals/` | 记忆改动必须过门禁，防止"记多了反而答差" |
| golden set 的 `difficulty=3` 多步案例 | `evals/golden/` | 长会话/跨会话测试样本 |
| 品牌资料库（pgvector） | W5 | 区分"知识库"与"记忆"的边界（本周核心概念题） |

> W7 检查点（ROADEMAP）：capstone 记得品牌调性、历史选题、禁用词；20+ 轮不爆上下文；跨会话可回忆；token 成本下降可量化。

**本周预算**：5 天 × 2–3h ≈ 12–15h。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接符号 | 推进的 DoD |
|---|---|---|---|
| D1 | `context/budget.py`（token 预算分配器） | `RequestLog`、`context_assemble` | 成本、延迟 |
| D2 | `memory/working.py`（会话摘要/tool result 截断） | `SessionStore` | 可用性（20+ 轮不爆） |
| D3 | `memory/long_term.py`（提取/写入/检索/遗忘四阶段） | Retriever 协议 | 质量（跨会话可回忆） |
| D4 | `context/degradation.py`（退化缓解）+ 边界定义 | `order_for_attention` | 质量 |
| D5 | 成本优化实验 + 合流 | `make eval`、history json | 成本（下降可量化） |

**唯一主线**：建立"预算 → 工作记忆 → 长期记忆 → 退化缓解"的 Context 流水线，并用 A/B 证明**token 成本下降但质量不劣化**。

---

## Gate 0 · 开工前（30 min）

- [ ] `make eval` 全绿、门禁生效（W7 要动喂给模型的内容，护栏先立住）
- [ ] 确认 tokenizer 方案：用哪个计数（厂商 tokenizer / tiktoken / 估算），**记进 ADR 别散在代码**
- [ ] 准备一个 20+ 轮的长会话脚本（压测"不爆上下文"）
- [ ] 想清"知识库 vs 记忆"的边界（下面 D4 会给判据），别把两者混成一个东西

---

## Day 1（3h）· Context Engineering 全景 + token 预算分配

**目标**：把"每次调用的 payload"拆成有预算的槽位，超预算有确定的裁剪顺序。**这是本周一切的地基**。

### 你要做的事
- [ ] `context/budget.py`：`ContextBudget`（各槽位上限）+ `allocate()`
- [ ] `context/builder.py`：`ContextBuilder`（按槽位组装最终 messages）
- [ ] `docs/w7/context-budget-spec.md`：槽位定义 + 裁剪优先级
- [ ] 单测：超预算裁剪顺序、各槽位边界、计数一致性

### 引导框架

```python
# context/budget.py —— 槽位化是核心
@dataclass(frozen=True)
class ContextBudget:
    system: int
    retrieved: int        # RAG 命中
    memory: int           # 长期记忆
    history: int          # 工作记忆/对话摘要
    tool_results: int
    reserve_output: int   # 给模型输出留的余量

def allocate(messages: list[Message], budget: ContextBudget,
             tokenizer) -> AllocatedContext:
    """按固定优先级裁剪，返回实际占用与裁剪记录（进日志，可归因）。"""
    ...
```

**算法步骤（今天最容易写错的地方）**
1. 槽位要**预留输出余量** `reserve_output`——不预留会在长输入时把 `max_tokens` 挤到 0，直接 400
2. 裁剪优先级写死并记进 spec：`tool_results 过期 > history 旧轮 > retrieved 低分 > memory`。**先裁最不重要的**，别一刀切
3. 每次裁剪记 `*_trimmed` 日志 + 裁剪 token 数（W3 trace、W4 归因要用）
4. tokenizer 计数必须与实际发送的一致——**计数口径不一致 = 预算形同虚设**（写进 ADR）
5. `ContextBuilder` 输出的是标准 `list[Message]`，通过 `ChatCall.messages` 喂给 `LLMGateway`，不改 gateway

**自检**
- [ ] 构造超长输入 → 按优先级裁剪，`reserve_output` 始终保留
- [ ] `pytest -k budget` 覆盖：恰好等于预算、超 2 倍预算、空槽位
- [ ] 计数与真实发送的 token 差在容差内（用 mock adapter 核对）

### 🔍 交给我的锚点
`ContextBudget`、spec、裁剪日志。我会看：有没有预留输出余量、裁剪优先级是否确定、计数口径是否一致。

---

## Day 2（3h）· 工作记忆：会话摘要 + tool result 截断

**目标**：让 20+ 轮对话不爆上下文。

### 你要做的事
- [ ] `memory/working.py`：`WorkingMemory`（对话摘要、tool result 截断）
- [ ] 滚动摘要：超过阈值把旧轮压成摘要（走 gateway，cheap alias）
- [ ] tool result 截断：大结果只留摘要 + 可重新获取的句柄
- [ ] 单测：摘要触发、截断后可恢复、20 轮不超限

### 引导框架

```python
# memory/working.py
class WorkingMemory:
    def __init__(self, store: SessionStore, gateway: LLMGateway,
                 *, summarize_after: int) -> None: ...
    async def compress(self, session_id: str) -> None:
        """旧轮 -> 摘要 Message；保留最近 N 轮原文。"""
        ...
    def truncate_tool_result(self, text: str, *, handle: str) -> str:
        """超长结果 -> 摘要 + '可调用 X 重新获取完整结果' 的句柄。"""
        ...
```

**算法步骤（今天最容易写错的地方）**
1. 摘要触发阈值按 **token 数**而非轮数（不同轮长度差很大），但 `summarize_after` 可兜底
2. 摘要本身是 `Message`，**要标注它是摘要**（如 metadata），别让模型把摘要当用户原话
3. tool result 截断保留"重新获取句柄"——否则截断了就永久丢失信息（可靠性）
4. 摘要调用走 `cheap` alias + 计量，成本进日志（W11 成本看板）
5. **幂等**：同一会话重复 compress 不重复压缩（对齐 W2/W6 主线）

**自检**
- [ ] 跑一个 20+ 轮脚本，最终 payload 始终在预算内，无 400
- [ ] 压缩后再问第 1 轮的内容，能回忆（摘要有效）
- [ ] 截断的 tool result 能通过句柄重新获取（反向验证信息没丢）

### 🔍 交给我的锚点
WorkingMemory、20 轮脚本输出、摘要成本。我会看：阈值按 token 还是轮数、摘要有没有标注、截断是否可恢复。

---

## Day 3（3h）· 长期记忆四阶段 + 品牌画像

**目标**：实现"提取 / 写入 / 检索 / 遗忘"，让 capstone 跨会话记得品牌调性、历史选题、禁用词。

### 你要做的事
- [ ] `memory/long_term.py`：`LongTermMemory`（四阶段）
- [ ] `memory/extract.py`：从会话提取结构化记忆（品牌调性/禁用词/选题）
- [ ] 记忆存储：复用 `Retriever` 协议（pgvector，别另建库）
- [ ] 遗忘策略：TTL / 置信度衰减 / 显式删除
- [ ] 单测：四阶段闭环、跨会话回忆、遗忘生效

### 引导框架

```python
# memory/long_term.py
@dataclass(frozen=True)
class MemoryItem:
    key: str              # "brand_tone" | "forbidden_words" | "past_topics"
    value: str
    confidence: float
    source_session: str
    created_at: datetime

class LongTermMemory:
    async def extract(self, session_id: str) -> list[MemoryItem]: ...   # 提取
    async def write(self, items: list[MemoryItem]) -> None: ...          # 写入
    async def recall(self, query: str, k: int) -> list[MemoryItem]: ...  # 检索
    async def forget(self, predicate) -> int: ...                        # 遗忘
```

**算法步骤**
1. **提取**在会话结束时触发（或显式），走 cheap LLM 抽取结构化项；**不要在每轮都提取**（成本）
2. **写入**复用 pgvector 的 `Retriever`/upsert——记忆和知识库共用底座但**用不同 namespace/metadata 区分**（边界）
3. **检索**在 `ContextBuilder` 的 `memory` 槽位注入，受 D1 预算约束
4. **遗忘**：置信度低于阈值或超 TTL 的清理；品牌调性这类长期项 TTL 设长，选题类设短
5. 冲突处理：同一 key 的新值如何覆盖旧值（写进 ADR，别静默追加导致越记越乱）

**自检**
- [ ] 会话 A 写入"禁用词：最、第一"→ 新会话 B 生成文案时自觉遵守（跨会话回忆）
- [ ] 低置信度记忆被 `forget` 清理，不污染后续
- [ ] `make eval` 的记忆相关 case 过门禁，且 success_rate 不劣化

### 🔍 交给我的锚点
LongTermMemory 四阶段、命名空间设计、ADR。我会看：**知识库与记忆的边界是否清晰**、提取是否在每轮误触发、冲突处理有没有定义。

---

## Day 4（2h）· 长上下文退化缓解 + 记忆/知识库边界

**目标**：缓解 lost-in-the-middle 在"带记忆的长 payload"下的复发，并把边界讲清（面试高频题）。

### 你要做的事
- [ ] `context/degradation.py`：结合 W6 `order_for_attention` 做全局重排
- [ ] `docs/w7/memory-vs-knowledge.md`：边界定义（判据表）
- [ ] 把记忆/检索/历史的重排统一进 `ContextBuilder`
- [ ] 单测：关键信息在首尾、边界判据覆盖

### 引导框架

**记忆 vs 知识库判据表（写进文档）**
| 维度 | 长期记忆 | 知识库（RAG） |
|---|---|---|
| 内容 | 品牌调性、禁用词、历史选题 | 竞品资料、行业文档、事实 |
| 来源 | 会话中产生、随用户演化 | 外部导入、相对静态 |
| 生命周期 | 会遗忘/更新/冲突 | 版本化、增量同步（W6） |
| 检索 | 按主体/会话回忆 | 按 query 语义+关键词（W5） |
| 存哪 | pgvector 不同 namespace | pgvector 文档 namespace |

**算法步骤**
1. `ContextBuilder` 组装时，把最关键的信息（system + 高置信度记忆 + top 检索）放 **payload 首尾**，次要历史放中间
2. 记忆与知识库用 namespace 隔离，检索互不污染
3. **别把知识库内容写进长期记忆**（反之亦然）——边界破了会导致"事实被当成偏好遗忘"
4. 用 golden 的 `difficulty=3` 案例验证长 payload 下质量不退化

**自检**
- [ ] `difficulty=3` 案例在带记忆的长 payload 下 success_rate 不低于基线
- [ ] 边界判据表每条都有正反例
- [ ] `pytest -k degradation` 全绿

### 🔍 交给我的锚点
`memory-vs-knowledge.md`、ContextBuilder 集成、边界测试。我会看：边界判据是否可操作、namespace 是否真隔离、重排有没有打乱引用（W6 联动）。

---

## Day 5（2.5h）· 成本优化实验 + 合流

**目标**：用 A/B 证明"压缩前后 token 下降可量化"，且质量过门禁。

### 你要做的事
- [ ] `docs/w7/context-cost-ab.md`：压缩前 vs 压缩后的 token/成本/质量对比
- [ ] 用 `make eval` + `RequestLog` 取真实 token 数
- [ ] 更新 `evals/history/YYYY-Www.json`；周五合流仪式
- [ ] 准备一段"20+ 轮不爆上下文"的可演示脚本（W12 demo 素材）

### 引导框架

**A/B 表模板**
| 配置 | 平均 input tokens | 平均 cost | success_rate | judge_score | 20 轮是否爆 |
|---|---|---|---|---|---|
| 无压缩（baseline） |  |  |  |  |  |
| 摘要+截断 |  |  |  |  |  |
| +记忆注入 |  |  |  |  |  |

**算法步骤**
1. 严格单变量，同一 golden、同一 alias；token 数从 `RequestLog` 取（不自己算）
2. 目标：input tokens 明显下降，同时 `success_rate`/`judge_score` 过 W4 门禁（**不能靠牺牲质量换成本**）
3. 若记忆注入让成本回升，如实记录取舍——这正是 Context Engineering 的核心权衡
4. 合流：`make eval` 过门禁 → 写 history json → `docs/log.md`

### 🔍 Gate 7 · 周末验收（全绿才算过 W7）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 预算分配 | 槽位化 + 预留输出余量，裁剪优先级确定 |
| 2 | 20+ 轮不爆 | 长会话脚本全程在预算内，无 400 |
| 3 | 工作记忆 | 摘要标注来源、tool 截断可恢复 |
| 4 | 长期记忆 | 四阶段闭环，跨会话可回忆品牌调性 |
| 5 | 边界清晰 | 记忆/知识库 namespace 隔离，判据可操作 |
| 6 | 质量不退化 | `difficulty=3` 案例过门禁 |
| 7 | 成本可量化 | 压缩前后 token/成本有对比数字 |
| 8 | 合流 | history 接上，端到端仍可跑 |

---

## 本周最容易返工的 6 处

1. **不预留输出余量** → 长输入把 `max_tokens` 挤到 0，直接 400
2. **token 计数口径与实际不一致** → 预算形同虚设
3. **摘要每轮都做** → 成本爆炸；应在会话结束/超阈值触发
4. **记忆与知识库混存无 namespace 隔离** → 事实被当偏好遗忘，边界破
5. **tool result 截断不留句柄** → 信息永久丢失（可靠性）
6. **靠牺牲质量换 token 下降** → 过不了门禁，实验不成立

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 上下文预算 | `context/{budget,builder,degradation}.py`、`docs/w7/context-budget-spec.md` |
| 记忆 | `memory/{working,long_term,extract}.py` |
| 边界文档 | `docs/w7/memory-vs-knowledge.md` |
| 实验 | `docs/w7/context-cost-ab.md` |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W7 条目） |

---

## P3 挂载点（见 ROADEMAP §7）

- **多语言**（P3-2）：记忆项可加 `locale` 维度（品牌调性按语言存），若 W4 已解锁可顺势做；默认档不受影响
- **前端**（P3-1）：W7 的会话/记忆可视化是深入档素材，最佳挂载点 W11 末 / W12
- 第二个 P3 决策点在 **W8 末**，本周先专注 P0 主线
