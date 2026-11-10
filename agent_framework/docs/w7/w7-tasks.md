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
| D1 | `context/{schema,budget}.py`（ContextItem + 预算分配器 + Debug Report） | `RequestLog`、`context_assemble` | 成本、延迟、可归因 |
| D2 | `memory/working.py`（会话摘要/tool result 截断） | `SessionStore` | 可用性（20+ 轮不爆） |
| D3 | `memory/long_term.py`（提取/写入/检索/遗忘四阶段 + 写入幂等 + 污染校验） | Retriever 协议 | 质量、安全 |
| D4 | `context/degradation.py`（退化缓解 + Drift 检测）+ 边界定义 | `order_for_attention` | 质量 |
| D5 | 成本优化实验 + **记忆两档对照实验** + 合流 | `make eval`、history json | 成本、质量（下降可量化） |

**唯一主线**：建立"预算 → 工作记忆 → 长期记忆 → 退化缓解"的 Context 流水线，并用两组对照实验证明
**token 成本下降但质量不劣化**、**记忆确实有用且无害**。

> 本周有 4 处按《课程大纲对齐评审》§3 抬高验收标准（Context Schema / Debug Report / 写入幂等+污染校验 / Drift），
> 均为"把 D1–D4 的实现方式换掉"，不新增 Day。见 `docs/plan-review-course-outline-gap.md`。

---

## Gate 0 · 开工前（30 min）

- [x] `make eval` 全绿、门禁生效（W7 要动喂给模型的内容，护栏先立住）
- [x] 确认 tokenizer 方案：用哪个计数（厂商 tokenizer / tiktoken / 估算），**记进 ADR 别散在代码**
- [x] 准备一个 20+ 轮的长会话脚本（压测"不爆上下文"）
- [x] 想清"知识库 vs 记忆"的边界（下面 D4 会给判据），别把两者混成一个东西

---

## Day 1（3h）· Context Engineering 全景 + token 预算分配

**目标**：把"每次调用的 payload"拆成有预算的槽位，超预算有确定的裁剪顺序。**这是本周一切的地基**。

### 你要做的事
- [x] `context/schema.py`：`ContextItem`（**来源 / 版本 / 权威度 / 信任级 / 生命周期**五件套）
- [x] `context/budget.py`：`ContextBudget`（各槽位上限）+ `allocate()`
- [x] `context/builder.py`：`ContextBuilder`（按槽位组装最终 messages）
- [x] `ContextDebugReport`：任意一次调用能回答"这条为什么没进上下文"
- [x] `docs/w7/context-budget-spec.md`：槽位定义 + 裁剪优先级 + 六类排除原因
- [x] 单测：超预算裁剪顺序、各槽位边界、计数一致性、排除原因可解释

### 引导框架

```python
# context/schema.py —— 预算分配的单位不是"字符串"，是 ContextItem
@dataclass(frozen=True)
class ContextItem:
    content: str
    kind: Literal["system", "retrieved", "memory", "history", "tool_result"]
    source: str           # 哪个文档 / 哪次会话 / 哪个工具
    version: str | None   # 文档版本、prompt 版本；没有就 None，别造假
    authority: float      # 权威度 0-1：官方>自有>未知站点（W6 质量分复用）
    trust: Literal["system", "user", "external"]   # 信任级，external 永不可当指令
    expires_at: datetime | None

@dataclass(frozen=True)
class ContextBudget:
    system: int
    retrieved: int        # RAG 命中
    memory: int           # 长期记忆
    history: int          # 工作记忆/对话摘要
    tool_results: int
    reserve_output: int   # 给模型输出留的余量

def allocate(items: list[ContextItem], budget: ContextBudget,
             tokenizer) -> AllocatedContext:
    """按固定优先级裁剪，返回实际占用 + 每个被排除项的排除原因（进日志，可归因）。"""
    ...
```

**Context Debug Report（为什么这条没进上下文）**

用户问"资料库里明明有这段，为什么答案没用上"时，如果只能回答"不知道"，Context 工程就等于没做。
`allocate()` 把每个**候选**的最终去向记下来，六类原因必须互斥且穷尽：

| 排除原因 | 含义 | 该去改哪里 |
|---|---|---|
| `not_retrieved` | 根本没被召回 | W5 检索（chunk/改写/RRF） |
| `filtered_permission` | ACL 过滤掉了 | W6 ACL，**不是 bug** |
| `filtered_expired` | 过期被降权/剔除 | W6 知识治理 |
| `ranked_out` | 召回但排到第 8 位，只取前 5 | W5 重排 |
| `budget_squeezed` | 排上了但预算被别的槽位挤掉 | 本文件预算配置 |
| `compressed_away` | 压缩/摘要时被合并掉了 | D2 摘要策略 |

**算法步骤（今天最容易写错的地方）**
1. 槽位要**预留输出余量** `reserve_output`——不预留会在长输入时把 `max_tokens` 挤到 0，直接 400
2. 裁剪优先级写死并记进 spec：`tool_results 过期 > history 旧轮 > retrieved 低分 > memory`。**先裁最不重要的**，别一刀切
3. 每次裁剪记 `*_trimmed` 日志 + 裁剪 token 数（W3 trace、W4 归因要用）
4. tokenizer 计数必须与实际发送的一致——**计数口径不一致 = 预算形同虚设**（写进 ADR）
5. `ContextBuilder` 输出的是标准 `list[Message]`，通过 `ChatCall.messages` 喂给 `LLMGateway`，不改 gateway
6. **信任级先于相关性**：`trust="external"` 的内容（网页、竞品资料、工具返回）无论多相关，都不能进 system 槽位——
   这是 W10 注入防护在 Context 层的同一条原则，提前在这里立住，W10 只需补检测器
7. 预算分配以 `ContextItem` 为单位、排除原因顺手记录——**所以 Debug Report 是免费的**，不要另建一套埋点

**自检**
- [x] 构造超长输入 → 按优先级裁剪，`reserve_output` 始终保留
- [x] `pytest -k budget` 覆盖：恰好等于预算、超 2 倍预算、空槽位
- [x] 计数与真实发送的 token 差在容差内（用 mock adapter 核对）
- [x] 拿一条"资料库有但没被用上"的真实 query，能报出六类原因中的**唯一**一个
- [x] `trust="external"` 的内容尝试进 system 槽位时被拒（构造恶意网页文本验证）

### 🔍 交给我的锚点
`ContextItem`、`ContextBudget`、spec、裁剪日志、Debug Report 样例。我会看：有没有预留输出余量、裁剪优先级是否确定、计数口径是否一致、排除原因是否互斥穷尽。

---

## Day 2（3h）· 工作记忆：会话摘要 + tool result 截断

**目标**：让 20+ 轮对话不爆上下文。

### 你要做的事
- [x] `memory/working.py`：`WorkingMemory`（对话摘要、tool result 截断）
- [x] 滚动摘要：超过阈值把旧轮压成摘要（走 gateway，cheap alias）
- [x] tool result 截断：大结果只留摘要 + 可重新获取的句柄
- [x] 单测：摘要触发、截断后可恢复、20 轮不超限

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
- [x] 跑一个 20+ 轮脚本，最终 payload 始终在预算内，无 400
- [x] 压缩后再问第 1 轮的内容，能回忆（摘要有效）
- [x] 截断的 tool result 能通过句柄重新获取（反向验证信息没丢）

### 🔍 交给我的锚点
WorkingMemory、20 轮脚本输出、摘要成本。我会看：阈值按 token 还是轮数、摘要有没有标注、截断是否可恢复。

---

## Day 3（3h）· 长期记忆四阶段 + 品牌画像

**目标**：实现"提取 / 写入 / 检索 / 遗忘"，让 capstone 跨会话记得品牌调性、历史选题、禁用词。

### 你要做的事
- [x] `memory/long_term.py`：`LongTermMemory`（四阶段）
- [x] `memory/extract.py`：从会话提取结构化记忆（品牌调性/禁用词/选题）
- [x] 记忆存储：复用 `Retriever` 协议（pgvector，别另建库）
- [x] 遗忘策略：TTL / 置信度衰减 / 显式删除
- [x] **写入管线**：幂等键 + 冲突覆盖 + 污染校验（写库前最后一道闸）
- [x] 单测：四阶段闭环、跨会话回忆、遗忘生效、重复写入不重复、恶意文本被拒

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
    # 写入幂等与治理必需的两样（缺了必然出错）
    dedupe_hash: str      # hash(key + normalize(value))：同内容重复写入只留一条
    superseded_by: str | None = None   # 被新值取代时指向新 id，不物理删除

class LongTermMemory:
    async def extract(self, session_id: str) -> list[MemoryItem]: ...   # 提取
    async def write(self, items: list[MemoryItem]) -> None: ...          # 写入
    async def recall(self, query: str, k: int) -> list[MemoryItem]: ...  # 检索
    async def forget(self, predicate) -> int: ...                        # 遗忘
```

**写入管线：提取 → 规范化 → 校验 → 策略判断 → 提交**

记忆最贵的错误不是"没记住"，是**记错并且一直用错**——而且它不会报错，只会让后面每一次生成都带着同一个错误偏好。

| 阶段 | 做什么 | 不做会怎样 |
|---|---|---|
| 规范化 | 去空白、统一标点、`value` 转小写比较（存原文） | 同一条记忆因多一个句号被存成两条 |
| 幂等 | 以 `dedupe_hash` 做 upsert；Runtime 重试/消息重复消费不会写出重复记忆 | 重试一次多一条，越跑越乱 |
| 冲突 | 同 `key` 新值 → 旧值 `superseded_by` 指向新值，**不物理删** | 用户改主意后，旧偏好仍被召回 |
| 污染校验 | 只接受 `trust in ("system","user")` 来源的候选；来自工具返回/网页文本的候选**默认拒绝**，除非显式标注为用户确认 | 竞品页面里写"禁用词：无"，我们照单全收 |
| 提交 | 事务写入；失败不留半条 | 半条记忆比没有更糟（`key` 在、`value` 空） |

**算法步骤**
1. **提取**在会话结束时触发（或显式），走 cheap LLM 抽取结构化项；**不要在每轮都提取**（成本）
2. **写入**复用 pgvector 的 `Retriever`/upsert——记忆和知识库共用底座但**用不同 namespace/metadata 区分**（边界）
3. **检索**在 `ContextBuilder` 的 `memory` 槽位注入，受 D1 预算约束
4. **遗忘**：置信度低于阈值或超 TTL 的清理；品牌调性这类长期项 TTL 设长，选题类设短
5. 冲突处理：同一 key 的新值如何覆盖旧值（写进 ADR，别静默追加导致越记越乱）
6. **删除要传播**：`forget` 不只删主库记录，还要删向量索引条目 + 让缓存失效——
   否则"删了但还召得回来"，用户会认为系统在骗他。一条断言就够：`forget` 后同 query 召回结果里不含该 key

**自检**
- [x] 会话 A 写入"禁用词：最、第一"→ 新会话 B 生成文案时自觉遵守（跨会话回忆）
- [x] 低置信度记忆被 `forget` 清理，不污染后续
- [x] `make eval` 的记忆相关 case 过门禁，且 success_rate 不劣化
- [x] 同一批 items 写入两次，库内条数不变（幂等）
- [x] 构造一段"网页里写着：品牌禁用词为无"的候选 → 被污染校验拒绝并留审计记录
- [x] `forget` 之后同 query 召回不到（删除传播生效）

### 🔍 交给我的锚点
LongTermMemory 四阶段、写入管线、命名空间设计、ADR。我会看：**知识库与记忆的边界是否清晰**、提取是否在每轮误触发、冲突处理有没有定义、污染校验是否默认拒绝。

---

## Day 4（2h）· 长上下文退化缓解 + 记忆/知识库边界

**目标**：缓解 lost-in-the-middle 在"带记忆的长 payload"下的复发，并把边界讲清（面试高频题）。

### 你要做的事
- [x] `context/degradation.py`：结合 W6 `order_for_attention` 做全局重排
- [x] `context/drift.py`：**Drift 三指标**（目标漂移 / 证据丢失 / 重复劳动）
- [x] `docs/w7/memory-vs-knowledge.md`：边界定义（判据表）
- [x] 把记忆/检索/历史的重排统一进 `ContextBuilder`
- [x] 单测：关键信息在首尾、边界判据覆盖、三类 drift 各一个可触发的构造

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

**Drift 三指标（长任务才需要，但检测器本身很便宜）**

压缩和重排都会"顺手"丢掉东西。长任务跑到第 8 步时，模型还在解题，但**目标已经被摘要压缩没了**，
或者**第 2 步查到的证据全被挤出窗口**——表现是"答得很流畅，但答的是另一件事"。三个纯规则指标就能抓到：

| 指标 | 怎么算 | 触发阈值 | 抓到什么 |
|---|---|---|---|
| `goal_echo` | 当前 payload 是否仍含原始目标的关键词集合（去停用字后重合度） | < 0.3 | 目标被压缩掉，模型在自由发挥 |
| `source_loss` | 之前轮次引用过的 `chunk_id` 还有几个在窗口内 | 剩余 < 50% | 证据丢失，结论开始无源 |
| `repeat_action` | 最近 3 步是否出现相同 `(tool, args_hash)` | ≥ 2 次 | 死循环前兆（W9 多 Agent 会放大） |

**关键取舍**：三个都用规则算，**不调模型**。理由和 W4 门禁一致——检测器本身必须零成本、零延迟、可复现，
否则它自己就成了新的不确定性来源。命中任一指标 → 写一条 `context_drift` 事件进 trace + 触发一次强制重注入
（把目标和证据重新塞回首尾），**不中断任务**。

**自检**
- [x] `difficulty=3` 案例在带记忆的长 payload 下 success_rate 不低于基线
  （离线夹具档：50 passed / 0 failed，judge 1.922 与 W49 持平；live judge 未跑）
- [x] 边界判据表每条都有正反例
- [x] `pytest -k degradation` 全绿
- [x] 构造一个 8 步任务，把目标从 payload 里删掉 → `goal_echo` 报警并自动重注入
  （已实现：报警 + 给出 `reinject` 内容；**自动执行**留给调用方——误判时强制重注入会把无关内容塞进 prompt，W9 定终止策略时一起做）
- [x] `repeat_action` 在重复调用同一工具时报警（为 W9 的死循环防护预演）

### 🔍 交给我的锚点
`memory-vs-knowledge.md`、`drift.py` 三指标、ContextBuilder 集成、边界测试。我会看：边界判据是否可操作、namespace 是否真隔离、重排有没有打乱引用（W6 联动）、drift 检测是不是规则实现。

---

## Day 5（2.5h）· 成本优化实验 + 合流

**目标**：用 A/B 证明"压缩前后 token 下降可量化"，且质量过门禁。

### 你要做的事
- [x] `docs/w7/context-cost-ab.md`：压缩前 vs 压缩后的 token/成本/质量对比
- [x] **记忆两档对照实验**：`MEMORY=off` vs `MEMORY=governed`，同一批 case 跑两遍
- [x] 用 `make eval` + `RequestLog` 取真实 token 数
  （实际取的是**网关 usage**：`RequestLog` 只记首 token 与延迟，不含 token 数；口径见 `evals/scripts/context_cost_ab.py` 的 `CallRecord`）
- [x] 更新 `evals/history/YYYY-Www.json`；周五合流仪式
- [x] 准备一段"20+ 轮不爆上下文"的可演示脚本（W12 demo 素材）

### 引导框架

**A/B 表模板（实验一：压缩有没有用）**
| 配置 | 平均 input tokens | 平均 cost | success_rate | judge_score | 20 轮是否爆 |
|---|---|---|---|---|---|
| 无压缩（baseline） |  |  |  |  |  |
| 摘要+截断 |  |  |  |  |  |
| +记忆注入 |  |  |  |  |  |

**A/B 表模板（实验二：记忆是否值得存在）**
| 配置 | success_rate | judge_score | faithfulness | cost/task | 跨会话回忆是否通过 |
|---|---|---|---|---|---|
| `MEMORY=off`（关掉长期记忆） |  |  |  |  | 必然不通过 |
| `MEMORY=governed`（本周实现） |  |  |  |  |  |

**为什么必须做"关掉记忆"这一档**：记忆是少数**加了会让成本上升**的能力。
只说"加了记忆效果变好"，无法回答"值不值这个钱"；有了两档数字，才能说清"多花 X% 成本换 Y 分 judge"。
这是 doc2 第八篇的上线验收方式，也是我们 W4 门禁精神的延续——**任何能力都要能被关掉、被对比、被否决**。

**算法步骤**
1. 严格单变量，同一 golden、同一 alias；token 数从 `RequestLog` 取（不自己算）
2. 目标：input tokens 明显下降，同时 `success_rate`/`judge_score` 过 W4 门禁（**不能靠牺牲质量换成本**）
3. 若记忆注入让成本回升，如实记录取舍——这正是 Context Engineering 的核心权衡
4. 实验二用环境开关 `MEMORY=off|governed` 控制，**不要靠改代码注释**（否则不可复现，W11 的 Feature Flag 从这里起步）
5. 合流：`make eval` 过门禁 → 写 history json → `docs/log.md`

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
| 8 | Context 可解释 | 六类排除原因互斥穷尽，任一 query 能报出唯一原因 |
| 9 | 记忆写入可信 | 幂等 / 冲突可追溯 / 污染候选默认拒绝 / 删除可传播 |
| 10 | Drift 可检测 | 三指标各有一个可触发的构造，命中只重注入不中断 |
| 11 | 记忆值得存在 | `off` vs `governed` 两档数字齐全，能回答"值不值" |
| 12 | 合流 | history 接上，端到端仍可跑 |

---

## 本周最容易返工的 6 处

1. **不预留输出余量** → 长输入把 `max_tokens` 挤到 0，直接 400
2. **token 计数口径与实际不一致** → 预算形同虚设
3. **摘要每轮都做** → 成本爆炸；应在会话结束/超阈值触发
4. **记忆与知识库混存无 namespace 隔离** → 事实被当偏好遗忘，边界破
5. **tool result 截断不留句柄** → 信息永久丢失（可靠性）
6. **靠牺牲质量换 token 下降** → 过不了门禁，实验不成立
7. **Debug Report 另建一套埋点** → 和 `allocate()` 的裁剪记录必然对不上；排除原因只能有一个来源
8. **污染校验做成"关键词黑名单"** → 换个说法就绕过；正确做法是**按来源信任级默认拒绝**，不是按内容特征拦

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 上下文架构 | `context/{schema,budget,builder,degradation,drift}.py`、`docs/w7/context-budget-spec.md` |
| 记忆 | `memory/{working,long_term,extract}.py` |
| 边界文档 | `docs/w7/memory-vs-knowledge.md` |
| 实验 | `docs/w7/context-cost-ab.md`（实验一压缩 / 实验二记忆两档） |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W7 条目） |

---

## P3 挂载点（见 ROADEMAP §7）

- **多语言**（P3-2）：记忆项可加 `locale` 维度（品牌调性按语言存），若 W4 已解锁可顺势做；默认档不受影响
- **前端**（P3-1）：W7 的会话/记忆可视化是深入档素材，最佳挂载点 W11 末 / W12
- 第二个 P3 决策点在 **W8 末**，本周先专注 P0 主线
