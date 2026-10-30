# W6 执行清单（生产 RAG II · 溯源与增量 · P1）

> **角色约定**：沿用 `docs/w1/w1-tasks.md` 第 3 行——架构、骨架、验收标准由我给，代码由你写，我不代写。
> 本周把 W5 的"能检索"升级成"可溯源、可增量、可控权限"。ROADEMAP 检查点：**新增文档免全量重建；引用准确率抽样 ≥ 90%**。
> 评级 P1 意味着：若进度落后，这是**第一个可压缩深度**的周（默认档仍要做：溯源 + 增量 + 调优清单），但**引用溯源不能砍**——它是 capstone 可信度的地基。

## 承接面：W1–W5 已交付

| 已有 | 位置 | W6 的接缝 |
|---|---|---|
| `Chunk.source` / `Chunk.metadata` | `knowledge/base.py`（W5 已带） | 本周溯源直接用，**不用重建索引** |
| `Retriever` 协议 + `HybridRetriever`/`RetrievalPipeline` | `retrieval/` | 增量/ACL 都在协议内加能力，不改上层 |
| `golden_v1.jsonl` 的 `references` | `evals/golden/` | 引用准确率抽样的 ground truth |
| `make eval` + `regression_gate` | `evals/` | 增量索引改动后必须过门禁 |
| `ScoredChunk.retriever` 来源标记 | `retrieval/hybrid.py` | 溯源要一路带到输出，别中途丢 |
| `RequestLog` trace / `attribute_failure` | W2/W4 | 引用缺失可归因为"检索层" |

> W6 检查点（ROADEMAP）：capstone 报告中的每条结论可溯源；新增文档免全量重建；引用准确率抽样 ≥ 90%。

**本周预算**：5 天 × 2–3h ≈ 10–13h（P1，可压缩）。

---

## 0. 本周骨架总览

| Day | 新增模块（落点） | 承接符号 | 推进的 DoD |
|---|---|---|---|
| D1 | `knowledge/citation.py`（引用溯源） | `Chunk.source`、`ScoredChunk` | 质量（可溯源） |
| D2 | `knowledge/incremental.py`（增量索引/版本/删除） | `PgVectorStore` upsert | 可靠性、成本 |
| D3 | `retrieval/acl.py`（ACL-aware 最小实现） | `Chunk.metadata.acl` | 安全（W10 前置） |
| D4 | `retrieval/context_assemble.py`（lost-in-middle 缓解） | pipeline 输出 | 质量 |
| D5 | `docs/w6/rag-tuning-checklist.md` + 抽样验收 + 合流 | golden references | 证据、可归因 |

**唯一主线**：让"报告里的每句话都能指出出自哪份文档的哪一段"。溯源链路今天贯通，后面 ACL/上下文组装都是在这条链路上的加固。

---

## Gate 0 · 开工前（30 min）

- [ ] 确认 W5 的 `Chunk.source` 对每个 chunk 都非空（跑那条校验断言）——这是本周不重建的前提
- [ ] `make eval` 全绿、门禁生效
- [ ] 准备一份"会更新的文档"和一份"要删除的文档"（测增量/删除同步）
- [ ] 准备 ≥20 条带 `references` 的 golden case 做引用准确率抽样

---

## Day 1（3h）· 引用溯源（输出必须带出处）

**目标**：从 chunk 到最终答案，**引用标记一路不丢**，输出可定位到 doc_id#chunk_id。

### 你要做的事
- [ ] `knowledge/citation.py`：`Citation` 数据结构 + 从 `ScoredChunk` 生成引用
- [ ] 在答案生成 prompt 里注入"引用编号 → 出处"的映射
- [ ] 输出解析：答案中的 `[n]` 引用对应哪个 chunk
- [ ] `evals/scripts/citation_accuracy.py`：抽样准确率（对 golden `references`）
- [ ] 单测：引用丢失、编号错位、无引用三种边界

### 引导框架

```python
# knowledge/citation.py
@dataclass(frozen=True)
class Citation:
    index: int             # 答案里的 [n]
    doc_id: str
    chunk_id: str
    source: str            # 来自 Chunk.source
    excerpt: str           # 支撑片段，供人工核对

def build_context(chunks: list[ScoredChunk]) -> tuple[str, dict[int, Citation]]:
    """拼带编号的 context 文本 + index->Citation 映射；编号与文本必须严格对应。"""
    ...
```

**算法步骤（今天最容易写错的地方）**
1. `build_context` 生成 `[1]...text...` 的编号文本，同时返回 `index -> Citation` 映射；**编号与映射必须同源生成**，否则会错位
2. prompt 明确要求"每个事实性陈述后跟 `[n]`，无出处不要编"（这也是 W10 反幻觉的前置）
3. 解析答案里的 `[n]`，映射到 Citation；**找不到对应编号 = 引用缺失**，计入准确率分母
4. 引用准确率 = 抽样中"引用存在且指向正确文档"的比例，目标 ≥90%
5. **引用不要塞 `provider_meta`**（ADR-007：那是透传给上游的），走正常 content 或结构化输出

**自检**
- [ ] `pytest -k citation` 覆盖编号错位检测、引用丢失检测
- [ ] 用 ≥20 条 golden 跑 `citation_accuracy.py`，输出准确率与失败样本
- [ ] 故意删掉一个 chunk 的 source → 该条被判"引用缺失"而非静默通过（反向验证）

### 🔍 交给我的锚点
`build_context`、citation_accuracy 抽样结果。我会看：编号与映射是否同源、引用缺失是否被检出、准确率是否 ≥90%。

---

## Day 2（3h）· 增量索引 + 版本/删除同步

**目标**：新增/更新/删除文档**不触发全量重建**（ROADEMAP W6 检查点）。

### 你要做的事
- [ ] `knowledge/incremental.py`：`IncrementalIndexer`（按 doc 版本 upsert/删除）
- [ ] 文档版本识别（内容哈希 / 外部 version 字段）
- [ ] 删除同步：文档删除时清掉其所有 chunk
- [ ] 单测：新增不改旧、更新替换、删除清理、重复 upsert 幂等

### 引导框架

```python
# knowledge/incremental.py
@dataclass(frozen=True)
class DocVersion:
    doc_id: str
    content_hash: str
    version: int

class IncrementalIndexer:
    def __init__(self, store: PgVectorStore) -> None: ...
    async def sync(self, doc_id: str, content: str) -> SyncResult:
        """hash 未变 -> skip；变了 -> 删旧 chunk + 写新 chunk（同一事务）。"""
        ...
    async def delete(self, doc_id: str) -> int: ...
```

**算法步骤（今天最容易写错的地方）**
1. 先算 `content_hash`，与已存版本比对，**未变直接 skip**（别每次重新 embed，成本杀手）
2. 更新走"删旧 chunk + 写新 chunk"的**同一事务**，中途失败不能留下半新半旧（否则溯源指向已删 chunk）
3. 删除按 `doc_id` 批量清 chunk；返回删除条数供核对
4. `sync`/`delete` 幂等：同一输入跑两次结果一致（对齐 W2 会话幂等、W11 幂等主线）
5. 记录 `index_version` 与 `model_snapshot`（ADR-009），embedding 模型换了要能标记"需重建"

**自检**
- [ ] 新增 1 份文档 → 旧文档的 chunk 数与向量未变（不是全量重建）
- [ ] 更新同一文档 → 旧 chunk 被替换，无残留
- [ ] 删除 → 该 doc 的所有 chunk 清零；重复调用不报错（反向验证幂等）

### 🔍 交给我的锚点
IncrementalIndexer、事务处理、幂等测试。我会看：更新是否事务化（不留半新半旧）、hash 比对是否真跳过、删除是否干净。

---

## Day 3（2h）· ACL-aware 检索最小实现

**目标**：检索结果按调用方权限过滤。**最小实现**即可（P1 周不追求复杂权限模型），但过滤必须发生在"进 LLM context 之前"。

### 你要做的事
- [ ] `retrieval/acl.py`：`ACLFilter`（从 `Chunk.metadata.acl` 过滤）
- [ ] 在 `RetrievalPipeline` 里接入过滤（召回后、进 context 前）
- [ ] 主体身份从请求上下文取（先用简单 principal，W10 接真鉴权）
- [ ] 单测：越权 chunk 被滤除、过滤后不足 k 的处理

### 引导框架

```python
# retrieval/acl.py
@dataclass(frozen=True)
class Principal:
    subject_id: str
    groups: frozenset[str]

def allowed(chunk: Chunk, principal: Principal) -> bool:
    """chunk.metadata['acl'] 与 principal.groups 求交；无 acl 默认按配置拒绝或放行（写进 ADR）。"""
    ...
```

**算法步骤**
1. 过滤放在**重排之后、拼 context 之前**——先过滤再重排会浪费重排算力
2. 过滤后若不足 k，**召回时多取一些**（over-fetch）再过滤，避免"有权限的结果太少"
3. **默认策略写进 ADR**：无 `acl` 字段的 chunk 是"拒绝"还是"放行"——安全周（W10）会审视，默认建议"拒绝"
4. 被滤除的 chunk 不进日志全文，只记数量（隐私，W3 ADR 已定）
5. 主体身份本周用简单的 header/principal 占位，别提前实现完整鉴权（那是 W10）

**自检**
- [ ] 构造一个越权 chunk → 不出现在 context 里（反向验证）
- [ ] 过滤导致结果不足时，over-fetch 能补足到 k
- [ ] `pytest -k acl` 离线全绿

### 🔍 交给我的锚点
ACLFilter、ADR 里的默认策略、越权测试。我会看：过滤位置对不对、无 acl 的默认策略是否安全、有没有提前做完整鉴权。

---

## Day 4（2h）· lost in the middle 与上下文污染缓解

**目标**：缓解"长 context 中间信息被忽略"和"无关 chunk 污染答案"——这是 RAG 质量的老问题，也是 W7 上下文工程的引子。

### 你要做的事
- [ ] `retrieval/context_assemble.py`：重排序（把最相关的放首尾）+ 去冗余
- [ ] 去冗余：相似 chunk 合并，避免同一段反复进 context 浪费 token
- [ ] 与 D1 的 `build_context` 整合（编号仍要连续、不错位）
- [ ] 单测：首尾放置、去冗余后编号连续、超预算裁剪

### 引导框架

```python
# context_assemble.py
def order_for_attention(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """最相关的放首尾，次要的放中间（lost-in-the-middle 缓解）。"""
    ...
def deduplicate(chunks: list[ScoredChunk], threshold: float) -> list[ScoredChunk]: ...
def trim_to_budget(chunks: list[ScoredChunk], token_budget: int) -> list[ScoredChunk]:
    """按预算裁剪，裁剪掉的不能留悬空引用编号。"""
    ...
```

**算法步骤（今天最容易写错的地方）**
1. 重排序只改"排列顺序"，**不改 chunk_id 与引用编号映射**——否则 D1 的溯源错位
2. 裁剪后**重新生成连续编号**，避免答案引用到已裁掉的 `[n]`
3. token 预算从配置读，复用 W7 要建的预算分配思路（本周先局部用）
4. 去冗余阈值写进配置；合并时保留 `source` 最完整的那个 chunk

**自检**
- [ ] 构造 10 个 chunk，裁剪到 5 个后答案引用编号仍连续且可解析
- [ ] 重复 chunk 被合并，context token 数下降
- [ ] `make eval` 的 success_rate 不因重排序劣化（过门禁）

### 🔍 交给我的锚点
context_assemble、裁剪后编号连续性测试。我会看：裁剪有没有留悬空引用、重排序有没有打乱溯源映射。

---

## Day 5（2h）· 调优清单 + 抽样验收 + 合流

**目标**：沉淀 `docs/w6/rag-tuning-checklist.md`，完成引用准确率抽样验收，周五合流。

### 你要做的事
- [ ] `docs/w6/rag-tuning-checklist.md`：RAG 全链路调优清单（切分/召回/重排/改写/溯源/上下文）
- [ ] 引用准确率抽样 ≥90% 的验收记录
- [ ] 增量索引"免全量重建"验收记录
- [ ] 更新 `evals/history/YYYY-Www.json`、`docs/log.md`；周五合流仪式

### 引导框架

**调优清单模板（每格：现象 / 可能原因 / 调哪个旋钮 / 怎么验证）**
| 现象 | 归因层（W4 playbook） | 旋钮 | 验证命令 |
|---|---|---|---|
| 答案没引用 | 溯源层 | citation 编号映射 | `citation_accuracy.py` |
| 召回不到 | 检索层 | chunk 大小 / RRF / 改写 | `retrieval_metrics.py` |
| 答案跑偏 | 上下文层 | 重排序 / 去冗余 / 预算 | `make eval` |
| 延迟高 | 检索层 | 重排预算 / over-fetch | 门禁 p95 |

**算法步骤**
1. 清单要能被**后来人照着排查**（W12 文档项、面试可讲）
2. 引用准确率抽样的样本、判定标准、失败案例都记录在案（可复现）
3. 合流：`make eval` 过门禁 → 写 history json → `docs/log.md`
4. 若本周压缩了 ACL 深度，在 log 里明确标注"P1 默认档完成，ACL 深入档待解锁"

### 🔍 Gate 6 · 周末验收（全绿才算过 W6）

| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 溯源贯通 | 答案引用可定位到 doc_id#chunk_id |
| 2 | 引用准确率 | 抽样 ≥90%，失败样本可查 |
| 3 | 增量索引 | 新增/更新/删除免全量重建，幂等 |
| 4 | 事务安全 | 更新不留半新半旧，删除干净 |
| 5 | ACL 最小实现 | 越权 chunk 不进 context，默认策略安全 |
| 6 | 上下文质量 | 裁剪后无悬空引用，success_rate 过门禁 |
| 7 | 调优清单 | checklist 可照着排查 |
| 8 | 合流 | history 接上，端到端仍可跑 |

---

## 本周最容易返工的 6 处

1. **W5 的 `Chunk.source` 没带** → 本周溯源直接返工重建索引（前置 Gate 0 必须先查）
2. **引用编号与映射不同源** → 答案 `[n]` 指向错误文档，溯源形同虚设
3. **增量更新非事务化** → 中途失败留下半新半旧，引用指向已删 chunk
4. **更新不比对 hash** → 每次重新 embed，成本失控
5. **裁剪后留悬空引用编号** → 答案引用到被裁掉的 chunk
6. **ACL 过滤放在重排之前** → 浪费重排算力；或默认策略"放行"埋安全隐患

---

## 产出物位置

| 产出 | 位置 |
|---|---|
| 溯源 | `knowledge/citation.py`、`evals/scripts/citation_accuracy.py` |
| 增量 | `knowledge/incremental.py` |
| ACL | `retrieval/acl.py` |
| 上下文 | `retrieval/context_assemble.py` |
| 清单/ADR | `docs/w6/rag-tuning-checklist.md`、`docs/w6/adr-w6-*.md` |
| 记录 | `evals/history/YYYY-Www.json`、`docs/log.md`（W6 条目） |

---

## P3 挂载点（见 ROADEMAP §7）

- 本周是 **P1**，是进度落后时第一个可压缩深度的周：默认档（溯源 + 增量 + 调优清单）**必做**，ACL 深入档（细粒度权限模型）可留到 W10 安全周一并做
- **多语言**（P3-2）：若 W4 决策点已解锁，可在 RAG 资料库加多语言 chunk + `locale` 元数据——但默认档不受影响
