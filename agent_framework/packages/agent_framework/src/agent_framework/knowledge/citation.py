"""引用溯源（W6 D1）：chunk → 编号 → 答案里的 `[n]` → 回到原文。

W6 的检查点只有一句人话：**报告里的每条结论都能指出出自哪份文档的哪一段**。
它由三段串起来，任何一段松掉，整条链就是"看起来有引用"：

1. `build_context` 把 chunk 编号成 `[1]…[n]`，**同时**产出 `index -> Citation` 映射；
2. 生成侧按 `CITATION_RULES` 的约束在每条事实后写 `[n]`；
3. `audit_citations` 把答案里的编号解析回来，判它是否指向正确的段。

三条纪律，每条对应一次真实的翻车：

- **编号与映射必须同源生成。** 分两次遍历（一次写文本、一次建映射）时，只要中途
  有一次 `enumerate` 起点写成 0，答案 `[1]` 就会指向第二段，而两边代码看起来都对。
  这里只有一个循环，`Citation.index` 与文本里的 `[n]` 由同一个变量产出。
- **找不到编号 = 引用缺失，进分母。** 解析失败的编号若被静默跳过，准确率的分母
  只剩"解析成功的那几条"，指标会随解析失败率升高而变好看——这是最坏的一类指标。
- **引用不进 `provider_meta`**（ADR-007）：那是透传给上游的原始字段，放进去等于把
  内部编号暴露成厂商可见的契约，换 adapter 就丢。

本模块零第三方依赖：不给"算一个引用"引入 tokenizer 或模板引擎。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .base import ScoredChunk

# 答案里的引用标记。只认半角 `[n]`：全角 `【n】`、`[n, m]` 这类写法先不支持，
# 因为宽容匹配会把正文里的 Markdown 链接 `[1](url)` 也算成引用，反而更难判。
CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")

# 桥接层在 source_uri 缺失时写的占位（见 rag_mvp/contrib/agent_framework.py）。
# 它是**显式标记**而不是空串，正是为了让这里能把它判成失败，而不是"看起来有出处"。
MISSING_SOURCE_PREFIX = "missing-source:"

# 每块进 context 的字符上限。太长的块会把整段上下文挤满，让"第 5 条之后的内容"
# 实际上没进模型；太短又会截断论据，让引用指向半句话。
DEFAULT_MAX_CHARS_PER_CHUNK = 800

# 生成侧必须遵守的规则。放在框架侧是因为它约束的是**引用格式**，
# 与"写什么内容"无关；产品层的 system prompt 仍然由产品自己写。
CITATION_RULES = (
    "引用规则（必须遵守）：\n"
    "1. 每条事实性陈述后标注来源编号，形如 [1]；同一句可用多个编号 [2][3]。\n"
    "2. 只能使用参考资料里出现过的编号；不要编造编号，也不要写 [0]。\n"
    "3. 参考资料没有覆盖的问题，直接回答「资料未覆盖」，不要用常识补全。\n"
    "4. 若两处资料互相矛盾，必须显式指出「两处说法不一致」，不要自行裁决。"
)

VERDICT_OK = "ok"
VERDICT_NO_CITATION = "no_citation"
VERDICT_UNKNOWN_INDEX = "unknown_index"
VERDICT_SOURCE_MISSING = "source_missing"
VERDICT_OFF_TARGET = "off_target"


@dataclass(frozen=True)
class Citation:
    """一条引用的完整出处。

    `excerpt` 是**给人核对用的**：抽样复核时先看摘录，不必打开原文档。
    它不是给模型看的（模型看的是 context 正文），所以截断它不影响答案质量。
    """

    index: int
    chunk_id: str
    document_id: str
    source_uri: str
    excerpt: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    title: str = ""
    # 入库时就被标记为"出处缺失"（桥接层写 `metadata["source_uri_missing"]=True`）。
    # 单独一个字段而不是只看 uri 前缀：前缀是桥接层的约定，框架不该依赖它的字面量。
    marked_missing: bool = False

    @property
    def source_missing(self) -> bool:
        """出处是否缺失/是占位符。"""

        return (
            self.marked_missing
            or not self.source_uri.strip()
            or self.source_uri.startswith(MISSING_SOURCE_PREFIX)
        )

    @property
    def locator(self) -> str:
        """人能照着找到原文的一句话：`标题 · 路径 · 第 3–4 页 · 第三章 > 3.2`。"""

        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.source_uri and not self.source_missing:
            parts.append(self.source_uri)
        elif self.source_missing:
            # 显式写出来，不要让它看起来像"没写页码"这种小事。
            parts.append("出处缺失")
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"第 {self.page_start}–{self.page_end} 页")
            else:
                parts.append(f"第 {self.page_start} 页")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        return " · ".join(parts) if parts else f"chunk {self.chunk_id}"


@dataclass(frozen=True)
class CitationContext:
    """带编号的 context 文本 + 编号映射。**两者同源**，不接受外部拼装。"""

    text: str
    citations: tuple[Citation, ...] = ()

    def by_index(self) -> dict[int, Citation]:
        return {citation.index: citation for citation in self.citations}

    def __len__(self) -> int:
        return len(self.citations)


def _locator_line(chunk: ScoredChunk) -> str:
    """块首行：`[n] 标题 | 路径 | 第 x 页 | 章节`。

    把出处写进 context（而不只是写进 Citation）是刻意的：模型看到路径与页码后，
    更不容易把两段相邻内容混成一段，抽样时也更容易判断"它到底引了哪一段"。
    """

    metadata = chunk.chunk.metadata
    parts: list[str] = []
    title = str(metadata.get("document_title") or "").strip()
    if title:
        parts.append(title)
    if chunk.chunk.source_uri:
        parts.append(chunk.chunk.source_uri)
    if chunk.chunk.page_start is not None:
        if (
            chunk.chunk.page_end is not None
            and chunk.chunk.page_end != chunk.chunk.page_start
        ):
            parts.append(f"第{chunk.chunk.page_start}–{chunk.chunk.page_end}页")
        else:
            parts.append(f"第{chunk.chunk.page_start}页")
    if chunk.chunk.section_path:
        parts.append(" > ".join(chunk.chunk.section_path))
    return " | ".join(parts)


def build_context(
    chunks: Sequence[ScoredChunk],
    *,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> CitationContext:
    """把检索结果拼成带编号的 context，并返回 `index -> Citation` 映射。

    **唯一一个同时决定"文本顺序"和"编号"的地方**：循环里 `index` 变量既写进
    `[{index}]`，也写进 `Citation(index=index, ...)`。下游任何环节（去冗余、
    裁剪、首尾放置）都必须在这一步**之前**完成，否则编号会指向被移动过的块。
    """

    if max_chars_per_chunk <= 0:
        raise ValueError("max_chars_per_chunk 必须大于 0")

    blocks: list[str] = []
    citations: list[Citation] = []

    for index, scored in enumerate(chunks, start=1):
        chunk = scored.chunk
        body = chunk.text.strip()
        if len(body) > max_chars_per_chunk:
            body = body[:max_chars_per_chunk].rstrip() + "…"

        locator = _locator_line(scored)
        header = f"[{index}] {locator}" if locator else f"[{index}]"
        blocks.append(f"{header}\n{body}")

        citations.append(
            Citation(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_uri=chunk.source_uri,
                excerpt=body[:300],
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=tuple(chunk.section_path),
                title=str(chunk.metadata.get("document_title") or ""),
                marked_missing=bool(chunk.metadata.get("source_uri_missing")),
            )
        )

    return CitationContext(text="\n\n".join(blocks), citations=tuple(citations))


def render_prompt(context: CitationContext, *, question: str) -> str:
    """组装最终的用户消息：规则 + 参考资料 + 问题。

    顺序是"规则在最前"：放到最后时，长 context 会把指令挤到中间/末尾，
    而指令在末尾时模型更容易把它当成"另一份资料"而不是约束。
    """

    return (
        f"{CITATION_RULES}\n\n"
        f"参考资料：\n{context.text}\n\n"
        f"问题：{question}"
    )


@dataclass(frozen=True)
class ParsedCitations:
    """答案里的编号解析结果。"""

    # 按出现顺序去重后的编号。重复引用同一段只算一次。
    indices: tuple[int, ...] = ()
    citations: tuple[Citation, ...] = ()
    # 映射里查不到的编号。这是**编号错位**的直接证据，不能静默丢弃。
    unknown: tuple[int, ...] = ()

    @property
    def has_citation(self) -> bool:
        return bool(self.indices)

    @property
    def resolved(self) -> int:
        return len(self.citations)


def parse_citations(answer: str, context: CitationContext) -> ParsedCitations:
    """解析答案里的 `[n]`，映射回 Citation。

    不校验"编号是否出现在句末"这类排版细节：那不是正确性问题，用规则去卡只会
    让答案为了过检而变形。这里只回答两件事——引了哪些段、有没有引到不存在的段。
    """

    mapping = context.by_index()
    seen: list[int] = []
    unknown: list[int] = []

    for match in CITATION_PATTERN.finditer(answer or ""):
        index = int(match.group(1))
        if index in seen or index in unknown:
            continue
        if index in mapping:
            seen.append(index)
        else:
            unknown.append(index)

    return ParsedCitations(
        indices=tuple(seen),
        citations=tuple(mapping[index] for index in seen),
        unknown=tuple(unknown),
    )


@dataclass(frozen=True)
class CitationExpectation:
    """抽样核对时的 ground truth。

    `chunk_ids` 来自 golden 标注（rag_mvp 的 `gold_chunk_ids`）。**判据是 chunk 级**，
    不是文档级：本语料只有一个文档，文档级判据会让准确率恒等于 100%，
    等于什么都没测。
    """

    chunk_ids: frozenset[str] = frozenset()
    document_ids: frozenset[str] = frozenset()

    def matches(self, citation: Citation) -> bool:
        if citation.chunk_id in self.chunk_ids:
            return True
        # 没有 chunk 级标注时退到文档级；有标注就不放水。
        return not self.chunk_ids and citation.document_id in self.document_ids


@dataclass(frozen=True)
class CitationAudit:
    """一条答案的引用体检结果。

    `verdict` 只保留第一个失败原因（报告里好读），`reasons` 全量保留（排查时好用）。
    """

    verdict: str
    reasons: tuple[str, ...] = ()
    parsed: ParsedCitations = field(default_factory=ParsedCitations)
    # 命中标注段的编号。**至少一条命中就算通过**——理由见 `audit_citations`。
    matched: tuple[int, ...] = ()
    # 未命中标注段的编号。它不判失败，但必须能被看见：否则 precision 一路下滑，
    # 报告上却全是"通过"。判断"是否只是同页相邻块"需要 chunk 元数据，
    # 这里只给编号，把解释权留给报告。
    unmatched: tuple[int, ...] = ()
    # 命中标注段的引用占全部引用的比例。没提供 ground truth 时为 None
    # （不是 1.0，也不是 0.0——"没测"和"测出来是满分"必须能区分）。
    precision: float | None = None

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_OK


def audit_citations(
    answer: str,
    context: CitationContext,
    *,
    expected: CitationExpectation | None = None,
) -> CitationAudit:
    """判一条答案的引用是否成立。

    四级判据，从"完全没有引用"到"引了但引错":

    - `no_citation`：答案里没有任何 `[n]`。**不是"格式问题"**——它意味着这段答案
      不可溯源，在 capstone 里等价于结论没有证据。
    - `unknown_index`：引用了映射里不存在的编号（典型是模型自己编 `[5]`）。
    - `source_missing`：引用的块出处是占位符（`missing-source:`）。这类块在入库时
      就已知缺来源，绝不能让它在抽样里混成"引用正确"。
    - `off_target`：引用的块**全都不在**标注集内。

    **通过口径是"至少一条命中标注段"，不是"条条命中"。** 这一点是实测逼出来的：
    首次抽样时 `golden-001` 引了 5 条，其中 c330 / c348 正是标注段，另外 3 条是
    与标注段同页的相邻块（c329 与 c330 同为 p152）。按"条条命中"判，这条失败；
    但它的答案确实指向了正确出处。

    根因是**标注本身是部分标注**：425 个块里只标了 69 个。要求模型只引被标注的块，
    等于用"标注是否穷尽"去衡量"引用是否正确"——模型引对了没标的地方反而扣分，
    而这类失败会归因到"生成层"，让人去改 prompt，改完指标仍不动。

    所以这里把两件事拆开：`verdict` 只回答"有没有引对"，`precision` 回答
    "引的东西有多干净"。后者必须报出来——只要前者，模型学会"把整个 context 都
    引一遍"就能稳过（参见抽样脚本里的 `blanket_citation` 检查）。
    """

    parsed = parse_citations(answer, context)
    reasons: list[str] = []

    if not parsed.indices and not parsed.unknown:
        reasons.append("答案中没有任何 [n] 引用")

    if parsed.unknown:
        reasons.append(f"引用了不存在的编号 {list(parsed.unknown)}")

    missing = [c.index for c in parsed.citations if c.source_missing]
    if missing:
        reasons.append(f"编号 {missing} 指向的块出处缺失")

    matched: list[int] = []
    unmatched: list[int] = []
    if expected is not None and parsed.citations:
        for citation in parsed.citations:
            if expected.matches(citation):
                matched.append(citation.index)
            else:
                unmatched.append(citation.index)
        if not matched:
            reasons.append(
                f"引用的段均不在标注集内：{[c.chunk_id for c in parsed.citations]}"
            )
        elif unmatched:
            # 不判失败，但一定写进 reasons：这类"噪声引用"多了，答案的可核对性
            # 会下降（读者要一条条点开才知道哪条真的支撑结论）。
            noise = [c.chunk_id for c in parsed.citations if c.index in set(unmatched)]
            reasons.append(
                f"另有 {len(unmatched)} 条引用不在标注集内（不判失败，计入 precision）：{noise}"
            )

    if parsed.unknown:
        verdict = VERDICT_UNKNOWN_INDEX
    elif missing:
        verdict = VERDICT_SOURCE_MISSING
    elif not parsed.indices:
        verdict = VERDICT_NO_CITATION
    elif expected is not None and not matched:
        verdict = VERDICT_OFF_TARGET
    else:
        verdict = VERDICT_OK

    precision = (
        len(matched) / len(parsed.citations)
        if expected is not None and parsed.citations
        else None
    )

    return CitationAudit(
        verdict=verdict,
        reasons=tuple(reasons),
        parsed=parsed,
        matched=tuple(matched),
        unmatched=tuple(unmatched),
        precision=precision,
    )


def citation_accuracy(audits: Iterable[CitationAudit]) -> float:
    """引用准确率 = 通过条数 / 总条数。

    分母是**所有抽样条数**，不是"解析成功的条数"：后者会让解析失败自动从分母消失，
    于是"漏引用越多、指标越好"。没有样本时返回 0.0 而不是 1.0——空集上的 1.0
    是最容易糊弄过去的那种绿。
    """

    rows = list(audits)
    if not rows:
        return 0.0
    return sum(1 for audit in rows if audit.ok) / len(rows)


def citation_payload(citations: Sequence[Citation]) -> list[dict[str, object]]:
    """对外响应里的引用块。手写字段，不 dump 内部 dataclass。

    与 `/v1/draft` 的 DTO 纪律一致：字段一旦进了响应就是对外契约，
    多带一个内部字段（如 `retriever`）以后就删不掉了。
    """

    return [
        {
            "index": citation.index,
            "chunk_id": citation.chunk_id,
            "document_id": citation.document_id,
            "source_uri": citation.source_uri,
            "page_start": citation.page_start,
            "page_end": citation.page_end,
            "section_path": list(citation.section_path),
            "excerpt": citation.excerpt,
            "locator": citation.locator,
        }
        for citation in citations
    ]


def missing_source_chunks(chunks: Sequence[ScoredChunk]) -> list[str]:
    """入库自检用：哪些块的出处是空的/占位符。

    单独给一个函数是因为它要在**入库时**跑（Gate 0 的前置检查），
    而不是等到抽样报告里才发现——那时文档已经躺进索引了。
    """

    bad: list[str] = []
    for scored in chunks:
        uri = (scored.chunk.source_uri or "").strip()
        if not uri or uri.startswith(MISSING_SOURCE_PREFIX):
            bad.append(scored.chunk.chunk_id)
    return bad


def group_by_document(chunks: Sequence[ScoredChunk]) -> Mapping[str, list[ScoredChunk]]:
    """按文档分组。引用面板要按文档聚合时用它，避免调用方各写一遍。"""

    grouped: dict[str, list[ScoredChunk]] = {}
    for scored in chunks:
        grouped.setdefault(scored.chunk.document_id, []).append(scored)
    return grouped
