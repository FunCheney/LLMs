#!/usr/bin/env python
"""W6 D1 · 引用准确率抽样（对 chunk 级 gold）。

Gate 6 #2 的执行者：**抽样的引用里，有多少条真的指向标注段**。

它测的不是"答案写得好不好"，而是"答案里那个 `[3]` 到底指哪一段"。这两件事
在报告里经常被混成一件，于是"引用准确率 95%"可能只是"模型很爱写方括号"。

三种跑法，用途不同：

    # 1) 实跑（加载模型 + 调 LLM，慢，出真数字）
    .venv/bin/python evals/scripts/citation_accuracy.py --limit 30 --record /tmp/cit.json

    # 2) 复算（读已记录的 run，不加载模型、不调 LLM，CI 用这个）
    .venv/bin/python evals/scripts/citation_accuracy.py --replay /tmp/cit.json

    # 3) 自检（纯逻辑，毫秒级）
    .venv/bin/python evals/scripts/citation_accuracy.py --self-test

**为什么要有 replay**：抽样结论必须能被别人在没模型、没密钥的机器上复现。
只留一张"我跑出来是 92.9%"的截图，等于把结论锁在跑过的那台机器上；而
replay 把"答案 + context + gold"整体落盘，任何人重跑审计逻辑都能得到同一个数。

**为什么失败要分类**：`verdict != ok` 只说明"引用不成立"，但它可能是三个完全
不同的原因——gold 根本没进 context（检索层该修）、进了但模型没引（生成层该修）、
引了不存在的编号（模型在编）。混成一个数字，下周就不知道该动哪个旋钮。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # 让 `python evals/scripts/x.py` 也能 import 本仓库包
    sys.path.insert(0, str(REPO_ROOT))

from evals.metrics import percentile  # noqa: E402  (线性插值分位，与 W1 基线同算法)

DEFAULT_SAMPLE = REPO_ROOT / "evals" / "golden" / "citation_sample_v1.jsonl"

# Gate 6 #2 的通过线。写在代码里而不是只写在文档里：门禁要能自己判红绿，
# "看报告的人记得 90 这个数"不是门禁。
DEFAULT_MIN_ACCURACY = 0.90

# 失败归因。前四个对应 W4 playbook 的四层，最后一个留给"语料本身没覆盖"。
FAIL_RETRIEVAL_MISS = "retrieval_miss"  # gold 没进 context → 检索层
FAIL_CITATION_MISS = "citation_miss"  # gold 在 context 但答案没引 → 生成层
FAIL_HALLUCINATED_INDEX = "hallucinated_index"  # 引了不存在的编号 → 生成层（幻觉）
FAIL_SOURCE_MISSING = "source_missing"  # 引用的块没有出处 → 入库层
FAIL_OFF_TARGET = "off_target"  # 引了别的段且 gold 不在 context → 检索层
FAIL_BLANKET = "blanket_citation"  # 把整个 context 都引了 → 生成层（等于没区分）

# "全量引用"的判定门槛。context 只有 2–3 块时全引是合理的（就那么几段），
# 到 4 块以上还全引，说明模型没有在挑证据——它只是给每段都挂了个编号，
# 这种答案"处处有引用"，但读者无从知道哪条真的支撑结论。
BLANKET_MIN_CONTEXT = 4


def is_blanket_citation(*, cited: int, context: int) -> bool:
    """是否把 context 一块不落地全引了。"""

    return context >= BLANKET_MIN_CONTEXT and cited >= context


# --------------------------------------------------------------------------- #
# 样本
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sample:
    """一条抽样题。

    `gold_chunk_ids` 是 chunk 级标注（来自 rag_mvp `golden_queries.yaml`）。
    **判据坚持 chunk 级**：这 425 个块全部来自同一份文档，若退到文档级判据，
    准确率会恒等于 100%——那是最典型的"指标看起来很好，因为它什么都没测"。
    """

    id: str
    question: str
    gold_chunk_ids: tuple[str, ...] = ()
    kind: str = "answerable"
    tags: tuple[str, ...] = ()
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.gold_chunk_ids)


def load_sample(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("//"):
                continue
            raw = json.loads(text)
            samples.append(
                Sample(
                    id=str(raw["id"]),
                    question=str(raw["question"]),
                    gold_chunk_ids=tuple(raw.get("gold_chunk_ids") or ()),
                    kind=str(raw.get("kind") or "answerable"),
                    tags=tuple(raw.get("tags") or ()),
                    notes=str(raw.get("notes") or ""),
                )
            )
    if not samples:
        raise ValueError(f"抽样集为空：{path}")
    return samples


# --------------------------------------------------------------------------- #
# 单条结果（纯数据，不依赖模型，可单测）
# --------------------------------------------------------------------------- #


@dataclass
class CaseResult:
    id: str
    question: str
    answer: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    failure_kind: str | None = None
    cited_chunk_ids: list[str] = field(default_factory=list)
    context_chunk_ids: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_in_context: bool = False
    context_ms: float = 0.0
    answer_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    degraded: list[str] = field(default_factory=list)
    kind: str = "answerable"
    # 命中标注段的引用占全部引用的比例。None = 没 ground truth（应拒答题）。
    precision: float | None = None
    blanket: bool = False

    @property
    def ok(self) -> bool:
        """通过 = 引到了标注段，且不是"全引一遍"。"""

        return self.verdict == "ok" and not self.blanket


def classify_failure(
    *,
    verdict: str,
    cited_chunk_ids: Sequence[str],
    context_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    blanket: bool = False,
) -> str | None:
    """把 `verdict` 再拆一层，指向"该修哪一层"。

    刻意不看 `reasons` 的文案：文案会改，判据不能跟着改。这里只依赖
    "引了哪些块 / context 里有哪些块 / 标注了哪些块"三个事实。
    """

    gold = set(gold_chunk_ids)
    in_context = gold & set(context_chunk_ids)

    if verdict != "ok":
        if verdict == "unknown_index":
            return FAIL_HALLUCINATED_INDEX
        if verdict == "source_missing":
            return FAIL_SOURCE_MISSING
        if not in_context:
            # 标注段压根没进 context：跟模型无关，先去修召回。
            # 这一条占多数时，"改 prompt"是白费功夫——这是本脚本最想拦住的一种误判。
            return FAIL_RETRIEVAL_MISS
        if verdict == "no_citation":
            return FAIL_CITATION_MISS
        if not (set(cited_chunk_ids) & gold):
            # gold 在 context 里，却引了别的段：定位到了正确段落附近但抄错了编号。
            return FAIL_CITATION_MISS
        return FAIL_OFF_TARGET

    # 引用成立，但把整个 context 都引了一遍 —— 那等于没有区分证据。
    # 放在最后判：检索没召回时先报 retrieval_miss 更有用，
    # "全引一遍"只是那之后的次级问题。
    return FAIL_BLANKET if blanket else None


def summarize(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """汇总。分母只算可答样本；应拒答样本单列，混进来会让两个指标互相污染。"""

    answerable = [case for case in cases if case.kind == "answerable" and case.gold_chunk_ids]
    insufficient = [case for case in cases if case not in answerable]
    ok = sum(1 for case in answerable if case.ok)

    verdicts = Counter(case.verdict for case in answerable)
    failures = Counter(case.failure_kind for case in answerable if case.failure_kind)

    latencies = sorted(case.answer_ms for case in answerable)
    precisions = [case.precision for case in answerable if case.precision is not None]
    total_cited = sum(len(case.cited_chunk_ids) for case in answerable)
    total_hit = sum(
        len(set(case.cited_chunk_ids) & set(case.gold_chunk_ids)) for case in answerable
    )
    return {
        "total": len(cases),
        "answerable": len(answerable),
        "insufficient": len(insufficient),
        "passed": ok,
        "accuracy": (ok / len(answerable)) if answerable else 0.0,
        # 引用干净度，两个口径都要，因为它们在报告里回答的是不同问题：
        #   precision       = 全部引用中命中标注段的比例（micro，主指标）
        #   precision_macro = 每条答案各自精确率的均值（长答案与短答案等权）
        # 两者差 6pp 时正是长答案引得多、命中的少——只看其中一个会看丢这件事。
        # **都不设门槛**：标注只覆盖 69/425 个块，precision 的上限由标注完整性决定，
        # 拿它当门禁等于让标注工作量决定验收结果。但它必须被看见：
        # 准确率靠"至少一条命中"通过时，precision 掉下来就是答案在灌水。
        "precision": (total_hit / total_cited) if total_cited else 0.0,
        "precision_macro": (sum(precisions) / len(precisions)) if precisions else 0.0,
        "cited_total": total_cited,
        "cited_matched": total_hit,
        "blanket_cases": sum(1 for case in answerable if case.blanket),
        "verdicts": dict(verdicts),
        "failure_kinds": dict(failures),
        # 应拒答的题正确的表现是"不编引用"。单列一栏，不并进准确率：
        # 两类样本的分母含义不同，合并后一个变好会掩盖另一个变坏。
        "insufficient_refused": sum(
            1 for case in insufficient if not case.cited_chunk_ids
        ),
        "gold_in_context": sum(1 for case in answerable if case.gold_in_context),
        # 与 W1 基线同一分位算法（线性插值，q ∈ [0,1]）。换算法就等于换口径，
        # 两边的 p95 从此不可比。样本太少时 percentile 返回 None，这里补 0.0
        # 并在报告里照常打印——"用 3 个点算出来的 p95"比显示 0 更有害。
        "answer_p50_ms": percentile(latencies, 0.50) or 0.0,
        "answer_p95_ms": percentile(latencies, 0.95) or 0.0,
        "input_tokens": sum(case.input_tokens for case in cases),
        "output_tokens": sum(case.output_tokens for case in cases),
        "degraded_cases": sum(1 for case in cases if case.degraded),
    }


# --------------------------------------------------------------------------- #
# 实跑
# --------------------------------------------------------------------------- #


def _settings(overrides: dict[str, Any]) -> Any:
    from agent_framework.config import Settings

    return Settings(**overrides)


async def _run_live(
    samples: Sequence[Sample],
    *,
    overrides: dict[str, Any],
    device: str | None,
    warmup: bool,
) -> tuple[list[CaseResult], dict[str, Any]]:
    """返回 `(逐条结果, id -> CitationContext)`。

    context 一起带出来是为了 `--record`：replay 要重新审计，而审计需要
    当时的 context 原文与编号映射。只存答案，replay 就无从判断"引用的编号
    当时指向哪一段"，只能信当时那个 verdict——那就不是复算，是抄结论。
    """
    from agent_framework.bootstrap import build_gateway
    from agent_framework.knowledge import CitationExpectation, audit_citations
    from growth_agent.answer import KnowledgeAnswerer
    from growth_agent.rag import build_runtime

    if device:
        overrides = {**overrides, "retrieval_device": device}
    settings = _settings(overrides)
    runtime = build_runtime(settings)
    gateway = build_gateway(settings)
    answerer = KnowledgeAnswerer(runtime.pipeline, gateway, settings=settings)

    # 首条查询会带上 tokenizer / 线程池 / 连接池的惰性初始化。混进延迟统计
    # 就是一行"第一条特别慢"的假数据，而它和模型质量毫无关系。
    if warmup:
        try:
            await answerer.answer("预热", top_k=1)
        except Exception as exc:  # noqa: BLE001 - 预热失败不该中断抽样
            print(f"[warn] 预热失败（不影响后续样本）：{exc}", file=sys.stderr)

    results: list[CaseResult] = []
    contexts: dict[str, Any] = {}
    try:
        for sample in samples:
            started = time.perf_counter()
            answer = await answerer.answer(sample.question)
            elapsed = (time.perf_counter() - started) * 1000.0

            # 用**同一个** audit 函数、补上 ground truth 再判一次，
            # 而不是另写一套"评测专用判定"——两套判定必然漂移。
            audit = audit_citations(
                answer.text,
                answer.context,
                expected=CitationExpectation(chunk_ids=frozenset(sample.gold_chunk_ids)),
            )
            context_ids = [c.chunk_id for c in answer.context.citations]
            cited_ids = [c.chunk_id for c in audit.parsed.citations]
            contexts[sample.id] = answer.context
            blanket = is_blanket_citation(
                cited=len(cited_ids), context=len(context_ids)
            )

            results.append(
                CaseResult(
                    id=sample.id,
                    question=sample.question,
                    answer=answer.text,
                    verdict=audit.verdict,
                    reasons=list(audit.reasons),
                    failure_kind=classify_failure(
                        verdict=audit.verdict,
                        cited_chunk_ids=cited_ids,
                        context_chunk_ids=context_ids,
                        gold_chunk_ids=sample.gold_chunk_ids,
                        blanket=blanket,
                    ),
                    cited_chunk_ids=cited_ids,
                    context_chunk_ids=context_ids,
                    gold_chunk_ids=list(sample.gold_chunk_ids),
                    gold_in_context=bool(set(sample.gold_chunk_ids) & set(context_ids)),
                    context_ms=(
                        answer.diagnostics.total_ms if answer.diagnostics else 0.0
                    ),
                    answer_ms=elapsed,
                    input_tokens=answer.input_tokens,
                    output_tokens=answer.output_tokens,
                    degraded=list(answer.diagnostics.degraded) if answer.diagnostics else [],
                    kind=sample.kind,
                    precision=audit.precision,
                    blanket=blanket,
                )
            )
            print(
                f"  {sample.id}  {audit.verdict:<18} {answer.text[:40].replace(chr(10), ' ')}",
                file=sys.stderr,
            )
    finally:
        await gateway.aclose()
        runtime.close()
    return results, contexts


# --------------------------------------------------------------------------- #
# 落盘 / 复算
# --------------------------------------------------------------------------- #


def _citation_dump(citation: Any) -> dict[str, Any]:
    """完整 dump 一条 Citation。

    少 dump 一个字段，replay 就可能得到不同结论（例如漏掉 `marked_missing`，
    一条"出处缺失"的样本会在复算时变成"引用正确"）。
    """

    return {
        "index": citation.index,
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "source_uri": citation.source_uri,
        "excerpt": citation.excerpt,
        "page_start": citation.page_start,
        "page_end": citation.page_end,
        "section_path": list(citation.section_path),
        "title": citation.title,
        "marked_missing": citation.marked_missing,
    }


def write_run(
    path: Path,
    cases: Sequence[CaseResult],
    contexts: dict[str, Any],
    *,
    meta: dict[str, Any],
) -> None:
    payload = {
        "meta": meta,
        "summary": summarize(cases),
        # **全部** cases 都要落盘，包括应拒答题。
        # 只存可答题时，replay 会报"28 条、应拒答 0"，看起来一切正常而实际上
        # 丢掉了一整类样本——"拒绝编引用"这件事再也无法复算。
        "cases": [
            {
                **{
                    key: value
                    for key, value in case.__dict__.items()
                    if key != "answer"
                },
                "answer": case.answer,
                "context": {
                    "text": contexts[case.id].text,
                    "citations": [
                        _citation_dump(citation)
                        for citation in contexts[case.id].citations
                    ],
                },
            }
            for case in cases
            if case.id in contexts
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def _context_from_dump(raw: dict[str, Any]) -> Any:
    from agent_framework.knowledge import Citation, CitationContext

    return CitationContext(
        text=raw["text"],
        citations=tuple(
            Citation(
                index=int(item["index"]),
                chunk_id=item["chunk_id"],
                document_id=item.get("document_id", ""),
                source_uri=item.get("source_uri", ""),
                excerpt=item.get("excerpt", ""),
                page_start=item.get("page_start"),
                page_end=item.get("page_end"),
                section_path=tuple(item.get("section_path") or ()),
                title=item.get("title", ""),
                marked_missing=bool(item.get("marked_missing")),
            )
            for item in raw["citations"]
        ),
    )


def replay(path: Path) -> list[CaseResult]:
    """读已记录的 run 重新审计。**不加载模型、不调 LLM。**"""

    from agent_framework.knowledge import CitationExpectation, audit_citations

    payload = json.loads(path.read_text(encoding="utf-8"))
    results: list[CaseResult] = []
    for raw in payload["cases"]:
        context = _context_from_dump(raw["context"])
        expected = CitationExpectation(chunk_ids=frozenset(raw.get("gold_chunk_ids") or ()))
        audit = audit_citations(raw["answer"], context, expected=expected)
        context_ids = [c.chunk_id for c in context.citations]
        cited_ids = [c.chunk_id for c in audit.parsed.citations]
        blanket = is_blanket_citation(cited=len(cited_ids), context=len(context_ids))
        results.append(
            CaseResult(
                id=raw["id"],
                question=raw["question"],
                answer=raw["answer"],
                verdict=audit.verdict,
                reasons=list(audit.reasons),
                failure_kind=classify_failure(
                    verdict=audit.verdict,
                    cited_chunk_ids=cited_ids,
                    context_chunk_ids=context_ids,
                    gold_chunk_ids=raw.get("gold_chunk_ids") or (),
                    blanket=blanket,
                ),
                cited_chunk_ids=cited_ids,
                context_chunk_ids=context_ids,
                gold_chunk_ids=list(raw.get("gold_chunk_ids") or ()),
                gold_in_context=bool(set(raw.get("gold_chunk_ids") or ()) & set(context_ids)),
                context_ms=float(raw.get("context_ms") or 0.0),
                answer_ms=float(raw.get("answer_ms") or 0.0),
                input_tokens=int(raw.get("input_tokens") or 0),
                output_tokens=int(raw.get("output_tokens") or 0),
                degraded=list(raw.get("degraded") or []),
                kind=raw.get("kind") or "answerable",
                precision=audit.precision,
                blanket=blanket,
            )
        )
    return results


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def format_report(cases: Sequence[CaseResult]) -> str:
    summary = summarize(cases)
    lines = [
        "",
        "=== 引用准确率抽样 ===",
        f"样本 {summary['total']} 条"
        f"（可答 {summary['answerable']} / 应拒答 {summary['insufficient']}）",
        f"准确率 {summary['accuracy']:.4f}（{summary['passed']}/{summary['answerable']}）"
        f"  门槛 {DEFAULT_MIN_ACCURACY:.2f} → "
        f"{'PASS' if summary['accuracy'] >= DEFAULT_MIN_ACCURACY else 'FAIL'}",
        f"gold 进 context {summary['gold_in_context']}/{summary['answerable']}"
        f"  应拒答未编引用 {summary['insufficient_refused']}/{summary['insufficient']}",
        # precision 与 blanket 一起看：准确率靠"至少一条命中"通过时，
        # 这两列是唯一能发现"答案在灌水"的地方。
        f"引用精确率 {summary['precision']:.4f}"
        f"（{summary['cited_matched']}/{summary['cited_total']} 条引用命中标注段；"
        f"按题平均 {summary['precision_macro']:.4f}，不设门槛）"
        f"  全量引用 {summary['blanket_cases']} 条",
        f"判定分布 {summary['verdicts']}",
        f"失败归因 {summary['failure_kinds']}",
        f"答案延迟 p50 {summary['answer_p50_ms']:.0f}ms  p95 {summary['answer_p95_ms']:.0f}ms"
        f"  tokens in/out {summary['input_tokens']}/{summary['output_tokens']}",
    ]
    failures = [case for case in cases if case.kind == "answerable" and not case.ok]
    if failures:
        lines.append("")
        lines.append("--- 失败样本（逐条可查）---")
        for case in failures:
            lines.append(
                f"[{case.id}] {case.verdict} / {case.failure_kind}\n"
                f"  问题: {case.question}\n"
                f"  原因: {'; '.join(case.reasons)}\n"
                f"  引用: {case.cited_chunk_ids}\n"
                f"  标注: {case.gold_chunk_ids}\n"
                f"  答案: {case.answer[:200].replace(chr(10), ' ')}\n"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #


def self_test() -> int:
    """纯逻辑自检：不加载模型、不读文件、不调 LLM。"""

    from agent_framework.knowledge import (
        Chunk,
        CitationExpectation,
        ScoredChunk,
        audit_citations,
        build_context,
    )

    def chunk(chunk_id: str, *, uri: str = "https://docs.python.org/x") -> ScoredChunk:
        return ScoredChunk(
            chunk=Chunk(
                chunk_id=chunk_id,
                document_id="doc-1",
                text=f"正文 {chunk_id}",
                source_uri=uri,
            ),
            score=1.0,
            retriever="dense",
            rank=1,
        )

    failures: list[str] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            failures.append(f"{name}: 期望 {expected!r}，实际 {actual!r}")

    # 1. 正确引用 → ok
    ctx = build_context([chunk("a"), chunk("b")])
    gold_a = CitationExpectation(chunk_ids=frozenset({"a"}))
    audit = audit_citations("结论甲[1]。", ctx, expected=gold_a)
    check("ok", audit.verdict, "ok")

    # 2. 编号错位：引了 [2]，但 gold 是 a → off_target，且归因到 citation_miss
    audit = audit_citations("结论甲[2]。", ctx, expected=gold_a)
    check("错位判定", audit.verdict, "off_target")
    check(
        "错位归因",
        classify_failure(
            verdict=audit.verdict,
            cited_chunk_ids=[c.chunk_id for c in audit.parsed.citations],
            context_chunk_ids=["a", "b"],
            gold_chunk_ids=["a"],
        ),
        FAIL_CITATION_MISS,
    )

    # 3. 无引用 → no_citation
    audit = audit_citations("结论甲。", ctx, expected=gold_a)
    check("无引用判定", audit.verdict, "no_citation")

    # 4. gold 不在 context → 归因检索层（不是生成层）
    check(
        "检索缺失归因",
        classify_failure(
            verdict="no_citation",
            cited_chunk_ids=[],
            context_chunk_ids=["b"],
            gold_chunk_ids=["a"],
        ),
        FAIL_RETRIEVAL_MISS,
    )

    # 5. 编造编号 → hallucinated_index
    audit = audit_citations("结论甲[9]。", ctx)
    check("编造编号判定", audit.verdict, "unknown_index")
    check(
        "编造编号归因",
        classify_failure(
            verdict=audit.verdict,
            cited_chunk_ids=[],
            context_chunk_ids=["a"],
            gold_chunk_ids=["a"],
        ),
        FAIL_HALLUCINATED_INDEX,
    )

    # 6. 出处缺失 → source_missing（不能算"引用正确"）
    ctx_missing = build_context([chunk("c", uri="missing-source:no-uri")])
    audit = audit_citations("结论丙[1]。", ctx_missing)
    check("出处缺失判定", audit.verdict, "source_missing")

    # 7. 分母不随解析失败缩小：空集准确率是 0.0，不是 1.0
    from agent_framework.knowledge import citation_accuracy

    check("空集准确率", citation_accuracy([]), 0.0)

    if failures:
        print("self-test FAILED")
        for item in failures:
            print("  -", item)
        return 1
    print("self-test OK（7 项）")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="抽样集 jsonl")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖召回条数")
    parser.add_argument("--token-budget", type=int, default=None, help="覆盖 context 预算")
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=30_000.0,
        help=(
            "检索链路预算。**默认给足**：这个脚本测的是引用质量，不是预算压力。"
            "用生产默认 800ms 时，本机 CPU 上首次召回就会 1.0s 超时，"
            "报告会把'预算不够'写成'检索失败'——那是两回事"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试用）")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="默认 0：抽样必须可复现，同一次改动跑两遍要得到同一个数",
    )
    parser.add_argument(
        "--no-governance",
        action="store_true",
        help="关掉治理三条，用来做 A/B（默认开）",
    )
    parser.add_argument("--no-warmup", action="store_true", help="不丢弃首条（会污染延迟）")
    parser.add_argument("--record", type=Path, default=None, help="把本次 run 落盘（供 replay）")
    parser.add_argument("--replay", type=Path, default=None, help="复算已落盘的 run")
    parser.add_argument("--json", type=Path, default=None, help="只写结构化摘要")
    parser.add_argument("--self-test", action="store_true", help="只跑纯逻辑自检")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.self_test:
        return self_test()

    if args.replay:
        cases = replay(args.replay)
        print(format_report(cases))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(summarize(cases), ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        summary = summarize(cases)
        return 0 if summary["accuracy"] >= DEFAULT_MIN_ACCURACY else 1

    samples = load_sample(args.sample)
    if args.limit:
        samples = samples[: args.limit]
    if not any(sample.answerable for sample in samples):
        print("抽样集里没有可答样本（全部无 gold_chunk_ids）", file=sys.stderr)
        return 2

    overrides: dict[str, Any] = {
        "answer_temperature": args.temperature,
        "retrieval_budget_ms": args.budget_ms,
    }
    if args.top_k:
        overrides["retrieval_top_k"] = args.top_k
    if args.token_budget:
        overrides["retrieval_token_budget"] = args.token_budget
    if args.no_governance:
        overrides["retrieval_governance"] = False

    print(
        f"抽样 {len(samples)} 条 | top_k={args.top_k or '默认'} "
        f"| temperature={args.temperature} | governance={'off' if args.no_governance else 'on'}",
        file=sys.stderr,
    )
    cases, contexts = asyncio.run(
        _run_live(
            samples,
            overrides=overrides,
            device=args.device,
            warmup=not args.no_warmup,
        )
    )

    print(format_report(cases))
    summary = summarize(cases)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    if args.record:
        write_run(
            args.record,
            cases,
            contexts,
            meta={
                "sample": str(args.sample),
                "top_k": args.top_k,
                "temperature": args.temperature,
                "governance": not args.no_governance,
                "device": args.device,
            },
        )
        print(f"run 已落盘：{args.record}（用 --replay 复算）", file=sys.stderr)
    return 0 if summary["accuracy"] >= DEFAULT_MIN_ACCURACY else 1


if __name__ == "__main__":
    raise SystemExit(main())
