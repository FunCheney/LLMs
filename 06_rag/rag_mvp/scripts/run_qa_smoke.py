"""
CLI：Phase 3 生成链路冒烟测试（L2，对应 data/eval/L2_generate_smoke.md）。

数据集：data/eval/qa_smoke.yaml（10 题验收问题集，含期望章节 / 期望要点 / 是否应拒答）。

两个轮次：
1. 第一轮「检索上下文预检」（默认，不调 LLM）：
   逐题检索 Top-K，检查期望章节是否被召回（等价于 ask_cli.py --dry-run 的判定，
   只是直接调 retriever，免去拼上下文）。正例题未召回 = 失败（记录为「检索未召回」，
   不怪 LLM）；负例题（expect_insufficient=true）只记录召回的噪声章节，不参与判定。
2. 第二轮「真实问答 + 落盘」（--with-llm，需 DEEPSEEK_API_KEY）：
   调 answer_question() 得到 RAGResponse，逐题存到 data/eval/reports/qa_phase3/<id>.json，
   并自动检查：
   - 拒答正确性：insufficient_evidence 是否等于 expect_insufficient；
   - 引用编号：答案中的 [n] 是否越界（1..检索条数之外视为模型编造编号）；
   - 正例题非拒答答案是否至少带一个 [n] 标记。
   另生成 review_checklist.md：把正确性 / 忠实度 / 引用 / 拒答四项
   「逐题人工验收」指标列成打勾清单（自动检查结果已预填）。

--review-only：不跑检索和 LLM，只读取已落盘的 JSON 重新跑自动检查并刷新报告
（改了检查逻辑后不用重打 LLM 即可复核）。

用法：
    uv run python scripts/run_qa_smoke.py                 # 第一轮：检索预检
    uv run python scripts/run_qa_smoke.py --with-llm      # 第一+二轮（需 .env 配置 DEEPSEEK_API_KEY）
    uv run python scripts/run_qa_smoke.py --review-only   # 复核已落盘结果
    uv run python scripts/run_qa_smoke.py --ids qa-001,qa-010 --with-llm
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval.matchers import find_first_matching_rank, hit_matches_rules
from eval.smoke_schema import SmokeMatchRules
from generation.pipeline import answer_question
from retrieval.factory import RetrieverConfig, create_retriever
from retrieval.types import RetrievalResult

# 答案中引用标记的格式：[1] [2] ...
CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
@dataclass
class QAQuestion:
    """一条验收问题及其基准（来自 qa_smoke.yaml）。"""

    id: str
    question: str
    tags: list[str]
    expect_insufficient: bool      # True = 库外负例题，期望拒答
    expected_sections: list[str]   # 期望答案引用的章节（用于召回判定）
    expected_points: list[str]     # 期望答案包含的核心要点（人工验收用）
    notes: str = ""


@dataclass
class QADataset:
    dataset_id: str
    description: str
    mode: str
    top_k: int
    questions: list[QAQuestion]


def load_qa_dataset(path: str | Path) -> QADataset:
    """加载 qa_smoke.yaml，并做基本校验。"""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"数据集不存在: {path}")
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    defaults = data.get("defaults") or {}
    questions: list[QAQuestion] = []
    for item in data.get("questions") or []:
        if not item.get("id") or not item.get("question"):
            raise SystemExit(f"数据集条目缺少 id/question: {item}")
        q = QAQuestion(
            id=item["id"],
            question=item["question"],
            tags=list(item.get("tags") or []),
            expect_insufficient=bool(item.get("expect_insufficient", False)),
            expected_sections=list(item.get("expected_sections") or []),
            expected_points=list(item.get("expected_points") or []),
            notes=str(item.get("notes") or ""),
        )
        # 正例题必须给出期望章节，否则第一轮召回无从判定
        if not q.expect_insufficient and not q.expected_sections:
            raise SystemExit(f"{q.id} 是正例题但未配置 expected_sections")
        questions.append(q)

    if not questions:
        raise SystemExit(f"数据集为空: {path}")

    return QADataset(
        dataset_id=str(data.get("dataset_id", "qa-smoke")),
        description=str(data.get("description") or ""),
        mode=str(defaults.get("mode", "hybrid")),
        top_k=int(defaults.get("top_k", 5)),
        questions=questions,
    )


# ---------------------------------------------------------------------------
# 第一轮：检索预检
# ---------------------------------------------------------------------------
@dataclass
class RecallCheck:
    """检索预检结果。hit=None 表示负例题（只观察噪声，不判定）。"""

    hit: bool | None
    first_rank: int | None
    top_sections: list[str] = field(default_factory=list)


def check_recall(question: QAQuestion, hits: list[RetrievalResult]) -> RecallCheck:
    top_sections = [hit.chunk.section_title for hit in hits]
    if question.expect_insufficient:
        # 负例题：Top-K 理应全是噪声，仅记录，不判通过/失败
        return RecallCheck(hit=None, first_rank=None, top_sections=top_sections)

    rules = SmokeMatchRules(any_section_title=question.expected_sections)
    hit = any(hit_matches_rules(h, rules) for h in hits)
    return RecallCheck(
        hit=hit,
        first_rank=find_first_matching_rank(hits, rules),
        top_sections=top_sections,
    )


# ---------------------------------------------------------------------------
# 第二轮：真实问答的自动检查
# ---------------------------------------------------------------------------
@dataclass
class GenerationCheck:
    """对一份 RAGResponse 的自动化检查（指标 3/4 的可判定部分）。"""

    insufficient: bool                 # 模型是否判定证据不足
    refusal_ok: bool                   # 拒答行为是否符合期望
    markers: list[int]                 # 答案中出现的 [n] 编号
    out_of_range: list[int]            # 越界编号（模型编造的引用）
    no_marker_positive: bool           # 正例题正常作答却一个标记都没有
    json_path: str                     # 落盘位置


def check_generation(
    question: QAQuestion,
    answer: str,
    citations: list[dict],
    insufficient: bool,
    retrieved_chunk_ids: list[str],
    json_path: str,
) -> GenerationCheck:
    # citations 传 dict 而不是 Citation，这样 review-only 模式读 JSON 也能复用本函数
    del citations  # 目前编号检查只依赖检索条数；citations 留给将来扩展
    markers = sorted({int(m) for m in CITATION_MARKER_PATTERN.findall(answer)})
    n_retrieved = len(retrieved_chunk_ids)
    out_of_range = [m for m in markers if m < 1 or m > n_retrieved]

    refusal_ok = insufficient == question.expect_insufficient
    no_marker_positive = (
        not question.expect_insufficient and not insufficient and not markers
    )
    return GenerationCheck(
        insufficient=insufficient,
        refusal_ok=refusal_ok,
        markers=markers,
        out_of_range=out_of_range,
        no_marker_positive=no_marker_positive,
        json_path=json_path,
    )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
@dataclass
class QASmokeResult:
    id: str
    question: str
    tags: list[str]
    expect_insufficient: bool
    notes: str
    recall: RecallCheck | None = None
    generation: GenerationCheck | None = None
    passed: bool = True
    failures: list[str] = field(default_factory=list)


def aggregate_result(
    question: QAQuestion,
    recall: RecallCheck | None,
    generation: GenerationCheck | None,
) -> QASmokeResult:
    """把两轮检查汇总成单题结论；failures 非空即未通过。"""
    failures: list[str] = []

    if recall is not None and recall.hit is False:
        failures.append("检索未召回：Top-K 未命中任何期望章节（属检索问题，不怪 LLM）")

    if generation is not None:
        if not generation.refusal_ok:
            failures.append(
                f"拒答不符期望：insufficient_evidence={generation.insufficient}，"
                f"期望 {question.expect_insufficient}"
            )
        if generation.out_of_range:
            failures.append(f"引用编号越界：{generation.out_of_range}（检索只给了 Top-K 编号）")
        if generation.no_marker_positive:
            failures.append("正例题答案没有任何 [n] 引用标记（指标 3 不达标）")

    return QASmokeResult(
        id=question.id,
        question=question.question,
        tags=question.tags,
        expect_insufficient=question.expect_insufficient,
        notes=question.notes,
        recall=recall,
        generation=generation,
        passed=not failures,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# 输出：控制台 / summary.json / 人工验收清单
# ---------------------------------------------------------------------------
def _result_to_dict(result: QASmokeResult) -> dict:
    return {
        "id": result.id,
        "question": result.question,
        "tags": result.tags,
        "expect_insufficient": result.expect_insufficient,
        "passed": result.passed,
        "failures": result.failures,
        "notes": result.notes,
        "recall": None
        if result.recall is None
        else {
            "hit": result.recall.hit,
            "first_rank": result.recall.first_rank,
            "top_sections": result.recall.top_sections,
        },
        "generation": None
        if result.generation is None
        else {
            "insufficient": result.generation.insufficient,
            "refusal_ok": result.generation.refusal_ok,
            "markers": result.generation.markers,
            "out_of_range": result.generation.out_of_range,
            "no_marker_positive": result.generation.no_marker_positive,
            "json_path": result.generation.json_path,
        },
    }


def write_summary(
    dataset: QADataset,
    mode: str,
    top_k: int,
    results: list[QASmokeResult],
    with_llm: bool,
    report_dir: Path,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_id": dataset.dataset_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "top_k": top_k,
        "rounds": {"retrieval_precheck": True, "llm": with_llm},
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "results": [_result_to_dict(r) for r in results],
    }
    path = report_dir / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def write_review_checklist(
    results: list[QASmokeResult],
    questions: dict[str, QAQuestion],
    report_dir: Path,
) -> Path:
    """生成逐题人工验收清单：四项指标 + 期望要点 + 答案与引用速览。"""
    lines: list[str] = [
        "# Phase 3 QA 人工验收清单",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "自动检查已预填到对应指标后；「正确性 / 忠实度」需人工对照期望要点打勾。",
    ]
    for result in results:
        q = questions[result.id]
        lines += [
            "",
            "---",
            "",
            f"## {result.id}：{result.question}",
            "",
            f"- tags：{', '.join(result.tags) or '-'}　期望拒答：{result.expect_insufficient}",
            f"- 自动检查：{'通过' if result.passed else '未通过'}"
            + (f"（{'；'.join(result.failures)}）" if result.failures else ""),
            "",
            "期望要点：",
            *[f"  - [ ] {point}" for point in q.expected_points],
            "",
            "四项指标：",
            "  - [ ] 1. 答案正确性（核心要点是否准确、有无答非所问）",
            "  - [ ] 2. 忠实度（结论是否 100% 来自 Context，无模型自由发挥）",
        ]
        gen = result.generation
        if gen is None:
            lines.append("  - [ ] 3. 引用准确性（未跑第二轮，暂无自动检查）")
            lines.append("  - [ ] 4. 无证据拒答（未跑第二轮，暂无自动检查）")
        else:
            marker_desc = (
                f"标记 {gen.markers}，越界 {gen.out_of_range or '无'}"
                if gen.markers
                else "答案中无 [n] 标记"
            )
            lines.append(
                f"  - [{'x' if gen.markers and not gen.out_of_range else ' '}] "
                f"3. 引用准确性（自动：{marker_desc}；仍需人工核对编号与 Citations 是否对应）"
            )
            refusal_desc = "符合期望" if gen.refusal_ok else "不符合期望"
            lines.append(
                f"  - [{'x' if gen.refusal_ok else ' '}] "
                f"4. 无证据拒答（自动：insufficient_evidence={gen.insufficient}，{refusal_desc}）"
            )
        lines += ["", f"答案与引用存档：`{gen.json_path if gen else '（未落盘）'}`"]
    path = report_dir / "review_checklist.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_console_report(
    results: list[QASmokeResult], mode: str, top_k: int, with_llm: bool
) -> None:
    print(f"\nMode: {mode}  k={top_k}  rounds: 检索预检{' + LLM问答' if with_llm else ''}")
    print("=" * 72)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        parts = [f"[{status}] {r.id} {r.question}"]
        if r.recall is not None:
            if r.recall.hit is None:
                parts.append("负例题(噪声观察)")
            elif r.recall.hit:
                parts.append(f"召回@{r.recall.first_rank}")
            else:
                parts.append("未召回")
        if r.generation is not None:
            g = r.generation
            parts.append(f"拒答={'✓' if g.refusal_ok else '✗'}")
            parts.append(
                f"引用{g.markers or '无'}" + (f" 越界{g.out_of_range}" if g.out_of_range else "")
            )
        print(" | ".join(parts))
        for failure in r.failures:
            print(f"    ✗ {failure}")
        if r.recall is not None and r.recall.hit is None:
            # 负例题：展示召回的噪声，便于预判 LLM 是否会被误导
            print(f"    噪声章节: {r.recall.top_sections}")
    passed = sum(1 for r in results if r.passed)
    print("=" * 72)
    print(f"passed {passed}/{len(results)}")


# ---------------------------------------------------------------------------
# 运行逻辑
# ---------------------------------------------------------------------------
def _check_indexes(mode: str, dense_index: Path, bm25_index: Path) -> None:
    """索引存在性 + 非空校验（空文件通常是上次构建中断留下的）。"""
    def _bad(path: Path, build_cmd: str) -> str | None:
        if not path.exists():
            return f"索引不存在: {path}\nRun: {build_cmd}"
        if path.stat().st_size == 0:
            return f"索引文件为空（0 字节）: {path}\n可能是上次构建中断，请重建: {build_cmd}"
        return None

    if mode in {"dense", "hybrid"}:
        msg = _bad(dense_index, "uv run python scripts/build_vector_index.py")
        if msg:
            raise SystemExit(msg)
    if mode in {"sparse", "hybrid"}:
        msg = _bad(bm25_index, "uv run python scripts/build_bm25_index.py")
        if msg:
            raise SystemExit(msg)


def _ensure_llm_ready() -> None:
    """第二轮需要真实 LLM：确认 API Key 可用，否则给出可操作提示。"""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY 未配置，无法运行第二轮（真实问答）。\n"
            "请在项目根目录的 .env 中加入 DEEPSEEK_API_KEY=... 后重试，\n"
            "或先只跑第一轮（去掉 --with-llm）。"
        )


def run_review_only(
    dataset: QADataset, report_dir: Path, top_k: int, fail_on_miss: bool
) -> int:
    """复核模式：读取已落盘的 RAGResponse JSON，重新跑自动检查。"""
    results: list[QASmokeResult] = []
    for q in dataset.questions:
        json_path = report_dir / f"{q.id}.json"
        generation = None
        if json_path.is_file():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            generation = check_generation(
                q,
                answer=payload.get("answer", ""),
                citations=payload.get("citations") or [],
                insufficient=bool(payload.get("insufficient_evidence")),
                retrieved_chunk_ids=payload.get("retrieved_chunk_ids") or [],
                json_path=str(json_path),
            )
        else:
            generation = None
        result = aggregate_result(q, recall=None, generation=generation)
        if generation is None:
            result.passed = False
            result.failures.append(f"未找到落盘结果 {json_path}（先跑 --with-llm）")
        results.append(result)

    summary_path = write_summary(dataset, "-", top_k, results, with_llm=True, report_dir=report_dir)
    questions = {q.id: q for q in dataset.questions}
    checklist_path = write_review_checklist(results, questions, report_dir)
    print_console_report(results, mode="-", top_k=top_k, with_llm=True)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Review checklist saved to: {checklist_path}")
    if fail_on_miss and any(not r.passed for r in results):
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 生成链路冒烟测试：第一轮检索预检（默认）+ 第二轮真实问答落盘（--with-llm）",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/qa_smoke.yaml"),
        help="QA 验收问题集 YAML（默认 data/eval/qa_smoke.yaml）",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid"],
        default=None,
        help="检索模式（默认取数据集的 defaults.mode）",
    )
    parser.add_argument("--k", type=int, default=None, help="Top-K（默认取数据集 defaults.top_k）")
    parser.add_argument("--dense-index", type=Path, default=Path("data/index/local.pkl"))
    parser.add_argument("--bm25-index", type=Path, default=Path("data/index/bm25.pkl"))
    parser.add_argument("--device", default=None, help="Torch device，如 cpu / mps")
    parser.add_argument("--model", default=None, help="LLM 模型名（默认 DEEPSEEK_MODEL 或 deepseek-chat）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="运行第二轮：真实调用 LLM 问答并落盘 RAGResponse（需 DEEPSEEK_API_KEY）",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="不跑检索/LLM，只读取已落盘 JSON 重新跑自动检查",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/eval/reports/qa_phase3"),
        help="报告输出目录（默认 data/eval/reports/qa_phase3）",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="只跑指定题目，逗号分隔，如 qa-001,qa-010",
    )
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        default=True,
        help="有题目未通过时以退出码 1 结束（默认）",
    )
    parser.add_argument(
        "--no-fail-on-miss",
        action="store_false",
        dest="fail_on_miss",
        help="无论结果如何都以退出码 0 结束",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_qa_dataset(args.dataset)

    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        unknown = wanted - {q.id for q in dataset.questions}
        if unknown:
            raise SystemExit(f"数据集中不存在这些 id: {sorted(unknown)}")
        dataset.questions = [q for q in dataset.questions if q.id in wanted]

    mode = args.mode or dataset.mode
    top_k = args.k or dataset.top_k
    report_dir: Path = args.report_dir

    if args.review_only:
        sys.exit(run_review_only(dataset, report_dir, top_k, args.fail_on_miss))

    _check_indexes(mode, args.dense_index, args.bm25_index)
    # if args.with_llm:
    #     _ensure_llm_ready()

    retriever = create_retriever(
        RetrieverConfig(
            mode=mode,
            dense_index_path=args.dense_index,
            bm25_index_path=args.bm25_index,
            device=args.device,
        )
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    results: list[QASmokeResult] = []
    total = len(dataset.questions)
    for index, q in enumerate(dataset.questions, start=1):
        print(f"[{index}/{total}] {q.id} {q.question}")

        # 第一轮：检索预检（dry-run，不调 LLM）
        hits = retriever.retrieve(q.question, top_k=top_k)
        recall = check_recall(q, hits)

        # 第二轮：真实问答 + 落盘
        generation = None
        if args.with_llm:
            response = answer_question(
                q.question,
                retriever,
                retrieval_mode=mode,
                top_k=top_k,
                model=args.model,
                temperature=args.temperature,
            )
            json_path = report_dir / f"{q.id}.json"
            json_path.write_text(
                json.dumps(response.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            generation = check_generation(
                q,
                answer=response.answer,
                citations=[c.__dict__ for c in response.citations],
                insufficient=response.insufficient_evidence,
                retrieved_chunk_ids=response.retrieved_chunk_ids,
                json_path=str(json_path),
            )

        results.append(aggregate_result(q, recall=recall, generation=generation))

    summary_path = write_summary(
        dataset, mode, top_k, results, with_llm=args.with_llm, report_dir=report_dir
    )
    print_console_report(results, mode, top_k, with_llm=args.with_llm)
    print(f"\nSummary saved to: {summary_path}")

    if args.with_llm:
        questions = {q.id: q for q in dataset.questions}
        checklist_path = write_review_checklist(results, questions, report_dir)
        print(f"Review checklist saved to: {checklist_path}")
        print("下一步：打开 checklist 逐题人工验收「正确性 / 忠实度」两项。")

    if args.fail_on_miss and any(not r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
