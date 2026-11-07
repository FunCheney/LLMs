"""引用准确率抽样脚本的单测（W6 D1 / Gate 6 #2）。

只测**纯逻辑**：分类归因、汇总分母、record→replay 往返一致。不加载模型、
不调 LLM——真正的抽样要跑 18–20s 冷启动 + 几十次生成，塞进 `make test`
会让全量用例从 11s 变成分钟级，最后所有人都用 `-k` 跳过它。

重点验三件事：

1. **失败归因指向正确的层**。把"检索没召回"误判成"模型没引用"，下周就会去改
   prompt —— 改完指标不动，再改一轮。这个判据是脚本最有价值的部分，必须钉住。
2. **分母不缩水**。应拒答题不进准确率分母；解析失败不减分母；空集返回 0.0
   而不是 1.0（空集上的 1.0 是最容易糊弄过去的绿）。
3. **replay 忠实**。落盘时少一个字段（比如 `marked_missing`），复算就会把
   "出处缺失"变成"引用正确"，而两次报告都写着"通过"。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "evals" / "scripts" / "citation_accuracy.py"
SAMPLE_PATH = REPO_ROOT / "evals" / "golden" / "citation_sample_v1.jsonl"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("citation_accuracy", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # 必须先登记到 sys.modules：脚本里有 dataclass，创建时要回查
    # `sys.modules[cls.__module__]`，不登记会 AttributeError。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script()


# --------------------------------------------------------------------------- #
# 抽样集本身
# --------------------------------------------------------------------------- #


def test_sample_file_has_enough_chunk_level_cases(script: ModuleType) -> None:
    """W6 要求 ≥20 条带 ground truth 的样本，且判据必须是 chunk 级。

    只有 URL 级 `references` 的样本不算：这 425 个块来自同一份文档，
    退到文档级判据后准确率恒等于 100%，等于没测。
    """

    samples = script.load_sample(SAMPLE_PATH)
    answerable = [sample for sample in samples if sample.answerable]

    assert len(answerable) >= 20
    assert all(sample.gold_chunk_ids for sample in answerable)
    # id 唯一，否则汇总时会把同一条算两次
    assert len({sample.id for sample in samples}) == len(samples)


def test_every_gold_chunk_id_is_a_real_chunk_id(script: ModuleType) -> None:
    """gold 里不能有拼错的 id。

    一个拼错的 id 会让该样本永远判失败，而报告上只显示"引用准确率低一点"——
    查起来要翻 30 条答案。
    """

    samples = script.load_sample(SAMPLE_PATH)
    for sample in samples:
        for chunk_id in sample.gold_chunk_ids:
            # 语料 id 形如 claude-code-0001-p112-p112-c242
            assert chunk_id.startswith("claude-code-"), (sample.id, chunk_id)
            assert "-c" in chunk_id, (sample.id, chunk_id)


# --------------------------------------------------------------------------- #
# 失败归因
# --------------------------------------------------------------------------- #


def test_classify_ok_has_no_failure_kind(script: ModuleType) -> None:
    assert (
        script.classify_failure(
            verdict="ok",
            cited_chunk_ids=["a"],
            context_chunk_ids=["a"],
            gold_chunk_ids=["a"],
        )
        is None
    )


def test_classify_retrieval_miss_when_gold_absent_from_context(
    script: ModuleType,
) -> None:
    """gold 没进 context → 检索层。这时改 prompt 是白费功夫。"""

    assert (
        script.classify_failure(
            verdict="no_citation",
            cited_chunk_ids=[],
            context_chunk_ids=["b", "c"],
            gold_chunk_ids=["a"],
        )
        == script.FAIL_RETRIEVAL_MISS
    )


def test_classify_citation_miss_when_gold_in_context_but_not_cited(
    script: ModuleType,
) -> None:
    """gold 在 context 里却没被引 → 生成层（prompt / 模型）。"""

    assert (
        script.classify_failure(
            verdict="no_citation",
            cited_chunk_ids=[],
            context_chunk_ids=["a", "b"],
            gold_chunk_ids=["a"],
        )
        == script.FAIL_CITATION_MISS
    )


def test_classify_wrong_target_is_citation_miss(script: ModuleType) -> None:
    """引了别的段（编号错位）也归生成层：检索已经把它送到嘴边了。"""

    assert (
        script.classify_failure(
            verdict="off_target",
            cited_chunk_ids=["b"],
            context_chunk_ids=["a", "b"],
            gold_chunk_ids=["a"],
        )
        == script.FAIL_CITATION_MISS
    )


def test_classify_hallucinated_index(script: ModuleType) -> None:
    assert (
        script.classify_failure(
            verdict="unknown_index",
            cited_chunk_ids=[],
            context_chunk_ids=["a"],
            gold_chunk_ids=["a"],
        )
        == script.FAIL_HALLUCINATED_INDEX
    )


def test_classify_source_missing(script: ModuleType) -> None:
    assert (
        script.classify_failure(
            verdict="source_missing",
            cited_chunk_ids=["a"],
            context_chunk_ids=["a"],
            gold_chunk_ids=["a"],
        )
        == script.FAIL_SOURCE_MISSING
    )


# --------------------------------------------------------------------------- #
# 防作弊：全量引用
# --------------------------------------------------------------------------- #


def test_blanket_requires_enough_context(script: ModuleType) -> None:
    """context 只有 2–3 块时全引是合理的（就那么几段），不算灌水。"""

    assert script.is_blanket_citation(cited=3, context=3) is False
    assert script.is_blanket_citation(cited=5, context=5) is True
    assert script.is_blanket_citation(cited=4, context=5) is False


def test_blanket_citation_does_not_pass(script: ModuleType) -> None:
    """把整个 context 都引一遍 → 不算通过。

    "至少一条命中"这个宽松口径的代价就是这条：模型只要全引，必然命中。
    所以这里必须补一道反制，否则准确率会被"每段挂个编号"刷到 100%。
    """

    case = _case(
        script,
        verdict="ok",
        blanket=True,
        cited_chunk_ids=["a", "b", "c", "d"],
        # 按真实路径赋值：`classify_failure` 决定 failure_kind，测试不另造一条规则。
        failure_kind=script.classify_failure(
            verdict="ok",
            cited_chunk_ids=["a", "b", "c", "d"],
            context_chunk_ids=["a", "b", "c", "d"],
            gold_chunk_ids=["a"],
            blanket=True,
        ),
    )
    summary = script.summarize([case])

    assert case.ok is False
    assert summary["accuracy"] == 0.0
    assert summary["blanket_cases"] == 1
    assert summary["failure_kinds"] == {script.FAIL_BLANKET: 1}


def test_blanket_does_not_mask_retrieval_miss(script: ModuleType) -> None:
    """检索没召回时先报 retrieval_miss：那才是要修的地方。"""

    assert (
        script.classify_failure(
            verdict="off_target",
            cited_chunk_ids=["x", "y", "z", "w"],
            context_chunk_ids=["x", "y", "z", "w"],
            gold_chunk_ids=["a"],
            blanket=True,
        )
        == script.FAIL_RETRIEVAL_MISS
    )


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


def _case(script: ModuleType, **kwargs: object) -> object:
    base: dict[str, object] = {
        "id": kwargs.pop("id", "s1"),
        "question": "q",
        "answer": "a",
        "verdict": "ok",
        "gold_chunk_ids": ["a"],
        "context_chunk_ids": ["a"],
        "cited_chunk_ids": ["a"],
        "kind": "answerable",
    }
    base.update(kwargs)
    return script.CaseResult(**base)


def test_summary_excludes_insufficient_from_denominator(script: ModuleType) -> None:
    """应拒答题不并进准确率：两类分母含义不同，合并会互相掩盖。"""

    cases = [
        _case(script, id="ok1"),
        _case(script, id="bad", verdict="no_citation", cited_chunk_ids=[]),
        # 应拒答且确实没编引用 —— 它不该拉高准确率，也不该拉低
        _case(script, id="neg", gold_chunk_ids=[], kind="insufficient", cited_chunk_ids=[]),
    ]

    summary = script.summarize(cases)

    assert summary["answerable"] == 2
    assert summary["insufficient"] == 1
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["insufficient_refused"] == 1


def test_precision_reports_micro_and_macro(script: ModuleType) -> None:
    """micro 与 macro 必须都给：它们差得远时，正是"长答案引得多、命中少"。

    只给按题平均（macro）会把一条引了 8 条的答案与一条引了 2 条的答案等权，
    于是"引用总量很大但命中率低"这种情况在报告上被平均掉了。
    """

    cases = [
        # 一条引 1 命中 1；一条引 3 命中 1
        _case(script, id="a", cited_chunk_ids=["g"], gold_chunk_ids=["g"], precision=1.0),
        _case(
            script,
            id="b",
            cited_chunk_ids=["g", "x", "y"],
            gold_chunk_ids=["g"],
            precision=1 / 3,
        ),
    ]

    summary = script.summarize(cases)

    assert summary["cited_total"] == 4
    assert summary["cited_matched"] == 2
    assert summary["precision"] == pytest.approx(0.5)  # micro
    assert summary["precision_macro"] == pytest.approx(2 / 3)  # macro


def test_summary_on_empty_is_zero_not_one(script: ModuleType) -> None:
    summary = script.summarize([])

    assert summary["accuracy"] == 0.0
    assert summary["answerable"] == 0


def test_summary_counts_failure_kinds(script: ModuleType) -> None:
    cases = [
        _case(script, id="r", verdict="no_citation", cited_chunk_ids=[], context_chunk_ids=["b"],
              failure_kind=script.FAIL_RETRIEVAL_MISS),
        _case(script, id="g", verdict="no_citation", cited_chunk_ids=[],
              failure_kind=script.FAIL_CITATION_MISS),
    ]

    summary = script.summarize(cases)

    assert summary["failure_kinds"] == {
        script.FAIL_RETRIEVAL_MISS: 1,
        script.FAIL_CITATION_MISS: 1,
    }


# --------------------------------------------------------------------------- #
# record → replay 往返
# --------------------------------------------------------------------------- #


def _fake_context(script: ModuleType) -> object:
    from agent_framework.knowledge import Chunk, ScoredChunk

    chunk = Chunk(
        chunk_id="a",
        document_id="doc-1",
        text="正文甲",
        source_uri="https://docs.python.org/x",
        page_start=3,
        page_end=4,
        section_path=("第三章", "3.2"),
    )
    scored = ScoredChunk(chunk=chunk, score=1.0, retriever="dense", rank=1)
    from agent_framework.knowledge import build_context

    return build_context([scored])


def test_record_and_replay_agree(script: ModuleType, tmp_path: Path) -> None:
    """同一份答案，实跑判什么，复算就判什么。"""

    context = _fake_context(script)
    cases = [
        _case(
            script,
            id="s1",
            answer="结论甲[1]。",
            verdict="ok",
            context_chunk_ids=["a"],
            cited_chunk_ids=["a"],
        )
    ]
    run_path = tmp_path / "run.json"
    script.write_run(run_path, cases, {"s1": context}, meta={"sample": "test"})

    replayed = script.replay(run_path)

    assert [case.verdict for case in replayed] == ["ok"]
    assert replayed[0].context_chunk_ids == ["a"]


def test_replay_preserves_missing_source(script: ModuleType, tmp_path: Path) -> None:
    """落盘漏掉 `marked_missing` 会把"出处缺失"复算成"引用正确"。

    这条是 write_run 字段完整性的守门测试：它不测审计逻辑，它测"你存够了没有"。
    """

    from agent_framework.knowledge import Citation, CitationContext

    context = CitationContext(
        text="[1] 出处缺失\n正文",
        citations=(
            Citation(
                index=1,
                chunk_id="a",
                document_id="doc-1",
                source_uri="missing-source:no-uri",
            ),
        ),
    )
    cases = [_case(script, id="s1", answer="结论[1]。", verdict="source_missing")]
    run_path = tmp_path / "run.json"
    script.write_run(run_path, cases, {"s1": context}, meta={})

    replayed = script.replay(run_path)

    assert replayed[0].verdict == "source_missing"


def test_replay_recomputes_verdict_instead_of_trusting_record(
    script: ModuleType, tmp_path: Path
) -> None:
    """记录里写 `ok`，但答案其实没有引用 → 复算必须判 no_citation。

    如果 replay 直接读记录里的 verdict，那就不是"复算"，是"抄结论"。
    """

    context = _fake_context(script)
    cases = [_case(script, id="s1", answer="结论甲（没有编号）。", verdict="ok")]
    run_path = tmp_path / "run.json"
    script.write_run(run_path, cases, {"s1": context}, meta={})

    replayed = script.replay(run_path)

    assert replayed[0].verdict == "no_citation"


def test_replay_keeps_insufficient_cases(script: ModuleType, tmp_path: Path) -> None:
    """应拒答题必须一起落盘。

    只存可答题时，replay 会报"28 条、应拒答 0"——数字看着正常，但"拒绝编引用"
    这件事再也无法复算，而这正是 Gate 6 里最容易变坏的一格。
    """

    context = _fake_context(script)
    cases = [
        _case(script, id="s1", answer="结论甲[1]。", verdict="ok"),
        _case(
            script,
            id="neg",
            answer="资料未覆盖。",
            verdict="no_citation",
            gold_chunk_ids=[],
            cited_chunk_ids=[],
            kind="insufficient",
        ),
    ]
    run_path = tmp_path / "run.json"
    script.write_run(run_path, cases, {"s1": context, "neg": context}, meta={})

    summary = script.summarize(script.replay(run_path))

    assert summary["total"] == 2
    assert summary["insufficient"] == 1
    assert summary["insufficient_refused"] == 1
    # 准确率分母仍然只有可答题
    assert summary["answerable"] == 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_self_test_flag_passes() -> None:
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--self-test"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "self-test OK" in completed.stdout


def test_sample_file_is_valid_jsonl() -> None:
    for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("//"):
            json.loads(line)
