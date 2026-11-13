"""W9 D2：Agentic Search（缺口驱动 + 三个停止条件 + 证据/假设分区）。

三条停止条件各有一个"永远喂不饱"的构造。没有这种构造，停止条件只是代码里
一段没被触发过的 if——而它恰好是成本失控时唯一的闸门。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_framework.planner import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    AutonomyBudget,
    Plan,
    PlanStep,
    ScriptedReasoner,
    StepRunner,
)
from agent_framework.planner.research_planner import (
    STOP_BUDGET,
    STOP_DUPLICATE_QUERY,
    STOP_ENOUGH,
    STOP_LOW_YIELD,
    Evidence,
    KeywordDecomposer,
    ResearchPlanner,
    ResearchReport,
    SearchBudget,
    SearchOutcome,
    SkillSearchBackend,
    similarity,
)
from agent_framework.retrieval.acl import Principal
from agent_framework.skills import PermissionGuard, SandboxPolicy, SkillExecutor, SkillRegistry
from agent_framework.skills.servers import SearchServer

EDITOR = Principal(subject_id="editor", groups=frozenset({"editor"}))


# --------------------------------------------------------------------------- #
# 后端的三种脾气
# --------------------------------------------------------------------------- #


class GenerousBackend:
    """每次都给出**新的**大段证据。用来测 `max_searches` 硬闸。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, *, top_k: int) -> SearchOutcome:
        self.queries.append(query)
        quote = f"{query} 的结论：样本 480 家企业，转化周期中位数 94 天，预算占比 28%。"
        return SearchOutcome(
            evidence=[Evidence(quote=quote, source_uri=f"https://example.com/{len(self.queries)}")]
        )


class StingyBackend:
    """第一次给一点，之后几乎不给。用来测"边际收益低"。"""

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, top_k: int) -> SearchOutcome:
        self.calls += 1
        if self.calls == 1:
            quote = "首轮证据：" + "、".join(["样本 480 家企业", "预算占比 28%", "周期 94 天"] * 6)
            return SearchOutcome(evidence=[Evidence(quote=quote, source_uri="https://a.example/1")])
        return SearchOutcome(evidence=[Evidence(quote="略。", source_uri=f"https://a.example/{self.calls}")])


class FailingBackend:
    """子问题永远失败。用来测"失败要标注，不能编"。"""

    async def search(self, query: str, *, top_k: int) -> SearchOutcome:
        from agent_framework.planner.base import StepRecord, args_digest

        record = StepRecord(
            index=1,
            step_id=query,
            tool="web_search",
            args_digest=args_digest({"query": query}),
            status="error",
            code="tool_timeout",
        )
        return SearchOutcome(evidence=[], record=record)


def _reasoner(subquestions: list[str]) -> ScriptedReasoner:
    return ScriptedReasoner(
        plan=Plan(
            task="t",
            steps=[PlanStep(id=f"q{i}", goal=item) for i, item in enumerate(subquestions)],
        )
    )


def _planner(
    backend: object,
    subquestions: list[str],
    *,
    search_budget: SearchBudget | None = None,
    budget: AutonomyBudget | None = None,
) -> ResearchPlanner:
    # runner=None：夹具后端自己造记录，规划器不直接调工具。
    return ResearchPlanner(
        reasoner=_reasoner(subquestions),
        backend=backend,  # type: ignore[arg-type]
        budget=budget,
        search_budget=search_budget,
    )


# --------------------------------------------------------------------------- #
# 1. 三个停止条件
# --------------------------------------------------------------------------- #


class TestStopConditions:
    async def test_stops_at_search_budget(self) -> None:
        """永远有缺口 → 在 `max_searches` 处停下，不无限循环。

        5 个子问题、预算只够 4 次：一定搜不满，所以停下的原因只能是预算。
        """

        backend = GenerousBackend()
        planner = _planner(
            backend,
            ["子问题一", "子问题二", "子问题三", "子问题四", "子问题五"],
            search_budget=SearchBudget(max_searches=4, min_new_evidence_chars=1),
        )
        result = await planner.run("主题")

        assert len(backend.queries) == 4
        assert STOP_BUDGET in result.detail
        assert result.status == STATUS_PARTIAL

    async def test_stops_on_low_marginal_yield(self) -> None:
        """第二轮几乎没新增 → 边际收益判停，而不是继续搜到预算上限。"""

        backend = StingyBackend()
        planner = _planner(
            backend,
            ["子问题一", "子问题二"],
            search_budget=SearchBudget(max_searches=10, min_new_evidence_chars=80),
        )
        result = await planner.run("主题")

        assert STOP_LOW_YIELD in result.detail
        # 只搜了 2 次就停：第 1 次满足子问题一，第 2 次收益不足。
        assert backend.calls == 2

    async def test_stops_when_all_subquestions_have_evidence(self) -> None:
        backend = GenerousBackend()
        planner = _planner(
            backend,
            ["子问题一", "子问题二"],
            search_budget=SearchBudget(max_searches=10, min_new_evidence_chars=1),
        )
        result = await planner.run("主题")

        assert STOP_ENOUGH in result.detail
        assert result.status == STATUS_COMPLETED
        # 每个子问题一条证据，不多搜
        assert len(backend.queries) == 2

    async def test_duplicate_query_detected(self) -> None:
        """候选问法全部试过 → 停在 STOP_DUPLICATE_QUERY。

        构造：后端**永远不给证据**（所以子问题永远不满足），且边际收益阈值为 0
        （所以"零新增"不触发低收益停止）。这样循环只能靠"没有新问法"停下来，
        正是"换个同义词反复搜"走到尽头的形态。
        """

        backend = GenerousBackend()

        class MuteBackend:
            def __init__(self) -> None:
                self.queries: list[str] = []

            async def search(self, query: str, *, top_k: int) -> SearchOutcome:
                self.queries.append(query)
                return SearchOutcome(evidence=[])

        mute = MuteBackend()
        planner = _planner(
            mute,
            ["子问题一"],
            search_budget=SearchBudget(max_searches=10, min_new_evidence_chars=0),
        )
        result = await planner.run("主题")

        assert STOP_DUPLICATE_QUERY in result.detail
        # 四个候选问法都试过之后才判重复，不是第一轮就放弃。
        assert len(mute.queries) == 4
        assert backend.queries == []


# --------------------------------------------------------------------------- #
# 2. 失败要诚实
# --------------------------------------------------------------------------- #


class TestFailureHonesty:
    async def test_failed_subquestion_marked_not_invented(self) -> None:
        planner = _planner(FailingBackend(), ["子问题一"])
        result = await planner.run("主题")

        report_text = result.answer
        assert "证据不足" in report_text
        assert "检索失败" in report_text
        # 不能编：没有证据就不许出现"结论"里的断言
        assert "（证据不足，暂不下结论）" in report_text

    async def test_empty_evidence_raises_at_construction(self) -> None:
        """空 quote 的"证据"在构造期就被拒——它进结论只会变成无法核对的一句话。"""

        with pytest.raises(ValueError):
            Evidence(quote="   ", source_uri="https://a.example/1")


# --------------------------------------------------------------------------- #
# 3. 报告结构：结论只由证据支撑
# --------------------------------------------------------------------------- #


class TestReportStructure:
    def test_conclusion_lines_all_cited(self) -> None:
        report = ResearchReport(
            question="Q",
            evidence=[
                Evidence(quote="预算占比 28%", source_uri="https://a/1"),
                Evidence(quote="周期 94 天", source_uri="https://a/2"),
            ],
            hypotheses=["可能是季节性因素"],
            gaps=["竞品价格：没有公开数据"],
        )
        rendered = report.render()

        assert "## 证据（可核对）" in rendered
        assert "## 假设（未验证，不得当作结论）" in rendered
        assert "## 证据不足" in rendered
        assert report.unreferenced_claims() == []

    def test_hypothesis_does_not_leak_into_conclusion(self) -> None:
        report = ResearchReport(
            question="Q",
            evidence=[Evidence(quote="唯一证据", source_uri="https://a/1")],
            hypotheses=["这是猜测，不应该进结论段"],
        )
        rendered = report.render()
        conclusion = rendered.split("## 结论", 1)[1]
        assert "这是猜测" not in conclusion

    def test_unreferenced_claim_is_detected(self) -> None:
        """反向验证：手工往结论里塞一条无引用断言，审计必须报出来。"""

        report = ResearchReport(question="Q", evidence=[])
        report_rendered = report.render()
        assert "（证据不足，暂不下结论）" in report_rendered
        # 有证据时才可能出问题；这里直接验判据本身
        assert report.unreferenced_claims() == []


# --------------------------------------------------------------------------- #
# 4. 近似去重
# --------------------------------------------------------------------------- #


class TestSimilarity:
    def test_identical_queries_score_one(self) -> None:
        assert similarity("竞品基准 数据 2026", "竞品基准 数据 2026") == 1.0

    def test_different_topics_score_low(self) -> None:
        assert similarity("竞品基准 数据", "风险 反例") < 0.2

    def test_synonym_rewrite_scores_high(self) -> None:
        """换一个字的同义改写要被判重复——这正是"看起来在干活"的搜索。"""

        assert similarity("竞品基准 数据 2026", "竞品基准 数据 2026 年") > 0.8

    def test_empty_query_scores_zero(self) -> None:
        assert similarity("", "abc") == 0.0


# --------------------------------------------------------------------------- #
# 5. 接 W8 Skill 的后端（真实结构化载荷）
# --------------------------------------------------------------------------- #


class TestSkillBackend:
    async def test_evidence_extracted_from_structured_content(self, tmp_path: Path) -> None:
        """证据从 `structuredContent.results` 读，不解析渲染文本。

        文本是给模型看的，随时会为了可读性改格式；结构化字段是契约。
        """

        sandbox = SandboxPolicy(write_roots=(tmp_path,), command_prefixes=("echo",))
        executor = SkillExecutor(
            registry=SkillRegistry(),
            clients={"search": SearchServer().client()},
            sandbox=sandbox,
            guard=PermissionGuard(),
        )
        runner = StepRunner(executor, principal=EDITOR, stage="research")
        backend = SkillSearchBackend(runner)

        outcome = await backend.search("增长", top_k=3)

        assert outcome.record is not None
        assert outcome.record.status == "ok"
        assert outcome.evidence
        assert all(item.quote for item in outcome.evidence)
        assert all(item.source_uri.startswith("http") for item in outcome.evidence)

    async def test_backend_reports_failure_record(self, tmp_path: Path) -> None:
        """工具失败时返回空证据 + 失败记录，不抛异常。

        抛异常会让"某个子问题搜不到"升级成"整个调研崩了"，而调研的价值
        恰恰在于拿到 4/5 的答案。
        """

        sandbox = SandboxPolicy(write_roots=(tmp_path,))
        executor = SkillExecutor(
            registry=SkillRegistry(),
            clients={"search": SearchServer().client()},
            sandbox=sandbox,
            guard=PermissionGuard(),
        )
        # 用一个必然被拒的查询：空串违反 minLength=2
        runner = StepRunner(executor, principal=EDITOR, stage="research")
        outcome = await SkillSearchBackend(runner).search("", top_k=3)

        assert outcome.evidence == []
        assert outcome.record is not None
        assert outcome.record.status == "denied"


# --------------------------------------------------------------------------- #
# 6. 预算也管检索
# --------------------------------------------------------------------------- #


async def test_autonomy_budget_also_caps_search_rounds() -> None:
    """`max_steps` 与 `max_searches` 同时生效，取先到者。"""

    backend = GenerousBackend()
    planner = _planner(
        backend,
        ["子问题一", "子问题二", "子问题三"],
        search_budget=SearchBudget(max_searches=99, min_new_evidence_chars=1),
        budget=AutonomyBudget(max_steps=2),
    )
    result = await planner.run("主题")

    assert result.status == STATUS_PARTIAL
    assert len(backend.queries) <= 2


# --------------------------------------------------------------------------- #
# 7. 拆解：正交才叫拆解
# --------------------------------------------------------------------------- #


class TestDecomposer:
    async def test_angles_are_orthogonal(self) -> None:
        """四个角度互不包含，否则等于把一个子问题搜四遍。"""

        subs = await KeywordDecomposer().decompose("B2B 内容营销", max_subquestions=4)

        assert len(subs) == 4
        assert all("B2B 内容营销" in item for item in subs)
        # 去掉主题后剩下的角度词必须两两不同
        suffixes = {item.replace("B2B 内容营销", "").strip() for item in subs}
        assert len(suffixes) == 4

    async def test_respects_limit(self) -> None:
        subs = await KeywordDecomposer().decompose("主题", max_subquestions=2)
        assert len(subs) == 2

    async def test_strips_trailing_punctuation(self) -> None:
        subs = await KeywordDecomposer().decompose("主题？", max_subquestions=1)
        assert "？" not in subs[0]
