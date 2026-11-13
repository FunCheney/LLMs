"""Agentic Search：缺口驱动的多轮检索（W9 D2）。

**名字很重要。** W5 的检索是"召回一次取 Top-K"；这里是"围绕信息缺口反复检索"。
两者的评测指标完全不同：前者看 recall/precision，后者看"几个子问题拿到了证据、
几轮收敛、有没有重复搜同一句话"。把这件事叫"RAG"会让人以为沿用 W5 的指标。

为什么不在 `PlanExecutePlanner` 上继承：静态 Plan 要求"跑之前就知道有哪几步"，
而 Agentic Search 的计划是**跑的过程中被发现的**——第二轮搜什么取决于第一轮
缺什么。硬套静态 Plan 会把"缺口驱动"这件事藏进一个每次循环都重建的 Plan 里，
读者以为它有计划，其实它只有游标。所以这里直接基于 `PlannerBase`，
并把"计划"显式表达成 `Gap` 列表。

三类停止条件缺一不可，它们防的是三种不同的烧钱方式：

| 条件 | 防的是 |
|---|---|
| 证据足够 | 模型永远觉得"还能再查一下" |
| 预算耗尽 | 成本彻底失控（唯一硬闸） |
| 边际收益低 | 换同义词反复搜同一件事——最贵也最像在干活的那种失败 |
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..retrieval.context_assemble import estimate_tokens
from .base import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    Decision,
    PlannerBase,
    Reasoner,
    StepRunner,
    TaskResult,
)
from .controls import AutonomyBudget, StopReason

__all__ = [
    "STOP_BUDGET",
    "STOP_DUPLICATE_QUERY",
    "STOP_ENOUGH",
    "STOP_LOW_YIELD",
    "Evidence",
    "Gap",
    "KeywordDecomposer",
    "ResearchPlanner",
    "ResearchReport",
    "SearchBudget",
    "SearchBackend",
    "SearchOutcome",
    "SkillSearchBackend",
    "similarity",
]


# --------------------------------------------------------------------------- #
# 停止原因（本模块专属，与 StopReason 分开）
# --------------------------------------------------------------------------- #

STOP_ENOUGH = "enough_evidence"
STOP_BUDGET = "search_budget"
STOP_LOW_YIELD = "low_marginal_yield"
STOP_DUPLICATE_QUERY = "duplicate_query"


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Evidence:
    """一条可引用的证据。**没有 quote 的不算证据**——只有标题的"证据"
    在结论里会变成"据说某报告提到"，而读者无法核对。"""

    quote: str
    source_uri: str
    title: str = ""
    subquestion: str = ""

    def __post_init__(self) -> None:
        if not self.quote.strip():
            raise ValueError("Evidence.quote 不能为空：没有原文摘录就不是证据")


@dataclass(slots=True)
class Gap:
    """信息缺口。**决定下一轮搜什么，而不是把上一轮重搜一遍。**"""

    subquestion: str
    # 缺什么：缺数字 / 缺反例 / 缺时效 / 缺来源。写出来才知道该换角度。
    missing: str = "证据"
    # 已经搜过的 query。**必须记**，否则模型换个同义词就把预算花在同一个坑里。
    searched: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    # 检索失败标记。不标的话，同一个失败子问题会被反复挑中，
    # 每轮都撞一次同样的错——"卡在一个死掉的子问题上"也是一种死循环。
    failed: bool = False

    @property
    def satisfied(self) -> bool:
        return bool(self.evidence) or self.failed


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """检索预算。

    `min_new_evidence_chars` 是"边际收益"的度量：一轮新增有效证据少于这个数
    就认为这一轮白跑。用字符数而不是条数，是因为"搜到 3 条一模一样的转载"
    在条数上很好看，实际零信息。
    """

    max_searches: int = 12
    min_new_evidence_chars: int = 120
    top_k: int = 5
    # 近似去重阈值：字符 bigram Jaccard。0.8 意味着"基本同一句话"。
    similarity_threshold: float = 0.8


@dataclass(slots=True)
class SearchOutcome:
    """一次检索的结果。

    `record` 由**执行方**填（走 Skill 的后端能给出真实的 ToolExecution），
    纯夹具后端给 None，由规划器补一条合成记录。这样 trace 里一定有东西，
    而执行只发生一次——早先的版本规划器与后端各调一次检索，
    同一句话被搜两遍、预算也只扣了一半。
    """

    evidence: list[Evidence] = field(default_factory=list)
    record: Any = None


class SearchBackend(Protocol):
    """检索能力。产品层接 W5 `RetrievalPipeline` 或 W8 `web_search` Skill。"""

    async def search(self, query: str, *, top_k: int) -> SearchOutcome: ...


class Decomposer(Protocol):
    async def decompose(self, task: str, *, max_subquestions: int) -> list[str]: ...


# --------------------------------------------------------------------------- #
# 拆解
# --------------------------------------------------------------------------- #

# 正交角度。**固定四问**在离线档够用，也解释了"什么叫正交"：
# 定义/规模/竞品/风险互不包含，不会出现"子问题 2 是子问题 1 的复述"。
_ANGLES: tuple[str, ...] = ("定义与边界", "规模与数据", "竞品与替代", "风险与反例")


@dataclass(slots=True)
class KeywordDecomposer:
    """确定性拆解：主题 × 四个正交角度。

    为什么不用模型：拆解质量是 Agentic Search 的上限，理应交给模型；
    但离线用例与对比实验需要一个**可复现**的拆解器——模型每次拆得不一样，
    "第 3 轮停了"就无法归因到停止条件还是拆解变了。所以两档拆解器都提供，
    默认给确定性的这个，在线档换成 `LLMReasoner` 的规划能力。
    """

    angles: tuple[str, ...] = _ANGLES

    async def decompose(self, task: str, *, max_subquestions: int) -> list[str]:
        core = task.strip().rstrip("。？?").strip()
        return [f"{core} 的{angle}" for angle in self.angles[:max_subquestions]]


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ResearchReport:
    """调研产物。**证据 / 假设 / 缺口三段分开**，结论只由证据支撑。

    这三段混在一起是 W4 归因里 `HALLUCINATION` 的主要来源：模型推测的
    "可能是季节性因素"和真实检索到的"转化周期中位数 94 天"在文本里长得一样，
    读者（以及下一个模型）分不出来。
    """

    question: str
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    rounds: int = 0
    searches: int = 0
    stop_reason: str = ""

    def render(self) -> str:
        lines = [f"# {self.question}", "", "## 证据（可核对）", ""]
        if self.evidence:
            for index, item in enumerate(self.evidence, start=1):
                title = f"{item.title} — " if item.title else ""
                lines.append(f"[{index}] {title}{item.quote}（{item.source_uri}）")
        else:
            lines.append("（没有拿到任何可引用证据）")

        lines += ["", "## 假设（未验证，不得当作结论）", ""]
        lines += [f"- {item}" for item in self.hypotheses] or ["（无）"]

        lines += ["", "## 证据不足", ""]
        lines += [f"- {item}" for item in self.gaps] or ["（无）"]

        lines += ["", "## 结论", ""]
        lines += self.conclusion_lines() or ["（证据不足，暂不下结论）"]
        return "\n".join(lines)

    def conclusion_lines(self) -> list[str]:
        """结论只用**有引用编号**的断言。

        一条结论对应一条证据（编号就是它在 `evidence` 里的位置），
        这样"这句话从哪来"在文本里就能追。没有证据的位置宁可不写。
        """

        return [
            f"- {item.quote} [{index}]"
            for index, item in enumerate(self.evidence, start=1)
            if item.quote.strip()
        ]

    def unreferenced_claims(self) -> list[str]:
        """结论里没有引用编号的行。空列表 = 每条断言都能追到出处。

        这是"结论段无未引用断言"的**可执行判据**，不是靠人读一遍。
        """

        import re

        pattern = re.compile(r"\[\d+\]")
        bad: list[str] = []
        in_conclusion = False
        for line in self.render().splitlines():
            if line.startswith("## "):
                in_conclusion = line.strip() == "## 结论"
                continue
            if in_conclusion and line.strip().startswith("- ") and not pattern.search(line):
                bad.append(line.strip())
        return bad


# --------------------------------------------------------------------------- #
# 规划器
# --------------------------------------------------------------------------- #


class ResearchPlanner(PlannerBase):
    """缺口驱动检索。每轮：算缺口 → 生成 query → 去重 → 检索 → 更新缺口。"""

    name = "research"

    def __init__(
        self,
        *,
        reasoner: Reasoner,
        backend: SearchBackend,
        runner: StepRunner | None = None,
        decomposer: Decomposer | None = None,
        budget: AutonomyBudget | None = None,
        search_budget: SearchBudget | None = None,
        clock: Any = None,
        observation_window: int = 8,
    ) -> None:
        kwargs: dict[str, Any] = {
            "reasoner": reasoner,
            "runner": runner,
            "budget": budget,
            "observation_window": observation_window,
        }
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(**kwargs)
        self._backend = backend
        # `decomposer` 显式给就用它（离线档要可复现）；不给就让 reasoner 拆——
        # 拆解质量是 Agentic Search 的上限，那本来就该由模型决定。
        self._decomposer = decomposer
        self._search_budget = search_budget or SearchBudget()

    async def run(self, task: str) -> TaskResult:
        subquestions = await self._decompose(task, limit=min(5, self._budget.max_steps))
        gaps = [Gap(subquestion=item) for item in subquestions]
        report = ResearchReport(question=task)
        stop = ""

        while True:
            reason = self._check_continue()
            if reason is not None:
                stop = reason.value
                break
            if all(gap.satisfied for gap in gaps):
                stop = STOP_ENOUGH
                break
            if report.searches >= self._search_budget.max_searches:
                stop = STOP_BUDGET
                break

            gap = next((item for item in gaps if not item.satisfied), None)
            if gap is None:
                stop = STOP_ENOUGH
                break

            query = self._next_query(gap, history=[q for item in gaps for q in item.searched])
            if query is None:
                # 换了角度也没有新问法：这是"搜了很多遍同一件事"的前兆，
                # 记成缺口而不是继续烧预算。
                stop = STOP_DUPLICATE_QUERY
                report.gaps.append(f"{gap.subquestion}：已试过的问法都重复，停止检索")
                gap.failed = True
                break

            gap.searched.append(query)
            report.searches += 1
            report.rounds += 1

            outcome = await self._backend.search(query, top_k=self._search_budget.top_k)
            self._record(outcome.record or _synthetic_record(query, self._trace.step_count + 1))

            if getattr(outcome.record, "status", "ok") not in ("ok",):
                # 单个子问题失败 → 标注证据不足，不中断整份报告。
                # 调研的价值在于"拿到 4/5 的答案"，全有全无是最差的产品决策。
                code = getattr(outcome.record, "code", "") or "error"
                report.gaps.append(f"{gap.subquestion}：检索失败（{code}）")
                gap.failed = True
                continue

            new_chars = _merge_evidence(gap, outcome.evidence)

            if new_chars < self._search_budget.min_new_evidence_chars:
                # 边际收益低：这一轮的预算花掉了但几乎没增量。停。
                stop = STOP_LOW_YIELD
                report.gaps.append(
                    f"{gap.subquestion}：新增证据 {new_chars} 字，低于阈值"
                    f" {self._search_budget.min_new_evidence_chars}"
                )
                break

        report.evidence = [item for gap in gaps for item in gap.evidence]
        report.stop_reason = stop
        return self._result(
            task=task,
            status=STATUS_COMPLETED if stop == STOP_ENOUGH else STATUS_PARTIAL,
            reason=_to_stop_reason(stop),
            answer=report.render(),
            detail=f"停止条件：{stop}；检索 {report.searches} 次，{report.rounds} 轮",
            steps_total=len(gaps),
        )

    # -- 内部 -------------------------------------------------------------- #

    async def _decompose(self, task: str, *, limit: int) -> list[str]:
        """主题 → 正交子问题。"""

        if self._decomposer is not None:
            return await self._decomposer.decompose(task, max_subquestions=limit)
        plan = await self.reasoner.propose_plan(task, max_steps=limit)
        goals = [step.goal.strip() for step in plan.steps if step.goal.strip()]
        # 拆不出子问题就退回"整题一问"：空列表会让循环立刻判"证据足够"然后交白卷。
        return goals[:limit] or [task.strip()]

    def _next_query(self, gap: Gap, *, history: Sequence[str] = ()) -> str | None:
        """按"缺什么"换角度生成 query，并做近似去重。

        模型最常见的烧钱方式不是搜不到，而是**换个同义词再搜一遍**：
        成本翻三倍，证据一条没多。所以候选问法要先跟自己搜过的比，
        再跟所有子问题搜过的比。
        """

        candidates = [
            gap.subquestion,
            f"{gap.subquestion} 数据 2026",
            f"{gap.subquestion} 报告 案例",
            f"{gap.subquestion} 反例 风险",
        ]
        seen = set(gap.searched) | set(history)
        for candidate in candidates:
            if candidate in seen:
                continue
            if any(
                similarity(candidate, old) >= self._search_budget.similarity_threshold
                for old in seen
            ):
                continue
            return candidate
        return None


def _merge_evidence(gap: Gap, fresh: Iterable[Evidence]) -> int:
    """并入新证据，返回**新增字符数**。同一条来源+摘录只算一次。

    口径用字符而不是 token：阈值要能和人工观察对上（"这一轮就多了两句话"），
    而 token 估算本身就有 1.6 字符/token 的粗糙度，再拿它当阈值只是双重噪音。
    """

    seen = {(item.source_uri, item.quote) for item in gap.evidence}
    added = 0
    for item in fresh:
        key = (item.source_uri, item.quote)
        if key in seen:
            continue
        seen.add(key)
        gap.evidence.append(item)
        added += len(item.quote) + len(item.title)
    return added


def _synthetic_record(query: str, index: int) -> Any:
    """夹具后端没有真实 ToolExecution 时补一条轨迹记录。

    没有它，纯夹具实验的 `path_repeat_rate` 与 `steps_used` 会是 0——
    评测报告上看着"很省"，实际是没记。
    """

    from .base import StepRecord, args_digest

    return StepRecord(
        index=index,
        step_id=query[:40],
        tool="web_search",
        args_digest=args_digest({"query": query}),
        status="ok",
        tokens=estimate_tokens(query),
        observation=query,
    )


def _to_stop_reason(stop: str) -> StopReason:
    mapping = {
        STOP_ENOUGH: StopReason.COMPLETED,
        STOP_BUDGET: StopReason.MAX_STEPS,
        STOP_LOW_YIELD: StopReason.NO_PROGRESS,
        STOP_DUPLICATE_QUERY: StopReason.NO_PROGRESS,
    }
    return mapping.get(stop, StopReason.ERROR)


# --------------------------------------------------------------------------- #
# 近似去重
# --------------------------------------------------------------------------- #


def similarity(left: str, right: str) -> float:
    """字符 bigram Jaccard。中文没有空格，词级切分要先分词；bigram 够用且零依赖。

    阈值 0.8 的含义：两个 query 八成以上的二元组相同，基本就是同一句话换个说法。
    """

    a = _bigrams(left)
    b = _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _bigrams(text: str) -> frozenset[str]:
    compact = "".join(text.split())
    if len(compact) < 2:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[i : i + 2] for i in range(len(compact) - 1))


# --------------------------------------------------------------------------- #
# 接 W8 Skill 的检索后端
# --------------------------------------------------------------------------- #


class SkillSearchBackend:
    """走 `SkillExecutor` 的检索后端：审计、沙箱、回放全都经过 W8 那套。

    结果从 MCP 的 `structuredContent.results` 读——**不解析渲染文本**。
    文本是给模型看的，格式会为了可读性变；结构化字段是契约。
    """

    def __init__(self, runner: StepRunner, *, tool: str = "web_search") -> None:
        self._runner = runner
        self._tool = tool

    async def search(self, query: str, *, top_k: int) -> SearchOutcome:
        record, payload = await self._runner.run_structured(
            Decision(
                kind="tool",
                tool=self._tool,
                arguments={"query": query, "max_results": top_k},
            )
        )
        if record.status != "ok" or not payload:
            return SearchOutcome(evidence=[], record=record)
        rows = payload.get("results") or []
        evidence: list[Evidence] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            snippet = str(row.get("snippet") or "").strip()
            if not snippet:
                continue
            evidence.append(
                Evidence(
                    quote=snippet,
                    source_uri=str(row.get("url") or ""),
                    title=str(row.get("title") or ""),
                    subquestion=query,
                )
            )
        return SearchOutcome(evidence=evidence, record=record)

