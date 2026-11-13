"""单 Agent vs 多 Agent 的三层对比实验（W9 D5）。

跑四种编排，同一批任务、同一个"任务世界"、同一份策略：

| 配置 | 编排 | 观察窗口 |
|---|---|---|
| `react_w3` | ReAct | 3（默认） |
| `react_w8` | ReAct | 8 |
| `plan_execute_w8` | Plan-and-Execute | 8（默认） |
| `multi_agent` | Supervisor（researcher + validator） | 无（共享状态） |

加 `react_w8` 这一档是为了**归因**：如果 `react_w3` 输而 `react_w8` 打平，
那么差异来自"记忆机制（窗口 vs 共享状态）"，不来自"多 Agent"这个名号。
只比 ReAct(3) 与多 Agent，会把窗口的功劳记在"分工"头上——这是最常见的错误归因。

**任务世界是确定性的**（`PolicySearch` 按查询精确返回，无网络），所以：

- 优点：结果可复现，跑两次必然同值；
- 代价：Multi-Trial 的极差**恒为 0**。它证明"可复现"，**不证明"稳定"**。
  稳定性要看在线档（`LLMReasoner`），本次没跑，报告里必须写明这一条。

产出：
- `<out>/traces.jsonl` —— 每条 case 的轨迹（喂给 `trajectory_eval.py`）
- `<out>/summary.json` —— 各配置 × 各难度档的指标
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_framework.multi_agent import Supervisor, SupervisorConfig
from agent_framework.planner import (
    AutonomyBudget,
    PlanExecutePlanner,
    PlanStep,
    PolicyReasoner,
    ReActPlanner,
    StepRunner,
    TaskPolicy,
)
from agent_framework.retrieval.acl import Principal
from agent_framework.skills import (
    PermissionGuard,
    SandboxPolicy,
    SkillExecutor,
    SkillRegistry,
)
from agent_framework.skills.servers import SearchServer, WorkspaceServer
from agent_framework.skills.servers.upstream import SearchHit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_eval import (  # noqa: E402
    aggregate_trajectories,
    invalid_call_rate,
    path_repeat_rate,
    steps_used,
    tool_selection_acc,
)

EDITOR = Principal(subject_id="editor", groups=frozenset({"editor"}))
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "w9_agent_v1.jsonl"

# 窗口 1 → 3 → 8 是一条**剂量曲线**：如果覆盖率随窗口单调上升而与配置名无关，
# 那么差异的来源就是"记忆机制"，不是"多 Agent"。
CONFIGS = (
    "react_w1",
    "react_w3",
    "react_w8",
    "plan_execute_w3",
    "plan_execute_w8",
    "react_w8_noisy",
    "multi_agent",
)
_WINDOWS = {
    "react_w1": 1,
    "react_w3": 3,
    "react_w8": 8,
    "plan_execute_w3": 3,
    "plan_execute_w8": 8,
    # 最后一档用同一个窗口，只把"听话程度"调低：用来证明过程层指标不是装饰品。
    "react_w8_noisy": 8,
}


# --------------------------------------------------------------------------- #
# 任务世界
# --------------------------------------------------------------------------- #


class ValuePolicy(TaskPolicy):
    """把"每步的正确产出"变成一个**可被文本检索到的唯一标记**。

    父类返回 query 本身，那样"答案对不对"没法测——query 天然出现在报告里。
    用自己的值（`值0`）之后，报告里有没有它就成了可判定的：查错了查询的步骤
    拿不到这个标记，覆盖率自然掉下来。
    """

    def value_of(self, index: int) -> str:
        return f"值{index}"


class NoisyPolicyReasoner(PolicyReasoner):
    """模拟"模型会选错工具"：第一步先调一个**不存在的工具**，然后自我纠正。

    为什么要这一档：`PolicyReasoner` 完全听话，所以过程层的四个指标在它身上
    恒为「工具选择 1.00 / 无效调用 0 / 路径重复 0」——看起来像四个装饰品。
    真实的失败恰恰发生在这里：模型编了一个工具名，被 OPA 层挡住，然后重试。
    这一档让那类失败在数据里出现一次。
    """

    _noised: bool = False

    async def next_action(self, state: Any) -> Any:
        from agent_framework.planner import Decision

        if not self._noised:
            self._noised = True
            return Decision(
                kind="tool",
                tool="web_search_v2",
                arguments={"query": "多 Agent 编排 定义"},
                tokens=self.tokens_per_call,
            )
        return await super().next_action(state)


class PolicySearch:
    """确定性的"搜索引擎"。**它只认精确查询**，这是整个实验的物理定律。

    三种返回：

    1. 查询匹配到某一步 → 返回该步的真值；
    2. 是复核步但没把要复核的值带进来 → 返回一个**看起来正常但内容错**的结论
       （真实系统里最常见的失败形态：工具成功了，答案错了）；
    3. 完全匹配不上（模型编了一个查询）→ 空结果。
    """

    def __init__(self, policy: ValuePolicy) -> None:
        self._policy = policy
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.calls.append(query)
        steps = self._policy.task.steps
        for index, step in enumerate(steps):
            base = str(step.arguments.get("query") or step.goal)
            if base not in query:
                continue
            source = self._policy.task.recall_from.get(index)
            if source is not None:
                true_value = self._policy.value_of(source)
                if f"（复核 {true_value}）" not in query:
                    return [self._hit(index, f"无结论{index}", query)]
            return [self._hit(index, self._policy.value_of(index), query)]
        return []

    @staticmethod
    def _hit(index: int, value: str, query: str) -> SearchHit:
        return SearchHit(
            title=f"结果{index}｜{query[:12]}",
            url=f"https://example.test/w9/{index}",
            snippet=f"这一步的结论是：{value}",
            rank=1,
        )


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #


def _executor(tmp: Path, upstream: PolicySearch) -> SkillExecutor:
    sandbox = SandboxPolicy(write_roots=(tmp,), command_prefixes=("echo",))
    return SkillExecutor(
        registry=SkillRegistry(),
        clients={
            "search": SearchServer(upstream).client(),
            "workspace": WorkspaceServer(root=tmp, sandbox=sandbox).client(),
        },
        sandbox=sandbox,
        guard=PermissionGuard(approver=_approve),
    )


async def _approve(skill: object, arguments: object, summary: str) -> str:
    return "approved"


def _task_policy(raw: dict[str, Any]) -> ValuePolicy:
    steps = tuple(
        PlanStep(
            id=f"s{index}",
            goal=str(step["goal"]),
            tool="web_search",
            arguments={"query": str(step["query"]), "max_results": 3},
        )
        for index, step in enumerate(raw["steps"])
    )
    recall = {
        index: int(step["recall_from"])
        for index, step in enumerate(raw["steps"])
        if "recall_from" in step
    }
    from agent_framework.planner.reasoners import PolicyTask

    return ValuePolicy(
        PolicyTask(
            id=str(raw["id"]),
            question=str(raw["question"]),
            steps=steps,
            recall_from=recall,
            difficulty=int(raw["difficulty"]),
        )
    )


def _expected_queries(policy: ValuePolicy) -> tuple[str, ...]:
    """多 Agent 档的检索词：**有完整共享状态时本应提出的词**。

    这是刻意为它构造的"能力上限"：真实系统里这个查询由 LLM 用共享状态生成。
    不这么构造，就等于让多 Agent 也用 ReAct 的窗口去回忆——那不是它的运作方式，
    而是把对手的缺陷安在它身上。
    """

    queries: list[str] = []
    for index, step in enumerate(policy.task.steps):
        base = str(step.arguments.get("query") or step.goal)
        source = policy.task.recall_from.get(index)
        if source is None:
            queries.append(base)
        else:
            queries.append(f"{base}（复核 {policy.value_of(source)}）")
    return tuple(queries)


@dataclass(slots=True)
class Outcome:
    """一条 case 的观测。三层指标各取几个数，缺一层就没法归因。"""

    case_id: str
    config: str
    difficulty: int
    status: str
    stop_reason: str
    coverage: float
    answer: str
    records: list[dict[str, Any]] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    # 每一步真正取回来的证据（工具返回的原文）。Outcome 从它判，不从答案判。
    evidence: list[str] = field(default_factory=list)
    tokens: int = 0
    seconds: float = 0.0
    turns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "config": self.config,
            "difficulty": self.difficulty,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "coverage": self.coverage,
            "tokens": self.tokens,
            "seconds": round(self.seconds, 4),
            "turns": self.turns,
            # Trajectory 层，单条也算一遍，便于查"是哪条拖低了均值"
            "tool_selection_acc": tool_selection_acc(self.records, self.expected_tools),
            "invalid_call_rate": invalid_call_rate(self.records),
            "steps_used": steps_used(self.records),
            "path_repeat_rate": path_repeat_rate(self.records),
        }


async def run_one(raw: dict[str, Any], config: str, tmp: Path) -> Outcome:
    policy = _task_policy(raw)
    upstream = PolicySearch(policy)
    executor = _executor(tmp, upstream)
    budget = AutonomyBudget(max_steps=10, max_tokens=80_000, max_seconds=60)
    expected = list(policy.expected_tools())

    if config == "multi_agent":
        result = await Supervisor(
            executor=executor,
            config=SupervisorConfig(max_steps=12, max_budget_tokens=80_000),
            principal=EDITOR,
            stage="research",
            questions=_expected_queries(policy),
        ).run(policy.task.question)
    else:
        window = _WINDOWS[config]
        runner = StepRunner(executor, principal=EDITOR, stage="research")
        reasoner: Any = (
            NoisyPolicyReasoner(policy=policy)
            if config.endswith("_noisy")
            else PolicyReasoner(policy=policy)
        )
        planner_cls = ReActPlanner if config.startswith("react") else PlanExecutePlanner
        result = await planner_cls(
            reasoner=reasoner, runner=runner, budget=budget, observation_window=window
        ).run(policy.task.question)

    return Outcome(
        case_id=policy.task.id,
        config=config,
        difficulty=policy.task.difficulty,
        status=result.status,
        stop_reason=result.stop_reason,
        coverage=_coverage(result.trace.records, policy),
        answer=result.answer,
        records=[
            {
                "tool": record.tool,
                "args_digest": record.args_digest,
                "status": record.status,
                "code": record.code,
            }
            for record in result.trace.records
        ],
        evidence=[record.observation for record in result.trace.records],
        expected_tools=expected,
        tokens=result.trace.tokens,
        seconds=result.trace.seconds,
        turns=result.trace.turns,
    )


def _coverage(records: Sequence[Any], policy: ValuePolicy) -> float:
    """Outcome 层的判据：**每一步的真值都真的被取回来了**。

    判据从**工具返回的证据**里读，不从答案文本里读。原因是踩出来的：
    `PolicyReasoner.compose` 是按任务书合成答案的，答案里天然有每一步的真值——
    用它判覆盖，四种编排会全部 1.00（实测过），实验等于没做。
    这也是一条方法论结论：离线档的"答案正确"不能当 outcome，只有"证据到没到"
    是可信的。在线档才有资格判答案。

    用"每步真值是否出现"而不是"关键词是否出现"：前者能抓住"查对了但没用上"。
    """

    steps = policy.task.steps
    if not steps:
        return 0.0
    joined = "\n".join(record.observation for record in records)
    hit = sum(1 for index in range(len(steps)) if policy.value_of(index) in joined)
    return hit / len(steps)


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #


def summarize(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for config in CONFIGS:
        for tier, name in ((None, "all"), (1, "d1"), (3, "d3")):
            group = [
                item
                for item in outcomes
                if item.config == config
                and (tier is None or _tier(item.difficulty) == tier)
            ]
            if not group:
                continue
            traj = aggregate_trajectories(
                (item.records, item.expected_tools) for item in group
            )
            rows[f"{config}|{name}"] = {
                "config": config,
                "tier": name,
                "cases": len(group),
                "coverage_mean": statistics.fmean(item.coverage for item in group),
                "coverage_range": _range([item.coverage for item in group]),
                "completed": sum(1 for item in group if item.status == "completed"),
                "tokens_mean": statistics.fmean(item.tokens for item in group),
                "seconds_mean": statistics.fmean(item.seconds for item in group),
                "turns_mean": statistics.fmean(item.turns for item in group),
                **traj.as_dict(),
            }
    return rows


def _tier(difficulty: int) -> int:
    """简单档 = 1/2（单步），复杂档 = 3（多步 + 回忆）。"""

    return 3 if difficulty >= 3 else 1


def _range(values: Sequence[float]) -> float:
    return max(values) - min(values) if values else 0.0


async def _main(out: Path, trials: int) -> int:
    raw_cases = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]

    outcomes: list[Outcome] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for config in CONFIGS:
            for trial in range(trials):
                for raw in raw_cases:
                    # 复杂档跑 3 次，简单档 1 次（确定性世界里重复不产生新信息）
                    if _tier(int(raw["difficulty"])) == 3 or trial == 0:
                        outcomes.append(await run_one(raw, config, tmp))

    out.mkdir(parents=True, exist_ok=True)
    (out / "traces.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "id": item.case_id,
                    "expected_tools": item.expected_tools,
                    "records": item.records,
                    "config": item.config,
                    "difficulty": _tier(item.difficulty),
                },
                ensure_ascii=False,
            )
            for item in outcomes
        )
        + "\n",
        encoding="utf-8",
    )
    summary = summarize(outcomes)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{'配置':<18}{'档':<6}{'条数':>5}{'覆盖':>8}{'极差':>7}{'完成':>6}"
          f"{'步数':>7}{'重复':>7}{'token':>8}{'秒':>7}")
    print("-" * 80)
    for row in summary.values():
        if row["tier"] == "all":
            continue
        print(
            f"{row['config']:<18}{row['tier']:<6}{row['cases']:>5}"
            f"{row['coverage_mean']:>8.2f}{row['coverage_range']:>7.2f}"
            f"{row['completed']:>6}{row['steps_used'] or 0:>7.2f}"
            f"{row['path_repeat_rate'] or 0:>7.2f}"
            f"{row['tokens_mean']:>8.0f}{row['seconds_mean']:>7.2f}"
        )
    print()
    print(f"明细：{out / 'traces.jsonl'}")
    print(f"汇总：{out / 'summary.json'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="单/多 Agent 三层对比（W9 D5）")
    parser.add_argument("--out", default="evals/out/w9_ab", help="输出目录")
    parser.add_argument("--trials", type=int, default=3, help="复杂档重复次数")
    args = parser.parse_args(argv)
    return asyncio.run(_main(Path(args.out), args.trials))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
