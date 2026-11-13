"""三种角色（W9 D3）。

**职责单一不是风格问题，是并发正确性问题。** 如果 supervisor 自己也检索、
也写报告，它就会同时是共享状态的写入方和调度方：调度时读到的快照很快就过期，
而它自己感觉不到。W9 的三个角色里，只有 researcher 和 validator 会写状态，
supervisor 只读——这条约束让版本冲突的范围缩小了一半。

角色返回的是**意图**（`RoleAction`），不是执行结果：真正调工具的是
`StepRunner`（W8 的五道门），执行完再由 supervisor 统一提交进共享状态。
角色自己直接写状态，就会出现"工具失败了但状态已经更新"的鬼故事。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .shared_state import Finding, SharedState

__all__ = [
    "ROLE_RESEARCHER",
    "ROLE_SUPERVISOR",
    "ROLE_VALIDATOR",
    "AgentRole",
    "ResearcherRole",
    "RoleAction",
    "RoleResult",
    "SupervisorRole",
    "ValidatorRole",
]

ROLE_SUPERVISOR = "supervisor"
ROLE_RESEARCHER = "researcher"
ROLE_VALIDATOR = "validator"


@dataclass(frozen=True, slots=True)
class RoleAction:
    """角色想调用的工具。**还没执行**——执行要过 W8 的五道门。"""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    # 这一步要回答哪个子问题。执行完用它决定把结果挂到哪个缺口上。
    question: str = ""


@dataclass(slots=True)
class RoleResult:
    """角色一次行动的产出。全是**待提交**的增量，不是已生效的状态。"""

    actions: list[RoleAction] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    tokens: int = 0
    # 这个角色认为还差什么。给 supervisor 的调度依据，不是给人看的文案。
    needs_fix: bool = False
    # "我还有活，**别换人**"。研究员逐个子问题查的时候要用：
    # 一旦查到第一条带引用的发现就交给 validator，多 Agent 的"多"就只剩
    # 一个检索 + 一次盖章，D5 的对比也就没什么可比的了。
    can_continue: bool = False
    note: str = ""


class AgentRole(Protocol):
    """角色的唯一接口。`act` 不许有副作用——副作用只能发生在 `StepRunner` 里。"""

    name: str

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult: ...


# --------------------------------------------------------------------------- #
# supervisor：只调度
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SupervisorRole:
    """只决定"下一个是谁"。**不检索、不写报告。**

    把它做成一个独立对象而不是 supervisor 里的几个 if，是为了让"职责单一"
    这件事可测：它没有 runner、没有 executor，想越界也没有工具可用。
    """

    name: str = ROLE_SUPERVISOR

    def pick(
        self,
        state: SharedState,
        *,
        last_role: str = "",
        validated: bool = False,
        continue_role: str = "",
    ) -> str | None:
        """返回下一个角色；None = 可以收尾了。

        顺序背后的判断：

        0. 上一位说"我还有活"（`can_continue`）→ 先让它做完。被打断的后果是
           半成品：研究员查到第 1 个问题就被换下去，报告只覆盖了 1/4。
        1. 一条发现都没有 → 先去检索（没有证据时做校验毫无意义）。
        2. 有发现但缺引用 → 回检索补引用。**不交给 validator 放行**：
           空引用进结论就是把猜测写成事实。
        3. 引用齐了且还没校验 → 交给 validator 判"证据够不够"。
        4. 校验过了 → 收尾。
        """

        # "还有活"只在同一位角色被真正启用时才算数：它自己都没被调度过，
        # 空字符串不能变成"再派一次空角色"。
        if continue_role and continue_role != ROLE_SUPERVISOR and not validated:
            return continue_role
        if not state.findings:
            return ROLE_RESEARCHER
        if state.uncited:
            # 已经在研究员手里还缺引用，说明它补不上——交给 validator 判失败，
            # 而不是无限地把它再派一次（那就是互踢皮球）。
            return ROLE_VALIDATOR if last_role == ROLE_RESEARCHER else ROLE_RESEARCHER
        if not validated:
            return ROLE_VALIDATOR
        return None

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        raise NotImplementedError(
            "supervisor 只调度，不执行；调度逻辑见 pick()"
        )


# --------------------------------------------------------------------------- #
# researcher：只检索
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ResearcherRole:
    """按子问题逐个检索。**不判断证据够不够**（那是 validator 的活）。"""

    questions: Sequence[str] = ()
    name: str = ROLE_RESEARCHER

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        pending = [item for item in self.questions if item not in state.asked]
        if not pending:
            return RoleResult(note="没有新的子问题可查")

        question = pending[0]
        return RoleResult(
            actions=[
                RoleAction(
                    tool="web_search",
                    arguments={"query": question, "max_results": 3},
                    question=question,
                )
            ],
            asked=[question],
            # 还有下一个子问题就要求"别换人"：换人等于只查了 1/N。
            can_continue=len(pending) > 1,
            # 检索轮的 token 估算：query 长度 + 预期返回。用估算值而不是 0，
            # 否则预算守卫在离线档永远不触发。
            tokens=len(question) * 2,
            note=f"检索：{question}",
        )


# --------------------------------------------------------------------------- #
# validator：只查引用充分性
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ValidatorRole:
    """只回答一个问题：**每条结论都能追到出处吗。**"""

    # 至少要有几条带引用的发现才算"证据够"。默认 1：一份报告至少要有一处出处。
    min_cited: int = 1
    name: str = ROLE_VALIDATOR

    async def act(
        self, task: str, state: SharedState, *, budget_left: int
    ) -> RoleResult:
        uncited = state.uncited
        if uncited:
            return RoleResult(
                needs_fix=True,
                note=(
                    f"{len(uncited)} 条发现没有引用，不能进结论："
                    + "；".join(item.claim[:30] for item in uncited[:3])
                ),
            )
        if state.cited_count < self.min_cited:
            return RoleResult(
                needs_fix=True,
                note=f"带引用的发现只有 {state.cited_count} 条，低于下限 {self.min_cited}",
            )
        return RoleResult(note=f"校验通过：{state.cited_count} 条带引用")
