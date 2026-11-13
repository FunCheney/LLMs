"""`Reasoner` 的三种实现（W9 D1）。

为什么单独一个文件：D5 的三档对比（ReAct / Plan-and-Execute / 多 Agent）
**必须共用同一个大脑**。如果每一档各配一个 reasoner，最后比出来的差异里就
混着"谁的提示词更好"——这个变量比编排范式大得多，会把结论完全吃掉。

三种实现的分工：

| 实现 | 用在哪 | 为什么不换掉 |
|---|---|---|
| `ScriptedReasoner` | 单测 | 断言"守卫在超限时真的中断"，需要确定性；模型的不确定性会让用例偶发红 |
| `PolicyReasoner` | 离线对比实验 | 由确定性"任务世界"驱动，同一策略跑三种编排，差异只来自编排 |
| `LLMReasoner` | 在线档 | 真实模型；它有随机性，所以结论要配合 Multi-Trial 看极差 |

`PolicyReasoner` 不是"为了凑数据编的桩"：它就是"如果模型完全听话、只知道
任务书、只看得见喂给它的观察"这个假设下的行为。这个假设**写进报告**，
读者才知道离线结论的边界在哪。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMClient
from ..llm.types import Message, MessageRole
from .base import (
    DECISION_FINISH,
    DECISION_TOOL,
    AgentState,
    Decision,
    Plan,
    PlanStep,
)

__all__ = ["LLMReasoner", "PolicyReasoner", "PolicyTask", "ScriptedReasoner", "TaskPolicy"]


# --------------------------------------------------------------------------- #
# 单测用：脚本化
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ScriptedReasoner:
    """按剧本回放。剧本可以是"第几次调用返回什么"的列表。

    `replan_plan` 给重规划用：`None` 表示"重规划原样返回上一版"，
    这是构造"重复计划死循环"用例的最短路径。
    """

    plan: Plan
    actions: list[Decision] = field(default_factory=list)
    compose_text: str = "完成。"
    replan_plan: Plan | None = None
    tokens_per_call: int = 10

    # 记录调用次数，用于断言"守卫触发后没有再调模型"——
    # 守卫若只把状态改成 partial 却继续循环，日志看不出来，这个计数能看出来。
    calls: int = 0
    replan_calls: int = 0

    async def propose_plan(self, task: str, *, max_steps: int) -> Plan:
        self.calls += 1
        return self.plan

    async def replan(self, plan: Plan, failed: PlanStep) -> Plan:
        self.replan_calls += 1
        if self.replan_plan is not None:
            return self.replan_plan
        return plan

    async def next_action(self, state: AgentState) -> Decision:
        self.calls += 1
        if self.actions:
            decision = self.actions.pop(0)
            return Decision(
                kind=decision.kind,
                tool=decision.tool,
                arguments=decision.arguments,
                text=decision.text,
                tokens=self.tokens_per_call,
            )
        # 剧本用完了还问，说明规划器多跑了一轮（守卫失效的典型症状）。
        return Decision(kind=DECISION_FINISH, text="剧本耗尽", tokens=self.tokens_per_call)

    async def compose(self, state: AgentState) -> Decision:
        self.calls += 1
        return Decision(kind=DECISION_FINISH, text=self.compose_text, tokens=self.tokens_per_call)


# --------------------------------------------------------------------------- #
# 离线对比实验用：策略驱动
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PolicyTask:
    """一个可被确定性求解的任务。

    `steps` 是"正确做法"的步骤序列：每步给出工具与参数。参考实现用它算
    `tool_selection_acc` 的分母，`PolicyReasoner` 用它决定下一步。

    `recall_from` 是关键：某一步的参数需要**引用更早某步的结果**。
    这是 ReAct 在窗口不够时真正会输的地方，也是"计划状态 vs 上下文窗口"
    这个机制的唯一来源。没有它，两种编排打平——而"打平"本身也是要报的结果。
    """

    id: str
    question: str
    steps: tuple[PlanStep, ...]
    # (第几步) -> (引用第几步的结果)。决策时需要回看早期结果。
    recall_from: dict[int, int] = field(default_factory=dict)
    # 哪一步会失败（用于"部分失败诚实标注"的用例）。
    failing_step: int | None = None
    difficulty: int = 3


class TaskPolicy:
    """任务世界：工具怎么响应。确定性、无网络。

    `value_of(index)` 是第 index 步会产出的**唯一**正确值，由该步的 query 决定。
    回忆步必须把这个值原样带进 query——带错就会去查一个不存在的查询，
    于是"工具成功、答案错"，这正是线上最常见的失败形态。
    """

    def __init__(self, task: PolicyTask) -> None:
        self._task = task

    @property
    def task(self) -> PolicyTask:
        return self._task

    def value_of(self, index: int) -> str:
        steps = self._task.steps
        if index >= len(steps):
            return ""
        return str(steps[index].arguments.get("query") or steps[index].goal)

    def expected_tools(self) -> tuple[str, ...]:
        return tuple(step.tool for step in self._task.steps if step.tool)


@dataclass(slots=True)
class PolicyReasoner:
    """按任务书执行。**同一份策略，三种编排复用**。

    关键在 `_arguments_for`：回忆步要引用早期某步的产出，而它只能从
    **可见状态**里拿到那个值。ReAct 的可见状态是最近 N 条观察，P&E 的是
    计划 + 已完成目标，所以窗口不够时 ReAct 会拿不到、只能填"未知"。

    填"未知"而不是报错，是有意的：真实系统里模型记不住就编一个，
    查询仍然成功，答案因此是错的却看起来正常。这个失败形态比抛异常难发现，
    也更值得在对比实验里被量化。
    """

    policy: TaskPolicy
    # 模拟的随机性：种子相同则完全可复现。用于 Multi-Trial 的极差统计。
    seed: int = 0
    tokens_per_call: int = 40
    _cursor: int = 0
    _calls: int = 0

    async def propose_plan(self, task: str, *, max_steps: int) -> Plan:
        self._calls += 1
        steps = list(self.policy.task.steps[:max_steps])
        return Plan(task=task, steps=steps)

    async def replan(self, plan: Plan, failed: PlanStep) -> Plan:
        """把失败的步骤摘掉，保留其余步骤与它们的状态。

        这里可以说"跳过失败步"是合理的策略：调研的价值在于拿到 4/5 的答案。
        但**不能把它悄悄变成成功**——跳过的步要在 compose 里被点名。
        """

        self._calls += 1
        steps = [
            PlanStep(
                id=step.id,
                goal=step.goal,
                tool=step.tool,
                arguments=step.arguments,
                depends_on=step.depends_on,
                status=step.status,
            )
            for step in plan.steps
            if step.id != failed.id
        ]
        return Plan(task=plan.task, steps=steps, budget_tokens=plan.budget_tokens)

    async def next_action(self, state: AgentState) -> Decision:
        self._calls += 1

        if state.step is not None:
            index = self._index_of(state.step.id)
            return Decision(
                kind=DECISION_TOOL,
                tool=state.step.tool or "",
                arguments=self._arguments_for(index, state),
                tokens=self.tokens_per_call,
            )

        # -- ReAct 路径：没有 step，只能凭观察和游标推下一步 ------------------ #
        steps = self.policy.task.steps
        if self._cursor >= len(steps):
            # 走完了就成文。让 ReAct 交白卷会让"completed"变成假绿。
            return await self.compose(state)
        index = self._cursor
        self._cursor += 1
        return Decision(
            kind=DECISION_TOOL,
            tool=steps[index].tool or "",
            arguments=self._arguments_for(index, state),
            tokens=self.tokens_per_call,
        )

    async def compose(self, state: AgentState) -> Decision:
        self._calls += 1
        parts: list[str] = []
        for index, step in enumerate(self.policy.task.steps):
            if index == self.policy.task.failing_step:
                # 失败步要**显式写成缺口**，不能让报告看起来圆满。
                parts.append(f"{step.goal}：证据不足（该步未完成）。")
                continue
            parts.append(f"{step.goal}：{self.policy.value_of(index)}")
        return Decision(
            kind=DECISION_FINISH,
            text="\n".join(parts) or "（无可用证据）",
            tokens=self.tokens_per_call,
        )

    # -- 内部 -------------------------------------------------------------- #

    def _index_of(self, step_id: str) -> int:
        for index, step in enumerate(self.policy.task.steps):
            if step.id == step_id:
                return index
        return 0

    def _arguments_for(self, index: int, state: AgentState) -> dict[str, Any]:
        """填参数。需要回忆早期结果时，看**可见状态里有没有那条信息**。"""

        step = self.policy.task.steps[index]
        arguments = dict(step.arguments)
        source = self.policy.task.recall_from.get(index)
        if source is None:
            return arguments

        needle = self.policy.value_of(source)
        visible = "\n".join(state.observations)
        # 只在可见的观察里找。P&E 会把它带在观察窗口里（窗口足够大），
        # ReAct 只有在窗口覆盖到那一步时才有。
        recalled = needle if needle and needle in visible else "未知"
        base = str(arguments.get("query") or step.goal)
        arguments["query"] = f"{base}（复核 {recalled}）"
        return arguments


# --------------------------------------------------------------------------- #
# 在线档：真模型
# --------------------------------------------------------------------------- #

PLAN_SYSTEM = (
    "你是任务规划器。把任务拆成 1..N 个可执行步骤，每步一个动作。"
    "只输出 JSON：{\"steps\":[{\"id\":\"s1\",\"goal\":\"...\",\"tool\":\"工具名或空\","
    "\"arguments\":{},\"depends_on\":[]}]}"
)

ACT_SYSTEM = (
    "你是执行器。给定当前步骤与已有观察，决定下一步动作。"
    "只输出 JSON：{\"action\":\"tool\",\"tool\":\"工具名\",\"arguments\":{}} 或 "
    "{\"action\":\"finish\",\"text\":\"这一步的结论\"}"
)

COMPOSE_SYSTEM = (
    "你是撰写者。用已完成的观察写成结论。**没有观察支撑的话不要写进结论**，"
    "缺证据的部分明确写「证据不足」。只输出 JSON：{\"action\":\"finish\",\"text\":\"...\"}"
)


@dataclass(slots=True)
class LLMReasoner:
    """真模型实现。**解析失败不抛异常**，而是变成一次带病因的 finish。

    抛异常会把"模型输出了一段不是 JSON 的东西"升级成"整个任务崩了"，
    而这类失败在真实流量里很常见。做成 finish + 显式病因，产品能给出
    "这次没规划出来"的文案，日志里也有原文可查。
    """

    client: LLMClient
    model_alias: str = ""
    tokens_per_call: int = 0

    async def propose_plan(self, task: str, *, max_steps: int) -> Plan:
        raw = await self._ask(
            PLAN_SYSTEM,
            f"任务：{task}\n最多 {max_steps} 步。可用工具见 system。",
        )
        steps: list[PlanStep] = []
        for index, item in enumerate(raw.get("steps") or []):
            if not isinstance(item, dict):
                continue
            steps.append(
                PlanStep(
                    id=str(item.get("id") or f"s{index + 1}"),
                    goal=str(item.get("goal") or ""),
                    tool=(str(item["tool"]) if item.get("tool") else None),
                    arguments=dict(item.get("arguments") or {}),
                    depends_on=tuple(str(dep) for dep in item.get("depends_on") or ()),
                )
            )
        if not steps:
            # 一步都没拆出来时给一个"直接作答"的兜底，而不是空计划——
            # 空计划会让 `is_finished()` 立刻返回 True 然后交白卷。
            steps = [PlanStep(id="s1", goal=task, tool=None)]
        return Plan(task=task, steps=steps[:max_steps])

    async def replan(self, plan: Plan, failed: PlanStep) -> Plan:
        """失败后只做一件事：把失败步标记为跳过，其余保持。

        没有让它"自由重写计划"：重写会让已完成步骤的 id 消失，
        `_merge_replan` 就认不出哪些做过，于是重复产生副作用。
        """

        steps = [
            PlanStep(
                id=step.id,
                goal=step.goal,
                tool=step.tool,
                arguments=step.arguments,
                depends_on=step.depends_on,
                status=step.status,
            )
            for step in plan.steps
            if step.id != failed.id
        ]
        return Plan(task=plan.task, steps=steps, budget_tokens=plan.budget_tokens)

    async def next_action(self, state: AgentState) -> Decision:
        if state.step is None:
            body = (
                f"任务：{state.task}\n已做：{list(state.done_goals)}\n"
                f"观察：{list(state.observations)}"
            )
        else:
            body = (
                f"任务：{state.task}\n当前步骤：{state.step.goal}\n"
                f"建议工具：{state.step.tool}\n已做：{list(state.done_goals)}\n"
                f"观察：{list(state.observations)}"
            )
        raw = await self._ask(ACT_SYSTEM, body)
        if str(raw.get("action") or "") == DECISION_FINISH:
            return Decision(kind=DECISION_FINISH, text=str(raw.get("text") or ""))
        return Decision(
            kind=DECISION_TOOL,
            tool=str(raw.get("tool") or (state.step.tool if state.step else "") or ""),
            arguments=dict(raw.get("arguments") or {}),
        )

    async def compose(self, state: AgentState) -> Decision:
        body = (
            f"任务：{state.task}\n已做：{list(state.done_goals)}\n"
            f"观察：{list(state.observations)}"
        )
        raw = await self._ask(COMPOSE_SYSTEM, body)
        return Decision(kind=DECISION_FINISH, text=str(raw.get("text") or ""))

    async def _ask(self, system: str, body: str) -> dict[str, Any]:
        response = await self.client.chat(
            [
                Message(role=MessageRole.SYSTEM, content=system),
                Message(role=MessageRole.USER, content=body),
            ]
        )
        text = response.message.content
        if not isinstance(text, str):
            return {"text": "[模型返回了非文本内容]"}
        return _parse_json_object(text)


def _parse_json_object(text: str) -> dict[str, Any]:
    """从模型输出里抠出第一个 JSON 对象。

    先整体解析，失败再找最外层花括号——模型常见的可爱行为是包一层
    ```json 代码块或者前后加一句"好的，我来规划："。
    """

    stripped = text.strip()
    try:
        loaded = json.loads(stripped)
        return loaded if isinstance(loaded, dict) else {"text": stripped}
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            loaded = json.loads(stripped[start : end + 1])
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
    return {"text": f"[规划输出不是 JSON] {stripped[:200]}"}

