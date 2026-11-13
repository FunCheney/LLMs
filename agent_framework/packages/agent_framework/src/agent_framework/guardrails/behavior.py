"""行为监控（W10 D4）：不是"拦一次"，而是"看频率"。

前面的两层守的是**单次调用**：这条输入有没有注入、这条输出有没有出处。
它们看不见下面这类问题：

- 一次任务里 `publish` 被调了 12 次（每次都合法，合起来是灾难）；
- 某个 principal 5 分钟内花了 40 元（每笔都在预算内）；
- 一条链路连续 30 次 `WAITING_APPROVAL` 没人处理（不是故障，是流程卡住）。

这三类都属于"单点合法、频率异常"，用规则拦不住，只能**计数 + 阈值**。

## 为什么用滑动窗口而不是固定窗口

固定窗口（每分钟清零）在边界上有经典的双倍突发：59 秒打 10 次、
下一分钟第 1 秒再打 10 次，两个窗口各 10 次都"合规"，实际 1 秒内 20 次。
滑动窗口记时间戳列表，代价是内存随事件数增长——所以加了 `max_events`
上限，超过就丢弃最旧的（宁可有微小误差，也不要 OOM）。

## 告警与拦断是两件事

`check()` 返回 `BehaviorAlert` 而不是抛异常：频率异常通常**不该拦断**当前调用
（它可能正是用户在批量处理），而是要让值班看见、让熔断器（W11 D2）去处理。
所以这里只产出告警，拦断策略留给调用方——把判断权和执行权分开，
才不会出现"批量导入被行为监控拦死"。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BEHAVIOR_RULES",
    "BehaviorAlert",
    "BehaviorMonitor",
    "BehaviorRule",
    "Event",
]

# 事件名。用常量而不是散字符串：告警规则按名字匹配，
# 拼错一个字的表现是"规则静默失效"，这是最难查的一类配置 bug。
EVENT_TOOL_CALL = "tool_call"
EVENT_DANGEROUS_CALL = "dangerous_call"
EVENT_APPROVAL_PENDING = "approval_pending"
EVENT_BUDGET_SPENT = "budget_spent"  # value = 金额（CNY）
EVENT_STEP = "step"

DEFAULT_WINDOW_S = 300.0


@dataclass(frozen=True)
class BehaviorRule:
    """一条频率规则。

    `limit` 的语义随 `event` 变：计数类事件是"次数上限"，`budget_spent` 是
    "金额上限"。混在一个字段里不优雅，但比开两个类好——规则表要能一眼读完。
    """

    name: str
    event: str
    limit: float
    window_s: float = DEFAULT_WINDOW_S
    # 只对这些事件生效的额外过滤（如 skill 名）。空 = 所有主体。
    scope: str = ""
    # 告警级别：warn 只记录，page 要叫人。
    severity: str = "warn"


# 默认三条。数字不是拍的：
# - dangerous_call 5/5min：发布这类动作，正常一次任务最多 2–3 次；
#   5 次意味着有循环在重试发布，而每次都合法。
# - budget_spent 2.0 CNY/5min：W3 的单任务成本基线是 0.03–0.4 元，
#   2 元 ≈ 一个用户 5 分钟内跑了 5 倍于最贵任务的开销。
# - approval_pending 3/10min：连着 3 条等人批还没人管，说明流程卡住了，
#   不是安全问题但会拖死吞吐。
BEHAVIOR_RULES: tuple[BehaviorRule, ...] = (
    BehaviorRule(
        name="dangerous_call_burst",
        event=EVENT_DANGEROUS_CALL,
        limit=5,
        window_s=300.0,
        severity="page",
    ),
    BehaviorRule(
        name="budget_burst",
        event=EVENT_BUDGET_SPENT,
        limit=2.0,
        window_s=300.0,
        severity="page",
    ),
    BehaviorRule(
        name="approval_backlog",
        event=EVENT_APPROVAL_PENDING,
        limit=3,
        window_s=600.0,
        severity="warn",
    ),
)


@dataclass(frozen=True)
class Event:
    """一条被记录的事件。`value` 默认 1，金额类事件填实际金额。"""

    event: str
    at: float
    value: float = 1.0
    scope: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorAlert:
    """一次越线。`observed` 是窗口内的实际值，报告里要写它而不是只写"超限"。"""

    rule: str
    event: str
    observed: float
    limit: float
    window_s: float
    severity: str

    def message(self) -> str:
        return (
            f"{self.rule}: 最近 {self.window_s:.0f}s 内 {self.event} = "
            f"{self.observed:.2f}，超过上限 {self.limit:g}"
        )


class BehaviorMonitor:
    """滑动窗口计数。

    `clock` 可注入：测试要构造"5 分钟内发生 6 次"，靠真 `time.monotonic`
    就得 `sleep`，而一旦某个测试开始 sleep，下一步一定是把它 skip 掉。
    """

    def __init__(
        self,
        rules: Iterable[BehaviorRule] | None = None,
        *,
        clock: Callable[[], float] | None = None,
        max_events: int = 4096,
    ) -> None:
        self._rules = tuple(rules if rules is not None else BEHAVIOR_RULES)
        self._clock = clock or time.monotonic
        self._events: dict[str, deque[Event]] = defaultdict(deque)
        self._max_events = max_events
        # 已经报过的 (rule, scope)，避免同一波突发每来一条事件就告警一次。
        # 只保留"最近一次越线"，下一次越线要等窗口滑过——极简的抑制策略，
        # 够用且不需要引入告警平台。
        self._alerted: dict[tuple[str, str], float] = {}

    @property
    def rules(self) -> tuple[BehaviorRule, ...]:
        return self._rules

    def record(
        self,
        event: str,
        *,
        value: float = 1.0,
        scope: str = "",
        **detail: Any,
    ) -> BehaviorAlert | None:
        """记录一条事件并即时判断。返回告警（若有）。"""

        now = self._clock()
        queue = self._events[event]
        queue.append(Event(event=event, at=now, value=value, scope=scope, detail=detail))
        while len(queue) > self._max_events:
            queue.popleft()

        return self._evaluate(event, scope=scope, now=now)

    def check(self, event: str | None = None, *, scope: str = "") -> tuple[BehaviorAlert, ...]:
        """不记录，只按当前窗口重算。用于"定期巡检"而不是"每次调用"。"""

        now = self._clock()
        events = [event] if event else list(self._events)
        alerts: list[BehaviorAlert] = []
        for name in events:
            alert = self._evaluate(name, scope=scope, now=now)
            if alert is not None:
                alerts.append(alert)
        return tuple(alerts)

    def snapshot(self) -> dict[str, Any]:
        """当前窗口内的计数快照。给 `/health` 或运维面板用。"""

        now = self._clock()
        out: dict[str, Any] = {}
        for event, queue in self._events.items():
            live = [item for item in queue if now - item.at <= DEFAULT_WINDOW_S]
            out[event] = {
                "count": len(live),
                "total": round(sum(item.value for item in live), 4),
                "oldest_s": round(now - live[0].at, 1) if live else None,
            }
        return out

    def reset(self) -> None:
        self._events.clear()
        self._alerted.clear()

    # -- 内部 -------------------------------------------------------------- #

    def _evaluate(self, event: str, *, scope: str, now: float) -> BehaviorAlert | None:
        queue = self._events.get(event)
        if not queue:
            return None

        for rule in self._rules:
            if rule.event != event:
                continue
            if rule.scope and rule.scope != scope:
                continue

            live = [item for item in queue if now - item.at <= rule.window_s]
            if rule.scope:
                live = [item for item in live if item.scope == scope]

            observed = sum(item.value for item in live)
            if observed <= rule.limit:
                continue

            key = (rule.name, scope)
            last = self._alerted.get(key)
            # 抑制：同一规则同一 scope，窗口内只叫一次。
            if last is not None and now - last <= rule.window_s:
                continue
            self._alerted[key] = now

            return BehaviorAlert(
                rule=rule.name,
                event=event,
                observed=observed,
                limit=rule.limit,
                window_s=rule.window_s,
                severity=rule.severity,
            )

        return None
