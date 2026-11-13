"""多智能体共享状态（W9 D3）。

**为什么要版本号。** 多个角色并发提交同一个状态时，"读-改-写"会在没有任何
报错的情况下丢掉一方的工作：A 读到 version=3，B 也读到 3，A 先写（4），
B 后写（也基于 3 算出 4）——B 的写入把 A 的覆盖了，而两边都以为成功。
这类丢失在日志里看不出来，只能靠写入时校验版本。

与 W6/W8 的幂等是同一条主线的另一面：那边防"重复执行"，这里防"并发覆盖"，
共同点是**乐观锁必须在写入点校验，不能在读取点假设**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Finding", "SharedState", "StaleStateError"]


class StaleStateError(RuntimeError):
    """版本冲突。**必须抛**，不能"重试一次就好"——
    自动重试会把"谁的工作被丢了"这个问题藏起来，而丢工作的人不会收到通知。
    """


@dataclass(frozen=True, slots=True)
class Finding:
    """一条发现。`citation` 为空说明它还是假设，不是结论。

    validator 的职责就是盯着这个空字符串：空引用进结论，报告就把猜测写成了事实。
    """

    claim: str
    citation: str = ""
    role: str = ""

    @property
    def cited(self) -> bool:
        return bool(self.citation.strip())


@dataclass(slots=True)
class SharedState:
    """角色之间共享的工作区。**唯一可写入口是 `commit`。**"""

    task: str
    findings: list[Finding] = field(default_factory=list)
    # 已经问过的问题。researcher 靠它避免重复检索（与 D2 的 `Gap.searched` 同源）。
    asked: list[str] = field(default_factory=list)
    version: int = 0
    budget_used_tokens: int = 0
    # 谁在什么时候改了什么。排查"结论怎么变成这样"时要能倒放。
    history: list[tuple[int, str, int]] = field(default_factory=list)

    # -- 读 ---------------------------------------------------------------- #

    @property
    def uncited(self) -> list[Finding]:
        """没有引用的发现。空列表 = 每条都能追到出处。"""

        return [item for item in self.findings if not item.cited]

    @property
    def cited_count(self) -> int:
        return len([item for item in self.findings if item.cited])

    def snapshot(self) -> dict[str, object]:
        return {
            "task": self.task,
            "version": self.version,
            "findings": len(self.findings),
            "cited": self.cited_count,
            "uncited": len(self.uncited),
            "asked": list(self.asked),
            "budget_used_tokens": self.budget_used_tokens,
        }

    # -- 写 ---------------------------------------------------------------- #

    def commit(
        self,
        role: str,
        *,
        findings: list[Finding] | None = None,
        asked: list[str] | None = None,
        tokens: int = 0,
        expected_version: int | None = None,
    ) -> int:
        """提交一批变更，返回新版本号。

        `expected_version` 给了就必须等于当前版本，否则抛 `StaleStateError`。
        **并发提交一定要给**：不给就等于"以我读到的为准"，而那正是会丢数据的写法。
        """

        if expected_version is not None and expected_version != self.version:
            raise StaleStateError(
                f"{role} 提交时版本已变（期望 {expected_version}，当前 {self.version}）："
                "本次变更基于过期快照，拒绝写入以免覆盖他人结果"
            )

        added = 0
        for item in findings or []:
            self.findings.append(item)
            added += 1
        for question in asked or []:
            if question not in self.asked:
                self.asked.append(question)
        self.budget_used_tokens += max(0, tokens)
        self.version += 1
        self.history.append((self.version, role, added))
        return self.version
