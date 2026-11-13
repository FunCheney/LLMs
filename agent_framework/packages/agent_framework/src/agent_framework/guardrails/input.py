"""输入侧守卫（W10 D4）：长度 / 类型 / 注入，三道合一。

为什么不把三件事拆成三个类：调用点的顺序是固定的（先看形状、再看内容、
最后隔离），拆开之后每个调用点都要自己排一遍顺序，而"顺序写错"的表现是
"先隔离再检测"——检测器看到的是被转义过的文本，规则命中率直接掉一半。

## 长度上限为什么是**两个**而不是一个

`max_chars` 管的是"这一段内容本身不合理"（一个 20 万字的用户输入不该被
当成一个问题）；`max_total` 管的是"整包 context 不合理"（10 个 5000 字的
检索块拼起来会吃掉整个窗口，W7 的预算器都来不及裁）。
两者混用会导致"单块没问题但整包爆了"或反之。

## 超长为什么不直接截断

截断用户输入会**改变问题的意思**（"……除了以下三种情况"里的"以下三种"
被切掉），而截断检索块会让引用指向不完整的段落。所以这里是**拒绝**，
让调用方自己决定是拆分还是放弃。唯一的例外是 `allow_truncate=True`，
给"展示用"的路径用（如前端预览）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .injection import (
    InjectionGuard,
    InjectionHit,
    isolate,
)

__all__ = [
    "BlockedInput",
    "InputCheck",
    "InputGuard",
    "InputLimits",
]


class BlockedInput(Exception):
    """输入被拦断。`code` 与统一错误表对齐（`CONTENT_BLOCKED` → 400/422）。"""

    code = "CONTENT_BLOCKED"

    def __init__(self, message: str, *, violations: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.violations = violations


@dataclass(frozen=True)
class InputLimits:
    """阈值。默认值按"正常用户不会这么写"定，而不是按"我们扛得住"定。"""

    # 单段输入上限。8000 字 ≈ 一份长报告的一节；再长说明该走拆解（W9）。
    max_chars: int = 8000
    min_chars: int = 1
    # 整包上限。10 个检索块 × 1500 字 + 用户问题，留一倍余量。
    max_total_chars: int = 60000
    # 单次请求里允许的不可信块数。超过说明检索配置错了（recall_multiplier 太大）。
    max_blocks: int = 64


@dataclass(frozen=True)
class InputCheck:
    """一次输入检查的结果。`sanitized` 是**可以进 context 的文本**。"""

    ok: bool
    sanitized: str = ""
    violations: tuple[str, ...] = ()
    hits: tuple[InjectionHit, ...] = ()
    detail: dict[str, object] = field(default_factory=dict)

    def raise_if_blocked(self) -> str:
        if not self.ok:
            raise BlockedInput(
                "；".join(self.violations) or "输入未通过检查", violations=self.violations
            )
        return self.sanitized


class InputGuard:
    """长度 + 注入 + 隔离。

    `prepare(text, source=...)` 是标准入口：检测通过则返回**已隔离**的文本，
    调用方拿到它直接拼进 context 就行，不必记得再包一层 untrusted——
    忘记包隔离是这类代码最常见的失误，所以把它做成"返回值的形态"。
    """

    def __init__(
        self,
        *,
        limits: InputLimits | None = None,
        injection: InjectionGuard | None = None,
        allow_truncate: bool = False,
    ) -> None:
        self._limits = limits or InputLimits()
        self._injection = injection or InjectionGuard()
        self._allow_truncate = allow_truncate

    @property
    def injection(self) -> InjectionGuard:
        return self._injection

    def check(self, text: str, *, source: str = "user") -> InputCheck:
        """只判断，不改文本（除隔离外）。

        返回的 `sanitized` **始终是隔离后的形态**：即使 `ok=False` 也给出，
        是为了让调用方在"部分拦截"场景（多个块里坏一块）下能处置——
        但拦断路径不该用它。
        """

        limits = self._limits
        violations: list[str] = []
        length = len(text)

        if length < limits.min_chars:
            violations.append(f"输入过短：{length} < {limits.min_chars}")
        if length > limits.max_chars:
            if self._allow_truncate:
                text = text[: limits.max_chars]
            else:
                violations.append(f"输入过长：{length} > {limits.max_chars}")

        verdict = self._injection.screen(text, source=source)
        if not verdict.allowed:
            # 命中规则的原文**不重复放进错误信息**：它会原样进日志与告警，
            # 而那正是攻击者想要的"让我的载荷出现在你们的运维面板上"。
            violations.append(f"疑似提示注入（{verdict.reason()}）")

        return InputCheck(
            ok=not violations,
            sanitized=isolate(text, source=source),
            violations=tuple(violations),
            hits=verdict.hits,
            detail={
                "source": source,
                "chars": length,
                "suspicious": verdict.suspicious,
            },
        )

    def prepare(self, text: str, *, source: str = "user") -> str:
        """检测 + 隔离，通过则返回可直接拼进 context 的文本，否则抛 `BlockedInput`。"""

        return self.check(text, source=source).raise_if_blocked()

    def prepare_many(self, items: list[tuple[str, str]]) -> tuple[str, ...]:
        """批量：`[(source, text), ...]` → 隔离后的块列表。

        一块被拦就整包抛错（fail-closed）：检索块之间会互相支撑，
        去掉一块之后模型可能基于剩余的块得出相反结论，而它不会知道自己
        少了一块。宁可整次请求失败，也不给一个"基于残缺证据"的答案。
        """

        if len(items) > self._limits.max_blocks:
            raise BlockedInput(
                f"不可信块过多：{len(items)} > {self._limits.max_blocks}",
                violations=(f"块数 {len(items)} 超过上限 {self._limits.max_blocks}",),
            )

        total = sum(len(text) for _, text in items)
        if total > self._limits.max_total_chars:
            raise BlockedInput(
                f"context 总量超限：{total} > {self._limits.max_total_chars}",
                violations=(f"总量 {total} 超过上限 {self._limits.max_total_chars}",),
            )

        return tuple(self.prepare(text, source=source) for source, text in items)
