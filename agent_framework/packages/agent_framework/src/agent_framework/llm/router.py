"""路由：task → tier → 候选模型列表（architecture §4.2）。

两条边界：

- router 只回答"候选是谁"，**不做 fallback**。"绕不绕第二圈"由 `LLMGateway` 决定，
  因为只有它知道上一个候选是失败了，还是"已开始输出，不可重试"。
- 未知 task 回落到 `default` 但**必须记 warning**：静默回落会让你以为评测跑的是
  strong，实际跑的是 cheap，而指标曲线看不出区别。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..config import Catalog
from ..observability import router_log


@runtime_checkable
class Router(Protocol):
    """路由抽象。

    第二实现是 `StaticRouter`（测试与调试用）——没有它，任何路由测试都得构造
    整份 models.yaml，抽象也就没法证明自己值得存在（architecture 原则 3）。
    """

    @property
    def name(self) -> str: ...

    def route(self, task: str) -> list[str]:
        """返回候选 alias 有序列表；首个为主，其余为 fallback 链。"""


class TaskRouter:
    """catalog 驱动：task → tier → [primary, *fallback]。"""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    @property
    def name(self) -> str:
        return "task"

    def route(self, task: str) -> list[str]:
        if task not in self._catalog.tasks:
            router_log().warning(
                "unknown task %r, falling back to 'default'; known tasks: %s",
                task,
                sorted(self._catalog.tasks),
            )

        return self._catalog.resolve(task)


class StaticRouter:
    """固定候选列表。"""

    def __init__(self, aliases: Sequence[str], *, name: str = "static") -> None:
        self._aliases = tuple(dict.fromkeys(aliases))
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def route(self, task: str) -> list[str]:
        # task 参数被刻意忽略：这个实现的存在意义就是"不按 task 分支"。
        return list(self._aliases)
