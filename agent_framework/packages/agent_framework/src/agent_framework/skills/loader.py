"""Skill 注册表与加载（W8 D4）：动态发现、热重载、渐进式加载。

检查点原文是「新增工具/Skill 不改主流程即可被调用」。这句话有一个很强的
推论：**加载失败必须是局部的**。一个 YAML 写错就整表清空，那"新增"立刻
变成"停机"，没人敢加东西。所以这里的行为是：

- 单个文件解析/校验失败 → 跳过它、把原因记进 `errors`，其余照常注册
- 重名 → **报错**而不是后者覆盖前者。静默覆盖会产生一类极难查的 bug：
  你以为在调 A，实际调的是同名 B，而日志里只有一个名字
- `reload()` 重扫全部已注册路径并递增 `generation`；调用方可以据此判断
  "我的可见面缓存过期了"

文件格式与 `library.BUILTIN_SKILLS` 的字典**完全同形**（见 `spec.SkillSpec.from_dict`）。
两条加载路径共用一套字段，避免"文件式 Skill 少一个 mission 也能加载"这种
只在一半输入上生效的校验。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from .library import builtin_skills
from .spec import SkillSpec, SkillSpecError

logger = logging.getLogger(__name__)

__all__ = ["SKILL_FILE_SUFFIXES", "SkillRegistry"]

SKILL_FILE_SUFFIXES = (".yaml", ".yml", ".json")


class SkillRegistry:
    """进程内注册表。不可变视图（`get` 返回的 `SkillSpec` 是 frozen）。"""

    def __init__(self, *, with_builtins: bool = True) -> None:
        self._skills: dict[str, SkillSpec] = {}
        self._paths: list[Path] = []
        self._errors: list[str] = []
        # 每次 reload 递增。调用方缓存了可见面就靠它判断要不要重算——
        # 没有它，热重载在调用方看来是"什么都没发生"。
        self._generation = 0

        if with_builtins:
            for spec in builtin_skills():
                self._skills[spec.name] = spec

    # -- 读 ---------------------------------------------------------------- #

    def get(self, name: str) -> SkillSpec | None:
        """`None` 表示不存在。**不抛 KeyError**：调用方（执行器）要把它
        转成一个明确的 `tool_not_found` 结果，而不是一个 500。"""
        return self._skills.get(name)

    def has(self, name: str) -> bool:
        return name in self._skills

    def all(self) -> tuple[SkillSpec, ...]:
        """全部 Skill，按名字排序（可复现：可见面的顺序不该随插入顺序变）。"""
        return tuple(self._skills[name] for name in sorted(self._skills))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def errors(self) -> tuple[str, ...]:
        """上次发现/重载跳过的文件与原因。**必须能被打印出来**——
        一个被静默跳过的 Skill 等于一个不存在的 Skill。"""
        return tuple(self._errors)

    # -- 写 ---------------------------------------------------------------- #

    def register(self, spec: SkillSpec, *, replace: bool = False) -> None:
        if spec.name in self._skills and not replace:
            raise SkillSpecError(
                f"Skill 重名：{spec.name!r} 已存在（来源 {self._skills[spec.name].source}），"
                f"新来源 {spec.source}。要覆盖请显式 replace=True"
            )
        spec.validate()
        self._skills[spec.name] = spec
        self._generation += 1

    def discover(
        self, paths: Sequence[str | Path] | str | Path, *, replace: bool = False
    ) -> list[SkillSpec]:
        """扫描目录或单个文件，注册其中全部 Skill。

        返回**成功注册**的那些。失败的在 `errors` 里——这个返回值设计让
        "加了 3 个、成功了 2 个"这种情况不可能被调用方误读成"加了 3 个"。
        """
        targets = [paths] if isinstance(paths, (str, Path)) else list(paths)
        loaded: list[SkillSpec] = []

        for target in targets:
            path = Path(target).expanduser()
            if path not in self._paths:
                self._paths.append(path)

            for file in _iter_skill_files(path):
                try:
                    spec = self._load_file(file)
                except (SkillSpecError, yaml.YAMLError, OSError, ValueError) as exc:
                    self._errors.append(f"{file}: {exc}")
                    logger.warning("skill 加载失败，已跳过：%s（%s）", file, exc)
                    continue

                try:
                    self.register(spec, replace=replace)
                except SkillSpecError as exc:
                    self._errors.append(f"{file}: {exc}")
                    logger.warning("skill 注册失败，已跳过：%s（%s）", file, exc)
                    continue

                loaded.append(spec)

        return loaded

    async def reload(self) -> tuple[SkillSpec, ...]:
        """重扫已注册的全部路径。

        `async` 是为了与"将来从远端拉 Skill 包"的形态一致（接口不变），
        当前实现是同步文件 IO。**不**在这里清空已有 Skill：
        清空再加载意味着"加载到一半失败 = 一个工具都没有"。
        """
        self._errors.clear()
        for path in list(self._paths):
            self.discover(path, replace=True)

        self._generation += 1
        return self.all()

    # -- 渐进式加载 -------------------------------------------------------- #

    def describe_all(self, names: Sequence[str] | None = None) -> str:
        """只注入 `name + description`。

        参数 `names` 用于可见面裁剪（`visibility.py` 算完之后再调）。
        """
        selected = (
            [self._skills[name] for name in names if name in self._skills]
            if names is not None
            else list(self.all())
        )
        return "\n".join(spec.describe() for spec in selected)

    def instructions_of(self, names: Iterable[str]) -> str:
        """选中之后才注入的正文。**只对本次要用的 Skill** 调用。"""
        blocks: list[str] = []
        for name in names:
            spec = self._skills.get(name)
            if spec is None:
                continue
            header = f"## {spec.name}\n使命：{spec.mission}"
            principles = "\n".join(f"- {item}" for item in spec.principles)
            body = f"{header}\n原则：\n{principles}\n步骤：\n{spec.instructions}"
            blocks.append(body)
        return "\n\n".join(blocks)

    # -- 内部 -------------------------------------------------------------- #

    def _load_file(self, path: Path) -> SkillSpec:
        raw = yaml.safe_load(path.read_text("utf-8"))
        if not isinstance(raw, dict):
            raise SkillSpecError(f"顶层必须是映射，收到 {type(raw).__name__}")

        # 允许一个文件里放 `{skills: [...]}`（一次定义多个），也允许单个定义。
        if "skills" in raw:
            raise SkillSpecError(
                "一个文件只能定义一个 Skill（当前不支持 skills: [] 批量）："
                "批量定义会让'哪个文件对应哪个 Skill'无法回答"
            )

        return SkillSpec.from_dict(raw, source=str(path))


def _iter_skill_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SKILL_FILE_SUFFIXES else []
    if not path.is_dir():
        return []
    return sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in SKILL_FILE_SUFFIXES
    )


def registry_snapshot(specs: Iterable[SkillSpec]) -> list[dict[str, Any]]:
    """给报告用的投影（`/v1/tools` 也用它）。"""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "entrypoint": spec.entrypoint,
            "permissions": sorted(spec.permissions),
            "sandbox": spec.sandbox,
            "stages": sorted(spec.stages),
            "dangerous": spec.dangerous,
            "source": spec.source,
            "checks": len(spec.done_criteria),
        }
        for spec in specs
    ]
