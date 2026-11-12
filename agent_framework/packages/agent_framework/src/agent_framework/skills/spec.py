"""Skill 规范（W8 D3）：六层结构 + 参数契约 + A14 校验的唯一实现。

**为什么 Skill 不能只有"操作步骤"。** 只写步骤的 Skill 会稳定产出一类失败：
模型照着步骤跑完，然后宣布"已完成"，而结果里缺了最关键的东西——没给引用、
没覆盖要求的维度、数字对不上。这不是模型不听话，是 Skill 没定义**什么算完成**。
六层里前五层回答"怎么做"，第六层回答"做到什么程度才算做到"。

| 层 | 字段 | 缺了会怎样 |
|---|---|---|
| 1 身份 | `name` / `description` | 模型不知道何时该用它 |
| 2 使命边界 | `mission` | 拿它去干不该干的活（万能 Skill） |
| 3 领域原则 | `principles` | 步骤对但违反行业硬规矩 |
| 4 执行步骤 | `instructions` | — （唯一大家都会写的一层） |
| 5 工具依赖 | `parameters` / `entrypoint` / `permissions` / `sandbox` | 调不动或越权 |
| 6 完成标准 | `done_criteria` | "已完成"无法被验证 |

两条设计纪律：

1. **`permissions` / `dangerous` 今天就声明。** D5 的白名单、W10 的 HITL
   都读它们；事后再补等于把已经上线的 Skill 全部重审一遍。
2. **`done_criteria` 只允许机器可判的四类。** 写"质量要高"等于没写——
   校验器判不了，就又回到"模型自己说好"。`min_items` 必须带 `count`：
   没有数字的"至少几条"不是一个可判定的条件。

参数契约直接复用 `ToolSpec` 的 JSON Schema 形态，所以 conformance A14 的
校验逻辑可以**同时**管工具与 Skill——这也是 `validate_arguments` 放在这里
而不是 `executor.py` 的原因：它是契约的一部分，不是某条链路的私货。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from ..llm.types import ToolSpec

__all__ = [
    "CHECK_KINDS",
    "KNOWN_PERMISSIONS",
    "PERMISSION_NET",
    "PERMISSION_PUBLISH",
    "PERMISSION_READ",
    "PERMISSION_SHELL",
    "PERMISSION_WRITE",
    "SANDBOX_LEVELS",
    "Check",
    "SkillSpec",
    "SkillSpecError",
    "TaskStage",
    "VerifySpec",
    "validate_arguments",
]

# 权限词表。**闭集**：写错一个词（`writes`）应当是加载期错误，不是
# "这个 Skill 永远不可见"——后者是静默失败，排查成本极高。
PERMISSION_READ = "read"
PERMISSION_NET = "net"
PERMISSION_WRITE = "write"
PERMISSION_SHELL = "shell"
PERMISSION_PUBLISH = "publish"
KNOWN_PERMISSIONS = frozenset(
    {PERMISSION_READ, PERMISSION_NET, PERMISSION_WRITE, PERMISSION_SHELL, PERMISSION_PUBLISH}
)

# 沙箱分级。`none` 只允许无副作用的只读能力（检索、查库）；
# 有任何写/执行能力的 Skill 至少 `subprocess`。
SANDBOX_LEVELS = ("none", "subprocess", "container")

CHECK_KINDS = ("contains", "cites", "min_items", "regex")

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{1,63}$")


class TaskStage(StrEnum):
    """任务阶段。**最小模型可见面**的第二个裁剪维度（见 `visibility.py`）。

    为什么阶段是四档而不是"发散/收敛"两档：裁得太粗没有效果（调研与写作
    要的工具差很多），裁得太细没人会在调用时正确标注（十档就需要一张对照表）。
    """

    RESEARCH = "research"
    DRAFT = "draft"
    PUBLISH = "publish"
    MAINTENANCE = "maintenance"


ALL_STAGES = frozenset(stage.value for stage in TaskStage)


@dataclass(frozen=True, slots=True)
class VerifySpec:
    """验真方式（第 6 层的机读版）：**回读真实状态**，不信 HTTP 200。

    声明式而不是回调：验真规则要能被 review、被 eval 断言、被审计打印。
    一个藏在 `lambda` 里的验真逻辑，等价于"没人知道它验了什么"。
    """

    # `server.tool`：回读用的工具（必须是只读的）。
    entrypoint: str
    # 从执行结果里取哪个字段当回读 id（`post_id` / `path` / ...）。
    id_field: str = "post_id"
    # 回读结果里哪个字段表示状态，以及哪些取值算"真的生效了"。
    status_field: str = "status"
    ok_status: tuple[str, ...] = ("published", "ok", "done")

    def read_back_args(self, payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """从执行结果里拼回读参数。取不到 id 就无法验真——返回 None，
        由调用方按 `unverified` 处理（**不能**默认当作成功）。"""
        if not payload:
            return None
        value = payload.get(self.id_field)
        if value in (None, ""):
            return None
        return {"id": str(value)}

    def is_ok(self, read_back: Mapping[str, Any] | None) -> bool:
        if not read_back:
            return False
        return str(read_back.get(self.status_field, "")) in self.ok_status


class SkillSpecError(ValueError):
    """Skill 定义不合法。加载期错误——**不**留到第一次调用才发现。"""


@dataclass(frozen=True, slots=True)
class Check:
    """一条机器可判的完成标准。

    `kind` 只有四种，多一种就意味着校验器多一段没人测的分支：

    - `contains`：输出里必须出现 `target` 这段原文
    - `cites`：输出里必须有 ≥`count` 个引用标记（`target` 是引用的正则，默认 `\\[\\d+\\]`）
    - `min_items`：输出里匹配 `target` 的条目 ≥`count` 个（默认按行首列表符计）
    - `regex`：输出必须匹配 `target`
    """

    kind: Literal["contains", "cites", "min_items", "regex"]
    # 找什么。`cites` / `min_items` 时是正则，`contains` 时是原文。
    target: str = ""
    # 不满足时**回喂给模型**的话。写"请补充"没用，要写"缺什么、去哪补"。
    hint: str = ""
    # 数量门槛。只有 `min_items` / `cites` 用；`contains` / `regex` 忽略。
    # 单独一个字段而不是塞进 `target` 字符串（`"3|..."`）：解析字符串的
    # 代码迟早会被某种格式的 target 打穿。
    count: int = 1

    def __post_init__(self) -> None:
        if self.kind not in CHECK_KINDS:
            raise SkillSpecError(f"未知的 done_criteria kind: {self.kind!r}（支持 {CHECK_KINDS}）")
        if not self.hint.strip():
            raise SkillSpecError(
                f"done_criteria[{self.kind}] 缺少 hint：不满足时模型不知道该补什么"
            )
        if self.kind == "contains" and not self.target:
            raise SkillSpecError("contains 必须有 target")
        if self.kind in {"cites", "min_items", "regex"}:
            try:
                re.compile(self.target)
            except re.error as exc:
                raise SkillSpecError(
                    f"done_criteria[{self.kind}] 的 target 不是合法正则：{self.target!r}（{exc}）"
                ) from exc
        if self.count < 1:
            raise SkillSpecError("count 必须 ≥ 1")


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """一个 Skill 的完整定义（六层）。构造后不可变——注册表会把它放进 dict、
    多个请求共享，可变对象在这里等于共享状态 bug。"""

    # -- 1 身份 ------------------------------------------------------------ #
    name: str
    # 给模型看的一句话。渐进式加载时**只有它**会被注入，所以它必须能独立
    # 回答"什么时候该用我"。上限 300 字符是经验值：再长就不如直接看全文。
    description: str

    # -- 2 使命边界 -------------------------------------------------------- #
    # 必须写清"不负责什么"。只写"负责什么"等于没有边界——模型会拿它去干
    # 相邻的活，而失败时的表现是"答得很像但不该由它答"。
    mission: str

    # -- 3 领域原则 -------------------------------------------------------- #
    # 做这行当的硬规矩（"引用必须给原文"、"不得评价竞品"）。
    principles: tuple[str, ...] = ()

    # -- 4 执行步骤 -------------------------------------------------------- #
    instructions: str = ""

    # -- 5 工具依赖 -------------------------------------------------------- #
    # JSON Schema（与 ToolSpec.parameters 同形，A14 直接复用）。
    parameters: dict[str, Any] = field(default_factory=dict)
    # 入口：`server.tool` 形式。执行器按 `server` 找 client，按 `tool` 调用。
    entrypoint: str = ""
    permissions: frozenset[str] = frozenset()
    sandbox: str = "none"
    # 哪些任务阶段该让它出现（`visibility.py` 的输入）。空 = 所有阶段可见，
    # 但那是**默认放开**，`lint()` 会提示显式声明——"所有阶段都需要"
    # 和"忘了写"在报告里必须能区分。
    stages: frozenset[str] = frozenset()

    # -- 6 完成标准 -------------------------------------------------------- #
    done_criteria: tuple[Check, ...] = ()
    # 验真的声明（回读哪个 id、什么状态算生效）。D5 的 verify 层读它。
    verify: VerifySpec | None = None

    # 高风险动作（对外发布、删除、付费）。D5 的审批门与 W10 的 HITL 读它。
    dangerous: bool = False
    # 定义来源（文件路径 / "builtin"）。排障时"这个 Skill 哪来的"必须能回答。
    source: str = "builtin"
    version: str = "1"

    # -- 校验 -------------------------------------------------------------- #

    def validate(self) -> None:
        """结构校验。**全部是硬错误**，没有"警告后继续"——
        一个字段写错的 Skill 一旦加载成功，它会以一个看起来正常的方式失败。"""
        if not _NAME_RE.match(self.name):
            raise SkillSpecError(
                f"Skill 名 {self.name!r} 不合法：需匹配 {_NAME_RE.pattern}"
                f"（同时也是合法的工具名，便于直接喂给模型）"
            )
        if not self.description.strip():
            raise SkillSpecError(f"{self.name}: description 不能为空（模型靠它决定何时调用）")
        if len(self.description) > 300:
            raise SkillSpecError(
                f"{self.name}: description 长 {len(self.description)} 字符，超过 300。"
                f"渐进式加载只注入这一段，太贵；细节写进 instructions"
            )
        if not self.mission.strip():
            raise SkillSpecError(f"{self.name}: mission 不能为空（没有边界就是万能 Skill）")
        if not self.instructions.strip():
            raise SkillSpecError(f"{self.name}: instructions 不能为空（至少要有执行步骤）")

        unknown = set(self.permissions) - KNOWN_PERMISSIONS
        if unknown:
            raise SkillSpecError(
                f"{self.name}: 未知权限 {sorted(unknown)}；支持 {sorted(KNOWN_PERMISSIONS)}"
            )
        if not self.permissions:
            # 无权限的 Skill 其实是"纯提示词"，它不需要 entrypoint；
            # 有 entrypoint 却不声明权限 = 想越过白名单。
            if self.entrypoint:
                raise SkillSpecError(
                    f"{self.name}: 声明了 entrypoint 就必须声明 permissions（否则绕过白名单）"
                )
        else:
            if not self.entrypoint:
                raise SkillSpecError(f"{self.name}: 有 permissions 就必须有 entrypoint")
            if "." not in self.entrypoint:
                raise SkillSpecError(
                    f"{self.name}: entrypoint {self.entrypoint!r} 需为 'server.tool' 形式"
                )

        if self.sandbox not in SANDBOX_LEVELS:
            raise SkillSpecError(f"{self.name}: sandbox 必须是 {SANDBOX_LEVELS} 之一")

        unknown_stages = set(self.stages) - ALL_STAGES
        if unknown_stages:
            raise SkillSpecError(
                f"{self.name}: 未知阶段 {sorted(unknown_stages)}；支持 {sorted(ALL_STAGES)}"
            )

        # 沙箱等级与权限的一致性：有写/执行能力就必须有隔离。
        if self.sandbox == "none" and self.permissions & {PERMISSION_WRITE, PERMISSION_SHELL}:
            raise SkillSpecError(
                f"{self.name}: 声明了 write/shell 权限就不能用 sandbox=none"
                f"（至少 subprocess）"
            )

        if self.permissions and not self.parameters:
            raise SkillSpecError(
                f"{self.name}: 有 entrypoint 就必须声明 parameters"
                f"（空 schema 会被 A14 拒掉一切参数）"
            )
        if not isinstance(self.parameters, dict):
            raise SkillSpecError(f"{self.name}: parameters 必须是 dict")

        # 危险动作必须有验真标准：没有 done_criteria 的"已发布"没法验真，
        # 而 D5 的 verify 层正是靠它判断"接口 200 但没生效"。
        if self.dangerous and not self.done_criteria:
            raise SkillSpecError(
                f"{self.name}: dangerous=True 必须声明 done_criteria（否则'已发布'无法验证）"
            )
        for check in self.done_criteria:
            if not isinstance(check, Check):
                raise SkillSpecError(f"{self.name}: done_criteria 里必须是 Check 实例")

        if self.verify is not None:
            if not isinstance(self.verify, VerifySpec):
                raise SkillSpecError(f"{self.name}: verify 必须是 VerifySpec")
            if "." not in self.verify.entrypoint:
                raise SkillSpecError(
                    f"{self.name}: verify.entrypoint {self.verify.entrypoint!r} "
                    f"需为 'server.tool' 形式"
                )
            # 只读能力不需要回读（它没有"生效"这回事）；有副作用的才需要。
            if not self.permissions & {PERMISSION_WRITE, PERMISSION_PUBLISH}:
                raise SkillSpecError(
                    f"{self.name}: 只有 write/publish 类 Skill 需要 verify"
                    f"（只读动作没有「生效」可验）"
                )

    def lint(self) -> tuple[str, ...]:
        """可运行但可疑的写法。与 `validate()` 分开：这些不是错误，
        但都该在 review 时被看见。"""
        problems: list[str] = []
        if not any(marker in self.mission for marker in ("不负责", "不用于", "不做", "不包含")):
            problems.append(f"{self.name}: mission 没写「不负责什么」（只写负责什么等于没边界）")
        if not self.principles:
            problems.append(f"{self.name}: 没有 principles（领域硬规矩会退化成模型自由裁量）")
        if self.dangerous and PERMISSION_PUBLISH not in self.permissions:
            problems.append(f"{self.name}: dangerous=True 但没声明 publish 权限（风险级会偏低）")
        if not self.stages:
            problems.append(f"{self.name}: 没声明 stages（默认所有阶段可见，等于没做可见面收敛）")
        return tuple(problems)

    # -- 转换 -------------------------------------------------------------- #

    def to_tool_spec(self) -> ToolSpec:
        """给模型看的工具声明。描述里带上使命边界——那是模型判断
        "该不该用这个工具"最需要的一句话。"""
        return ToolSpec(
            name=self.name,
            description=f"{self.description}（不负责：{_not_in_charge(self.mission)}）",
            parameters=self.parameters,
        )

    def describe(self) -> str:
        """渐进式加载的**唯一**注入形态：`name + description`（约 20 token）。

        `instructions` 可能有几百 token。3 个 Skill 时全注入无所谓，
        30 个时它会把 context 直接吃掉——所以"选中后才注入"不是优化，
        是这个机制能扩到 30 个的前提。
        """
        return f"- {self.name}: {self.description}"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, source: str = "inline") -> SkillSpec:
        """从 YAML/dict 构造。未知字段直接报错：拼错的键（`permisions`）
        被静默忽略，等于给了一个没有权限限制的 Skill。"""
        known = {
            "name",
            "description",
            "mission",
            "principles",
            "instructions",
            "parameters",
            "entrypoint",
            "permissions",
            "sandbox",
            "stages",
            "done_criteria",
            "verify",
            "dangerous",
            "version",
        }
        unknown = set(raw) - known
        if unknown:
            raise SkillSpecError(f"未知字段 {sorted(unknown)}；允许 {sorted(known)}")

        checks = tuple(
            Check(
                kind=str(item["kind"]),  # type: ignore[arg-type]
                target=str(item.get("target") or ""),
                hint=str(item.get("hint") or ""),
                count=int(item.get("count", 1)),
            )
            for item in _as_sequence(raw.get("done_criteria"))
        )

        spec = cls(
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            mission=str(raw.get("mission") or ""),
            principles=tuple(str(item) for item in _as_sequence(raw.get("principles"))),
            instructions=str(raw.get("instructions") or ""),
            parameters=dict(raw.get("parameters") or {}),
            entrypoint=str(raw.get("entrypoint") or ""),
            permissions=frozenset(str(item) for item in _as_sequence(raw.get("permissions"))),
            sandbox=str(raw.get("sandbox") or "none"),
            stages=frozenset(str(item) for item in _as_sequence(raw.get("stages"))),
            done_criteria=checks,
            verify=_as_verify(raw.get("verify")),
            dangerous=bool(raw.get("dangerous", False)),
            source=source,
            version=str(raw.get("version") or "1"),
        )
        spec.validate()
        return spec


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _as_verify(value: Any) -> VerifySpec | None:
    """`verify` 既可以是一个 dict，也可以写成单个 `server.tool` 字符串
    （最常见的情况只需要 entrypoint）。"""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return VerifySpec(entrypoint=value)
    if isinstance(value, Mapping):
        known = {"entrypoint", "id_field", "status_field", "ok_status"}
        unknown = set(value) - known
        if unknown:
            raise SkillSpecError(f"verify 未知字段 {sorted(unknown)}；允许 {sorted(known)}")
        ok_status = tuple(str(item) for item in _as_sequence(value.get("ok_status")))
        return VerifySpec(
            entrypoint=str(value.get("entrypoint") or ""),
            id_field=str(value.get("id_field") or "post_id"),
            status_field=str(value.get("status_field") or "status"),
            ok_status=ok_status or ("published", "ok", "done"),
        )
    raise SkillSpecError(f"verify 必须是字符串或映射，收到 {type(value).__name__}")


def _not_in_charge(mission: str) -> str:
    """从 mission 里挑出"不负责"那半句，喂给工具描述。"""
    for marker in ("不负责", "不用于", "不做", "不包含"):
        index = mission.find(marker)
        if index >= 0:
            tail = mission[index:].split("；")[0].split("。")[0].strip()
            return tail or mission
    return mission[:60]


# --------------------------------------------------------------------------- #
# A14 参数校验：工具与 Skill 共用的唯一实现
# --------------------------------------------------------------------------- #


def validate_arguments(
    parameters: Mapping[str, Any] | None, arguments: Mapping[str, Any] | None
) -> list[str]:
    """JSON Schema（子集）→ 违规列表。空列表 = 通过。

    刻意只支持 A14 需要的子集：`type` / `properties` / `required` /
    `additionalProperties` / `enum` / `minimum` / `maximum` / `minLength` /
    `maxLength`。不引 `jsonschema` 库的理由和 redact 不做全文扫描一样：
    Skill 的参数契约是我们自己写的，用到的模式就是这些；为了没用到的一半
    规范引一个依赖，换来的是"校验行为和线上不一致"的风险。

    违反项**全部返回**而不是抛第一个错：A14 的价值是"一次看清模型怎么跑偏的"，
    逐个报错会让调用方重试三次才凑齐一份完整的错误报告。
    """
    schema = parameters or {}
    args = dict(arguments or {})
    problems: list[str] = []

    if not isinstance(args, dict):
        return ["tool arguments 必须是对象"]

    props = schema.get("properties") or {}
    if not isinstance(props, Mapping):
        props = {}
    required = [str(item) for item in (schema.get("required") or [])]

    unknown = set(args) - set(props)
    # `additionalProperties` 缺省视为**允许**（JSON Schema 的默认语义）。
    # 想严格就必须显式写 false——这条默认值不能替调用方改，
    # 改了会让所有手写 schema 的 Skill 突然全部拒绝合法参数。
    if unknown and schema.get("additionalProperties") is False:
        problems.append(f"schema 之外的参数 {sorted(unknown)}")

    missing = [name for name in required if name not in args]
    if missing:
        problems.append(f"缺少必填参数 {sorted(missing)}")

    for name, value in args.items():
        rule = props.get(name)
        if not isinstance(rule, Mapping):
            continue
        problems.extend(_check_value(name, value, rule))

    return problems


_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _check_value(name: str, value: Any, rule: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    expected = rule.get("type")

    if isinstance(expected, str) and expected in _TYPE_CHECKS:
        allowed = _TYPE_CHECKS[expected]
        # bool 是 int 的子类，`True` 会被当成合法 integer —— 这是 Python
        # 类型系统的坑，必须在工具边界上显式堵住。
        is_bool_mismatch = isinstance(value, bool) and expected in {"integer", "number"}
        if is_bool_mismatch or not isinstance(value, allowed):
            problems.append(f"参数 {name} 应为 {expected}，收到 {type(value).__name__}")
            return problems

    enum = rule.get("enum")
    if isinstance(enum, Sequence) and enum and value not in enum:
        problems.append(f"参数 {name} 必须是 {list(enum)} 之一")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            problems.append(f"参数 {name}={value} 小于下限 {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            problems.append(f"参数 {name}={value} 超过上限 {maximum}")

    if isinstance(value, str):
        min_length = rule.get("minLength")
        max_length = rule.get("maxLength")
        if isinstance(min_length, int) and len(value) < min_length:
            problems.append(f"参数 {name} 长度小于 {min_length}")
        if isinstance(max_length, int) and len(value) > max_length:
            problems.append(f"参数 {name} 长度超过 {max_length}")

    return problems
