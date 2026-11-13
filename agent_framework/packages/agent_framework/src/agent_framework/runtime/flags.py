"""Feature Flag（W11 D5）：不改代码、不发版就能退回旧行为。

## 一条铁律：默认值 = **旧行为**

如果默认值是"新行为"，那么回滚就要改代码、重新发版——于是 Kill Switch 等于没有。
所以每个 Flag 的 `default` 一律是 `False`（关 = 旧路径），开启是**显式动作**。

`test_every_flag_defaults_to_the_old_behaviour` 会逐个断言这一点。
这条测试看起来废话，但它是唯一能挡住"顺手把默认值改成 True"的东西。

## 灰度：按主体稳定分桶，不是随机

`10%` 的含义必须是"**同一批人**持续走新逻辑"，而不是"每个请求 10% 概率"。
随机的话，同一个用户会在新旧行为之间来回跳，任何一个跨多次请求的 bug 都复现不出来，
而用户看到的系统像在抽风。

分桶用 `sha256(flag:principal)` 取模 10000：同一主体、同一 flag 永远落在同一档，
不同 flag 之间互相独立（不会"进了 A 就必然进 B"）。

## 一个 Flag 是有寿命的

`removal_deadline` 不是形式主义：一个开了两年的 Flag，两条路径都会被改，
一年后没人敢关它。所以 `FlagSet.stale()` 会把过期的列出来——**它必须能被查出来**，
否则"临时开关"会永久留在代码里，而临时是最长久的那个词。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

__all__ = [
    "FLAGS",
    "Flag",
    "FlagSet",
    "FlagState",
    "bucket",
    "parse_features",
]


@dataclass(frozen=True, slots=True)
class Flag:
    """一个开关。`default` 永远是旧行为（`False`）。"""

    name: str
    description: str
    # 旧行为是"关"。改成 True 之前请先读模块文档第一段。
    default: bool = False
    # 什么时候必须清掉这个 Flag。到期不清，代码里就会永久多一条分支。
    removal_deadline: date | None = None


# 当前在册的开关。**加一个 Flag 要在这里登记**：不登记的名字在 `FlagSet` 里
# 会抛 KeyError——拼错的开关名最坏的结果是"以为开了其实没开"，那和没做一样。
#
# 两个**刻意不登记**的候选，写在这里是因为"为什么不是 Flag"比"是 Flag"更容易被问：
#
# - `circuit_breaker`：已经有 `BREAKER_ENABLED`（W11 D2）。同一个行为两个开关，
#   一定会出现"我关了 A 它还在跑"——而排查时两个人各看一个开关。
#   一个**类型化布尔**比一个字符串开关更不容易拼错，保留前者。
# - `batched_rewrite`：把三平台改写合并成一次调用，W8 已实测否决
#   （LinkedIn 的段落结构被带进 X，X 文案出现空行，`validate()` 拦下）。
#   给一个被否决的方案配开关，等于给"开着但没人敢开"的代码办长期居留。
FLAGS: dict[str, Flag] = {
    "semantic_cache": Flag(
        name="semantic_cache",
        description="相似 query 复用上一次的结果（W11 D5 成本冲刺）",
        removal_deadline=date(2027, 1, 31),
    ),
    "difficulty_router": Flag(
        name="difficulty_router",
        description="简单任务路由到 cheap 档（architecture §8）",
        removal_deadline=date(2027, 1, 31),
    ),
}


@dataclass(frozen=True, slots=True)
class FlagState:
    """一次判定结果。`reason` 必须能解释"为什么是这个结论"。"""

    name: str
    enabled: bool
    reason: str
    bucket: int | None = None
    rollout: float | None = None


def _normalize(value: str) -> str:
    return value.strip().lower()


def parse_features(raw: str | None) -> dict[str, str]:
    """解析 `FEATURES="semantic_cache=10%,difficulty_router=on"`。

    值只接受三种：`on` / `off` / `0–100%`。其余一律抛错——
    一个拼错的值（比如 `10` 忘了 `%`）如果被静默当成 `on`，等于 100% 放量。
    """

    parsed: dict[str, str] = {}
    for item in (raw or "").split(","):
        chunk = _normalize(item)
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"FEATURES 条目缺少 '='：{item!r}（示例：a=on,b=10%,c=off）")
        name, _, value = chunk.partition("=")
        name, value = _normalize(name), _normalize(value)
        if not name:
            raise ValueError(f"FEATURES 条目缺少名字：{item!r}")
        if value not in {"on", "off"} and not value.endswith("%"):
            raise ValueError(f"FEATURES 的值只能是 on/off/N%（收到 {value!r}）")
        if value.endswith("%"):
            percent = value[:-1]
            if not percent.isdigit() or int(percent) > 100:
                raise ValueError(f"FEATURES 的百分比非法：{value!r}")
        parsed[name] = value
    return parsed


def bucket(flag: str, principal: str) -> int:
    """稳定分桶：0–9999。同一 (flag, principal) 永远落同一档。"""

    digest = hashlib.sha256(f"{flag}:{principal}".encode()).hexdigest()
    return int(digest[:8], 16) % 10_000


@dataclass
class FlagSet:
    """所有开关的当前配置 + 判定逻辑。"""

    values: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: object) -> FlagSet:
        """从 `Settings.features` 造。**未登记的名字在这里就报错**。

        静默忽略拼错的开关（`semantic_cach=on`）是最坏的一种失败：
        运维以为开了、指标没动、而代码里没有任何一处会因此变红。
        启动时直接炸掉，比上线三天后才发现便宜得多。
        """

        values = parse_features(getattr(settings, "features", ""))
        unknown = sorted(set(values) - set(FLAGS))

        if unknown:
            raise ValueError(
                f"FEATURES 里出现未登记的开关 {unknown}；已知：{sorted(FLAGS)}"
            )
        return cls(values=values)

    # -- 判定 --------------------------------------------------------------- #

    def rollout_of(self, name: str) -> float:
        """该 flag 的放量比例（0–1）。未配置 / `off` → 0。"""

        self._require(name)
        raw = _normalize(self.values.get(name, ""))

        if not raw or raw == "off":
            return 0.0
        if raw == "on":
            return 1.0
        return int(raw[:-1]) / 100

    def state(self, name: str, *, principal: str | None = None) -> FlagState:
        """判定一个 flag 对某个主体是否生效。"""

        self._require(name)
        raw = _normalize(self.values.get(name, ""))
        rollout = self.rollout_of(name)

        if not raw:
            return FlagState(name, FLAGS[name].default, "未配置，取默认（旧行为）")

        if raw == "off":
            return FlagState(name, False, "显式关闭")

        if raw == "on":
            return FlagState(name, True, "全量开启", rollout=1.0)

        # 按比例放量：主体为空时**不分桶**，直接关。
        # 匿名流量做灰度没有意义——它无法被"同一批人持续走新逻辑"这个前提覆盖。
        if not principal:
            return FlagState(name, False, "按比例放量但主体未知，默认关", rollout=rollout)

        slot = bucket(name, principal)
        enabled = slot < int(rollout * 10_000)
        return FlagState(
            name,
            enabled,
            f"分桶 {slot}/10000 {'<' if enabled else '>='} {int(rollout * 10_000)}",
            bucket=slot,
            rollout=rollout,
        )

    def enabled(self, name: str, *, principal: str | None = None) -> bool:
        return self.state(name, principal=principal).enabled

    # -- 运维 --------------------------------------------------------------- #

    def stale(self, *, today: date | None = None) -> list[str]:
        """已过清除期限的 flag。`removal_deadline` 为空的是常态开关，不算过期。"""

        now = today or date.today()
        return sorted(
            flag.name
            for flag in FLAGS.values()
            if flag.removal_deadline is not None and flag.removal_deadline < now
        )

    def snapshot(self, *, today: date | None = None) -> dict[str, object]:
        """进 `asset_snapshot` 的那一份。`stale` 一起带上，复盘时能看见技术债。"""

        return {
            "values": dict(self.values),
            "effective": {
                name: self.rollout_of(name) for name in FLAGS if name in self.values
            },
            "stale": self.stale(today=today),
        }

    def _require(self, name: str) -> None:
        if name not in FLAGS:
            raise KeyError(f"未登记的 flag {name!r}（已知：{sorted(FLAGS)}）")
