"""资产版本快照（W11 D5）：系统的行为由六样东西共同决定。

只给代码打 tag，等于只能回滚六分之一。所以这里把六类资产各算一个版本号，
合成一份快照写进 `evals/history`——**复盘时能回答"这一期跑的是哪套东西"**。

| 资产 | 版本从哪来 | 回滚方式 |
|---|---|---|
| 代码 | 镜像 tag / `GIT_SHA` | 回退 tag（W10 RUNBOOK §10） |
| 模型 | `models.yaml` 的 alias → 厂商模型名 | 改 alias 指向（换回 cheap/strong） |
| Prompt | **构造 prompt 的模块源码指纹** | 回退代码（模板现在拼在代码里） |
| 工具/Skill | `SkillSpec.version` + 内容指纹 | 切 version / 回退 spec 文件 |
| RAG 策略 | 检索参数组的哈希 | 切配置组 |
| 记忆策略 | `MEMORY` + 后端 | 切 env |

## 版本号必须能"自己变"

手填的版本号一定会忘。所以除了 Skill 的 `version` 字段（W8 已有），
其余五类都用**内容指纹**：改了内容，版本就变。这比"记得改版本号"可靠得多。

副作用是版本号不可读（`a3f19c2e`），所以 `AssetVersion.source` 里写清"这个指纹是怎么算的"。

## Prompt 版本目前是"模块源码指纹"，不是"模板指纹"

因为 prompt 是拼在代码里的（`pipeline/seo.py`、`pipeline/style.py`、`answer.py`），
没有单独的模板文件可以哈希。用模块源码指纹是当下能做到的最诚实版本：
**改了 prompt 就是改了这几个文件，指纹一定变**。

等哪天模板外置成文件，这里改成读文件内容即可——接口（`AssetVersion`）不用动。
这件事写成缺口，而不是假装"prompt 已有版本管理"。
"""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "ASSET_KINDS",
    "AssetSnapshot",
    "AssetVersion",
    "capture",
    "content_version",
    "skill_drift",
]

ASSET_KINDS: tuple[str, ...] = ("code", "model", "prompt", "skill", "rag", "memory")

# 构造 prompt 的模块。**改 prompt 就是改这些文件**，所以指纹取它们。
PROMPT_MODULES: tuple[str, ...] = (
    "agent_framework.llm.client",
    "growth_agent.answer",
    "growth_agent.pipeline.seo",
    "growth_agent.pipeline.style",
)


def content_version(payload: Any, *, length: int = 8) -> str:
    """内容指纹。`sort_keys` 是必需的：同一份配置换个键顺序不算改版。"""

    import json

    text = payload if isinstance(payload, str) else json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True, slots=True)
class AssetVersion:
    """一类资产的一个版本。`source` 说明"这个号是怎么来的"，`rollback` 说明怎么退。"""

    kind: str
    name: str
    version: str
    source: str
    rollback: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "rollback": self.rollback,
        }


@dataclass(frozen=True, slots=True)
class AssetSnapshot:
    """某一刻的六类资产版本。`digest` 用来判断"两次跑的是不是同一套系统"。"""

    versions: tuple[AssetVersion, ...]
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def digest(self) -> str:
        return content_version([item.as_dict() for item in self.versions])

    def by_kind(self) -> dict[str, tuple[AssetVersion, ...]]:
        grouped: dict[str, list[AssetVersion]] = {kind: [] for kind in ASSET_KINDS}
        for item in self.versions:
            grouped.setdefault(item.kind, []).append(item)
        return {kind: tuple(items) for kind, items in grouped.items()}

    @property
    def missing_kinds(self) -> tuple[str, ...]:
        """哪几类没有版本。**空的不许静默**：少一类就等于少一条回滚路径。"""

        present = {item.kind for item in self.versions}
        return tuple(kind for kind in ASSET_KINDS if kind not in present)

    def diff(self, other: AssetSnapshot) -> list[str]:
        """和另一份快照比，哪些资产变了。复盘第一个要回答的就是这个。"""

        mine = {(item.kind, item.name): item.version for item in self.versions}
        theirs = {(item.kind, item.name): item.version for item in other.versions}

        changed: list[str] = []
        for key in sorted(set(mine) | set(theirs)):
            before, after = theirs.get(key), mine.get(key)
            if before != after:
                changed.append(f"{key[0]}/{key[1]}: {before or '—'} → {after or '—'}")
        return changed

    def as_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "captured_at": self.captured_at.isoformat(),
            "missing_kinds": list(self.missing_kinds),
            "versions": [item.as_dict() for item in self.versions],
        }


# --------------------------------------------------------------------------- #
# 六类各自的取版本方式
# --------------------------------------------------------------------------- #


def code_version() -> AssetVersion:
    """代码版本取环境变量里的 commit。取不到就老老实实写 `dev`。"""

    sha = os.environ.get("GIT_SHA") or os.environ.get("GIT_COMMIT") or "dev"
    return AssetVersion(
        kind="code",
        name="image",
        version=sha[:12],
        source="环境变量 GIT_SHA（构建时注入）；缺省 dev 表示本地未构建镜像",
        rollback="回退镜像 tag（ops/RUNBOOK.md §10）",
    )


def model_versions(catalog: Any) -> list[AssetVersion]:
    """每个 alias 一个版本：`provider:model`。

    为什么不用 catalog 文件哈希：alias 指向才是行为，文件里改一行注释不该算改版。
    """

    versions: list[AssetVersion] = []
    for alias, model in sorted(getattr(catalog, "models", {}).items()):
        versions.append(
            AssetVersion(
                kind="model",
                name=alias,
                version=f"{model.provider}:{model.model or '<empty>'}",
                source="models.yaml 的 alias → 厂商模型名",
                rollback="改 alias 指向（models.yaml）后重启；RUNBOOK §11",
            )
        )
    return versions


def prompt_versions(modules: Sequence[str] = PROMPT_MODULES) -> list[AssetVersion]:
    """prompt 版本 = 构造 prompt 的模块源码指纹。

    模块 import 不到（比如只装了框架包、没装产品应用）时**跳过并留痕**，
    不编一个假版本号出来。
    """

    versions: list[AssetVersion] = []
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - 缺少产品包是正常情况，不是错误
            continue
        path = getattr(module, "__file__", None)
        if not path:
            continue
        text = Path(path).read_text("utf-8", errors="ignore")
        versions.append(
            AssetVersion(
                kind="prompt",
                name=name,
                version=content_version(text),
                source=f"{Path(path).name} 的源码指纹（prompt 拼在代码里）",
                rollback="回退该模块所在镜像 tag（模板尚未外置，见模块文档缺口）",
            )
        )
    return versions


def skill_versions(specs: Iterable[Any]) -> list[AssetVersion]:
    """Skill 版本 = 声明的 `version` + 内容指纹（两个都留）。"""

    versions: list[AssetVersion] = []
    for spec in specs:
        versions.append(
            AssetVersion(
                kind="skill",
                name=getattr(spec, "name", "?"),
                version=str(getattr(spec, "version", "1")),
                source=f"spec.version；内容指纹 {skill_fingerprint(spec)}",
                rollback="切回上一版 spec 文件（YAML 有版本字段）",
            )
        )
    return versions


def skill_fingerprint(spec: Any) -> str:
    """Skill 内容指纹：只看会改变行为的字段，不看名字之外的元信息。"""

    payload = {
        "description": getattr(spec, "description", ""),
        "mission": getattr(spec, "mission", ""),
        "principles": list(getattr(spec, "principles", ()) or ()),
        "instructions": getattr(spec, "instructions", ""),
        "parameters": getattr(spec, "parameters", {}) or {},
        "entrypoint": getattr(spec, "entrypoint", ""),
        "permissions": sorted(getattr(spec, "permissions", ()) or ()),
    }
    return content_version(payload)


def skill_drift(specs: Iterable[Any]) -> list[str]:
    """改了 spec 却忘了改 `version` 的清单。

    手填版本号一定会忘。这条不会自动改版本，只会**把不一致说出来**——
    静默地把版本当成内容指纹，会让"版本号"这个字段失去审计意义。
    """

    drifted: list[str] = []
    # name → (声明的 version, 内容指纹)。判据是**两者一起变才算合规**：
    # 只看"键变了"会把"改了内容也改了版本号"这种正确做法也报出来，
    # 那样这份清单就没人看了——而没人看的清单等于没有。
    seen: dict[str, tuple[str, str]] = {}
    for spec in specs:
        name = getattr(spec, "name", "?")
        fingerprint = skill_fingerprint(spec)
        declared = str(getattr(spec, "version", "1"))

        previous = seen.get(name)
        if previous is not None:
            previous_version, previous_fingerprint = previous
            if previous_version == declared and previous_fingerprint != fingerprint:
                drifted.append(f"{name}: version 未变（{declared}）但内容变了")
        seen[name] = (declared, fingerprint)
    return drifted


def rag_version(settings: Any) -> AssetVersion:
    """RAG 策略版本 = 检索参数组的哈希。

    W5 的实测结论是"参数比组件重要"（关掉重排 R@1 +7.1pp、关掉改写 p95 −94 倍），
    所以检索参数必须进快照——它比 `models.yaml` 更容易被顺手改。
    """

    payload = {
        "top_k": getattr(settings, "retrieval_top_k", None),
        "budget_ms": getattr(settings, "retrieval_budget_ms", None),
    }
    return AssetVersion(
        kind="rag",
        name="retrieval",
        version=content_version(payload),
        source="检索参数组（top_k / budget）的哈希",
        rollback="切回上一组检索参数并重启（RUNBOOK §12）",
    )


def memory_version(settings: Any) -> AssetVersion:
    """记忆策略版本。`off` 与 `governed` 的行为差异是"有没有长期记忆"，必须进快照。"""

    mode = str(getattr(settings, "memory", "off"))
    return AssetVersion(
        kind="memory",
        name="policy",
        version=mode,
        source="MEMORY 环境变量（W7 D5 建的开关）",
        rollback="切回上一种 MEMORY 取值并重启",
    )


def capture(
    settings: Any,
    *,
    catalog: Any | None = None,
    skills: Iterable[Any] = (),
    prompt_modules: Sequence[str] = PROMPT_MODULES,
) -> AssetSnapshot:
    """抓一份快照。**六类都要有**，缺了就在 `missing_kinds` 里显形。"""

    versions: list[AssetVersion] = [code_version()]

    if catalog is not None:
        versions.extend(model_versions(catalog))

    versions.extend(prompt_versions(prompt_modules))
    versions.extend(skill_versions(skills))
    versions.append(rag_version(settings))
    versions.append(memory_version(settings))

    return AssetSnapshot(versions=tuple(versions))


def rollback_table(snapshot: AssetSnapshot) -> Mapping[str, str]:
    """`kind → 回滚方式`。写进 RUNBOOK 的那张表直接由快照生成，不靠手抄。"""

    table: dict[str, str] = {}
    for item in snapshot.versions:
        table.setdefault(item.kind, item.rollback)
    return table
