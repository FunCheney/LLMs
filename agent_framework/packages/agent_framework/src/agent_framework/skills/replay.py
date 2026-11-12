"""录制回放（W8 D2）：外部工具的离线可复现层。

ROADMAP §8 点名的风险项：搜索/抓取 API 不稳定且有成本，直接接进评测
会让 CI 天天变红，而且**变红的原因与被测代码无关**。这一层用 W1 夹具的
同一套思路（录制真实字节 → 回放 → 测试断言不受外部波动影响）解决它。

四条纪律，每条都对应一次真实的翻车方式：

1. **未命中就炸，绝不偷偷联网。** `record=False` 时找不到夹具抛
   `ReplayMissError`。静默降级成真调用是最坏的设计：CI 偶发联网，
   指标时红时绿，而排查的人会先怀疑自己的改动。
2. **回放键用"规范化参数"的哈希。** `{"q": "a", "n": 1}` 与
   `{"n": 1, "q": "a"}` 必须打到同一条夹具（排序后序列化）。
   JSON 的键顺序不该影响语义，更不该决定 CI 是否通过。
3. **落盘前脱敏。** 夹具会进 git，而抓回来的响应里经常带着 key、
   cookie、带签名的 URL。脱敏在写入侧做一次（`redact.sanitize`），
   读取侧不再脱敏——两侧都做等于"改了脱敏规则后旧夹具永远不一致"。
4. **回放标记必须回流到调用方。** `_meta.replayed = True`，且成本记 0。
   不标的话，W11 的成本看板会把"这周跑了 3000 次搜索"和
   "这周真的花了 30 块"混成一件事。

夹具布局（便于 review，一条 diff 只对应一次调用）：

    tests/fixtures/tools/_tools.json          # tools/list 的返回
    tests/fixtures/tools/search_web/<hash>.json
    tests/fixtures/tools/fetch_url/<hash>.json

文件体：`{"tool", "args", "result", "meta": {"synthetic", "recorded_at", "schema"}}`
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .mcp.types import MCPTransport, tool_name_part
from .redact import sanitize

logger = logging.getLogger(__name__)

__all__ = [
    "FIXTURE_SCHEMA",
    "RecordedToolTransport",
    "ReplayMissError",
    "ReplayStats",
    "canonical_args",
    "fixture_key",
]

# 夹具格式版本。加字段不改版本；**语义**变了才 +1，并在读取时拒绝未来版本
# （读一个自己不认识的格式，比报错更危险）。
FIXTURE_SCHEMA = 1

# 单个响应体的落盘上限（字符）。超过就不录，只记一条 warning：
# 一个 5MB 的 HTML 塞进 git 之后，这个仓库的 clone 时间会一直是别人的问题。
MAX_FIXTURE_CHARS = 200_000


class ReplayMissError(RuntimeError):
    """回放未命中。**这是设计中的失败**，不是意外——所以它是一条清晰的报错，
    而不是一个能被 `except Exception` 顺手吞掉的空结果。"""


def canonical_args(arguments: dict[str, Any] | None) -> str:
    """参数规范化：键排序 + 紧凑分隔符 + 不转义中文。

    `sort_keys=True` 是必须的（见纪律 2）；`ensure_ascii=False` 是为了
    夹具可读——`\\u4e2d\\u6587` 在 diff 里没法 review。
    """
    return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fixture_key(tool: str, arguments: dict[str, Any] | None) -> str:
    """`sha256(tool + canonical_args)[:16]`。

    把 tool 一起算进去：不同工具收到同样参数是常态（`{"url": ...}`），
    只用参数哈希会让它们撞到同一个文件，然后回放出错误的结果——
    这类错误不会报错，只会让答案悄悄变错。
    """
    digest = hashlib.sha256(f"{tool}\x00{canonical_args(arguments)}".encode())
    return digest.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ReplayStats:
    """回放统计。报告里"命中/未命中/新录制"三个数必须分开列。"""

    hits: int = 0
    misses: int = 0
    records: int = 0
    # tools/list 单独计数：它的命中率不代表工具调用的命中率。
    list_hits: int = 0
    list_records: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.records

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "records": self.records,
            "list_hits": self.list_hits,
            "list_records": self.list_records,
        }


class RecordedToolTransport:
    """包在真传输外面的一层。`record=True` 时透传并落盘，`False` 时只读夹具。

    它是 `MCPTransport` 的实现，所以 `MCPClient` 完全不知道自己在回放——
    "回放"不该是一个贯穿全链路的分支，那会让每条路径都有两个版本。
    """

    def __init__(
        self,
        *,
        fixtures_root: str | Path,
        record: bool = False,
        upstream: MCPTransport | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._root = Path(fixtures_root)
        self._record = record
        self._upstream = upstream
        self._clock = clock or (lambda: datetime.now(UTC).isoformat(timespec="seconds"))
        self._stats = ReplayStats()

        if record and upstream is None:
            # 录制模式没有上游 = 配置错误。立刻炸，别等到第一次调用时
            # 才发现"录了个寂寞"。
            raise ValueError("record=True 必须提供 upstream（真实传输层）")

    @property
    def stats(self) -> ReplayStats:
        return self._stats

    # -- MCPTransport ------------------------------------------------------- #

    async def list_tools(self) -> dict[str, Any]:
        path = self._root / "_tools.json"

        if not self._record:
            if not path.is_file():
                self._note_miss()
                raise ReplayMissError(
                    f"回放未命中：{path} 不存在。"
                    f"先在联网环境跑 `python evals/scripts/tool_boundary_check.py --record` "
                    f"或显式构造 record=True 的传输层来生成夹具；"
                    f"离线档不会替你联网。"
                )
            payload = _read_fixture(path)
            self._stats = _replace(self._stats, list_hits=self._stats.list_hits + 1)
            return _as_result(payload, replayed=True)

        assert self._upstream is not None  # noqa: S101 - 构造时已保证
        raw = await self._upstream.list_tools()
        cleaned = sanitize(raw)
        _write_fixture(
            path,
            {"tool": "__list__", "args": {}, "result": cleaned, "meta": self._meta(cleaned)},
        )
        self._stats = _replace(self._stats, list_records=self._stats.list_records + 1)
        return raw

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._root / tool_name_part(name) / f"{fixture_key(name, arguments)}.json"

        if not self._record:
            if not path.is_file():
                self._note_miss()
                raise ReplayMissError(
                    f"回放未命中：{path}。tool={name} args={canonical_args(arguments)}\n"
                    f"离线档不会联网。要新增这条，请联网运行录制"
                    f"（RecordedToolTransport(record=True, upstream=...)）。"
                )
            payload = _read_fixture(path)
            record = payload.get("result")
            if not isinstance(record, dict):
                raise ReplayMissError(f"夹具格式非法（result 不是对象）：{path}")
            self._stats = _replace(self._stats, hits=self._stats.hits + 1)
            return _as_result(payload, replayed=True)

        assert self._upstream is not None  # noqa: S101
        raw = await self._upstream.call_tool(name, arguments)
        cleaned = sanitize(raw)

        body = json.dumps(cleaned, ensure_ascii=False)
        if len(body) > MAX_FIXTURE_CHARS:
            # 超大响应不录，但**不报错**：调用方拿到的是真实结果，
            # 只是下次回放会未命中。让它报错等于"抓到一个大页面就整个任务失败"。
            logger.warning(
                "tool fixture skipped (too large): tool=%s chars=%d path=%s",
                name,
                len(body),
                path,
            )
        else:
            _write_fixture(
                path,
                {
                    "tool": name,
                    "args": sanitize(arguments),
                    "result": cleaned,
                    "meta": self._meta(cleaned),
                },
            )
            self._stats = _replace(self._stats, records=self._stats.records + 1)

        # 真调用返回**未脱敏**的原始结果：脱敏是"为了落盘"，不是为了改行为。
        # 若把打码后的结果返回给执行器，真调用档与回放档的行为就不同了。
        return raw

    # -- 内部 -------------------------------------------------------------- #

    def _meta(self, result: Any) -> dict[str, Any]:
        return {
            # 真录制一律 synthetic=False。手写夹具要 true（W1 ADR-010 同一约定），
            # 由人手工构造时自己标。
            "synthetic": False,
            "recorded_at": self._clock(),
            "schema": FIXTURE_SCHEMA,
        }

    def _note_miss(self) -> None:
        self._stats = _replace(self._stats, misses=self._stats.misses + 1)


def _replace(stats: ReplayStats, **changes: int) -> ReplayStats:
    """`ReplayStats` 不可变，逐字段重建。用 `dataclasses.replace` 会多一次
    `__eq__` 检查，这里只为记账，没必要。"""
    values = stats.as_dict()
    values.update(changes)
    return ReplayStats(**values)


def _read_fixture(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayMissError(f"夹具不是合法 JSON：{path}（{exc}）") from exc

    if not isinstance(payload, dict):
        raise ReplayMissError(f"夹具顶层必须是对象：{path}")

    schema = (payload.get("meta") or {}).get("schema", FIXTURE_SCHEMA)
    if isinstance(schema, int) and schema > FIXTURE_SCHEMA:
        raise ReplayMissError(
            f"夹具格式版本 {schema} 高于本代码支持的 {FIXTURE_SCHEMA}：{path}"
        )

    return payload


def _as_result(payload: dict[str, Any], *, replayed: bool) -> dict[str, Any]:
    """夹具 → 返回体，并打上回放标记。

    标记走 `_meta`（协议预留位）而不是顶层：顶层多一个 `replayed` 字段
    会被严格的 server/客户端当非法字段——**回放出来的字节必须和真调用
    逐字段同形**，否则"回放通过但真调用失败"这类问题永远测不出来。
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        return {"content": [{"type": "text", "text": ""}]}

    stamped = dict(result)
    meta = dict(stamped.get("_meta") or {})
    meta["replayed"] = replayed
    if replayed:
        # 回放不花钱。写死 0 而不是抄夹具里的 cost：夹具里那个数是
        # "录的时候花了多少"，不是"这次花了多少"。
        meta["cost"] = 0.0
    stamped["_meta"] = meta
    return stamped


def _write_fixture(path: Path, payload: dict[str, Any]) -> None:
    """原子落盘：先写临时文件再 rename。

    直接 `write_text` 在中断时会留下半个 JSON，而这个半 JSON 会被下一次
    回放当成"格式非法"——一次中断换来一个长期红灯，不值得。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    tmp.replace(path)
