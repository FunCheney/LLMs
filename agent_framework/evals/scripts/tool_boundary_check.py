#!/usr/bin/env python
"""W8 边界与受控执行的自检脚本（Gate 8 的证据来源）。

用法：
    python evals/scripts/tool_boundary_check.py                 # 全量（零密钥、零联网）
    python evals/scripts/tool_boundary_check.py --record        # 生成/更新录制夹具
    python evals/scripts/tool_boundary_check.py --json out.json # 落盘报告

**为什么需要一个脚本而不是"看单测绿了"**：单测证明每个机制单独成立，
这个脚本证明它们在**同一次运行里**同时成立，并且把三类越界、三个 badcase、
可见面比例、审计字段的实测值打成一张表。Gate 8 的 12 项里有 8 项可以直接
引这里的数字——报告里能贴数字，比贴"测试通过"有说服力。

所有外部调用都走内置桩上游（`StubSearchUpstream` / `StubFetchUpstream`），
所以它**不需要密钥、不联网**、可以进 CI。`--record` 只更新夹具，
不改变断言。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = REPO_ROOT / "packages/agent_framework/tests/fixtures/tools"

sys.path.insert(0, str(REPO_ROOT / "packages/agent_framework/src"))
sys.path.insert(0, str(REPO_ROOT / "apps/growth_agent/src"))

from agent_framework.llm.types import ToolCall, ToolSpec  # noqa: E402
from agent_framework.retrieval import Principal  # noqa: E402
from agent_framework.skills import (  # noqa: E402
    STATUS_DUPLICATE,
    STATUS_PENDING,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    ActionRuntime,
    AuditLog,
    NumericCapRule,
    PermissionGuard,
    RecordedToolTransport,
    SandboxPolicy,
    SkillExecutor,
    SkillRegistry,
    TaskStage,
    VisibilityPolicy,
    builtin_skills,
    validate_arguments,
)
from agent_framework.skills.servers import (  # noqa: E402
    SearchServer,
    WorkspaceServer,
)

BODY = "这是一篇足够长的正文，用来满足发布工具的最小长度限制。" * 2

ANONYMOUS = Principal.anonymous()
VIEWER = Principal(subject_id="viewer", groups=frozenset({"viewer"}))
EDITOR = Principal(subject_id="editor", groups=frozenset({"editor"}))
PUBLISHER = Principal(subject_id="publisher", groups=frozenset({"publisher", "editor"}))
OPS = Principal(subject_id="ops", groups=frozenset({"ops"}))


@dataclass
class Report:
    sections: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    workspace: Path | None = None

    def add(self, section: str, payload: Any) -> None:
        self.sections[section] = payload

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> int:
        return sum(1 for item in self.checks if item["passed"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "passed": self.passed,
            "failed": len(self.checks) - self.passed,
            "sections": self.sections,
        }


@dataclass(frozen=True, slots=True)
class _Env:
    """自检环境的装配结果。

    早先这个装配函数返回 4 元组，`check_action_runtime` 又手拼了第二份
    （唯一差别是没了 `rules`），于是「参数篡改」那条 check 被一个**根本没装
    业务规则**的 runtime 验了——红得莫名其妙，因为验的不是要验的东西。
    两份装配必然漂移；现在只留一份，差异用参数表达，返回值改成具名属性，
    以后加字段不会再牵动所有解包点。
    """

    executor: SkillExecutor
    runtime: ActionRuntime
    workspace: WorkspaceServer
    audit: AuditLog
    sandbox: SandboxPolicy


def _env(root: Path, *, approver: Any = None) -> _Env:
    root.mkdir(parents=True, exist_ok=True)
    sandbox = SandboxPolicy(
        write_roots=(root,),
        command_prefixes=("echo",),
        network_allow=("example.com",),
    )
    workspace = WorkspaceServer(root=root, sandbox=sandbox)
    clients = {
        "workspace": workspace.client(),
        "search": SearchServer().client(),
    }
    audit = AuditLog()
    executor = SkillExecutor(
        registry=SkillRegistry(),
        clients=clients,
        sandbox=sandbox,
        guard=PermissionGuard(approver=approver),
        audit=audit,
    )
    runtime = ActionRuntime(
        executor=executor, rules=(NumericCapRule(field="budget", maximum=200_000),)
    )
    return _Env(
        executor=executor,
        runtime=runtime,
        workspace=workspace,
        audit=audit,
        sandbox=sandbox,
    )


def _call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"check-{name}", name=name, arguments=arguments)


# --------------------------------------------------------------------------- #
# 1. MCP 边界与动态发现
# --------------------------------------------------------------------------- #


async def check_mcp_boundary(report: Report) -> None:
    # 自己持有 transport：动态发现要往 server 侧加工具，从 client 里挖私有属性
    # 测试就成了「依赖实现细节」，client 一改内部结构它就假红。
    server = SearchServer()
    transport = server.transport()
    client = server.client(transport=transport)
    specs = await client.list_tools()
    tool_specs = all(isinstance(spec, ToolSpec) for spec in specs)
    report.check(
        "1 MCP 边界：业务只拿到内部 ToolSpec",
        tool_specs and specs[0].name == "search__search_web",
        f"tools={[s.name for s in specs]}",
    )

    before = [spec.name for spec in await client.list_tools()]

    async def new_tool(params: dict) -> dict:
        return {"content": [{"type": "text", "text": "ok"}]}

    transport.add_tool(
        {"name": "trend_chart", "description": "新工具", "inputSchema": {"type": "object"}},
        new_tool,
    )
    after = [spec.name for spec in await client.list_tools()]
    report.check(
        "2 动态发现：server 加工具后重新 list 即可见",
        set(after) > set(before),
        f"{before} → {after}",
    )


# --------------------------------------------------------------------------- #
# 2. 录制回放
# --------------------------------------------------------------------------- #


async def check_replay(report: Report, tmp: Path, *, record: bool) -> None:
    if record:
        FIXTURES_ROOT.mkdir(parents=True, exist_ok=True)
        recorder = RecordedToolTransport(
            fixtures_root=FIXTURES_ROOT, record=True, upstream=SearchServer().transport()
        )
        await recorder.list_tools()
        await recorder.call_tool("search_web", {"query": "增长"})
        await recorder.call_tool("search_web", {"query": "AI"})

    player = RecordedToolTransport(fixtures_root=FIXTURES_ROOT)
    client = SearchServer().client(transport=player)
    await client.list_tools()
    outcome = await client.call_tool("search__search_web", {"query": "增长"})
    report.check(
        "3 录制回放：离线档命中且不打网络",
        outcome.ok and outcome.replayed,
        f"replayed={outcome.replayed} cost={outcome.cost}",
    )

    # 未命中必须明确报错，而不是偷偷联网
    from agent_framework.skills import ReplayMissError

    miss_root = tmp / "empty_fixtures"
    miss_root.mkdir(exist_ok=True)
    miss = RecordedToolTransport(fixtures_root=miss_root)
    raised = False
    try:
        await miss.call_tool("search_web", {"query": "从未录过"})
    except ReplayMissError:
        raised = True
    report.check("3 录制回放：未命中明确报错不联网", raised, "ReplayMissError")

    # 脱敏
    leaked = False
    for path in FIXTURES_ROOT.rglob("*.json"):
        text = path.read_text("utf-8")
        if "sk-live" in text or "supersecret" in text:
            leaked = True
    report.check("4 脱敏：夹具里没有明文密钥", not leaked)


# --------------------------------------------------------------------------- #
# 3. 沙箱三道边界
# --------------------------------------------------------------------------- #


async def check_sandbox(report: Report, root: Path) -> None:
    executor = _env(root).executor

    path_result = await executor.execute(
        _call("write_brief", path="../../escape.md", content="内容" * 20),
        principal=EDITOR,
        stage="draft",
    )
    escaped = (root.parent.parent / "escape.md").exists()
    report.check(
        "9 沙箱-路径：越界被拦且未产生副作用",
        path_result.code == "sandbox_denied" and not escaped,
        f"code={path_result.code} escaped={escaped}",
    )

    from agent_framework.skills.spec import SkillSpec

    shell = SkillSpec(
        name="shell_probe",
        description="跑一条命令。",
        mission="负责跑命令；不负责其他。",
        instructions="1. 跑。",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        entrypoint="workspace.run_command",
        permissions=frozenset({"read", "shell"}),
        sandbox="subprocess",
        stages=frozenset({"maintenance"}),
    )
    executor.registry.register(shell)
    command_result = await executor.execute(
        _call("shell_probe", command="rm -rf /tmp/w8_should_not_exist"),
        principal=OPS,
        stage="maintenance",
    )
    report.check(
        "9 沙箱-命令：rm -rf 在副作用之前被拦",
        command_result.code == "sandbox_denied" and "command_denied" in command_result.detail,
        f"code={command_result.code}",
    )

    net_result = await executor.execute(
        _call("web_fetch", url="https://evil.example.org/page"),
        principal=EDITOR,
        stage="draft",
    )
    report.check(
        "9 沙箱-网络：白名单外域名被拦",
        net_result.code == "sandbox_denied" and "network_denied" in net_result.detail,
        f"code={net_result.code}",
    )


# --------------------------------------------------------------------------- #
# 4. 可见面与渐进式加载
# --------------------------------------------------------------------------- #


def check_visibility(report: Report) -> None:
    policy = VisibilityPolicy()
    skills = builtin_skills()
    principals = {
        "anonymous": ANONYMOUS,
        "viewer": VIEWER,
        "editor": EDITOR,
        "publisher": PUBLISHER,
        "ops": OPS,
    }

    matrix: dict[str, dict[str, Any]] = {}
    worst = 0.0
    for stage in TaskStage:
        matrix[stage.value] = {}
        for label, principal in principals.items():
            decision = policy.surface(skills, stage=stage, principal=principal)
            worst = max(worst, decision.ratio)
            matrix[stage.value][label] = {
                "visible": list(decision.names()),
                "ratio": round(decision.ratio, 3),
            }
    report.add("visibility_matrix", matrix)
    report.check(
        "8 可见面：任一 (principal, stage) < 60%",
        worst < 0.6,
        f"最大可见比例 {worst:.0%}（全集 {len(skills)} 个）",
    )

    registry = SkillRegistry()
    progressive_len = 0
    full_len = 0
    for stage in TaskStage:
        decision = policy.surface(skills, stage=stage)
        progressive_len += len(decision.describe_all())
        full_len += len(registry.instructions_of(decision.names()))
    ratio = progressive_len / full_len if full_len else 0.0
    report.check(
        "8 渐进式加载：描述层 < 全量 instructions 的 20%",
        ratio < 0.2,
        f"describe={progressive_len} 字符，instructions={full_len} 字符，比例 {ratio:.1%}",
    )


# --------------------------------------------------------------------------- #
# 5. 三个 badcase + 审计
# --------------------------------------------------------------------------- #


async def _always_approve(skill: Any, arguments: Any, summary: str) -> str:
    return "approved"


async def check_action_runtime(report: Report, tmp: Path) -> None:
    env = _env(tmp / "action_root", approver=_always_approve)
    executor, runtime, workspace = env.executor, env.runtime, env.workspace

    publish = ToolCall(
        id="pub-1", name="publish_post", arguments={"title": "标题", "body": BODY}
    )
    first = await runtime.submit(publish, principal=PUBLISHER, stage="publish")
    second = await runtime.submit(publish, principal=PUBLISHER, stage="publish")
    third = await runtime.submit(publish, principal=PUBLISHER, stage="publish")
    report.add(
        "badcase_duplicate",
        {
            "statuses": [first.status, second.status, third.status],
            "side_effects": len(workspace.posts()),
            "loop_warning": [first.loop_warning, second.loop_warning, third.loop_warning],
        },
    )
    statuses = [first.status, second.status, third.status]
    posts = len(workspace.posts())
    report.check(
        "10 badcase-重复请求：只产生 1 次副作用",
        first.status == STATUS_VERIFIED
        and second.status == STATUS_DUPLICATE
        and third.status == STATUS_DUPLICATE
        and posts == 1,
        f"statuses={statuses}，副作用 {posts} 次",
    )

    # badcase 2：参数篡改（业务规则）
    from agent_framework.skills.spec import SkillSpec

    executor.registry.register(
        SkillSpec(
            name="issue_budget",
            description="记录预算审批。",
            mission="负责记录预算；不负责付款。",
            instructions="1. 写记录。",
            parameters={
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "maximum": 1_000_000},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["budget", "path", "content"],
                "additionalProperties": False,
            },
            entrypoint="workspace.write_file",
            permissions=frozenset({"read", "write"}),
            sandbox="subprocess",
            stages=frozenset({"draft"}),
        )
    )
    tampered = await runtime.submit(
        ToolCall(
            id="budget-1",
            name="issue_budget",
            arguments={"budget": 1_000_000, "path": "budget.md", "content": "预算记录"},
        ),
        principal=PUBLISHER,
        stage="draft",
        intent_id="tamper",
    )
    report.add("badcase_tamper", {"status": tampered.status, "detail": tampered.detail})
    report.check(
        "10 badcase-参数篡改：业务规则拒绝（不是权限拒绝）",
        tampered.status == "invalid_action" and "numeric_cap" in tampered.detail,
        tampered.detail,
    )

    # badcase 3：接口假成功
    fake_root = tmp / "fake_root"
    fake_root.mkdir(exist_ok=True)
    fake_ws = WorkspaceServer(root=fake_root, sandbox=env.sandbox, fail_soft=True)
    fake_executor = SkillExecutor(
        registry=SkillRegistry(),
        clients={"workspace": fake_ws.client()},
        sandbox=env.sandbox,
        guard=PermissionGuard(approver=_always_approve),
    )
    fake_runtime = ActionRuntime(executor=fake_executor)
    fake = await fake_runtime.submit(publish, principal=PUBLISHER, stage="publish")
    report.add(
        "badcase_fake_success",
        {
            "status": fake.status,
            "verify": None if fake.verify is None else fake.verify.detail,
        },
    )
    report.check(
        "10 badcase-接口假成功：unverified 而非已发布",
        fake.status == STATUS_UNVERIFIED and len(fake_ws.posts()) == 0,
        f"status={fake.status}",
    )

    # 审批门占位
    no_hitl = WorkspaceServer(root=fake_root, sandbox=env.sandbox)
    no_hitl_executor = SkillExecutor(
        registry=SkillRegistry(),
        clients={"workspace": no_hitl.client()},
        sandbox=env.sandbox,
        guard=PermissionGuard(),
    )
    pending = await ActionRuntime(executor=no_hitl_executor).submit(
        publish, principal=PUBLISHER, stage="publish"
    )
    report.check(
        "10 审批门：无 HITL 时 irreversible 动作 pending",
        pending.status == STATUS_PENDING and len(no_hitl.posts()) == 0,
        f"status={pending.status}",
    )

    # 审计字段完整性
    entry = env.audit.entries()[-1].as_dict()
    required = {
        "action_id",
        "tool",
        "skill",
        "status",
        "risk",
        "principal",
        "args_digest",
        "arg_keys",
        "latency_ms",
        "replayed",
        "cost",
        "approval",
        "stage",
    }
    missing = sorted(required - set(entry))
    report.add(
        "audit_sample",
        {key: entry[key] for key in sorted(required & set(entry))},
    )
    report.check("6 全审计：Gate 要求的字段齐全", not missing, f"缺失 {missing or '无'}")

    dumped = json.dumps(
        [item.as_dict() for item in env.audit.entries()], ensure_ascii=False
    )
    leak = BODY[:20] in dumped
    report.check("6 全审计：参数值不落盘（只留摘要）", not leak)


# --------------------------------------------------------------------------- #
# 6. 用「只给定可见面」的模型行为做一个最小断言
# --------------------------------------------------------------------------- #


async def check_executor_order(report: Report, root: Path) -> None:
    executor = _env(root).executor

    invisible = await executor.execute(
        _call("publish_post", title="t", body=BODY), principal=EDITOR, stage="research"
    )
    report.check(
        "5 执行闭环：可见面之外 → tool_not_visible（不是 permission_denied）",
        invisible.code == "tool_not_visible",
        invisible.code,
    )

    invalid = await executor.execute(
        _call("write_brief", path="a.md"), principal=VIEWER, stage="draft"
    )
    report.check(
        "5 执行闭环：参数非法先于权限拦截（顺序固定）",
        invalid.code == "invalid_arguments",
        invalid.code,
    )

    denied = await executor.execute(
        _call("write_brief", path="briefs/a.md", content="内容" * 20),
        principal=VIEWER,
        stage="draft",
    )
    report.check(
        "5 执行闭环：看得见但没授权 → permission_denied",
        denied.code == "permission_denied",
        denied.code,
    )

    ok = await executor.execute(
        _call(
            "write_brief",
            path="briefs/a.md",
            content="## 结论\n本月新增注册 1200，环比 +18%。\n\n## 待确认\n渠道归因口径。",
        ),
        principal=EDITOR,
        stage="draft",
    )
    report.check(
        "5 执行闭环：成功路径产出 tool 消息（回注可用）",
        ok.ok and ok.message is not None and ok.message.role.value == "tool",
        # 早先这里只打 code（`invalid_arguments`），看不出是哪个参数不合法；
        # 打 detail 才能一眼看到「content 不足 20 字符」这类。
        f"code={ok.code} detail={ok.detail}",
    )


# --------------------------------------------------------------------------- #
# 7. A14 与 Skill 完成标准
# --------------------------------------------------------------------------- #


def check_spec_and_validator(report: Report) -> None:
    from agent_framework.skills import SkillValidator, get_builtin

    problems = validate_arguments(
        {"type": "object", "properties": {"q": {"type": "string"}},
         "required": ["q"], "additionalProperties": False},
        {"q": "x", "site": "y"},
    )
    report.check("5 A14：schema 外参数被拒", bool(problems), str(problems))

    publish = get_builtin("publish_post")
    unmet, _ = SkillValidator().verify(publish, "已经发出去了，放心。")
    report.check(
        "7 Skill 可验证：无证据的「已完成」被拦",
        bool(unmet),
        f"未满足 {len(unmet)} 条：{unmet}",
    )

    specs = builtin_skills()
    lint_clean = all(not spec.lint() for spec in specs)
    report.check(
        "7 Skill 六层：内置库 mission/principles/stages 齐全",
        lint_clean,
        f"{len(specs)} 个 Skill",
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


async def run(report: Report, tmp: Path, *, record: bool) -> None:
    await check_mcp_boundary(report)
    await check_replay(report, tmp, record=record)
    await check_sandbox(report, tmp / "sandbox_root")
    check_visibility(report)
    await check_action_runtime(report, tmp)
    await check_executor_order(report, tmp / "order_root")
    check_spec_and_validator(report)


def _print(report: Report) -> None:
    print("\nW8 边界与受控执行自检")
    print("=" * 78)
    for item in report.checks:
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"[{mark}] {item['name']}")
        if item["detail"]:
            print(f"       {item['detail']}")
    print("-" * 78)
    print(f"{report.passed}/{len(report.checks)} 通过")
    print("\n可见面矩阵（stage × principal → 可见数/全集）：")
    matrix = report.sections.get("visibility_matrix", {})
    principals = ["anonymous", "viewer", "editor", "publisher", "ops"]
    print(f"{'stage':<13}" + "".join(f"{name:<12}" for name in principals))
    for stage, row in matrix.items():
        cells = "".join(
            f"{len(row[name]['visible'])}/{6} ({row[name]['ratio']:.0%})".ljust(12)
            for name in principals
        )
        print(f"{stage:<13}{cells}")


def main() -> int:
    parser = argparse.ArgumentParser(description="W8 边界与受控执行自检")
    parser.add_argument("--record", action="store_true", help="重新录制工具夹具")
    parser.add_argument("--json", type=Path, default=None, help="把报告写成 JSON")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="w8-check-"))
    for name in ("sandbox_root", "order_root", "action_root"):
        (tmp / name).mkdir(parents=True, exist_ok=True)

    report = Report()
    try:
        asyncio.run(run(report, tmp, record=args.record))
    finally:
        _print(report)
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", "utf-8"
            )
            print(f"报告已写入 {args.json}")
        shutil.rmtree(tmp, ignore_errors=True)

    return 0 if report.passed == len(report.checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
