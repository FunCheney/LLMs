"""ACL-aware 检索（W6 D3）：把"他无权看这段"挡在进 context 之前。

这是**最小实现**，不是权限模型。三条边界刻意的选择，都是为了让它在 W10
接真鉴权时不用重写：

1. **默认拒绝未标注的块。**（`ACLPolicy.allow_unlabeled=False`）这条不能反过来：
   默认放行意味着"忘了打标签的文档就是公开的"，而漏打标签是新文档最常见的状态。
   W6 当前语料全部没有 acl 字段，所以**运行时默认不启用 ACL**（不传 principal），
   一旦启用就按拒绝走——"没开"和"开了但谁都能看"是两件事，不能混。
2. **过滤发生在精排之后、进 context 之前。** 放在精排之前有两个后果：
   白花精排算力在无权内容上；以及过滤后顺序与精排名次对不上，
   引用编号会指向"被过滤过但没重排"的位置。
3. **判定留在应用层，不下推成 SQL。** 权限是"组集合求交"，下推成 jsonb 包含会把
   策略拆到两处（SQL 一处、Python 一处），审计时看不全。代价是向量库会多取一些
   ——pipeline 的 over-fetch 就是为这个准备的。

日志纪律：被滤掉的块**只记数量不记内容**（W3 ADR）。把无权文本写进日志，
等于用日志绕过权限。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..knowledge.base import Chunk

# 通配组：任何 principal 都满足。用于"确实公开"的文档，
# 与"忘了打标签"区分开——后者是 no_acl，走 allow_unlabeled 策略。
PUBLIC_GROUP = "*"

REASON_ALLOWED_PUBLIC = "allowed_public"
REASON_ALLOWED_GROUP = "allowed_group"
REASON_ALLOWED_SUBJECT = "allowed_subject"
REASON_DENIED_NO_ACL = "denied_no_acl"
REASON_DENIED_NO_PRINCIPAL = "denied_no_principal"
REASON_DENIED_NOT_MEMBER = "denied_not_member"


@dataclass(frozen=True)
class Principal:
    """调用方主体。W10 之前只有"是谁 + 属于哪些组"，不做角色与继承。"""

    subject_id: str
    groups: frozenset[str] = frozenset()

    @classmethod
    def anonymous(cls) -> Principal:
        """匿名调用方。

        它是**有身份的主体**（"匿名"本身是一种身份），不是"没有 principal"。
        后者在启用 ACL 时直接拒绝整次检索——"还没接鉴权"不该悄悄变成"人人可见"。
        """

        return cls(subject_id="anonymous")


@dataclass(frozen=True)
class ACLDecision:
    """一次判定的结果。`reason` 让"为什么看不到"可归因，不带任何正文。"""

    allowed: bool
    reason: str


@dataclass(frozen=True)
class ACLPolicy:
    """acl 字段的解析策略。

    `allow_unlabeled=False` 是默认值，也是安全默认：没有 acl 字段的块视为
    "未标注"，按拒绝处理。要放行必须显式配置，并在 ADR 里写明理由。
    """

    allow_unlabeled: bool = False

    def labels(self, chunk: Chunk) -> tuple[frozenset[str], frozenset[str]]:
        """取出 `(groups, subjects)`。

        支持两种写法：`{"groups": [...], "subjects": [...]}` 与裸列表（视为 groups）。
        两种都收是因为入库侧（切分/抓取脚本）不可能统一——收窄成一种的结果
        是"另一种写法的文档全部变成未标注"，然后被默认拒绝挡住。
        """

        raw = chunk.metadata.get("acl")
        if raw is None:
            return frozenset(), frozenset()

        if isinstance(raw, Mapping):
            groups = raw.get("groups") or ()
            subjects = raw.get("subjects") or ()
            return frozenset(_as_text_set(groups)), frozenset(_as_text_set(subjects))

        if isinstance(raw, str):
            return frozenset({raw}), frozenset()

        if isinstance(raw, Sequence):
            return frozenset(_as_text_set(raw)), frozenset()

        # 其他类型（int/dict 的畸形值）按"未标注"处理，而不是抛错：
        # 一条脏 metadata 不该让整次检索 500，但它也绝不能因此被放行。
        return frozenset(), frozenset()

    def decide(self, chunk: Chunk, principal: Principal) -> ACLDecision:
        groups, subjects = self.labels(chunk)

        if not groups and not subjects:
            if self.allow_unlabeled:
                return ACLDecision(True, REASON_ALLOWED_PUBLIC)
            return ACLDecision(False, REASON_DENIED_NO_ACL)

        if PUBLIC_GROUP in groups:
            return ACLDecision(True, REASON_ALLOWED_PUBLIC)

        if principal.subject_id and principal.subject_id in subjects:
            return ACLDecision(True, REASON_ALLOWED_SUBJECT)

        if principal.groups & groups:
            return ACLDecision(True, REASON_ALLOWED_GROUP)

        return ACLDecision(False, REASON_DENIED_NOT_MEMBER)

    def check(self, chunk: Chunk, principal: Principal) -> bool:
        return self.decide(chunk, principal).allowed


def make_acl_check(
    principal: Principal | None,
    *,
    policy: ACLPolicy | None = None,
) -> Callable[[Chunk], bool] | None:
    """给 `RetrievalPipeline` 用的过滤器。`principal=None` 表示**不启用** ACL。

    返回 None 而不是"永远 True"的函数：pipeline 用 `is None` 判断是否统计
    `filtered_out`。一个恒真的过滤器会让诊断里的 `filtered_out=0` 看起来像
    "过滤跑了但没有越权内容"，而实际是"根本没过滤"。
    """

    if principal is None:
        return None

    resolved = policy or ACLPolicy()

    def _check(chunk: Chunk) -> bool:
        return resolved.check(chunk, principal)

    return _check


def _as_text_set(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence):
        return {str(item) for item in value if str(item).strip()}
    return set()


def filter_visible(
    chunks: Sequence[Chunk],
    principal: Principal | None,
    *,
    policy: ACLPolicy | None = None,
) -> tuple[list[Chunk], int]:
    """批量过滤，返回 `(可见的块, 被挡掉的条数)`。

    给"拿一批 chunk 先做权限筛选"的脚本用（导出、审计、增量同步预览）。
    检索链路本身走 pipeline，不用这个函数。
    """

    check = make_acl_check(principal, policy=policy)
    if check is None:
        return list(chunks), 0

    visible = [chunk for chunk in chunks if check(chunk)]
    return visible, len(chunks) - len(visible)
