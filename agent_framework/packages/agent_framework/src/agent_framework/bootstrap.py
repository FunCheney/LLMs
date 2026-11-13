"""组装 gateway 的唯一入口。

为什么单独一个文件：`config.py` 不能 import `registry.py`（会形成
config → registry → providers → config 的环），而 `server/app.py` 里塞组装逻辑
会让测试无法只测"配置 + 路由 + 客户端"这三层。
"""

from __future__ import annotations

from .config import Catalog, ConfigError, Settings, load_catalog
from .llm.base import Capability
from .llm.client import LLMGateway
from .llm.difficulty_router import DifficultyRouter
from .llm.registry import Registry
from .llm.router import Router, TaskRouter
from .observability import router_log
from .resilience.circuit_breaker import CircuitBreaker
from .resilience.concurrency import ConcurrencyLimiter
from .resilience.semantic_cache import SemanticCache
from .runtime.assets import capture
from .runtime.flags import FlagSet


def build_gateway(
    settings: Settings | None = None,
    *,
    catalog: Catalog | None = None,
) -> LLMGateway:
    """按配置造一条完整链路。

    启动时只做**结构校验**（catalog 加载 + 引用完整性），
    不实例化任何 adapter——密钥缺失不能挡启动（config.py 纪律 2）。

    W11 D5 起，三个"可回退的行为"都从这里分叉：路由（难度分流）、
    缓存（语义缓存）、熔断。**这是唯一开关生效的地方**——
    如果某个 flag 在别的模块里也被读一次，那它就有了两个判定点，
    而两个判定点迟早会出现一个 on 一个 off 的中间态。
    """

    resolved_settings = settings or Settings()
    resolved_catalog = catalog or load_catalog(resolved_settings.models_yaml)
    flags = FlagSet.from_settings(resolved_settings)
    registry = Registry(resolved_catalog, resolved_settings)

    return LLMGateway(
        registry=registry,
        router=build_router(resolved_catalog, resolved_settings, flags=flags),
        catalog=resolved_catalog,
        settings=resolved_settings,
        breaker=build_breaker(resolved_settings),
        cache=build_cache(resolved_catalog, resolved_settings, flags=flags),
    )


def build_router(
    catalog: Catalog,
    settings: Settings,
    *,
    flags: FlagSet,
) -> Router:
    """`difficulty_router` 关着（默认）就是 W10 的 `TaskRouter`，一字不差。"""

    if not flags.enabled("difficulty_router"):
        return TaskRouter(catalog)

    cheap = settings.difficulty_cheap_alias
    strong = settings.difficulty_strong_alias
    # 未显式配置时从 models.yaml 取——模型名的唯一定义处不能变成两处。
    cheap = cheap or _route_primary(catalog, "cheap")
    strong = strong or _route_primary(catalog, "strong")

    if not cheap or not strong:
        # 配不出两端就**退回旧行为并说出来**，不要造一个"看起来开了"的分流器：
        # 一个只会在同一档里打转的路由器，比不分流更难排查。
        router_log().warning(
            "difficulty_router 已开启但 cheap/strong 无法确定（cheap=%r strong=%r），"
            "退回 TaskRouter", cheap, strong,
        )
        return TaskRouter(catalog)

    return DifficultyRouter(
        catalog,
        cheap=cheap,
        strong=strong,
        threshold=settings.difficulty_threshold,
        # 能力检查**不可省**：`_ensure_capabilities` 是在重试循环之外抛的，
        # 所以把带 tools 的请求排到不支持 tools 的便宜档上，整次调用直接失败、
        # 不会 fallback 到强档——比分流前更糟。这不是"难度"问题，是"这条路不存在"。
        capability_check=lambda alias, call: _supports(catalog, alias, call),
    )


def _supports(catalog: Catalog, alias: str, call: object) -> bool:
    """这个 alias 能不能满足本次请求的**硬需求**（目前只有 tools）。"""

    if not getattr(call, "tools", None):
        return True

    try:
        model = catalog.get_model(alias)
    except ConfigError:
        return False

    return Capability.TOOLS in model.capabilities


def _route_primary(catalog: Catalog, tier: str) -> str:
    route = catalog.routes.get(tier)
    return route.primary if route is not None else ""


def build_cache(
    catalog: Catalog,
    settings: Settings,
    *,
    flags: FlagSet,
) -> SemanticCache | None:
    """`semantic_cache` 关着（默认）返回 `None`，gateway 直通。

    `asset_digest` 用**启动那一刻**的六类资产快照：prompt 或模型换了，
    旧答案就不再是"这套系统会给出的答案"，重启即作废。
    这也是为什么入口只在 bootstrap——缓存版本必须与进程的资产版本同频。
    """

    if not flags.enabled("semantic_cache"):
        return None

    snapshot = capture(settings, catalog=catalog)

    return SemanticCache(
        threshold=settings.cache_threshold,
        ttl_s=settings.cache_ttl_s,
        max_entries=settings.cache_max_entries,
        asset_digest=snapshot.digest,
    )


def build_breaker(settings: Settings) -> CircuitBreaker | None:
    """`BREAKER_ENABLED=false` 时返回 None（gateway 直通，行为回到 W10）。

    默认**开着**：一个默认关闭的保护等于没有保护，而"它其实一直没生效"
    这件事只有在出事那天才会被发现。
    """

    if not settings.breaker_enabled:
        return None

    return CircuitBreaker(
        failure_threshold=settings.breaker_threshold,
        reset_after_s=settings.breaker_reset_s,
    )


def build_limiter(
    catalog: Catalog,
    settings: Settings | None = None,
) -> ConcurrencyLimiter:
    """按 `models.yaml` 的 limits 造并发闸门。

    在 gateway **之外**而不是之内：gateway 的职责是"把一次调用打出去"，
    入口过载属于服务层（谁收请求谁拒绝）。混进 gateway 会让"重试一次"
    与"重新排队一次"看起来像同一件事，而过载时最怕的就是这个。
    """

    limits = {
        alias: catalog.limits.limit_for(alias, model.provider)
        for alias, model in catalog.models.items()
    }

    return ConcurrencyLimiter(
        limits,
        max_inflight=catalog.limits.max_inflight,
        # 未登记的 key（新加的 alias、评测临时钉的模型）走全局默认档，
        # 而不是代码里另一个写死的数字。
        default_limit=catalog.limits.max_inflight,
        default_timeout_s=settings.overload_wait_s if settings is not None else 0.25,
    )
