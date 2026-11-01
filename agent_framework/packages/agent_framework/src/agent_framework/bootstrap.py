"""组装 gateway 的唯一入口。

为什么单独一个文件：`config.py` 不能 import `registry.py`（会形成
config → registry → providers → config 的环），而 `server/app.py` 里塞组装逻辑
会让测试无法只测"配置 + 路由 + 客户端"这三层。
"""

from __future__ import annotations

from .config import Catalog, Settings, load_catalog
from .llm.client import LLMGateway
from .llm.registry import Registry
from .llm.router import TaskRouter
from .resilience.concurrency import ConcurrencyLimiter


def build_gateway(
    settings: Settings | None = None,
    *,
    catalog: Catalog | None = None,
) -> LLMGateway:
    """按配置造一条完整链路。

    启动时只做**结构校验**（catalog 加载 + 引用完整性），
    不实例化任何 adapter——密钥缺失不能挡启动（config.py 纪律 2）。
    """

    resolved_settings = settings or Settings()
    resolved_catalog = catalog or load_catalog(resolved_settings.models_yaml)
    registry = Registry(resolved_catalog, resolved_settings)

    return LLMGateway(
        registry=registry,
        router=TaskRouter(resolved_catalog),
        catalog=resolved_catalog,
        settings=resolved_settings,
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
