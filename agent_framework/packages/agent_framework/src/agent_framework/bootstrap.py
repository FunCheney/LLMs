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
