
import os

class Config:
    """配置类"""

    # Kimi API 配置
    MOONSHOT_API_KEY: str = os.getenv("MOONSHOT_API_KEY", "")
    # 向后兼容：如果没有 MOONSHOT_API_KEY，尝试使用 KIMI_API_KEY
    if not MOONSHOT_API_KEY:
        MOONSHOT_API_KEY = os.getenv("KIMI_API_KEY", "")

    KIMI_BASE_URL: str = "https://api.moonshot.cn/v1"

    # 模型配置
    DEFAULT_MODEL: str = "kimi-k3"  # 使用最新的 Kimi K3 模型

    # 搜索配置
    MAX_SEARCH_ITERATIONS: int = 5  # 最大搜索迭代次数（与 agent 默认值保持一致）
    SEARCH_TIMEOUT: float = float(os.getenv("SEARCH_TIMEOUT", "30"))