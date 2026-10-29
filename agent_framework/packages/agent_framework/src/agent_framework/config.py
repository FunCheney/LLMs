"""配置加载与启动校验（architecture §4.1）。

三条纪律：

1. **结构性错误在启动时就炸**：路由引用了不存在的模型、缺 pricing、`per` 写错，
   全部在 `load_catalog()` 里拦下。留到运行时的代价是"第一次真实请求才 500"。
2. **凭据缺失在首次使用时炸**：密钥缺失不能挡启动，否则离线测试永远起不来，
   最后只能往 CI 里塞假 key——那才是真正的泄露路径。
3. **alias 与 wire 模型名分离**（ADR-009）：`models` 的 key 是 alias（路由、日志、
   评测集都用它），`model` 字段才是真正发给厂商的名字。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .llm.base import Capability

DEFAULT_MODELS_YAML = Path(__file__).with_name("models.yaml")

# 单价统一按"每百万 token"计。允许按千计会让成本差 1000 倍且**不报错**，
# 这是最贵的一类手误，所以直接拒绝（V5）。
REQUIRED_PRICING_UNIT = 1_000_000

_KEY_ENV_TO_FIELD: dict[str, str] = {
    "OPENAI_API_KEY": "openai_api_key",
    "DEEPSEEK_API_KEY": "deepseek_api_key",
    "DASHSCOPE_API_KEY": "dashscope_api_key",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
}

# 公开别名：录制器要用它把"某个 provider 的 key 换成故意无效值"，从而安全地
# 录到真实的鉴权失败体（错误路径不能靠编，否则 401→502 的映射等于没验过）。
KEY_ENV_TO_FIELD = _KEY_ENV_TO_FIELD


class ConfigError(ValueError):
    """配置错误。

    与 LLMError 区分开：这不是上游的问题，是我们自己的配置写错了，
    对外应该是 500 + CONFIG_ERROR，不该混进 502 的上游故障里。
    """


class Pricing(BaseModel):
    """模型单价。"""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    currency: Literal["USD", "CNY"] = "USD"
    per: int = REQUIRED_PRICING_UNIT

    # 预留：DeepSeek 的缓存命中价是未命中的 1/25~1/50，W11 做成本优化时启用。
    cache_read: float | None = None

    @model_validator(mode="after")
    def validate_unit(self) -> Pricing:
        if self.per != REQUIRED_PRICING_UNIT:
            raise ValueError(
                f"pricing.per 必须为 {REQUIRED_PRICING_UNIT}（每百万 token），"
                f"当前为 {self.per}；按千计价会让成本差 1000 倍"
            )

        return self


class ProviderConfig(BaseModel):
    """一个厂商的连接配置。密钥只给环境变量名，绝不落 yaml。"""

    type: Literal["openai_compat", "anthropic", "mock"]
    base_url: str = ""
    api_key_env: str | None = None
    timeout_s: float = 60.0

    # 只对 openai_compat 生效：同一协议在各家的差异开关（覆盖 PROVIDER_FLAGS 的预设）。
    flags: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """模型配置。字典的 key 是 alias，`model` 是 wire 名。"""

    provider: str
    model: str = ""
    tier: str = "cheap"
    capabilities: list[Capability] = Field(default_factory=list)
    pricing: Pricing

    # Anthropic 的 max_tokens 必填：这里配好，registry 传成 default_max_tokens。
    max_tokens: int | None = None

    # Qwen 同一 provider 下 3.7 不能回传 reasoning、3.8 必须回传，
    # 只能按模型覆盖（PER_MODEL 策略），不能放在 provider 级。
    reasoning_replay: bool | None = None


class RouteConfig(BaseModel):
    primary: str
    fallback: list[str] = Field(default_factory=list)


class Catalog(BaseModel):
    """models.yaml 的内存形态。"""

    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    routes: dict[str, RouteConfig]
    tasks: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> Catalog:
        """V1–V4：引用完整性。

        一次把**所有**问题收集齐再抛，而不是遇到第一个就返回——
        改配置的人希望一次看到全部错误，而不是修一个跑一次。
        """

        problems: list[str] = []

        if "default" not in self.routes:
            problems.append("routes: 缺少 'default'，未知 task 的回落没有着落")

        for alias, model in self.models.items():
            if model.provider not in self.providers:
                problems.append(
                    f"models.{alias}.provider: 未知 provider '{model.provider}'"
                )

            # tier 必须对应一条 route：否则"按 tier 取不到候选"要等到运行时才暴露。
            if model.tier not in self.routes:
                problems.append(f"models.{alias}.tier: '{model.tier}' 不在 routes 里")

        for name, route in self.routes.items():
            for alias in (route.primary, *route.fallback):
                if alias not in self.models:
                    problems.append(f"routes.{name}: 引用了不存在的模型 '{alias}'")

        for task, tier in self.tasks.items():
            if tier not in self.routes:
                problems.append(f"tasks.{task}: '{tier}' 不是有效的 route")

        if problems:
            raise ValueError("models.yaml 校验失败：\n  - " + "\n  - ".join(problems))

        return self

    # -- 查询 ------------------------------------------------------------ #

    def get_model(self, alias: str) -> ModelConfig:
        model = self.models.get(alias)

        if model is None:
            raise ConfigError(f"未知模型 alias '{alias}'；可用：{sorted(self.models)}")

        return model

    def get_provider(self, alias: str) -> tuple[str, ProviderConfig]:
        model = self.get_model(alias)

        return model.provider, self.providers[model.provider]

    def route_for(self, task: str) -> RouteConfig:
        """task → tier → route。未知 task 回落到 default（由 router 记 warning）。"""

        tier = self.tasks.get(task, "default")
        route = self.routes.get(tier)

        if route is None:
            raise ConfigError(f"task '{task}' 的 tier '{tier}' 没有对应 route")

        return route

    def resolve(self, task: str) -> list[str]:
        """task → 候选 alias 有序列表（去重、保序）。"""

        route = self.route_for(task)

        return list(dict.fromkeys([route.primary, *route.fallback]))

    def resolve_models(self, task: str) -> list[ModelConfig]:
        return [self.get_model(alias) for alias in self.resolve(task)]


class Settings(BaseSettings):
    """环境侧配置。密钥只在这里出现，且只以 SecretStr 形态。"""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    models_yaml: Path = DEFAULT_MODELS_YAML

    # 首 token 超时是**流式重试的判定边界**（architecture §3.7），不是整体人工超时。
    first_token_timeout_s: float = 10.0
    request_timeout_s: float = 60.0

    max_attempts: int = Field(default=3, ge=1, le=5)
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0

    # 默认不记 prompt 原文：合规与隐私优先，需要排查时再显式打开。
    log_prompts: bool = False

    openai_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    def api_key(self, env_name: str | None) -> str:
        """按 models.yaml 的 `api_key_env` 取密钥。

        pydantic-settings 已经读过 `.env.local`，这里只做"变量名 → 字段"的映射，
        不再自己解析 yaml 或调 load_dotenv——否则会出现两份配置真相。
        """

        if not env_name:
            raise ConfigError("provider 未声明 api_key_env，无法取密钥")

        field_name = _KEY_ENV_TO_FIELD.get(env_name)

        if field_name is None:
            # 用映射表而不是 getattr 拼字符串：拼错会变成 AttributeError，
            # 而这里要的报错是"这个变量名我们不认识"。
            raise ConfigError(
                f"未知的 api_key_env '{env_name}'；支持：{sorted(_KEY_ENV_TO_FIELD)}"
            )

        secret: SecretStr | None = getattr(self, field_name)

        if secret is None or not secret.get_secret_value():
            raise ConfigError(
                f"缺少凭据 {env_name}：请在 .env.local 中设置 {env_name}=..."
            )

        return secret.get_secret_value()


def load_catalog(path: str | Path | None = None) -> Catalog:
    """读取并校验 models.yaml。任何结构错误都在这里炸，不留到运行时。"""

    target = Path(path) if path is not None else DEFAULT_MODELS_YAML

    if not target.is_file():
        raise ConfigError(f"models.yaml 不存在：{target}")

    try:
        raw = yaml.safe_load(target.read_text("utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target}: YAML 解析失败：{exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{target}: 顶层必须是映射")

    try:
        return Catalog.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{target}: 配置校验失败\n{exc}") from exc
