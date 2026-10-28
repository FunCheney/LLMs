from pathlib import Path
from typing import Literal, Any, List, Set

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from agent_framework.llm import Capability


class Pricing(BaseModel):
    input: float
    output: float
    currency: Literal["USD", "EUR"]
    per: int = 1_000_000
    cache_read: float | None = None

class ProviderConfig(BaseModel):
    type: Literal["openai_compat", "anthropic", "mock"]
    base_url: str
    api_key_env: str | None = None
    timeout_s : int = 60
    flags: dict[str, Any] = Field(default_factory=dict)  # 仅 openai_compat 消费


class ModelConfig(BaseModel):
    provider: str  # 指向 providers 的 key
    model: str  # **wire 上的真实模型名**
    tier: str  # 必须能在 routes 里找到
    capabilities: list[Capability]
    pricing: Pricing
    max_tokens: int | None = None  # Anthropic 必填，registry 传成 default_max_tokens
    reasoning_replay: bool | None = None  # Qwen 3.8 必须回传 / 3.7 不能回传

class RouterConfig(BaseModel):
    primary: str
    fallback: List[str] = Field(default_factory=list)


class Catalog(BaseModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    routers: dict[str, RouterConfig]
    tasks: dict[str, str]

    def resolve(self, task: str) -> list[str]:
        """task → tier → [primary, *fallback]（去重、保序）。"""
        if task not in self.tasks:
            raise KeyError(f"unknown task {task!r}, available: {sorted(self.tasks)}")
        tier = self.tasks[task]
        if tier not in self.routers:
            raise KeyError(f"unknown tier {tier!r}")
        router = self.routers[tier]
        seen : Set[str] = set()
        chain: List[str] = []
        for name in (router.primary, *router.fallback):
            if name in seen:
                continue
            # 4) 模型必须真实存在（早失败，避免运行时 404）
            if name not in self.models:
                raise KeyError(
                    f"router {tier!r} references unknown model {name!r}"
                )
            seen.add(name)
            chain.append(name)

        return chain

    def resolve_models(self, task: str) -> list[ModelConfig]:
        return [self.models[n] for n in self.resolve(task)]



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local", env_file_encoding="utf-8", extra="ignore"
    )

    models_yaml: Path = Path("models.yaml")
    first_token_timeout_s: float = 10.0
    request_timeout_s: float = 60.0
    max_attempts: int = 3
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0
    log_prompts: bool = False           # 默认关闭，默认不记 prompt 原文

    openai_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    def api_key(self, env_name: str | None) -> str:
        """按 models.yaml 的 api_key_env 取密钥；缺了给可执行报错。"""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        with self.models_yaml.open("r") as f:
            for doc in yaml.safe_load_all(f):
                providers = doc["providers"]
                for key, value in providers.items():
                    if key == "openai":
                        openai_api_key = value.get("api_key_env")
                    elif key == "deepseek":
                        deepseek_api_key = value.get("api_key_env")
                    elif key == "qwen":
                        dashscope_api_key = value.get("api_key_env")
                    elif key == "anthropic":
                        anthropic_api_key = value.get("api_key_env")
                    else:
                        raise KeyError(key)

                api_key = providers.get(env_name).get("api_key_env")
                # print(api_key)
        if api_key is None:
            raise ValueError(f"No api_key provided for {env_name}")

        return os.environ.get(api_key)



def load_catalog(path: str | Path) -> Catalog:
    models_yaml: Path = Path(path)
    with models_yaml.open("r") as f:
        raw =  yaml.safe_load(f)
    catalog = Catalog.model_validate(raw)
    return catalog







