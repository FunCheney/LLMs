"""配置层测试：结构错误必须在启动时炸，凭据缺失必须在首次使用时炸。

两类错误刻意分开断言：配置写错是**我们的 bug**（500 / CONFIG_ERROR），
密钥缺失是**部署问题**（同样是 ConfigError，但不该挡住启动）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_framework.config import (
    DEFAULT_MODELS_YAML,
    Catalog,
    ConfigError,
    Pricing,
    Settings,
    load_catalog,
)

GOOD_YAML = """
providers:
  openai:
    type: openai_compat
    base_url: "https://api.openai.test/v1"
    api_key_env: OPENAI_API_KEY

models:
  gpt-mini:
    provider: openai
    model: gpt-4o-mini
    tier: cheap
    pricing: { input: 0.15, output: 0.6, currency: USD, per: 1000000 }

routes:
  default: { primary: gpt-mini }
  cheap: { primary: gpt-mini }

tasks:
  chat: cheap
"""


def write_yaml(tmp_path: Path, text: str, name: str = "models.yaml") -> Path:
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")

    return target


def test_good_yaml_loads(tmp_path: Path) -> None:
    catalog = load_catalog(write_yaml(tmp_path, GOOD_YAML))

    assert catalog.resolve("chat") == ["gpt-mini"]
    assert catalog.get_model("gpt-mini").model == "gpt-4o-mini"


def test_repository_models_yaml_is_valid() -> None:
    """仓库里的 models.yaml 必须随时可加载：它是启动路径上的第一道门。"""

    catalog: Catalog = load_catalog(DEFAULT_MODELS_YAML)

    assert "default" in catalog.routes
    assert catalog.resolve("chat")[0] == "qwen3.7-plus"
    # alias ≠ wire 名（ADR-009）：Qwen 的滚动别名必须钉成日期快照。
    assert catalog.get_model("qwen3.7-plus").model == "qwen3.7-plus-2026-05-26"


def test_missing_pricing_reports_field_path(tmp_path: Path) -> None:
    broken = GOOD_YAML.replace(
        "    pricing: { input: 0.15, output: 0.6, currency: USD, per: 1000000 }\n", ""
    )

    with pytest.raises(ConfigError) as excinfo:
        load_catalog(write_yaml(tmp_path, broken))

    message = str(excinfo.value)
    # 报错里必须能看见字段路径，否则排查全靠猜。
    assert "pricing" in message
    assert "gpt-mini" in message


def test_pricing_per_must_be_million(tmp_path: Path) -> None:
    """按千计价会让成本差 1000 倍且**不报错**，这是最贵的手误。"""

    broken = GOOD_YAML.replace("per: 1000000", "per: 1000")

    with pytest.raises(ConfigError) as excinfo:
        load_catalog(write_yaml(tmp_path, broken))

    assert "1000000" in str(excinfo.value)


def test_pricing_rejects_bad_unit_directly() -> None:
    with pytest.raises(ValueError, match="1000000"):
        Pricing(input=1.0, output=2.0, per=1000)


def test_route_to_unknown_alias_reports_field_path(tmp_path: Path) -> None:
    broken = GOOD_YAML.replace(
        "  cheap: { primary: gpt-mini }", "  cheap: { primary: ghost }"
    )

    with pytest.raises(ConfigError) as excinfo:
        load_catalog(write_yaml(tmp_path, broken))

    message = str(excinfo.value)
    assert "routes.cheap" in message
    assert "ghost" in message


def test_unknown_provider_reports_field_path(tmp_path: Path) -> None:
    broken = GOOD_YAML.replace("    provider: openai", "    provider: ghost")

    with pytest.raises(ConfigError) as excinfo:
        load_catalog(write_yaml(tmp_path, broken))

    assert "models.gpt-mini.provider" in str(excinfo.value)


def test_missing_default_route_is_rejected(tmp_path: Path) -> None:
    broken = GOOD_YAML.replace("  default: { primary: gpt-mini }\n", "")

    with pytest.raises(ConfigError) as excinfo:
        load_catalog(write_yaml(tmp_path, broken))

    assert "default" in str(excinfo.value)


def test_tier_without_route_is_rejected(tmp_path: Path) -> None:
    broken = GOOD_YAML.replace("    tier: cheap", "    tier: strong")

    with pytest.raises(ConfigError) as excinfo:
        load_catalog(write_yaml(tmp_path, broken))

    assert "models.gpt-mini.tier" in str(excinfo.value)


def test_all_problems_are_collected_at_once(tmp_path: Path) -> None:
    """一次报全部问题：改配置的人不想"修一个跑一次"。"""

    broken = GOOD_YAML.replace("    provider: openai", "    provider: ghost").replace(
        "  cheap: { primary: gpt-mini }", "  cheap: { primary: phantom }"
    )

    with pytest.raises(ConfigError) as excinfo:
        load_catalog(write_yaml(tmp_path, broken))

    message = str(excinfo.value)
    assert "ghost" in message
    assert "phantom" in message


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="不存在"):
        load_catalog(tmp_path / "nope.yaml")


def test_broken_yaml_syntax_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="YAML"):
        load_catalog(write_yaml(tmp_path, "models: [unclosed\n"))


def test_missing_key_is_lazy_and_explains_env_name() -> None:
    """密钥缺失不挡启动，但首次取用时必须说清缺哪个变量。"""

    settings = Settings(_env_file=None, openai_api_key=None)

    with pytest.raises(ConfigError) as excinfo:
        settings.api_key("OPENAI_API_KEY")

    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_unknown_key_env_is_rejected() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ConfigError, match="未知的 api_key_env"):
        settings.api_key("SOME_UNKNOWN_KEY")


def test_api_key_returns_plaintext_from_secret() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-test-1234567890")

    assert settings.api_key("OPENAI_API_KEY") == "sk-test-1234567890"


def test_unknown_task_falls_back_to_default_route() -> None:
    catalog = load_catalog(DEFAULT_MODELS_YAML)

    assert catalog.resolve("no-such-task") == catalog.resolve("default")
