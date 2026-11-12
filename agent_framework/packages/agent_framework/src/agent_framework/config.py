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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)
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


class LimitsConfig(BaseModel):
    """并发限额（W2 D2）。**数字只从 yaml 来，代码里不许出现 `Semaphore(8)`**。

    写法是"三层覆盖"：alias 覆盖 > provider 覆盖 > 全局默认。
    不用"每家写死一个值"的原因：同一家下 cheap/strong 两档的池子大小本来就不同
    （claude-sonnet 的并发额度远小于 gpt-4o-mini），把它压成 provider 级必然写错一半。
    """

    model_config = ConfigDict(populate_by_name=True)

    # yaml 里写 `global:`，Python 侧不能叫 global，所以用 alias。
    max_inflight: int = Field(default=16, alias="global", ge=1)
    per_provider: dict[str, int] = Field(default_factory=dict)
    per_alias: dict[str, int] = Field(default_factory=dict)

    def limit_for(self, alias: str, provider: str) -> int:
        """alias 覆盖 → provider 覆盖 → 全局默认。"""

        if alias in self.per_alias:
            return self.per_alias[alias]

        if provider in self.per_provider:
            return self.per_provider[provider]

        return self.max_inflight



class Catalog(BaseModel):
    """models.yaml 的内存形态。"""

    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    routes: dict[str, RouteConfig]
    tasks: dict[str, str] = Field(default_factory=dict)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

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

        # limits 里的 key 打错（比如把 qwen 写成 qwen3）不会报错、只会静默不生效——
        # 那正好是"压测时以为限了、其实没限"的成因，所以启动时就查。
        for provider in self.limits.per_provider:
            if provider not in self.providers:
                problems.append(f"limits.per_provider: 未知 provider '{provider}'")

        for alias in self.limits.per_alias:
            if alias not in self.models:
                problems.append(f"limits.per_alias: 未知模型 alias '{alias}'")

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

    # -- W2：会话 / 过载 / 队列 ----------------------------------------- #
    # 默认 memory：离线测试与 CI 不该被一个 Redis 挡住。生产用 `SESSION_BACKEND=redis`。
    session_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_s: int = Field(default=86400, ge=60)
    session_max_turns: int = Field(default=50, ge=1)

    # 过载时最多等多久再回 503。等太久会把 p99 拖成秒级，比直接拒绝更糟。
    overload_wait_s: float = Field(default=0.25, ge=0)

    # 长任务队列上限与消费者数：满了给 503 + Retry-After，而不是把内存吃光。
    task_queue_size: int = Field(default=100, ge=1)
    task_workers: int = Field(default=2, ge=1)

    # -- W3：观测后端 ------------------------------------------------ #
    # 缺 key 时强制降级为 NoopExporter（见 exporters/langfuse.py）：
    # 观测组件挂掉不能影响主链路，这是可观测性组件的基本操守。
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)

    # -- W5：检索 ---------------------------------------------------- #
    # 预算**整体一个数**，流水线按剩余时间递减。拆成三个独立超时的话，
    # 总延迟是三者之和，而配置读起来像"总预算"，压测时会归因错。
    retrieval_budget_ms: float = Field(default=800.0, gt=0)
    retrieval_top_k: int = Field(default=8, ge=1)
    # 召回放大倍数：后面还有 ACL 过滤与去重会砍，只取 top_k 容易给不满。
    retrieval_recall_multiplier: float = Field(default=3.0, ge=1.0)

    # 默认 dense，不是 hybrid——这一条是**实测选出来的，不是拍的**。
    # 2026-11-05 在 28 条正例 / 425 chunk 上跑六档（见 docs/w5/rag-ab-report.md）：
    # dense R@1 .786 / R@5 .964 / R@10 1.000 / MRR .863 全场最优；
    # hybrid 是 .571 / .893 / .964 / .706。R@1 差 6 条、MRR 差 15.7pp。
    # 原因不在 RRF 本身，而在本语料的中文 BM25 分词质量太差（sparse 单独跑
    # R@1 只有 .250），RRF 用 1/(k+rank) 把一路噪声等权融进了正确结果，
    # 把原本排第一的 chunk 挤下去了。RRF 的假设是"两路各有独立信息"，
    # 这里不成立——所以关掉，而不是调 k 或加权（没有证据支持某个权重）。
    retrieval_mode: Literal["sparse", "dense", "hybrid"] = "dense"
    # 后端选择只影响装配，不影响任何业务代码（都走 Retriever 协议）。
    vector_store: Literal["local", "milvus", "pgvector"] = "local"

    # 精排**保留为可配置能力，默认关**。不是"没做完"，是实测结论：
    # 2026-11-05 两套口径各测一遍，框架口径（recall×3 → 24 候选，只重排前 20）
    # 与 rag_mvp 口径（100 候选全局重排）逐位一致，都是负收益：
    #   dense          R@1 .7857 / R@10 1.0000 / MRR .8628 / nDCG .7416
    #   dense + rerank R@1 .7143 / R@10 .9643 / MRR .8146 / nDCG .7233
    # 机制：cross-encoder 偏好"主题邻近的段落"，而 golden 要的是"恰好回答
    # 那一个问题的那一段"，于是把排第一的 chunk 换成了同页邻居
    # （golden-022 是最干净的证据，见 docs/w5/rag-ab-report.md）。
    #
    # 要开就**同时**设两个值，缺一个都会静默降级（启动时 rerank_preflight()
    # 会把这些不自洽报出来，见 growth_agent/rag.py）：
    #   RETRIEVAL_RERANK=true
    #   RETRIEVAL_BUDGET_MS=8500        # ≥ rerank_timeout + 500ms 召回余量
    retrieval_rerank: bool = False
    retrieval_rerank_candidates: int = Field(default=20, ge=1)
    # 8000 而不是 3000：本机 CPU 上 20 条 cross-encoder 是 p50 4.7s / p95 5.4s
    # （实测，同上报告），3000 会让**每一次**精排都超时降级。那种"配了但没
    # 生效"比不做更坏——它让人得出"精排无效"的结论，而真相是参数没给够。
    retrieval_rerank_timeout_ms: float = Field(default=8000.0, gt=0)

    # 本地模型跑在哪：None = 交给 sentence-transformers 自动选（macOS 上会挑 mps）。
    # **开精排的 macOS 机器必须设 cpu**：cross-encoder 跑在 pipeline 的工作线程里，
    # 而 torch 2.14 的 MPS 后端不支持"一个进程里两个线程各持一个 encoder"，
    # 表现为 `AGXG13XFamilyCommandBuffer ... A command encoder is already encoding`
    # 断言失败（进程直接死，不是异常，抓不到也降级不了）。
    # 容器/Linux 是 CPU，不设也没事；这条只在"Mac 上开精排"时是必须的。
    retrieval_device: Literal["cpu", "mps", "cuda"] | None = None

    # 改写默认关：它是一次 LLM 调用，且收益要用 A/B 证明后才该保留。
    retrieval_rewrite: bool = False
    retrieval_rewrite_timeout_ms: float = Field(default=2500.0, gt=0)

    # pgvector 连接串。和 redis_url 一样，只走 env，不进代码。
    pgvector_dsn: SecretStr | None = None

    # -- W6：溯源 / 上下文组装 / 治理 / ACL --------------------------- #
    # 单次回答的 context 预算（token，粗估）。3000 ≈ 4800 汉字 ≈ 6–8 个块。
    # 再大不会提升答案质量，反而触发 lost-in-the-middle。
    retrieval_token_budget: int = Field(default=3000, gt=0)
    # 留给"引用规则 + 问题 + 答案"的余量。不留的话参考资料会填满窗口，
    # 表现为 finish_reason=length 与半句答案，而日志上看是"检索很成功"。
    retrieval_context_reserve_tokens: int = Field(default=600, ge=0)
    # 近重复合并阈值（4-gram Jaccard）。1.0 = 只合并逐字相同，0 = 全部合并。
    retrieval_dedupe_threshold: float = Field(default=0.9, gt=0, le=1.0)
    # 首尾放置（lost-in-the-middle 缓解）。默认开：它只改"给模型看的顺序"，
    # 不改检索名次，所以不影响任何检索指标。
    retrieval_attention_ordering: bool = True
    # 每块进 context 的字符上限。太长会让前几块吃掉整个预算。
    retrieval_citation_max_chars: int = Field(default=800, gt=0)
    # 治理三条（质量降权 / 过期降权 / 冲突标注）。默认开：
    # 它们只乘权重、只加提示，不删数据，关掉不会变快只会变脏。
    retrieval_governance: bool = True

    # ACL 默认**关**：当前语料没有任何 acl 标注，开了等于全拒。
    # 但策略默认是拒绝未标注（见 acl.ACLPolicy），不要把它改成"放行"来自欺欺人——
    # "没开 ACL" 与 "开了但谁都能看" 在审计里是两件事。
    retrieval_acl_enabled: bool = False
    retrieval_acl_allow_unlabeled: bool = False

    # 回答用的路由档位与长度。走 models.yaml 的 tasks，不写死模型名。
    answer_task: str = "chat"
    answer_max_tokens: int = Field(default=800, ge=1, le=8192)
    # 温度默认不是 0：事实型回答也需要一点措辞自由度，否则长答案会退化成模板。
    # 但**评测必须显式设 0**（见 evals/scripts/citation_accuracy.py 的 `--temperature`）：
    # 抽样准确率若随采样波动，同一次改动跑两遍会得到两个结论。
    answer_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # -- W7：记忆 / 上下文工程 ----------------------------------------- #
    # 两档开关是实验设计的**变量**（见 docs/w7/context-cost-ab.md 实验二）：
    # off = 关掉长期记忆，"记忆是否值得存在"要靠两档数字回答，不能靠注释。
    # 字段名就叫 memory（env 即 MEMORY），不叫 memory_enabled——开关值
    # 将来可能不止两档（如 "readonly"），布尔名会封死这个演进。
    memory: Literal["off", "governed"] = "off"
    # 长期记忆单次召回条数。3 够覆盖 brand_tone + forbidden_words + 当前主题；
    # 调大之前先看 memory 槽位预算（600 token）装不装得下。
    memory_recall_k: int = Field(default=3, ge=1, le=10)
    # 低于此置信度的提取候选直接拒写（污染防线之一，另两道：来源信任级、dedupe）。
    memory_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # 会话级压缩的 token 阈值：未压缩历史 + 现摘要超过它就触发滚动摘要。
    # 主判据是 token 不是轮数（轮长差一个量级），summarize_after 是兜底。
    working_memory_token_threshold: int = Field(default=6000, ge=500)
    working_memory_summarize_after: int = Field(default=20, ge=1)
    # 压缩后保留最近多少条消息的原文（当前问题的直接上下文，不进摘要）。
    working_memory_keep_recent: int = Field(default=8, ge=1)
    # tool result 截断长度（字符）。截断必须留可重新获取的句柄（见 working.py）。
    memory_tool_result_max_chars: int = Field(default=2000, ge=100)
    # 记忆存储的表名（pgvector namespace 隔离：与知识库表 rag_chunks 不同表）。
    memory_table: str = "agent_memories"

    # -- W8：工具 / Skill / MCP ---------------------------------------- #
    # 两档开关。off = 不装配工具链路（`/v1/tools` 返回明确的"未开启"，
    # 不是空列表——空列表会被读成"没有可用工具"，是另一个故障）。
    tools: Literal["off", "governed"] = "off"
    # 工具可写入的根目录。**默认是一个受限目录**而不是仓库根：
    # 沙箱的默认策略是"只允许这个子树"，配成 `.` 等于让模型能改源码。
    tools_workspace_root: str = "./.agent_workspace"
    # 额外加载的 Skill 目录（逗号分隔）。新增 Skill 文件放这里 → reload 即可用。
    tools_skills_dir: str = ""
    # 命令白名单（逗号分隔前缀）。**默认空 = 不允许执行任何命令**：
    # 一条默认放行的白名单，等于把"执行任意命令"当成默认能力。
    tools_command_allow: str = ""
    # 出网白名单（逗号分隔域名）。默认空 = 不允许出网。
    tools_network_allow: str = ""
    # 单条命令的硬超时与工具结果进入 context 的截断长度。
    tools_command_timeout_s: float = Field(default=15.0, gt=0)
    tools_max_result_chars: int = Field(default=2000, ge=200)
    # 幂等缓存存活时间（秒）。默认 24h，与 action.py 的 DEFAULT_TTL_S 同源。
    tools_idempotency_ttl_s: float = Field(default=86400.0, gt=0)

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
