"""LLM 适配层（W1）。

待实现的文件（契约见 docs/architecture.md 第 3 节）：

- types.py     内部数据契约：Message / ToolSpec / ToolCall / LLMResponse / StreamEvent / Usage
- errors.py    异常层次 + retryable 语义 + 厂商状态码映射（映射只放这里，不放 adapter）
- base.py      LLMClient 抽象 + Capability 枚举
- client.py    装饰器链：trace/日志 -> 计量 -> 重试 -> 能力校验 -> adapter
- registry.py  name -> adapter 工厂（按 models.yaml 的 type 懒加载）
- router.py    task -> tier -> 候选模型列表（含 fallback 链）
- providers/   openai_compat / anthropic / mock

纪律：adapter 内禁止重试、日志、业务判断（ADR-003）。
"""

__all__: list[str] = []
