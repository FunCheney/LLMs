"""厂商适配器（W1 Day3）。

每个 adapter 只做 wire format <-> internal format 的双向翻译：

- openai_compat.py  OpenAI / DeepSeek / Qwen / Ollama / vLLM（base_url + capability 差异化）
- anthropic.py      system 顶层参数、tool_result 走 user block、独立流式事件序列
- mock.py           读 tests/fixtures/ 回放，离线测试与后续评测量产

禁止：重试、日志、缓存、任何业务判断。
"""

__all__: list[str] = []
