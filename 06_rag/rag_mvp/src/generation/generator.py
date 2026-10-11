"""
答案生成：调用 DeepSeek 大模型，基于检索到的上下文回答问题。

- 使用 LangChain 的 ChatDeepSeek 封装；
- API Key 从 .env 读取（DEEPSEEK_API_KEY），模型名可用 DEEPSEEK_MODEL 覆盖；
- temperature 默认 0，尽量让输出稳定、可复现。
"""

import os

from dotenv import load_dotenv

from generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def create_chat_model(model: str | None = None, temperature: float = 0.0):
    """构建 ChatDeepSeek 实例；缺少 API Key 或依赖时给出明确报错。"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. "
            "Add it to .env or export it in your shell."
        )

    model_name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # 延迟导入：没装 langchain-deepseek 时也能 import 本模块
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError as exc:
        raise RuntimeError(
            "langchain-deepseek is not installed. Run: uv sync"
        ) from exc

    return ChatDeepSeek(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )


def generate_answer(
    question: str,
    context: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """把 system prompt + (上下文, 问题) 交给模型，返回纯文本答案。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = create_chat_model(model=model, temperature=temperature)
    user_message = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    # 不同模型返回的 content 可能是 str 或结构化对象，统一转成字符串
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()
