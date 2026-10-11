import os

from dotenv import load_dotenv

from generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def create_chat_model(model: str | None = None, temperature: float = 0.0):
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. "
            "Add it to .env or export it in your shell."
        )

    model_name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

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
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()
