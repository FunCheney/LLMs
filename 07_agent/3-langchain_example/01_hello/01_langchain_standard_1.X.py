from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI # 用 OpenAI 兼容接口和模型对话
import os
import dotenv
# LangChainException：LangChain 在调用模型失败时会抛出的异常类型。
# 在 main() 里用 except LangChainException 单独接住这类错误，就能打出「模型调用失败」的日志，和配置错误、其他未知错误区分开。
from langchain_core.exceptions import LangChainException

dotenv.load_dotenv()

import logging

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # 当前模块的 logger，后面用 logger.info(...) 即可


def init_llm() -> ChatOpenAI:
    """
    初始化 LLM 客户端
    :return:
     ChatOpenAI: 初始化好的「对话客户端」，可以对其调用 .invoke(问题) 或 .stream(问题)。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL")

    print(api_key)
    print(base_url)
    print(model)

    # 创建客户端段
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model_name=model,
        temperature=0.7,  # 控制「随机程度」：0 更确定、重复性高；1 更随机、更有创意。一般 0.5～0.8 即可。
        max_tokens=2048,  # 单次回复最多生成多少个 token（约等于字数），防止回复过长或超限。
    )

    return llm


def main():
    """主函数：封装核心逻辑，符合 Python 工程化规范。"""
    try:
        # 先拿到「可对话的客户端」
        llm = init_llm()
        logger.info("LLM客户端初始化成功")

        # ----- 方式一：invoke（一次性拿完整回复） -----
        # 发一个问题，程序会等模型全部答完，再一次性把 response 给你。适合短问答。
        question = "你是谁"
        response = llm.invoke(question)
        logger.info(f"问题：{question}")
        logger.info(f"回答：{response.content}")  # .content 里是模型的纯文字回复

        # ----- 方式二：stream（流式，边生成边输出） -----
        # 模型边想边返回，每次返回一小段（chunk），用 for 循环一段段打印，就像打字机效果。适合长文或需要「实时看到输出」的场景。
        print("==================== 以下是流式输出（另一种调用方式）")
        print("*" * 50)
        response_stream = llm.stream("介绍下 langchain，300字以内")
        for chunk in response_stream:
            print(chunk.content, end="")  # end="" 表示不换行，紧挨着打
        print()  # 流式结束后补一个换行，避免和后续输出粘在一起

    # ----- 异常处理：根据错误类型打不同日志，方便排查 -----
    # try 里面的代码一旦报错，会跳到下面某个 except；若都不匹配，再往上抛。
    except ValueError as e:
        # 例如：.env 里没配 QWEN_API_KEY，init_llm_client 里会 raise ValueError
        logger.error(f"配置错误：{str(e)}")
    except LangChainException as e:
        # 例如：网络失败、API 限流、模型返回异常等，LangChain 会抛出 LangChainException
        logger.error(f"模型调用失败：{str(e)}")
    except Exception as e:
        # 其他没预料到的错误都归到这里，避免程序静默崩溃
        logger.error(f"未知错误：{str(e)}")


# 这里直接写 main() 即可，因为本文件的 main 是普通函数（def main），调用即执行。
# 若 main 是异步函数（async def main），则必须写 asyncio.run(main())，否则协程不会真正运行。
if __name__ == "__main__":
    main()