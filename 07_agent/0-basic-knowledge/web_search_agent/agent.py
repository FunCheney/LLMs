import json
import logging
import time
import os
from dotenv import load_dotenv
import requests

from openai import OpenAI
from typing import Any, Dict, List, Optional

from openai.types.chat.chat_completion import Choice

load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def _reasoning_safe_temperature(model, requested=1.0):
    """Reasoning models (Kimi K3, GPT-5, ...) only accept temperature=1.
    Return 1 for those; otherwise the requested value so non-reasoning
    providers (Doubao, DeepSeek, older Moonshot) are unchanged."""
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested


# ReAct 轨迹的步骤类型与展示标签（思考 → 行动 → 观察 → 最终答案）
STEP_LABELS = {
    "thought": ("💭", "思考"),
    "action": ("🔧", "行动"),
    "observation": ("👀", "观察"),
    "answer": ("✅", "最终答案"),
}

def format_trace_step(step: Dict[str, Any], max_len: int = 500) -> str:
    """把一条 ReAct 轨迹步骤渲染成一行可读文本。

    这正是本章强调的“轨迹（trajectory）”——用户消息、模型思考、工具调用、
    工具结果都被清晰地区分开来，让 ReAct 循环“想→做→看”一目了然。
    """
    icon, label = STEP_LABELS.get(step["type"], ("•", step["type"]))
    prefix = f"{icon} [{step.get('iteration', '-')}] {label}"

    if step["type"] == "action":
        args = json.dumps(step.get("args", {}), ensure_ascii=False)
        return f"{prefix}: 调用工具 {step.get('tool')}  参数={args}"

    content = str(step.get("content", "")).strip()
    if len(content) > max_len:
        content = content[:max_len] + f"…（省略 {len(content) - max_len} 字）"
    return f"{prefix}: {content}"


class WebSearchAgent:
    def __init__(self, base_url: str, api_key: str, model: str = "kimi-k3", verbose: bool = False):
        """
        Initialize the class.
        Args:
            base_url (str): The base url of the website.
            api_key (str): The API Key.
            model (str, optional): The model of the website. Defaults to "kimi-k3".
            verbose (bool, optional): Whether to display the log. Defaults to False. 是否实时打印 ReAct 轨迹（思考/行动/观察）
        """
        self.base_url = base_url or os.getenv("KIMI_BASE_URL")
        self.api_key = api_key or os.getenv("KIMI_API_KEY")
        self.model = model
        self.verbose = verbose

        from config import Config
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            # 应用配置的搜索超时，避免后端挂起时请求默认阻塞约 10 分钟
            timeout=Config.SEARCH_TIMEOUT,
        )
        self.model = model
        self.conversation_history = []
        # ReAct 轨迹：按顺序记录每一步的思考/行动/观察，便于展示与调试
        self.trace: List[Dict[str, Any]] = []
        self.api_turns: List[Dict[str, Any]] = []
        self.formula_uri = "moonshot/web-search:latest"
        self._formula_tools: Optional[List[Dict[str, Any]]] = None
        self._request_timeout = Config.SEARCH_TIMEOUT
        self.temperature = 0.6
        # 推理模型（Kimi K3）需要充足的输出预算，避免最终答案被截断
        self.max_tokens = 32768


    def _emit(self, step:Dict[str, Any]):
         """记录一条 ReAct 轨迹步骤，并在 verbose 模式下实时打印。"""
         self.trace.append(step)
         if self.verbose:
             print(format_trace_step(step))

    def _get_tools(self) -> List[Dict[str, Any]]:
        """Fetch and cache Kimi's authoritative Formula declaration."""
        # if getattr(self, "using_openrouter", False):
        #     return []
        if self._formula_tools is not None:
            return self._formula_tools

        url = (
            f"{self.base_url.rstrip('/')}/formulas/"
            f"{self.formula_uri}/tools"
        )
        started = time.monotonic()
        response = None
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self._request_timeout,
            )
            payload = response.json()
            response.raise_for_status()
            tools = payload.get("tools")
            if not isinstance(tools, list) or not tools:
                raise RuntimeError("Formula declaration response has no tools")
            if not any(
                tool.get("type") == "function"
                and tool.get("function", {}).get("name") == "web_search"
                for tool in tools
            ):
                raise RuntimeError(
                    "Formula declaration does not contain function web_search"
                )
        except Exception as exc:
            error_payload: Dict[str, Any] = {
                "class": type(exc).__name__,
                "message": str(exc),
            }
            if response is not None:
                try:
                    error_payload["response"] = response.json()
                except ValueError:
                    error_payload["response_text"] = response.text
            self.api_turns.append({
                "kind": "formula_tools",
                "formula_uri": self.formula_uri,
                "request": {"method": "GET", "url": url},
                "http_status": getattr(response, "status_code", None),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": error_payload,
            })
            raise

        self.api_turns.append({
            "kind": "formula_tools",
            "formula_uri": self.formula_uri,
            "request": {"method": "GET", "url": url},
            "http_status": response.status_code,
            "response": payload,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
        self._formula_tools = tools
        return tools

    def _execute_formula(self, name: str, raw_arguments: str) -> str:
        """Execute one Kimi Formula Fiber exactly as the model requested."""
        url = (
            f"{self.base_url.rstrip('/')}/formulas/"
            f"{self.formula_uri}/fibers"
        )
        body = {"name": name, "arguments": raw_arguments}
        started = time.monotonic()
        response = None
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
                timeout=self._request_timeout,
            )
            payload = response.json()
            response.raise_for_status()
            if payload.get("status") != "succeeded":
                raise RuntimeError(
                    f"Formula Fiber did not succeed: {payload.get('status')!r}"
                )
            context = payload.get("context") or {}
            result = context.get("output")
            if result in (None, ""):
                result = context.get("encrypted_output")
            if result in (None, ""):
                raise RuntimeError("Succeeded Formula Fiber returned no output")
        except Exception as exc:
            error_payload: Dict[str, Any] = {
                "class": type(exc).__name__,
                "message": str(exc),
            }
            if response is not None:
                try:
                    error_payload["response"] = response.json()
                except ValueError:
                    error_payload["response_text"] = response.text
            self.api_turns.append({
                "kind": "formula_fiber",
                "formula_uri": self.formula_uri,
                "request": {
                    "method": "POST",
                    "url": url,
                    "body": body,
                },
                "http_status": getattr(response, "status_code", None),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": error_payload,
            })
            raise

        self.api_turns.append({
            "kind": "formula_fiber",
            "formula_uri": self.formula_uri,
            "request": {"method": "POST", "url": url, "body": body},
            "http_status": response.status_code,
            "response": payload,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def _get_system_prompt(self) -> str:
            """
            获取系统提示
            """
            return f"""你是 Kimi，一个智能搜索助手。

    请按照以下步骤处理：
    1. 分析用户问题，识别关键信息需求
    2. 使用 web_search 官方工具搜索相关信息
    3. 如果需要更多信息，可以多次调用搜索工具
    4. 综合所有信息，生成准确、全面的答案

    注意：
    - 搜索时使用精准的关键词
    - 优先获取最新、最权威的信息
    - 答案要结构清晰，有理有据
    """

    def _chat(self, messages: List[Dict[str, Any]]) -> Choice:
        """
        调用 Kimi API 进行对话

        Args:
            messages: 消息列表

        Returns:
            API 响应的 Choice 对象
        """
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=_reasoning_safe_temperature(self.model, self.temperature),
            # Kimi K3 是推理模型，会先产出较长的 reasoning_content，需要给最终回答
            # 留足输出预算（Moonshot 要求 max_tokens>=2048），否则答案可能被截断为空。
            max_tokens=self.max_tokens,
        )
        if str(self.model).lower() == "kimi-k3":
            kwargs["reasoning_effort"] = "max"
        tools = self._get_tools()
        if tools:  # OpenRouter 兜底时无内置搜索工具，省略 tools 参数
            kwargs["tools"] = tools
        started = time.monotonic()
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            self.api_turns.append({
                "kind": "chat_completion",
                "request": json.loads(json.dumps(kwargs, ensure_ascii=False, default=str)),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": {"class": type(exc).__name__, "message": str(exc)},
            })
            raise
        response = (
            completion.model_dump() if hasattr(completion, "model_dump")
            else completion.dict() if hasattr(completion, "dict")
            else {"raw_response": str(completion)}
        )
        self.api_turns.append({
            "kind": "chat_completion",
            "request": json.loads(json.dumps(kwargs, ensure_ascii=False, default=str)),
            "response": json.loads(json.dumps(response, ensure_ascii=False, default=str)),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
        return completion.choices[0]

    def search_and_answer(self, user_question: str, max_iterations: int = 5) -> str:
        """
        执行搜索并生成答案

        Args:
            user_question: 用户问题
            max_iterations: 最大搜索迭代次数（防止无限循环）

        Returns:
            最终答案
        """
        # 构建系统提示
        system_prompt = self._get_system_prompt()

        # 重置对话历史并添加新的系统提示
        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        # 重置 ReAct 轨迹
        self.trace = []
        self.api_turns = []
        # Each independent question keeps its own real declaration receipt.
        self._formula_tools = None
        logger.info("开始调用 Kimi 搜索工具...")

        try:
            finish_reason = None
            iteration = 0

            # 循环处理，直到获得最终答案或达到最大迭代次数
            while (finish_reason is None or finish_reason == "tool_calls") and iteration < max_iterations:
                iteration += 1
                logger.info(f"迭代 {iteration}/{max_iterations}")

                # 调用 Kimi API
                choice = self._chat(self.conversation_history)
                finish_reason = choice.finish_reason

                # 捕获模型的思考过程（Kimi K3 等推理模型通过 reasoning_content 暴露思考模式）
                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning:
                    self._emit({"iteration": iteration, "type": "thought", "content": reasoning})

                if finish_reason == "tool_calls":
                    # 处理工具调用
                    logger.info(f"模型请求调用 {len(choice.message.tool_calls)} 个工具")

                    # 添加助手的消息（包含工具调用）到历史。
                    # 注意：必须把消息重建为纯 dict，而不是直接塞入 SDK 返回的
                    # pydantic message 对象——后者会附带 reasoning_content / refusal
                    # 等额外字段，回传给 Moonshot 时会触发 "tokenization failed" 400 错误。
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    })

                    # 执行每个工具调用
                    for tool_call in choice.message.tool_calls:
                        tool_call_name = tool_call.function.name
                        try:
                            tool_call_arguments = json.loads(
                                tool_call.function.arguments or "{}"
                            )
                        except json.JSONDecodeError:
                            # Models sometimes emit slightly invalid JSON; match
                            # chapter4 async-agent and keep the ReAct loop alive.
                            tool_call_arguments = {}
                            logger.warning(
                                "工具参数不是合法 JSON，已按空对象继续: %r",
                                tool_call.function.arguments,
                            )

                        logger.info(f"执行工具: {tool_call_name}, 参数: {tool_call_arguments}")
                        # 行动：记录一次工具调用
                        self._emit({"iteration": iteration, "type": "action",
                                    "tool": tool_call_name, "args": tool_call_arguments})

                        if tool_call_name == "web_search":
                            # Formula requires the original serialized
                            # arguments, even though the parsed copy above is
                            # retained for a readable ReAct trace.
                            tool_result = self._execute_formula(
                                tool_call_name,
                                tool_call.function.arguments or "{}",
                            )
                        else:
                            tool_result = f"Error: unable to find tool by name '{tool_call_name}'"

                        tool_content = (
                            tool_result
                            if isinstance(tool_result, str)
                            else json.dumps(tool_result, ensure_ascii=False)
                        )
                        # 观察：记录工具返回结果
                        self._emit({"iteration": iteration, "type": "observation",
                                    "tool": tool_call_name, "content": tool_content})
                        # 构建工具响应消息并添加到历史
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_content
                        })
                elif finish_reason == "length":
                    # 输出预算（max_tokens）耗尽导致截断：返回已生成内容并明确标注，
                    # 而不是把半截答案当作完整答案，也不误报“无法获取足够信息”
                    # （content 为空时，思考过程已耗尽整个预算）。
                    partial = (choice.message.content or "").strip()
                    logger.warning("回答因达到 max_tokens 上限被截断 (finish_reason=length)")
                    note = "（注意：回答因达到 max_tokens 上限被截断，请增大 max_tokens 后重试。）"
                    final = f"{partial}\n\n{note}" if partial else note
                    self._emit({"iteration": iteration, "type": "answer", "content": final})
                    # 存入历史时保留截断提示（final），否则 get_conversation_history()
                    # 会丢失截断语义，后续复用历史时可能把不完整回答当作普通回答。
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final
                    })
                    return final
                else:
                    # 获得最终答案
                    if choice.message.content:
                        answer = choice.message.content
                        logger.info("成功生成答案")
                        self._emit({"iteration": iteration, "type": "answer", "content": answer})

                        # 添加最终答案到历史
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": answer
                        })

                        return answer

            # 如果达到最大迭代次数仍未完成
            if iteration >= max_iterations:
                logger.warning(f"达到最大迭代次数 {max_iterations}")
                return MAX_ITERATIONS_MESSAGE

            return NO_INFO_MESSAGE

        except Exception as e:
            logger.error(f"{SEARCH_ERROR_PREFIX}: {str(e)}")
            return f"{SEARCH_ERROR_PREFIX}: {str(e)}"



# search_and_answer 不抛异常，而是以字符串形式返回失败兜底文案。
# 下列前缀 / 文案是判断“一次搜索是否失败”的唯一来源，供调用方（如
# examples.batch_search）复用，避免把失败响应误判为 success。
SEARCH_ERROR_PREFIX = "搜索过程中出现错误"
MAX_ITERATIONS_MESSAGE = "抱歉，搜索过程超过了最大迭代次数，请稍后重试。"
NO_INFO_MESSAGE = "抱歉，我无法获取足够的信息来回答您的问题。"


if __name__ == "__main__":
    # print(os.getenv("KIMI_BASE_URL"))
    agent = WebSearchAgent(base_url=os.getenv("KIMI_BASE_URL"), api_key=os.getenv("KIMI_API_KEY"))
    # 获取有那些工具
    print(agent._get_tools())

    # 示例问题
    test_question = "请搜索 Moonshot AI Context Caching 技术，告诉我这是什么。"

    print(f"问题: {test_question}")
    print("-" * 60)
    print("搜索中...")

    # 获取答案
    answer = agent.search_and_answer(test_question)

    print("\n答案:")
    print("-" * 60)
    print(answer)
