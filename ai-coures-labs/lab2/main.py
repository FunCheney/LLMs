"""
实验2：有状态对话的快照验证
学生需要实现 chat_with_memory 函数，管理多个独立会话的历史记录
"""
from session_manager import SessionManager
import httpx


def chat_with_memory(message: str, session_id: str) -> dict:
    """
    带记忆功能的对话函数，支持多会话隔离
    """
    # 1. 获取或创建会话
    session = SessionManager.get_session(session_id)

    if session is None or len(session) == 0:
        session = []
        SessionManager.add_session(session_id)

    # 2. 计算 history_length（本次对话前的历史消息数）
    history_length = len(session)

    # 3. 构建 Prompt，包含历史上下文 + 当前消息
    prompt = build_prompt(message, session)
    print(prompt)

    try:
        with httpx.Client(timeout=30.0) as client:
            # 4. 调用 Ollama API 获取回复
            response = client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                }
            )
            response.raise_for_status()

            # 解析响应
            result_data = response.json()
            response_text = result_data.get("response", "").strip()
            # 5. 保存用户消息和助手回复到历史记录
            # 先保存用户消息
            SessionManager.add_message(session_id, "user", message)
            # 再保存助手回复
            SessionManager.add_message(session_id, "assistant", response_text)
            # 6. 返回结构化结果
            return {
                "response": response_text,  # 返回回复文本
                "history_length": history_length,  # 本次对话前的历史消息数
                "session_id": session_id
            }

    except httpx.ConnectTimeout as e:
        raise ConnectionError(f"连接超时: {e}")
    except httpx.HTTPStatusError as e:
        raise ConnectionError(f"HTTP错误: {e}")
    except Exception as e:
        raise RuntimeError(f"对话失败: {e}")


def build_prompt(cur_msg: str, history_msg: list) -> str:
    """使用ChatML格式构建Prompt"""

    conversation_history = ""
    for msg in history_msg:
        role = "用户" if msg["role"] == "user" else "助手"
        conversation_history += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"

    prompt = f"""你是一个有帮助的AI助手。请根据对话历史回答用户的问题。
        {conversation_history}
        <|im_start|>用户
        {cur_msg}<|im_end|>
        <|im_start|>助手
    """

    return prompt


# 测试代码（可选，用于学生本地调试）
if __name__ == "__main__":
    # 清空历史
    SessionManager.clear_session()

    # 测试单会话
    print("=== 测试单会话 ===")
    session_id = "test_session"

    result1 = chat_with_memory("你好", session_id)
    print(f"第1次对话 - history_length: {result1['history_length']}, response: {result1['response'][:50]}")

    result2 = chat_with_memory("我叫张三", session_id)
    print(f"第2次对话 - history_length: {result2['history_length']}, response: {result2['response'][:50]}")

    result3 = chat_with_memory("我刚才说了什么？", session_id)
    print(f"第3次对话 - history_length: {result3['history_length']}, response: {result3['response'][:50]}")

    # 测试多会话隔离
    print("\n=== 测试会话隔离 ===")
    SessionManager.clear_session()

    result_a1 = chat_with_memory("苹果", "session_A")
    print(f"Session A 第1次 - history_length: {result_a1['history_length']}")

    result_b1 = chat_with_memory("香蕉", "session_B")
    print(f"Session B 第1次 - history_length: {result_b1['history_length']}")

    result_a2 = chat_with_memory("我之前说了什么？", "session_A")
    print(f"Session A 第2次 - history_length: {result_a2['history_length']}, response: {result_a2['response'][:50]}")
