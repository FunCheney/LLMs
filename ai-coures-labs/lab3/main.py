"""
实验3：记忆系统的内容检索
学生需要使用 LangChain 的 ConversationBufferMemory 管理会话历史
"""
from typing import Dict

# 提示：需要导入 LangChain 相关模块
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
#  (或新版本的等效导入)
from langchain_classic.memory import ConversationBufferMemory
# (或新版本的等效导入)
from langchain_classic.chains import LLMChain


# 全局 Memory 映射：{session_id: ConversationBufferMemory 实例}
SESSION_MEMORIES: Dict[str, ConversationBufferMemory] = {}  # TODO: 将 any 替换为正确的类型


def chat_with_langchain_memory(message: str, session_id: str) -> dict:
    """
    使用 LangChain 的 ConversationBufferMemory 进行对话

    参数:
        message: 用户消息
        session_id: 会话ID

    返回:
        字典，包含:
        - response (str): AI回复
        - memory_variables (dict): ConversationBufferMemory 的内部变量

    实现要求:
        1. 使用 ConversationBufferMemory 管理每个 session 的历史
        2. 将 Memory 对象与 Ollama LLM 集成
        3. memory_variables 应包含 'history' 键
        4. 不同 session 的 Memory 必须独立
    """
    global SESSION_MEMORIES

    # 1. 检查 session_id 是否存在对应的 Memory，不存在则创建
    if session_id not in SESSION_MEMORIES:
        SESSION_MEMORIES[session_id] = ConversationBufferMemory(
            memory_key="history",
            return_messages=False
        )

    memory = SESSION_MEMORIES[session_id]

    # 2. 创建 Ollama LLM 实例
    llm = Ollama(model="llama3:latest")  # 可以根据需要更改模型

    # 3. 创建 PromptTemplate（包含历史上下文）
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template="""基于以下对话历史回答问题：

        {history}
        
        当前问题: {input}
        
        请提供有帮助的回答:"""
    )

    # 4. 创建 LLMChain，连接 Prompt、LLM 和 Memory
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=False  # 设置为 True 可以查看详细执行过程
    )

    # 5. 运行链并获取响应
    response = chain.invoke({"input": message})

    # 获取当前的 memory 变量
    memory_variables = memory.load_memory_variables({})

    # 6. 返回响应和 memory_variables
    return {
        "response": response["text"],
        "memory_variables": memory_variables
    }


def get_memory_summary(session_id: str) -> str:
    """
    获取指定会话的完整历史记录摘要

    参数:
        session_id: 会话ID

    返回:
        格式化的历史记录字符串，如:
        "User: 你好\nAI: 你好！有什么可以帮助你的吗？\nUser: ..."

        如果会话不存在，返回空字符串或提示信息

    实现要求:
        1. 返回人类可读的格式
        2. 包含完整的用户输入和AI回复
        3. 对不存在的 session_id，返回空字符串或提示
    """
    global SESSION_MEMORIES

    # 1. 检查 session_id 是否存在
    if session_id not in SESSION_MEMORIES:
        return f"会话 {session_id} 不存在"

    memory = SESSION_MEMORIES[session_id]

    # 2. 获取 Memory 的 buffer
    memory_variables = memory.load_memory_variables({})

    # 3. 格式化消息历史为可读字符串
    history_text = memory_variables.get("history", "")

    # 如果历史为空，返回提示
    if not history_text.strip():
        return ""

    return history_text


def clear_memory(session_id: str = None):
    """
    清除会话记忆（辅助函数，用于测试）

    参数:
        session_id: 要清除的会话ID，如果为 None 则清除所有会话
    """
    global SESSION_MEMORIES

    if session_id is None:
        SESSION_MEMORIES.clear()
        print("所有会话记忆已清除")
    elif session_id in SESSION_MEMORIES:
        del SESSION_MEMORIES[session_id]
        print(f"会话 {session_id} 的记忆已清除")
    else:
        print(f"会话 {session_id} 不存在")


# 测试代码（可选，用于学生本地调试）
if __name__ == "__main__":
    # 清空记忆
    clear_memory()

    # 测试对话
    print("=== 测试 LangChain Memory ===")
    session_id = "test_session"

    result1 = chat_with_langchain_memory("我的电话是13800138000", session_id)
    print(f"第1次对话:")
    print(f"  Response: {result1['response'][:50]}")
    print(f"  Memory variables keys: {result1['memory_variables'].keys()}")

    result2 = chat_with_langchain_memory("我的邮箱是test@example.com", session_id)
    print(f"\n第2次对话:")
    print(f"  Response: {result2['response'][:50]}")

    # 获取历史摘要
    summary = get_memory_summary(session_id)
    print(f"\n历史摘要:\n{summary}")

    # 验证信息是否保存
    print(f"\n验证信息持久化:")
    print(f"  包含电话号码: {'13800138000' in summary}")
    print(f"  包含邮箱: {'test@example.com' in summary}")
