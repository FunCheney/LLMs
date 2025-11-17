
from typing import Dict, List

# 全局会话存储：{session_id: [消息列表]}
# 每条消息格式: {"role": "user" | "assistant", "content": "消息内容"}
SESSION_HISTORY: Dict[str, List[Dict[str, str]]] = {}

class SessionManager:
    def __init__(self, session):
        self.session = session


    def clear_session(session_id: str = None):
        """
        清除会话历史（辅助函数，用于测试）

        参数:
            session_id: 要清除的会话ID，如果为 None 则清除所有会话
        """
        global SESSION_HISTORY

        if session_id is None:
            SESSION_HISTORY.clear()
        elif session_id in SESSION_HISTORY:
            del SESSION_HISTORY[session_id]

    def get_session(session_id: str) -> List[Dict[str, str]]:
        """
        判断会话是否存在
        :return:
        """
        if session_id is None:
            return []
        if session_id not in SESSION_HISTORY:
            return []
        return SESSION_HISTORY[session_id]

    def add_session(session_id: str):
        """
        保存会话
        :return:
        """
        SESSION_HISTORY[session_id] = []

    def add_message(session_id: str, role: str, content: str):
        """
        向会话添加消息

        Args:
            session_id: 会话ID
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
        """
        session = SESSION_HISTORY[session_id]
        session.append({"role": role, "content": content})
