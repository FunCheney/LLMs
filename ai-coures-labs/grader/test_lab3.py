"""
实验3测评脚本：记忆系统的内容检索
测试学生实现的 chat_with_langchain_memory 和 get_memory_summary 函数
"""
import pytest
from student_code.lab3.main import (
    chat_with_langchain_memory,
    get_memory_summary,
    clear_memory
)
from grader.fixtures import ollama_health_check, clean_session_state


@pytest.mark.lab3
class TestLab3LangChainMemory:
    """实验3测评测试类"""

    def setup_method(self):
        """每个测试方法执行前清空所有会话"""
        clear_memory()

    def test_uses_conversation_buffer_memory(self, ollama_health_check):
        """
        测试1：验证使用了 ConversationBufferMemory
        权重：15%
        """
        session_id = "test_memory_type"
        result = chat_with_langchain_memory("测试消息", session_id)

        # 检查返回值包含 memory_variables
        assert 'memory_variables' in result, \
            "返回值必须包含 'memory_variables' 键"

        # 检查 memory_variables 是字典类型
        assert isinstance(result['memory_variables'], dict), \
            f"memory_variables 必须是字典类型，实际 {type(result['memory_variables'])}"

        print(f"✓ Memory variables keys: {result['memory_variables'].keys()}")

    def test_memory_variables_in_output(self, ollama_health_check):
        """
        测试2：验证返回值包含 memory_variables 键
        权重：15%
        """
        session_id = "test_output_structure"
        result = chat_with_langchain_memory("你好", session_id)

        required_keys = ['response', 'memory_variables']
        for key in required_keys:
            assert key in result, \
                f"返回字典缺少必需的键: '{key}'，当前包含: {list(result.keys())}"

        # 验证 response 是字符串
        assert isinstance(result['response'], str), \
            f"response 必须是字符串，实际 {type(result['response'])}"
        assert len(result['response']) > 0, \
            "response 不能为空"

        # 验证 memory_variables 包含 history 相关键
        # ConversationBufferMemory 通常会有 'history' 键
        memory_vars = result['memory_variables']
        assert 'history' in memory_vars, \
            f"memory_variables 应包含 'history' 键，当前包含: {list(memory_vars.keys())}"

    def test_information_persistence(self, ollama_health_check):
        """
        测试3：验证信息持久化
        权重：30%
        """
        session_id = "test_persistence"

        # 发送多条包含特定信息的消息
        chat_with_langchain_memory("我的生日是1990年5月20日", session_id)
        chat_with_langchain_memory("我在北京工作", session_id)
        chat_with_langchain_memory("我的爱好是摄影", session_id)

        # 获取历史摘要
        summary = get_memory_summary(session_id)

        # 验证摘要包含所有信息
        assert "1990" in summary or "5月20日" in summary or "生日" in summary, \
            f"历史摘要应包含生日信息，实际内容:\n{summary}"

        assert "北京" in summary, \
            f"历史摘要应包含地点信息（北京），实际内容:\n{summary}"

        assert "摄影" in summary, \
            f"历史摘要应包含爱好信息（摄影），实际内容:\n{summary}"

        print(f"✓ 信息持久化测试通过，摘要包含所有关键信息")
        print(f"历史摘要:\n{summary[:200]}...")

    def test_summary_format(self, ollama_health_check):
        """
        测试4：验证 get_memory_summary 返回格式化的历史记录
        权重：20%
        """
        session_id = "test_format"

        # 发送3条消息
        chat_with_langchain_memory("第一条消息", session_id)
        chat_with_langchain_memory("第二条消息", session_id)
        chat_with_langchain_memory("第三条消息", session_id)

        summary = get_memory_summary(session_id)

        # 验证摘要是字符串
        assert isinstance(summary, str), \
            f"摘要必须是字符串类型，实际 {type(summary)}"

        # 验证摘要不为空
        assert len(summary) > 0, \
            "摘要不能为空"

        # 验证包含 User 和 AI 标识（或类似的角色标识）
        summary_lower = summary.lower()
        has_user_marker = any(marker in summary_lower for marker in ['user', 'human', '用户'])
        has_ai_marker = any(marker in summary_lower for marker in ['ai', 'assistant', '助手'])

        assert has_user_marker, \
            f"摘要应包含用户角色标识（User/Human/用户），实际内容:\n{summary[:100]}"

        assert has_ai_marker, \
            f"摘要应包含AI角色标识（AI/Assistant/助手），实际内容:\n{summary[:100]}"

        # 验证包含对话内容
        assert "第一条消息" in summary or "第二条消息" in summary or "第三条消息" in summary, \
            f"摘要应包含对话内容，实际:\n{summary[:200]}"

        print(f"✓ 摘要格式验证通过")
        print(f"摘要示例:\n{summary[:150]}...")

    def test_nonexistent_session(self, ollama_health_check):
        """
        测试5：验证查询不存在的 session 时的行为
        权重：20%
        """
        nonexistent_id = "session_does_not_exist_12345"
        summary = get_memory_summary(nonexistent_id)

        # 验证返回空字符串或明确提示
        assert isinstance(summary, str), \
            f"摘要必须是字符串类型，实际 {type(summary)}"

        # 对于不存在的会话，应该返回空字符串或包含提示信息的字符串
        # 这里我们接受空字符串
        assert len(summary) == 0, \
            f"不存在的会话应返回空字符串，实际返回: {summary[:100]}"

        print(f"✓ 不存在会话的处理正确")

    def test_cross_session_isolation(self, ollama_health_check):
        """
        测试6：验证不同 session 的 summary 互不包含对方内容
        权重：加分项
        """
        session_a = "alice"
        session_b = "bob"

        # Session A 发送消息
        chat_with_langchain_memory("我喜欢Python编程", session_a)
        chat_with_langchain_memory("我在学习AI技术", session_a)

        # Session B 发送消息
        chat_with_langchain_memory("我喜欢Java开发", session_b)
        chat_with_langchain_memory("我在研究区块链", session_b)

        # 获取两个会话的摘要
        summary_a = get_memory_summary(session_a)
        summary_b = get_memory_summary(session_b)

        # 验证 Session A 的摘要包含自己的内容
        assert "Python" in summary_a, \
            f"Session A 的摘要应包含 'Python'，实际:\n{summary_a[:100]}"

        assert "AI" in summary_a or "ai" in summary_a.lower(), \
            f"Session A 的摘要应包含 'AI'，实际:\n{summary_a[:100]}"

        # 验证 Session A 的摘要不包含 Session B 的内容
        assert "Java" not in summary_a, \
            f"Session A 的摘要不应包含 Session B 的内容（Java），实际:\n{summary_a[:100]}"

        assert "区块链" not in summary_a, \
            f"Session A 的摘要不应包含 Session B 的内容（区块链），实际:\n{summary_a[:100]}"

        # 验证 Session B 的摘要包含自己的内容
        assert "Java" in summary_b, \
            f"Session B 的摘要应包含 'Java'，实际:\n{summary_b[:100]}"

        # 验证 Session B 的摘要不包含 Session A 的内容
        assert "Python" not in summary_b, \
            f"Session B 的摘要不应包含 Session A 的内容（Python），实际:\n{summary_b[:100]}"

        print(f"✓ 跨会话隔离测试通过")
        print(f"Session A 摘要:\n{summary_a[:100]}...")
        print(f"Session B 摘要:\n{summary_b[:100]}...")

        """
                测试5：验证查询不存在的 session 时的行为
                权重：20%
                """
        nonexistent_id = "session_does_not_exist_12345"
        summary = get_memory_summary(nonexistent_id)

        # 验证返回空字符串或明确提示
        assert isinstance(summary, str), \
            f"摘要必须是字符串类型，实际 {type(summary)}"

        # 对于不存在的会话，应该返回空字符串或包含提示信息的字符串
        # 这里我们接受空字符串
        assert len(summary) == 0, \
            f"不存在的会话应返回空字符串，实际返回: {summary[:100]}"

        print(f"✓ 不存在会话的处理正确")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])