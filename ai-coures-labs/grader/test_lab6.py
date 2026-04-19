"""
实验6测评脚本：工具调用的结果验证
测试学生实现的 agent_executor 函数
"""
import pytest
from student_code.lab6.main import (
    agent_executor,
    get_stock_price,
    add_numbers,
    get_weather
)
from grader.fixtures import ollama_health_check, clean_session_state


@pytest.mark.lab6
class TestLab6ToolCalling:
    """实验6测评测试类"""

    def test_stock_price_tool_selection(self, ollama_health_check):
        """
        测试1：验证股票价格工具选择
        权重：35%
        """
        tools = [get_stock_price, add_numbers, get_weather]
        query = "苹果公司的股价是多少？"

        result = agent_executor(query, tools)

        # 验证工具选择
        assert 'tool_used' in result, \
            "返回值必须包含 'tool_used' 键"

        assert result['tool_used'] == 'get_stock_price', \
            f"应选择 get_stock_price 工具，实际选择: {result['tool_used']}"

        # 验证参数传递
        assert 'tool_input' in result, \
            "返回值必须包含 'tool_input' 键"

        assert 'symbol' in result['tool_input'], \
            f"工具输入应包含 'symbol' 参数，实际: {result['tool_input']}"

        assert result['tool_input']['symbol'].upper() == 'AAPL', \
            f"股票代码应为 AAPL，实际: {result['tool_input']['symbol']}"

        print(f"✓ 股票价格工具选择正确")
        print(f"  工具: {result['tool_used']}")
        print(f"  参数: {result['tool_input']}")

    def test_add_numbers_tool_selection(self, ollama_health_check):
        """
        测试2：验证加法工具选择
        权重：35%
        """
        tools = [get_stock_price, add_numbers, get_weather]
        query = "计算15加27等于多少"

        result = agent_executor(query, tools)

        assert result['tool_used'] == 'add_numbers', \
            f"应选择 add_numbers 工具，实际选择: {result['tool_used']}"

        print(f"✓ 加法工具选择正确: {result['tool_used']}")

    def test_parameter_passing_correctness(self, ollama_health_check):
        """
        测试3：验证参数传递正确性
        权重：25%
        """
        tools = [get_stock_price, add_numbers, get_weather]
        query = "10加20等于多少？"

        result = agent_executor(query, tools)

        # 验证参数
        assert 'tool_input' in result, \
            "返回值必须包含 'tool_input' 键"

        tool_input = result['tool_input']

        # 验证包含正确的参数
        assert 'a' in tool_input and 'b' in tool_input, \
            f"add_numbers 工具应包含 'a' 和 'b' 参数，实际: {tool_input}"

        # 验证参数值（允许顺序不同）
        values = {tool_input['a'], tool_input['b']}
        assert values == {10, 20}, \
            f"参数值应为 10 和 20，实际: a={tool_input['a']}, b={tool_input['b']}"

        print(f"✓ 参数传递正确: {tool_input}")

    def test_tool_output_retrieval(self, ollama_health_check):
        """
        测试4：验证工具输出获取
        权重：25%
        """
        tools = [get_stock_price, add_numbers, get_weather]
        query = "查询AAPL股价"

        result = agent_executor(query, tools)

        # 验证工具输出
        assert 'tool_output' in result, \
            "返回值必须包含 'tool_output' 键"

        assert result['tool_output'] == 175.0, \
            f"AAPL股价应为 175.0，实际: {result['tool_output']}"

        print(f"✓ 工具输出正确: {result['tool_output']}")

    def test_weather_query_handling(self, ollama_health_check):
        """
        测试5：验证天气查询处理
        权重：15%
        """
        tools = [get_stock_price, add_numbers, get_weather]
        query = "北京的天气怎么样？"

        result = agent_executor(query, tools)

        # 验证工具选择
        assert result['tool_used'] == 'get_weather', \
            f"应选择 get_weather 工具，实际选择: {result['tool_used']}"

        # 验证参数
        assert 'city' in result['tool_input'], \
            f"工具输入应包含 'city' 参数，实际: {result['tool_input']}"

        assert result['tool_input']['city'] == '北京', \
            f"城市应为 '北京'，实际: {result['tool_input']['city']}"

        # 验证输出
        assert isinstance(result['tool_output'], dict), \
            f"天气输出应为字典，实际: {type(result['tool_output'])}"

        assert 'temp' in result['tool_output'], \
            f"天气输出应包含 'temp' 键，实际: {result['tool_output']}"

        assert result['tool_output']['temp'] == 25, \
            f"北京温度应为 25，实际: {result['tool_output']['temp']}"

        print(f"✓ 天气查询处理正确")
        print(f"  工具: {result['tool_used']}")
        print(f"  输出: {result['tool_output']}")

    def test_final_answer_structure(self, ollama_health_check):
        """
        测试6：验证最终回答结构
        权重：加分项
        """
        tools = [get_stock_price, add_numbers, get_weather]
        query = "15加27等于多少"

        result = agent_executor(query, tools)

        # 验证包含所有必需字段
        required_keys = ['tool_used', 'tool_input', 'tool_output', 'final_answer']
        for key in required_keys:
            assert key in result, \
                f"返回值缺少必需的键: '{key}'，当前包含: {list(result.keys())}"

        # 验证 final_answer 是字符串
        assert isinstance(result['final_answer'], str), \
            f"final_answer 必须是字符串，实际: {type(result['final_answer'])}"

        # 验证 final_answer 非空
        assert len(result['final_answer']) > 0, \
            "final_answer 不能为空"

        print(f"✓ 最终回答结构正确")
        print(f"  回答: {result['final_answer'][:50]}...")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
