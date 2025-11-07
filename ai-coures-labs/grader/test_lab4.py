"""
实验4测评脚本：LangChain 链的确定性输出
测试学生实现的 generate_ad 函数
"""
import pytest
from student_code.lab4.main import generate_ad
from grader.fixtures import ollama_health_check, clean_session_state


@pytest.mark.lab4
class TestLab4LangChainChain:
    """实验4测评测试类"""

    def test_output_structure(self, ollama_health_check):
        """
        测试1：验证返回字典包含所有必需键
        权重：20%
        """
        input_data = {"product": "测试产品", "feature": "测试特性"}
        result = generate_ad(input_data)

        # 检查必需的键
        required_keys = ['ad_copy', 'word_count', 'template_used']
        for key in required_keys:
            assert key in result, \
                f"返回字典缺少必需的键: '{key}'，当前包含: {list(result.keys())}"

        # 检查数据类型
        assert isinstance(result['ad_copy'], str), \
            f"ad_copy 必须是字符串，实际 {type(result['ad_copy'])}"

        assert isinstance(result['word_count'], int), \
            f"word_count 必须是整数，实际 {type(result['word_count'])}"

        assert isinstance(result['template_used'], str), \
            f"template_used 必须是字符串，实际 {type(result['template_used'])}"

    def test_word_count_accuracy(self, ollama_health_check):
        """
        测试2：验证 word_count 与 ad_copy 长度一致
        权重：25%
        """
        input_data = {"product": "智能音箱", "feature": "语音控制"}
        result = generate_ad(input_data)

        actual_length = len(result['ad_copy'])
        reported_count = result['word_count']

        assert actual_length == reported_count, \
            f"字数统计不准确：ad_copy 实际长度为 {actual_length}，word_count 为 {reported_count}"

        print(f"✓ 字数统计准确：{reported_count} 字")

    def test_product_name_inclusion(self, ollama_health_check):
        """
        测试3：验证产品名在广告文案中
        权重：20%
        """
        input_data = {"product": "智能手表", "feature": "心率监测"}
        result = generate_ad(input_data)

        ad_copy = result['ad_copy']
        product = input_data['product']

        assert product in ad_copy, \
            f"广告文案应包含产品名称 '{product}'，实际文案: {ad_copy}"

        print(f"✓ 文案包含产品名: {ad_copy[:50]}...")

    def test_feature_inclusion(self, ollama_health_check):
        """
        测试4：验证特性关键词在文案中
        权重：20%
        """
        input_data = {"product": "无线耳机", "feature": "降噪功能"}
        result = generate_ad(input_data)

        ad_copy = result['ad_copy']
        feature = input_data['feature']

        # 检查完整特性或特性关键词
        feature_keywords = feature.split()
        has_feature = feature in ad_copy or any(keyword in ad_copy for keyword in feature_keywords)

        assert has_feature, \
            f"广告文案应包含特性 '{feature}' 或其关键词，实际文案: {ad_copy}"

        print(f"✓ 文案包含特性关键词: {ad_copy[:50]}...")

    def test_word_count_reasonable_range(self, ollama_health_check):
        """
        测试5：验证文案长度在合理范围
        权重：15%
        """
        input_data = {"product": "扫地机器人", "feature": "自动避障"}
        result = generate_ad(input_data)

        word_count = result['word_count']

        assert 10 <= word_count <= 100, \
            f"文案长度应在 10-100 字之间，实际长度: {word_count} 字\n文案: {result['ad_copy']}"

        print(f"✓ 文案长度合理: {word_count} 字")

    @pytest.mark.parametrize("input_data,expected_product", [
        ({"product": "智能门锁", "feature": "指纹识别"}, "智能门锁"),
        ({"product": "空气净化器", "feature": "除甲醛"}, "空气净化器"),
        ({"product": "电动牙刷", "feature": "超声波清洁"}, "电动牙刷"),
        ({"product": "智能水杯", "feature": "记录饮水量"}, "智能水杯"),
        ({"product": "降噪耳机", "feature": "主动降噪"}, "降噪耳机"),
    ])
    def test_multiple_products(self, ollama_health_check, input_data, expected_product):
        """
        测试6：批量测试多种不同产品
        权重：加分项
        """
        result = generate_ad(input_data)

        # 验证输出结构
        assert 'ad_copy' in result
        assert 'word_count' in result

        # 验证字数一致性
        assert len(result['ad_copy']) == result['word_count'], \
            f"产品 '{expected_product}' 的字数统计不一致"

        # 验证产品名包含
        assert expected_product in result['ad_copy'], \
            f"产品 '{expected_product}' 的文案应包含产品名"

        # 验证长度合理
        assert 10 <= result['word_count'] <= 100, \
            f"产品 '{expected_product}' 的文案长度不合理: {result['word_count']}"

        print(f"✓ {expected_product}: {result['ad_copy'][:40]}... ({result['word_count']}字)")

    def test_template_consistency(self, ollama_health_check):
        """
        测试7：验证模板使用的一致性
        权重：额外测试
        """
        input_data = {"product": "智能手环", "feature": "运动追踪"}

        # 多次调用，验证模板名称一致
        result1 = generate_ad(input_data)
        result2 = generate_ad(input_data)

        assert 'template_used' in result1
        assert 'template_used' in result2

        # 模板名称应该是一致的
        assert result1['template_used'] == result2['template_used'], \
            "相同输入应使用相同的模板"

        print(f"✓ 模板一致性验证通过: {result1['template_used']}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
