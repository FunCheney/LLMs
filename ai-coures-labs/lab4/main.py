"""
实验4：LangChain 链的确定性输出
学生需要使用 LangChain 的 PromptTemplate 和 LLMChain 生成广告文案
"""


# 提示：需要导入 LangChain 相关模块
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Ollama
# from langchain.chains import LLMChain (或新版本的等效导入)


def generate_ad(input_dict: dict) -> dict:
    """
    使用 LangChain 链生成广告文案

    参数:
        input_dict: 字典，包含:
            - product (str): 产品名称
            - feature (str): 核心特性

    返回:
        字典，包含:
        - ad_copy (str): 生成的广告文案
        - word_count (int): 文案字数（使用 len(ad_copy) 统计）
        - template_used (str): 使用的模板名称

    实现要求:
        1. 使用 PromptTemplate 定义广告文案生成模板
        2. 创建 LLMChain，连接 PromptTemplate 和 Ollama LLM
        3. 链的输出必须经过后处理，提取文案并计算字数
        4. 模板必须包含 {product} 和 {feature} 两个变量
        5. word_count 必须与 ad_copy 的实际长度一致
    """
    # TODO: 学生需要实现此函数
    #
    # 实现步骤建议:
    # 1. 创建 PromptTemplate，包含产品和特性变量
    # 2. 创建 Ollama LLM 实例
    # 3. 创建 LLMChain，连接 Prompt 和 LLM
    # 4. 运行链，传入产品和特性参数
    # 5. 提取文案，计算字数
    # 6. 返回结构化结果

    # 提示：
    # 1. 创建 PromptTemplate，包含产品和特性变量
    # 2. 创建 Ollama LLM 实例
    # 3. 创建 LLMChain，连接 Prompt 和 LLM
    # 4. 运行链，传入产品和特性参数
    # 5. 提取文案，计算字数
    # 6. 返回结构化结果

    raise NotImplementedError("请实现 generate_ad 函数")


# 测试代码（可选，用于学生本地调试）
if __name__ == "__main__":
    # 测试基础版本
    print("=== 测试基础广告生成 ===")
    test_inputs = [
        {"product": "智能手表", "feature": "心率监测"},
        {"product": "无线耳机", "feature": "降噪功能"},
        {"product": "扫地机器人", "feature": "自动避障"}
    ]

    for input_data in test_inputs:
        try:
            result = generate_ad(input_data)
            print(f"\n产品: {input_data['product']}")
            print(f"文案: {result['ad_copy']}")
            print(f"字数: {result['word_count']}")
            print(f"模板: {result['template_used']}")
        except Exception as e:
            print(f"错误: {e}")
