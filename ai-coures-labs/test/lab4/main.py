import json
import re


from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_classic.chains import LLMChain

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
    """
    # 提取输入数据
    product = input_dict["product"]
    feature = input_dict["feature"]

    if not product or not feature:
        raise ValueError("输入字典必须包含 'product' 和 'feature' 键")

    # 1. 创建 PromptTemplate，包含产品和特性变量
    prompt_template = PromptTemplate(
        input_variables=[product, feature],
        template="""
        要求：
        根据产品名称和产品特性生成文案，广告文案中要包含产品名称。
        输出：严格按照以下JSON格式输出，不要添加任何其他内容:
        {{
            "product": "{product}",
            "feature": "{feature}",
            "ad_copy": "广告文案",
            "template_used": "模板名称"
        }}

        其中，ad_copy 是针对产品和特性生成的广告文案,其中要包含产品的名称 {product} 与核心特性 {feature}，
            template_used 是你使用的广告文案模板的名称。
        """
    )

    # 2. 创建 Ollama LLM 实例
    llm = Ollama(model="llama3:latest")

    # 3. 创建 LLMChain，连接 PromptTemplate 和 Ollama LLM
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=False  # 设置为 True 可以查看详细执行过程
    )

    # 4. 运行链，传入产品和特性参数
    try:
        response = chain.invoke(input_dict)
    except Exception as e:
        raise RuntimeError(f"调用链时出错: {e}")

    # 5. 提取生成的广告文案及模板
    try:
        # print("re: {}", response)
        response_text = str(response.get("text", "").strip())

        # 1. 提取 JSON 部分：可以通过正则匹配 JSON 结构部分
        json_part = re.search(r'\{.*\}', response_text, re.DOTALL).group(0)

        # 2. 替换单引号为双引号
        json_part_fixed = json_part.replace("'", '"')

        # 3. 修复 template_used 字段格式问题（由于模板值为字符串，且其值没有正确的双引号）
        json_part_fixed = re.sub(r'"template_used":\s*"[^"]+"', r'"template_used": "Product Feature Focus"',
                                 json_part_fixed)

        # 4. 加载为 JSON 对象
        try:
            data = json.loads(json_part_fixed)
            print(data)
            # 提取 ad_copy
            ad_copy = data.get("ad_copy")
            print("ad_copy:", ad_copy)
            return {
                "ad_copy": ad_copy,
                "word_count": len(ad_copy),
                "template_used": data.get("template_used"),
            }
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

    except (KeyError, SyntaxError) as e:
        raise RuntimeError(f"解析响应时出错: {e}")


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
