"""
实验1：结构化提示词与输出
学生需要实现 classify_text 函数，使用 Pydantic 模型返回结构化的文本分类结果
"""
from typing import List
from pydantic import BaseModel, Field
import httpx
import json


class TextClassification(BaseModel):
    """
    文本分类结果的数据模型
    """
    category: str = Field(
        ...,
        description="文本分类类别，必须是以下之一：'新闻', '技术', '体育', '娱乐', '财经'"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="分类置信度，范围0.0-1.0"
    )
    keywords: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="从文本中提取的1-5个关键词"
    )


def classify_text(text: str) -> TextClassification:
    """
    对输入文本进行分类，返回结构化的分类结果

    参数:
        text: 待分类的文本内容

    返回:
        TextClassification 实例，包含分类类别、置信度和关键词

    实现要求:
        1. 使用 Ollama API 调用 qwen3:8b 模型
        2. 设计结构化 Prompt，要求模型输出 JSON 格式
        3. 解析模型输出并验证为 TextClassification 模型
        4. category 必须是预定义的5个类别之一
        5. confidence_score 必须在 0-1 范围内
        6. keywords 列表长度为 1-5

    提示:
        - 在 Prompt 中明确指定输出格式和有效类别
        - 可以使用 Few-Shot 示例提高输出稳定性
        - 使用 Pydantic 的自动验证确保数据有效性
    """
    # 实现步骤建议:
    # 1. 构建结构化 Prompt，要求模型输出 JSON 格式
    # 2. 调用 Ollama API (http://localhost:11434/api/generate)
    # 3. 解析响应中的 JSON 字符串
    # 4. 使用 TextClassification.model_validate() 创建实例
    # 5. 返回验证后的 Pydantic 模型实例

    # 示例 Prompt 结构（学生需要完善）:
    prompt = f"""请对以下文本进行分类,并按照 json 的格式输出
    文本内容: "{text}"
    要求：
        1. 分类类别必须是以下五种之一：新闻、技术、体育、娱乐、财经
        2. 提供分类置信度（0.0-1.0之间的小数）
        3. 提取1-5个最能代表文本内容的关键词
        
        请严格按照以下JSON格式输出，不要添加任何其他内容：
        {{
            "category": "分类类别",
            "confidence_score": 置信度,
            "keywords": ["关键词1", "关键词2", ...]
        }}
    
    示例1：
        文本："苹果公司发布新款iPhone，搭载A18芯片"
        输出：{{"category": "技术", "confidence_score": 0.95, "keywords": ["苹果", "iPhone", "A18芯片", "发布"]}}
    
    示例2：
        文本："中国男篮在亚运会夺得冠军"
        输出：{{"category": "体育", "confidence_score": 0.92, "keywords": ["中国男篮", "亚运会", "冠军"]}}
    
    现在请分析给定的文本并输出JSON格式的结果：
    """

    # print(prompt)

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1  # 降低随机性，提高输出稳定性
                    }
                }
            )
            response.raise_for_status()

            # 解析响应
            result_data = response.json()
            response_text = result_data.get("response", "").strip()
            # print(response_text)
            # 尝试从响应中提取JSON（模型可能在响应前后添加了其他内容）
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                classification_data = json.loads(json_str)
            else:
                # 如果没有找到完整的JSON，尝试直接解析整个响应
                classification_data = json.loads(response_text)

            # 创建并验证 Pydantic 模型实例
            classification_result = TextClassification.model_validate(classification_data)

            return classification_result

    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析模型输出的JSON: {e}\n原始响应: {response_text}")
    except httpx.RequestError as e:
        raise ConnectionError(f"调用Ollama API失败: {e}")
    except Exception as e:
        raise RuntimeError(f"分类过程中发生错误: {e}")


# 测试代码（可选，用于学生本地调试）
if __name__ == "__main__":
    # 测试示例
    test_texts = [
        "OpenAI发布GPT-5，性能提升10倍",
        "中国队在巴黎奥运会夺得金牌",
        "A股市场今日大涨，沪指突破3000点"
    ]

    for text in test_texts:
        try:
            result = classify_text(text)
            print(f"\n文本: {text}")
            print(f"分类: {result.category}")
            print(f"置信度: {result.confidence_score}")
            print(f"关键词: {result.keywords}")
        except Exception as e:
            print(f"错误: {e}")
