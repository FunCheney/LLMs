import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载  中的环境变量
load_dotenv()


class HelloAgentsLLM:

    def __init__(self, model: str = None, base_url: str = None, api_key: str = None, timeout: int = None):
        """
            初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("MODE")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.api_key = api_key or os.getenv("API_KEY")
        self.timeout = timeout or os.getenv("TIMEOUT")

        if not all([self.model, self.base_url, self.api_key]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def think(self, message: List[Dict[str, str]], temperature: float = 0) -> str:

        print(f"🧠 正在调用 {self.model} 模型...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                stream=True
            )
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collect_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collect_content.append(content)
            print()  # 在流式输出结束后换行

            return "".join(collect_content)
        except Exception as e:
            print(e)
            return ""


if __name__ == "__main__":

    try:

        BASE_URL = "https://api-inference.modelscope.cn/v1/"
        MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
        llm = HelloAgentsLLM(MODEL_ID, BASE_URL, None, 60)

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llm.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except Exception as e:
        print(e)
