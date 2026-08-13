import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv(encoding='utf-8')

chat_model = init_chat_model(
    model=os.getenv("DEEPSEEK_MODEL"),
    model_provider='openai',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    max_tokens=100
)

response = chat_model.invoke("写一句关于秋天的词，14 字以内")
print(response)
print(type(response))
print(response.content)
print(type(response.content))


# 多次调用观察参数效果（如 temperature 对多样性的影响）
for i in range(3):
    print(f'---- 第 {i + 1} 次调用 ----')
    print(chat_model.invoke("写一句关于秋天的词，14 字以内").content)


