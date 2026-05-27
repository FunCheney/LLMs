# 嵌入
# 嵌入是表示学习的一种形式，通常用于将高纬度数据映射到低微空间中的表示形式。嵌入可以是词嵌入，图像嵌入等。

from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv(verbose=True)



# 使用在线大模型
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_URL"))

def get_embedding(text, model = "text-"):

    data = client.embeddings.create(input=text, model=model).data

    return [x.embedding for x in data]


vec = get_embedding(["苹果"])
print(f'第一个向量 {vec[0]}')
print(f'向量的纬度 {len(vec)}')

# 使用本地