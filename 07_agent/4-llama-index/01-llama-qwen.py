from http.client import responses

from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.dashscope import DashScope
from dotenv import load_dotenv
import os

from nltk.corpus.reader import documents

load_dotenv()

model = "qwen3.6-plus-2026-04-02"

api_key = os.getenv("QWEN_API_KEY")
api_base_url = os.getenv("QWEN_API_URL")

llm = DashScope(model=model, api_key=api_key, api_base_url=api_base_url,
                is_chat_model=True, max_tokens=1024)

response = llm.complete("推荐一个为期5天的北京旅游攻略")

print(response)

# 实现简单的 RAG 流程
Settings.llm = DashScope(model=model, api_key=api_key, api_base_url=api_base_url)

# 加载嵌入模型
Settings.embed_model = DashScopeEmbeddings(model="text-embedding-v4")

# 从文件目录加载文件，自动选择对应的文档加载器加载
documents = SimpleDirectoryReader(input_files=['data/test.txt']).load_data()

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 将索引转化为查询引擎
query_engine = index.as_query_engine()

responses = query_engine.query("公司的上下班时间")
print(responses)
