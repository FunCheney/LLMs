import os
# 从新的包导入
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# 如果需要，可继续使用HuggingFace镜像加速
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 使用新的HuggingFaceEmbeddings类（参数用法不变）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 初始化SemanticChunker，注意这里使用的是中文标点
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    sentence_split_regex=r'(?<=[。！？])'  # 关键修复：为中文文本设置正确的分句正则
)

# 加载并分割文档
loader = TextLoader("./data/蜂医.txt", encoding='utf-8')
documents = loader.load()
docs = text_splitter.split_documents(documents)

print(len(docs))

print(docs)