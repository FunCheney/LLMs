import os
# 如果需要，可继续使用HuggingFace镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 从新的包导入 uv add langchain-experimental
# LangChain 将一些实验性、可能不稳定或存在潜在问题的组件移到了 langchain-experimental 包中，以保持主包的精简和稳定。
# SemanticChunker 因为依赖 Embedding 模型进行语义判断，属于这类实验性功能，所以不在主 langchain 包中。
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
# 从 langchain 0.2.0 版本开始，官方对包结构进行了重构。
# TextLoader 这类文档加载器被从核心的 langchain 包中移出，迁移到了独立的集成包 langchain-community
from langchain_community.document_loaders import TextLoader
# 使用新的HuggingFaceEmbeddings类（参数用法不变）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
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