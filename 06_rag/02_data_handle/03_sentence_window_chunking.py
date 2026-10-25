"""
句子窗口检索的思想可以概括为：为检索精确性而索引小块，为上下文丰富性而检索大块。

其工作流程如下：

（1）索引阶段：在构建索引时，文档被分割成单个句子。每个句子都作为一个独立的“节点（Node）”存入向量数据库。同时，每个句子节点都会在元数据（metadata）中存储其上下文窗口，即该句子原文中的前N个和后N个句子。这个窗口内的文本不会被索引，仅仅是作为元数据存储。

（2）检索阶段：当用户发起查询时，系统会在所有单一句子节点上执行相似度搜索。因为句子是表达完整语义的最小单位，所以这种方式可以非常精确地定位到与用户问题最相关的核心信息。

（3）后处理阶段：在检索到最相关的句子节点后，系统会使用一个名为 MetadataReplacementPostProcessor 的后处理模块。该模块会读取到检索到的句子节点的元数据，并用元数据中存储的完整上下文窗口来替换节点中原来的单一句子内容。

（4）生成阶段：最后，这些被替换了内容的、包含丰富上下文的节点被传递给LLM，用于生成最终的答案。
"""
import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from llama_index.llms.deepseek import DeepSeek


Settings.llm = DeepSeek(model=os.getenv("DEEPSEEK_MODEL"), temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

documents = SimpleDirectoryReader(
    input_files=List["./data/《证券分析》本杰明•格雷厄姆.pdf"],
).load_data()

node_parser = SentenceWindowNodeParser.from_defaults(window_size=3, window_metadata_key="window",
                                                  original_text_metadata_key="original_text", )

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
sentence_nodes = node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(sentence_nodes)

# 3.2 常规分块索引 (基准)
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)

# 4. 构建查询引擎
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 5. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"查询: {query}\n")

print("--- 句子窗口检索结果 ---")
window_response = sentence_query_engine.query(query)
print(f"回答: {window_response}\n")

print("--- 常规检索结果 ---")
base_response = base_query_engine.query(query)
print(f"回答: {base_response}\n")
