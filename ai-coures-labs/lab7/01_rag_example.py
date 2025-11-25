

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from typing import List, Dict, Any, Optional
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings






def load_data():
    path_langchain_intro = "../test_data/knowledge_base/langchain_intro.txt"
    loader = TextLoader(path_langchain_intro)
    langchain_intro = loader.load()

    path_prompt = "../test_data/knowledge_base/prompt_engineering.txt"
    loader = TextLoader(path_prompt)
    prompt_info = loader.load()

    vector_db = "../test_data/knowledge_base/vector_db.txt"
    loader = TextLoader(vector_db)
    vector_info = loader.load()



def load_documents(doc_dir: str) -> List[Document]:
    """加载目录下所有 txt 文件"""
    if not os.path.exists(doc_dir):
        return []

    try:
        loader = DirectoryLoader(
            path=doc_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"成功加载 {len(documents)} 个文档。")
        return documents
    except Exception as e:
        print(f"[load_documents] 加载文档时发生错误: {e}")
        return []



def create_vector_store(documents: List, collection_name: str = "default"):
    """
    创建向量存储

    参数:
        documents: Document 对象列表
        collection_name: 集合名称

    返回:
        ChromaDB 向量存储实例
    """
    persist_directory = "./chroma_db_test"
    vect_store = Chroma.from_documents(
        documents,
        embedding=OllamaEmbeddings(
        model="llama3:latest",
        base_url="http://localhost:11434"),
        persist_directory=persist_directory,
        collection_name=collection_name)

    return vect_store

def retrieve_documents(query: str, vector_store, top_k: int = 3) -> List:
    """
    检索相关文档

    参数:
        query: 查询文本
        vector_store: 向量存储实例
        top_k: 返回的文档数量

    返回:
        检索到的 Document 对象列表
    """
    try:
        score = vector_store.similarity_search_with_score(query, k=top_k)
        print("retrieve_documents 成功")
        return score
    except Exception as e:
        print(e)
    return []



if __name__ == '__main__':
    documents = load_documents("../test_data/knowledge_base/")
    store = create_vector_store(documents, collection_name="test")
    l = retrieve_documents("langchain 是什么", store)
    print(l)
