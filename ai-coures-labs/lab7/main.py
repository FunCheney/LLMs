"""
实验7：RAG系统的检索与生成验证
学生需要构建完整的RAG系统，学习文档加载、向量化、检索和生成的全流程
"""
# 提示：需要导入 LangChain 相关模块
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from typing import List, Dict
import os


def ask_rag(question: str, collection_name: str = "default") -> dict:
    """
    使用 RAG 系统回答问题

    参数:
        question: 用户提出的问题
        collection_name: ChromaDB 集合名称（默认 "default"）

    返回:
        字典，包含以下键:
        - answer (str): 生成的答案
        - source_chunks (List[str]): 检索到的文档块列表
        - similarity_scores (List[float]): 相似度分数列表
        - has_answer (bool): 是否找到答案
    """
    # 1. 加载文档
    path_langchain_intro = "../test_data/knowledge_base/langchain_intro.txt"
    path_prompt = "../test_data/knowledge_base/prompt_engineering.txt"
    vector_db = "../test_data/knowledge_base/vector_db.txt"
    documents = load_documents(path_langchain_intro)
    prompt = load_documents(path_prompt)
    vector = load_documents(vector_db)

    # 2. 分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    info = text_splitter.split_documents(documents)
    prompt_documents = text_splitter.split_documents(prompt)
    vector_documents = text_splitter.split_documents(vector)
    doc_list = info + prompt_documents + vector_documents
    vec_store = create_vector_store(doc_list, collection_name)
    # 4. 检索相关文档
    result = retrieve_documents(question, vec_store, top_k=3)
    clean_result = [item[0] if isinstance(item, tuple) else item for item in result]

    # 5. 生成答案
    answer = generate_answer(question, clean_result)

    # 返回结果
    return {
        "answer": answer,
        "source_chunks": [doc['text'] for doc in result],
        "similarity_scores": [score for _, score in result],
        "has_answer": len(result) > 0
    }


def load_documents(doc_dir: str) -> List[Document]:
    """
    加载文档目录

    参数:
        doc_dir: 文档目录路径

    返回:
        Document 对象列表
    """
    loader = TextLoader(doc_dir)
    documents = loader.load()
    # print(type(documents))
    # 确保每个加载的内容是 Document 类型
    if isinstance(documents[0], tuple):
        # 如果是元组，提取 content 并构造 Document 对象
        documents = [Document(page_content=doc[0], metadata=doc[1]) for doc in documents]

    # 调试：打印加载的文档内容
    # print(f"加载的文档类型: {type(documents)}")
    # if documents:
    #     print(f"第一个文档内容: {documents[0].page_content[:100]}...")  # 打印第一个文档的内容

    return documents


def create_vector_store(documents: List, collection_name: str = "default"):
    """
    创建向量存储

    参数:
        documents: Document 对象列表
        collection_name: 集合名称

    返回:
        ChromaDB 向量存储实例
    """
    persist_directory = "./chroma_db"
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
    return vector_store.similarity_search_with_score(query, k=top_k)



def generate_answer(query: str, retrieved_docs: List) -> str:
    """
    基于检索到的文档生成答案

    参数:
        query: 用户问题
        retrieved_docs: 检索到的文档列表

    返回:
        生成的答案文本
    """
    # 创建临时向量存储（仅用于本次查询）
    vectorstore = Chroma.from_documents(
        documents=retrieved_docs,
        embedding=OllamaEmbeddings(model="llama3", base_url="http://localhost:11434")
    )

    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": len(retrieved_docs)})

    # 创建自定义提示模板
    prompt_template = """基于以下提供的上下文信息，请回答问题。如果你不知道答案，就说不知道，不要编造信息。

    上下文信息：
    {context}

    问题：{question}

    请根据上下文信息提供准确、简洁的答案：
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 创建 RetrievalQA 链
    qa_chain = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="llama3", temperature=0),
        retriever=retriever,
        chain_type="stuff",  # 使用stuff链类型
        chain_type_kwargs={"prompt": PROMPT}  # 传递自定义的prompt
    )

    # 生成答案
    result = qa_chain.invoke({"query": query})
    print(result)
    return result["result"]



def calculate_similarity_scores(retrieved_docs_with_scores: List[tuple[Document, float]]) -> List[float]:
    """将距离转为 0-1 相似度并降序排列"""
    # Chroma 返回的是距离 (distance)，越小越相似
    # 转换为相似度分数：similarity = 1 / (1 + distance)
    scores = [1 / (1 + dist) for _, dist in retrieved_docs_with_scores]
    # 降序排列，让最高的分数在前面
    return sorted(scores, reverse=True)

# 测试代码
if __name__ == "__main__":
    # 确保知识库目录存在
    kb_dir = "../test_data/knowledge_base"
    if not os.path.exists(kb_dir):
        print(f"警告: 知识库目录 {kb_dir} 不存在")
        print("请先创建知识库文档再进行测试")
    else:
        # 测试1：LangChain相关问题
        print("=== 测试1：LangChain相关问题 ===")
        try:
            result1 = ask_rag("LangChain是什么？")
            print(f"答案: {result1['answer'][:100]}...")
            print(f"找到相关内容: {result1['has_answer']}")
            print(f"相关文档块数量: {len(result1['source_chunks'])}")
            print(f"相似度分数: {result1['similarity_scores']}")
        except Exception as e:
            print(f"错误: {e}")

        # 测试2：向量数据库相关问题
        print("\n=== 测试2：向量数据库相关问题 ===")
        try:
            result2 = ask_rag("向量数据库的主要用途是什么？")
            print(f"答案: {result2['answer'][:100]}...")
            print(f"找到相关内容: {result2['has_answer']}")
            print(f"检索到的文档块:")
            for i, chunk in enumerate(result2['source_chunks'][:2], 1):
                print(f"  {i}. {chunk[:80]}...")
        except Exception as e:
            print(f"错误: {e}")

        # 测试3：知识库外问题
        print("\n=== 测试3：知识库外问题 ===")
        try:
            result3 = ask_rag("今天的天气怎么样？")
            print(f"答案: {result3['answer'][:100]}...")
            print(f"找到相关内容: {result3['has_answer']}")
            print(f"相关文档块数量: {len(result3['source_chunks'])}")
        except Exception as e:
            print(f"错误: {e}")

        # 测试4：Prompt工程相关问题
        print("\n=== 测试4：Prompt工程相关问题 ===")
        try:
            result4 = ask_rag("Prompt工程的技术有哪些？")
            print(f"答案: {result4['answer'][:100]}...")
            print(f"相似度分数: {result4['similarity_scores']}")
            print(f"检索到的文档块数量: {len(result4['source_chunks'])}")
        except Exception as e:
            print(f"错误: {e}")