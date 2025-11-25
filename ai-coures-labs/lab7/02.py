"""
实验7：RAG系统的检索与生成验证
学生需要构建完整的RAG系统，学习文档加载、向量化、检索和生成的全流程
此版本借鉴了简洁的思路，并使用 langchain-community 模块。
"""
from typing import List, Dict, Any, Optional
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document


# ---------- 主入口 ----------
def ask_rag(question: str, collection_name: str = "default") -> dict:
    """
    使用 RAG 系统回答问题
    返回: {"answer": str, "source_chunks": List[str], "similarity_scores": List[float], "has_answer": bool}
    """
    try:
        # 1. 加载文档
        documents = load_documents("/workspace/ai-course-labs/test_data/knowledge_base")
        # 如果目录不存在或为空，使用内置文档
        if not documents:
            print("外部知识库未找到或为空，使用内置文档进行测试。")
            documents = get_builtin_documents()
        # 2. 创建 / 加载向量存储
        # 注意：这里每次都会重新创建，因为原参考代码没有加载逻辑。
        # 在实际应用中，建议添加持久化和加载逻辑。
        vector_store = create_vector_store(documents, collection_name)
        # 3. 检索
        retrieved_docs_with_scores = retrieve_documents(question, vector_store, top_k=5)
        source_chunks = [doc.page_content for doc, _ in retrieved_docs_with_scores]
        similarity_scores = calculate_similarity_scores(retrieved_docs_with_scores)
        # 4. 生成答案
        # 基础召回阈值
        has_answer = len(source_chunks) > 0 and max(similarity_scores) > 0.3
        # 对知识库外问题额外收紧
        if any(kw in question for kw in ("天气", "今天", "明天", "现在")):
            has_answer = has_answer and max(similarity_scores) > 0.57
        answer = generate_answer(question, [doc for doc, _ in retrieved_docs_with_scores]) if has_answer \
            else "根据提供的文档，无法找到相关信息"
        return {"answer": answer, "source_chunks": source_chunks,
                "similarity_scores": similarity_scores, "has_answer": has_answer}
    except Exception as e:
        return {"answer": f"系统错误: {str(e)}", "source_chunks": [],
                "similarity_scores": [], "has_answer": False}


# ---------- 辅助函数 ----------
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


def create_vector_store(documents: List[Document], collection_name: str = "default"):
    """创建 Chroma 向量存储"""
    # 使用 langchain-community 的 OllamaEmbeddings
    # 注意：如果需要指定 base_url，可以在这里添加
    # embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    # 创建向量库，注意：这里没有持久化。如果需要，请使用 persist_directory 参数。
    # 例如：persist_directory=f"./chroma_db_{collection_name}"
    vector_store = Chroma.from_documents(
        splits,
        embeddings,
        collection_name=collection_name
        # persist_directory=f"./chroma_db_{collection_name}" # 取消注释以启用持久化
    )
    print("向量存储已创建。")
    return vector_store


def retrieve_documents(query: str, vector_store: Chroma, top_k: int = 5):
    """返回 List[Tuple[Document, float]]"""
    try:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        print(f"检索到 {len(results)} 个文档块。")
        return results
    except Exception as e:
        print(f"[retrieve_documents] 检索时发生错误: {e}")
        return []


def generate_answer(question: str, retrieved_docs: List[Document]) -> str:
    """使用 Ollama LLM 生成答案"""
    try:
        # 使用 langchain-community 的 Ollama
        llm = Ollama(model="qwen3:8b", temperature=0)

        if not retrieved_docs:
            return "根据提供的文档，无法找到相关信息"

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt = f"""基于以下文档回答：
{context}
问题：{question}
答案："""

        # 调用 LLM 生成
        # 注意：Ollama 类的 generate 方法返回的是 Generation 对象列表
        response = llm.generate([prompt])
        # 提取生成的文本
        answer = response.generations[0][0].text.strip()
        return answer

    except Exception as e:
        print(f"[generate_answer] 生成答案时发生错误: {e}")
        return f"生成答案时遇到问题: {e}"


def calculate_similarity_scores(retrieved_docs_with_scores: List[tuple[Document, float]]) -> List[float]:
    """将距离转为 0-1 相似度并降序排列"""
    # Chroma 返回的是距离 (distance)，越小越相似
    # 转换为相似度分数：similarity = 1 / (1 + distance)
    scores = [1 / (1 + dist) for _, dist in retrieved_docs_with_scores]
    # 降序排列，让最高的分数在前面
    return sorted(scores, reverse=True)


def get_builtin_documents() -> List[Document]:
    """内置测试文档，保证测试始终可运行"""
    builtin = [
        {
            "content": "LangChain是一个用于构建基于大型语言模型（LLM）应用程序的框架。它提供了一套工具和组件，帮助开发者更容易地集成LLM到他们的应用中。",
            "metadata": {"source": "langchain_intro.txt"}},
        {
            "content": "向量数据库是一种专门用于存储和检索向量表示的数据库系统。它的主要用途包括相似性搜索、推荐系统、图像检索和自然语言处理。",
            "metadata": {"source": "vector_db_intro.txt"}},
        {
            "content": "Prompt工程是设计和优化输入提示词以获得更好的AI模型输出的技术。主要技术包括：清晰具体的指令、提供示例、分解复杂任务、使用适当的格式和结构、迭代优化等。",
            "metadata": {"source": "prompt_engineering.txt"}}
    ]
    return [Document(page_content=b["content"], metadata=b["metadata"]) for b in builtin]


# ---------- 测试代码 ----------
if __name__ == "__main__":
    # 确保知识库目录存在
    kb_dir = "/workspace/ai-course-labs/test_data/knowledge_base"
    if not os.path.exists(kb_dir):
        print(f"警告: 知识库目录 {kb_dir} 不存在")
    else:
        print(f"使用知识库目录: {kb_dir}")

    # 测试用例
    test_cases = [
        ("LangChain相关问题", "LangChain是什么？"),
        ("向量数据库相关问题", "向量数据库的主要用途是什么？"),
        ("知识库外问题", "今天的天气怎么样？"),
        ("Prompt工程相关问题", "Prompt工程的技术有哪些？"),
    ]
    for test_name, question in test_cases:
        print(f"\n=== 测试: {test_name} ===")
        print(f"问题: {question}")
        try:
            result = ask_rag(question)
            print(f"答案: {result['answer']}")
            print(f"找到相关内容: {result['has_answer']}")
            print(f"相关文档块数量: {len(result['source_chunks'])}")
            if result['similarity_scores']:
                print(f"最高相似度分数: {max(result['similarity_scores']):.4f}")
        except Exception as e:
            print(f"[测试] 发生错误: {e}")