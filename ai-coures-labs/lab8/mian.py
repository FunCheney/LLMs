"""
实验8：人工介入的触发逻辑
学生需要在RAG系统基础上增加质量控制机制，当系统置信度不足时自动触发人工审核流程
"""
import os

# 提示：需要导入实验7的函数和 LangChain 相关模块
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from typing import List



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
    try:
        score = vector_store.similarity_search_with_score(query, k=top_k)
        print("retrieve_documents 成功")
        return score
    except Exception as e:
        print(e)
    return []


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




def ask_rag_with_review(question: str, confidence_threshold: float = 0.7) -> dict:
    """
    带人工审核机制的 RAG 系统

    参数:
        question: 用户提出的问题
        confidence_threshold: 触发人工审核的置信度阈值（默认 0.7）

    返回:
        字典，包含以下键:
        - status (str): 响应状态（"success" / "low_confidence" / "review_needed"）
        - answer (str): 生成的答案（status=review_needed时可为空）
        - confidence_score (float): 综合置信度分数（0-1）
        - review_reason (str): 需要审核的原因（仅当status=review_needed时）
        - source_chunks (list[str]): 检索到的文档块

    实现要求:
        1. 复用实验7的RAG系统进行检索和生成
        2. 根据以下维度评估置信度：
           - 检索相似度（权重50%）：Top-1文档的相似度分数
           - 文档覆盖度（权重30%）：相似度>0.5的文档数量
           - 答案长度（权重20%）：生成答案的字符数
        3. 置信度判定逻辑：
           - 若 Top-1 相似度 < 0.7，置信度判定为低
           - 若相似度>0.5的文档数 < 2，置信度判定为中
           - 否则置信度为高
        4. 根据置信度返回不同状态：
           - 高置信度：status="success"
           - 中置信度：status="low_confidence"
           - 低置信度：status="review_needed"

    提示:
        - 使用 similarity_search_with_score() 获取相似度分数
        - 计算综合置信度分数（0-1范围）
        - 当 confidence_score < confidence_threshold 时触发审核
    """
    # TODO: 实现带人工审核机制的 RAG 系统
    # 提示:

    # 1. 调用实验7的 ask_rag 函数获取 RAG 结果
    try:
        # 1.1 加载文档
        documents = load_documents("../test_data/knowledge_base")
        # 1.2 创建存储向量
        vector_store = create_vector_store(documents, collection_name="default")
        # 1.3 检索
        retrieved_docs_with_scores = retrieve_documents(question, vector_store, top_k=3)
        # print(retrieved_docs_with_scores)
        source_chunks = [doc.page_content for doc, _ in retrieved_docs_with_scores]
        similarity_scores = calculate_similarity_scores(retrieved_docs_with_scores)
        # 1.4 生成答案
        answer = generate_answer(question, [doc for doc, _ in retrieved_docs_with_scores])
        # 2. 计算综合置信度（基于检索相似度、文档覆盖度、答案长度）
        confidence_score = calculate_confidence(similarity_scores, source_chunks, answer)
        # 3. 根据置信度判定响应状态（success / low_confidence / review_needed）
        review = should_trigger_review(confidence_score, confidence_threshold)
        if review:
            review_reason = generate_review_reason(similarity_scores, source_chunks, answer)
        else:
            review_reason = ""
        # 4. 返回包含 status, confidence_score, answer/review_reason 的字典
        status = determine_status(confidence_score, confidence_threshold, similarity_scores)

        return {
            "status": status,
            "answer": answer,
            "confidence_score": confidence_score,
            "review_reason": review_reason,
        }
    except Exception as e:
        print(e)

    return {}




# 辅助函数示例
def calculate_confidence(similarity_scores: List[float], chunks: List[str], answer: str) -> float:
    """
    计算综合置信度分数（0-1）

    评估维度:
        - 检索相似度（50%）：Top-1文档的相似度分数
        - 文档覆盖度（30%）：相似度>0.5 的文档数量（归一化到 0-1）
        - 答案长度（20%）：答案字符数（归一化到 0-1，以 10 字为满分）
    """

    # -----------------------------
    # 1. 检索相似度（50%）
    # -----------------------------
    if similarity_scores:
        top1_score = max(similarity_scores)
    else:
        top1_score = 0.0

    sim_conf = top1_score  # 本身就是 0-1 区间

    # -----------------------------
    # 2. 文档覆盖度（30%）
    # -----------------------------
    if similarity_scores:
        high_conf_docs = sum(1 for s in similarity_scores if s > 0.5)
        # 归一化：高于阈值的文档数量 / 总文档数
        coverage_conf = high_conf_docs / len(similarity_scores)
    else:
        coverage_conf = 0.0

    # -----------------------------
    # 3. 答案长度（20%）
    # -----------------------------
    # 规则：长度 >= 10 字 → 1.0；短于 10 字按比例
    if answer:
        length_conf = min(len(answer) / 10, 1.0)
    else:
        length_conf = 0.0

    # -----------------------------
    # 综合得分
    # -----------------------------
    confidence = (
        sim_conf * 0.5 +
        coverage_conf * 0.3 +
        length_conf * 0.2
    )

    return round(confidence, 4)

def should_trigger_review(confidence_score: float, threshold: float) -> bool:
    """
    当综合置信度 < threshold 时触发人工审核
    """
    return confidence_score < threshold


def determine_confidence_level(similarity_scores: List[float]) -> str:
    """
    根据你定义的规则判断：low / medium / high
    """

    if not similarity_scores:
        return "low"

    top1 = max(similarity_scores)
    high_docs = sum(1 for s in similarity_scores if s > 0.5)

    # ---- 置信度规则（你定义的） ----
    if top1 < 0.7:
        return "low"
    elif high_docs < 2:
        return "medium"
    else:
        return "high"


def determine_status(confidence_score: float, threshold: float, similarity_scores: List[float]) -> str:
    """
    返回最终状态：success / low_confidence / review_needed
    """
    # 1) 根据规则判断置信度等级
    level = determine_confidence_level(similarity_scores)

    if level == "high":
        return "success"
    elif level == "medium":
        return "low_confidence"
    else:  # "low"
        return "review_needed"


def generate_review_reason(similarity_scores: List[float], chunks: List[str], answer: str) -> str:
    """
    当置信度低时，生成原因
    """
    if not similarity_scores:
        return "未检索到相关文档，相似度评分为空"

    top1 = max(similarity_scores)
    high_docs = sum(1 for s in similarity_scores if s > 0.5)

    # ---- 按你的逻辑生成审核原因 ----
    if top1 < 0.7:
        return f"Top-1 相似度偏低（{top1:.2f} < 0.70）"

    if high_docs < 2:
        return f"相似度高于 0.5 的文档数量不足（仅 {high_docs} 个）"

    return "置信度不足"

# 测试代码
if __name__ == "__main__":
    # 确保知识库目录存在
    kb_dir = "../test_data/knowledge_base"
    if not os.path.exists(kb_dir):
        print(f"警告: 知识库目录 {kb_dir} 不存在")
        print("请先创建知识库文档再进行测试")
    else:
        # 测试1：高置信度场景
        print("=== 测试1：高置信度场景 ===")
        try:
            result1 = ask_rag_with_review("LangChain框架的主要用途是什么？")
            print(f"状态: {result1['status']}")
            print(f"置信度: {result1['confidence_score']:.2f}")
            print(f"答案: {result1['answer'][:80]}...")
        except Exception as e:
            print(f"错误: {e}")

        # 测试2：低置信度触发审核
        print("\n=== 测试2：低置信度触发审核 ===")
        try:
            result2 = ask_rag_with_review("那个东西怎么用？")
            print(f"状态: {result2['status']}")
            print(f"置信度: {result2['confidence_score']:.2f}")
            if 'review_reason' in result2:
                print(f"审核原因: {result2['review_reason']}")
        except Exception as e:
            print(f"错误: {e}")

        # 测试3：中等置信度警告
        print("\n=== 测试3：中等置信度警告 ===")
        try:
            result3 = ask_rag_with_review("向量数据库和关系型数据库的区别？")
            print(f"状态: {result3['status']}")
            print(f"置信度: {result3['confidence_score']:.2f}")
            print(f"答案: {result3['answer'][:80]}...")
        except Exception as e:
            print(f"错误: {e}")

        # 测试4：自定义阈值
        print("\n=== 测试4：自定义高阈值 ===")
        try:
            result4 = ask_rag_with_review("Prompt工程相关内容", confidence_threshold=0.9)
            print(f"状态: {result4['status']}")
            print(f"置信度: {result4['confidence_score']:.2f}")
            print(f"相关文档块数量: {len(result4['source_chunks'])}")
        except Exception as e:
            print(f"错误: {e}")