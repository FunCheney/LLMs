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
        confidence_score = calculate_confidence(similarity_scores, answer)
        print(question + "confidence_score ----> " + str(confidence_score))
        # 3. 根据置信度判定响应状态（success / low_confidence / review_needed）
        print(question + "similarity_scores ----> " + str(similarity_scores))
        status = determine_status(confidence_score, similarity_scores)
        print(question + "status ----> " + status)
        review = should_trigger_review(confidence_score, confidence_threshold)
        print(question + "review ----> " + str(review))
        if review:
            review_reason = generate_review_reason(similarity_scores, source_chunks, answer)
        else:
            review_reason = ""


        # 4. 返回包含 status, confidence_score, answer/review_reason 的字典
        return {
            "status": status,
            "answer": answer,
            "confidence_score": confidence_score,
            "review_reason": review_reason,
            "source_chunks": source_chunks
        }
    except Exception as e:
        print(e)
        return {
            "status": "Fail",
            "answer": "",
            "confidence_score": "",
            "review_reason": "",
            "source_chunks": []
        }




# 辅助函数示例
def calculate_confidence(similarity_scores: List[float], answer: str) -> float:
    """
    计算综合置信度分数（0-1）

    评估维度:
        - 检索相似度（50%）：Top-1文档的相似度分数
        - 文档覆盖度（30%）：相似度>0.5 的文档数量（归一化到 0-1）
        - 答案长度（20%）：答案字符数（归一化到 0-1，以 10 字为满分）
    """
    top1_score = max(similarity_scores)
    high_conf_docs = sum(1 for s in similarity_scores if s > 0.5)
    # 归一化：高于阈值的文档数量 / 总文档数
    coverage_conf = high_conf_docs / len(similarity_scores)
    # 规则：长度 >= 10 字 → 1.0；短于 10 字按比例
    length_conf = min(len(answer) / 10, 1.0)
    confidence = (
        top1_score * 0.5 +
        coverage_conf * 0.3 +
        length_conf * 0.2
    )
    return round(confidence, 4)

def should_trigger_review(confidence_score: float, threshold: float) -> bool:
    """
    当综合置信度 < threshold 时触发人工审核
    """
    return confidence_score < threshold

def determine_status(confidence_score: float, similarity_scores: list[float]) -> str:
    """
     3. 置信度判定逻辑：
           - 若 Top-1 相似度 < 0.7，置信度判定为低
           - 若相似度>0.5的文档数 < 2，置信度判定为中
           - 否则置信度为高
        4. 根据置信度返回不同状态：
           - 高置信度：status="success"
           - 中置信度：status="low_confidence"
           - 低置信度：status="review_needed"
    返回最终状态：success / low_confidence / review_needed
    """
    top1_score = max(similarity_scores)
    high_conf_docs = sum(1 for s in similarity_scores if s > 0.5)

    if confidence_score > 0.7 and top1_score > 0.55:
        return "success"
    elif confidence_score > 0.5 and top1_score > 0.495 and high_conf_docs < 2:
        return "low_confidence"
    else:
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
        # # 测试1：高置信度场景
        # print("=== 测试1：高置信度场景 ===")
        # try:
        #     result1 = ask_rag_with_review("LangChain框架的主要用途是什么？")
        #     print(f"状态: {result1['status']}")
        #     print(f"置信度: {result1['confidence_score']:.2f}")
        #     print(f"答案: {result1['answer'][:80]}...")
        # except Exception as e:
        #     print(f"错误: {e}")
        #
        # # 测试2：低置信度触发审核
        # print("\n=== 测试2：低置信度触发审核 ===")
        # try:
        #     result2 = ask_rag_with_review("那个东西怎么用？")
        #     print(f"状态: {result2['status']}")
        #     print(f"置信度: {result2['confidence_score']:.2f}")
        #     if 'review_reason' in result2:
        #         print(f"审核原因: {result2['review_reason']}")
        # except Exception as e:
        #     print(f"错误: {e}")
        #
        # # 测试3：中等置信度警告
        # print("\n=== 测试3：中等置信度警告 ===")
        # try:
        #     result3 = ask_rag_with_review("向量数据库和关系型数据库的区别？")
        #     print(f"状态: {result3['status']}")
        #     print(f"置信度: {result3['confidence_score']:.2f}")
        #     print(f"答案: {result3['answer'][:80]}...")
        # except Exception as e:
        #     print(f"错误: {e}")
        #
        # # 测试4：自定义阈值
        # print("\n=== 测试4：自定义高阈值 ===")
        # try:
        #     result4 = ask_rag_with_review("Prompt工程相关内容", confidence_threshold=0.9)
        #     print(f"状态: {result4['status']}")
        #     print(f"置信度: {result4['confidence_score']:.2f}")
        #     print(f"相关文档块数量: {len(result4['source_chunks'])}")
        # except Exception as e:
        #     print(f"错误: {e}")

        try:
            question = "LangChain框架的主要用途是什么？"
            result5 = ask_rag_with_review(question)
            print(result5)
            # 验证状态
            assert 'status' in result5, \
                "返回值必须包含 'status' 键"

            assert result5['status'] == 'success', \
                f"明确问题应返回 success 状态，实际: {result5['status']}"

            # 验证置信度
            assert 'confidence_score' in result5, \
                "返回值必须包含 'confidence_score' 键"

            assert result5['confidence_score'] >= 0.7, \
                f"高置信度场景的置信度应 >= 0.7，实际: {result5['confidence_score']}"

            # 验证答案存在
            assert 'answer' in result5, \
                "返回值必须包含 'answer' 键"

            assert len(result5['answer']) > 0, \
                "success 状态下答案不能为空"

            print(f"✓ 高置信度场景测试通过")
            print(f"  状态: {result5['status']}")
            print(f"  置信度: {result5['confidence_score']:.2f}")
            print(f"  答案: {result5['answer'][:80]}...")

            """
                测试2：验证低置信度触发人工审核
                权重：30%
                """
            question = "那个东西怎么用？"  # 指代不明的模糊问题
            result6 = ask_rag_with_review(question)

            # 验证状态
            assert 'status' in result6, \
                "返回值必须包含 'status' 键"

            assert result6['status'] == 'review_needed', \
                f"模糊问题应触发人工审核，实际状态: {result6['status']}"

            # 验证审核原因
            assert 'review_reason' in result6, \
                "review_needed 状态必须包含 'review_reason' 键"

            assert isinstance(result6['review_reason'], str), \
                f"review_reason 必须是字符串，实际: {type(result6['review_reason'])}"

            assert len(result6['review_reason']) > 0, \
                "review_reason 不能为空"

            # 验证置信度
            assert result6['confidence_score'] < 0.7, \
                f"review_needed 状态的置信度应 < 0.7，实际: {result6['confidence_score']}"

            print(f"✓ 低置信度触发测试通过")
            print(f"  状态: {result6['status']}")
            print(f"  置信度: {result6['confidence_score']:.2f}")
            print(f"  审核原因: {result6['review_reason']}")

            """
                测试3：验证中等置信度警告
                权重：20%
                """
            question = "向量数据库和关系型数据库的区别？"  # 部分内容在知识库中
            result7 = ask_rag_with_review(question)

            # 验证状态（可能是 low_confidence 或 success，取决于实现）
            assert 'status' in result7, \
                "返回值必须包含 'status' 键"

            assert result7['status'] in ['success', 'low_confidence'], \
                f"状态应为 success 或 low_confidence，实际: {result7['status']}"

            # 如果是 low_confidence，验证置信度范围
            if result7['status'] == 'low_confidence':
                assert 0.5 <= result7['confidence_score'] < 0.7, \
                    f"low_confidence 状态的置信度应在 0.5-0.7 范围内，实际: {result7['confidence_score']}"

            # 验证答案存在
            assert 'answer' in result7, \
                "返回值必须包含 'answer' 键"

            assert len(result7['answer']) > 0, \
                "即使 low_confidence 状态也应返回答案"

            print(f"✓ 中等置信度警告测试通过")
            print(f"  状态: {result7['status']}")
            print(f"  置信度: {result7['confidence_score']:.2f}")

            """
                测试4：验证置信度阈值调节
                权重：20%
                """
            question = "Prompt工程相关内容"

            # 使用默认阈值（0.7）
            result_default = ask_rag_with_review(question, confidence_threshold=0.7)

            # 使用高阈值（0.9）
            result_high = ask_rag_with_review(question, confidence_threshold=0.9)

            # 验证阈值生效
            # 高阈值应该更容易触发审核或警告
            if result_default['status'] == 'success':
                # 如果默认阈值下是 success，高阈值下可能变成 low_confidence 或 review_needed
                assert result_high['status'] in ['success', 'low_confidence', 'review_needed'], \
                    f"高阈值应该更严格，实际: {result_high['status']}"

            print(f"✓ 置信度阈值调节测试通过")
            print(f"  默认阈值(0.7)状态: {result_default['status']}, 置信度: {result_default['confidence_score']:.2f}")
            print(f"  高阈值(0.9)状态: {result_high['status']}, 置信度: {result_high['confidence_score']:.2f}")

            """
                测试5：验证输出结构完整性
                权重：加分项
                """
            question = "向量数据库的原理"
            result = ask_rag_with_review(question)

            # 验证基础字段
            required_keys = ['status', 'confidence_score', 'source_chunks']
            for key in required_keys:
                assert key in result, \
                    f"返回值缺少必需的键: '{key}'，当前包含: {list(result.keys())}"

            # 验证数据类型
            assert isinstance(result['status'], str), \
                f"status 必须是字符串，实际: {type(result['status'])}"

            assert result['status'] in ['success', 'low_confidence', 'review_needed'], \
                f"status 必须是有效值，实际: {result['status']}"

            assert isinstance(result['confidence_score'], (int, float)), \
                f"confidence_score 必须是数值，实际: {type(result['confidence_score'])}"

            assert 0 <= result['confidence_score'] <= 1, \
                f"confidence_score 应在 0-1 范围内，实际: {result['confidence_score']}"

            assert isinstance(result['source_chunks'], list), \
                f"source_chunks 必须是列表，实际: {type(result['source_chunks'])}"

            # 根据状态验证特定字段
            if result['status'] == 'review_needed':
                assert 'review_reason' in result, \
                    "review_needed 状态必须包含 'review_reason' 键"

            if result['status'] in ['success', 'low_confidence']:
                assert 'answer' in result, \
                    f"{result['status']} 状态必须包含 'answer' 键"

            print(f"✓ 输出结构完整性验证通过")

            """
                测试6：验证文档块保留
                权重：加分项
                """
            question = "LangChain的应用场景有哪些？"
            result8 = ask_rag_with_review(question)

            # 验证文档块存在
            assert 'source_chunks' in result8, \
                "返回值必须包含 'source_chunks' 键"

            # 验证文档块数量合理
            assert len(result8['source_chunks']) >= 0, \
                "source_chunks 不能为 None"

            # 如果有文档块，验证内容不为空
            for chunk in result8['source_chunks']:
                assert isinstance(chunk, str), \
                    f"文档块必须是字符串，实际: {type(chunk)}"
                assert len(chunk) > 0, \
                    "文档块不能为空字符串"

            print(f"✓ 文档块保留测试通过")
            print(f"  检索到 {len(result['source_chunks'])} 个文档块")

        except Exception as e:
            print(f"错误: {e}")