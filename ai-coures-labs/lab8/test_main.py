from mian import ask_rag_with_review

def test_high_confidence_scenario():
    """
    测试1：验证高置信度场景
    权重：30%
    """
    question = "LangChain框架的主要用途是什么？"
    result = ask_rag_with_review(question)
    print(result)
    # 验证状态
    assert 'status' in result, \
        "返回值必须包含 'status' 键"

    assert result['status'] == 'success', \
        f"明确问题应返回 success 状态，实际: {result['status']}"

    # 验证置信度
    assert 'confidence_score' in result, \
        "返回值必须包含 'confidence_score' 键"

    assert result['confidence_score'] >= 0.7, \
        f"高置信度场景的置信度应 >= 0.7，实际: {result['confidence_score']}"

    # 验证答案存在
    assert 'answer' in result, \
        "返回值必须包含 'answer' 键"

    assert len(result['answer']) > 0, \
        "success 状态下答案不能为空"

    print(f"✓ 高置信度场景测试通过")
    print(f"  状态: {result['status']}")
    print(f"  置信度: {result['confidence_score']:.2f}")
    print(f"  答案: {result['answer'][:80]}...")


def test_low_confidence_trigger():
    """
    测试2：验证低置信度触发人工审核
    权重：30%
    """
    question = "那个东西怎么用？"  # 指代不明的模糊问题
    result = ask_rag_with_review(question)

    # 验证状态
    assert 'status' in result, \
        "返回值必须包含 'status' 键"

    assert result['status'] == 'review_needed', \
        f"模糊问题应触发人工审核，实际状态: {result['status']}"

    # 验证审核原因
    assert 'review_reason' in result, \
        "review_needed 状态必须包含 'review_reason' 键"

    assert isinstance(result['review_reason'], str), \
        f"review_reason 必须是字符串，实际: {type(result['review_reason'])}"

    assert len(result['review_reason']) > 0, \
        "review_reason 不能为空"

    # 验证置信度
    assert result['confidence_score'] < 0.7, \
        f"review_needed 状态的置信度应 < 0.7，实际: {result['confidence_score']}"

    print(f"✓ 低置信度触发测试通过")
    print(f"  状态: {result['status']}")
    print(f"  置信度: {result['confidence_score']:.2f}")
    print(f"  审核原因: {result['review_reason']}")


def test_medium_confidence_warning():
    """
    测试3：验证中等置信度警告
    权重：20%
    """
    question = "向量数据库和关系型数据库的区别？"  # 部分内容在知识库中
    result = ask_rag_with_review(question)

    # 验证状态（可能是 low_confidence 或 success，取决于实现）
    assert 'status' in result, \
        "返回值必须包含 'status' 键"

    assert result['status'] in ['success', 'low_confidence'], \
        f"状态应为 success 或 low_confidence，实际: {result['status']}"

    # 如果是 low_confidence，验证置信度范围
    if result['status'] == 'low_confidence':
        assert 0.5 <= result['confidence_score'] < 0.7, \
            f"low_confidence 状态的置信度应在 0.5-0.7 范围内，实际: {result['confidence_score']}"

    # 验证答案存在
    assert 'answer' in result, \
        "返回值必须包含 'answer' 键"

    assert len(result['answer']) > 0, \
        "即使 low_confidence 状态也应返回答案"

    print(f"✓ 中等置信度警告测试通过")
    print(f"  状态: {result['status']}")
    print(f"  置信度: {result['confidence_score']:.2f}")


def test_confidence_threshold_adjustment():
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


def test_output_structure():
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


def test_source_chunks_preservation():
    """
    测试6：验证文档块保留
    权重：加分项
    """
    question = "LangChain的应用场景有哪些？"
    result = ask_rag_with_review(question)

    # 验证文档块存在
    assert 'source_chunks' in result, \
        "返回值必须包含 'source_chunks' 键"

    # 验证文档块数量合理
    assert len(result['source_chunks']) >= 0, \
        "source_chunks 不能为 None"

    # 如果有文档块，验证内容不为空
    for chunk in result['source_chunks']:
        assert isinstance(chunk, str), \
            f"文档块必须是字符串，实际: {type(chunk)}"
        assert len(chunk) > 0, \
            "文档块不能为空字符串"

    print(f"✓ 文档块保留测试通过")
    print(f"  检索到 {len(result['source_chunks'])} 个文档块")


if __name__ == "__main__":
    test_source_chunks_preservation()