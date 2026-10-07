### Phase 1：文档处理
阶段 1 · 文档接入
基础设施
选择并下载真实文档。
解析 PDF / HTML。
保留标题、章节、页码和原始来源。
输出 chunks.jsonl。
人工检查至少 20 个 Chunk。

交付物： 可以被检索的结构化文档数据。

当前状态：已完成。PDF / HTML 解析后保留章节、页码和 `source_uri`，输出 `data/processed/chunks.jsonl`，并用 `scripts/review_chunks.py` 抽检。

交付物： data/processed/chunks.jsonl（444 条），字段含章节、页码、source_uri 等。

能力栈：

解析：PDF（PyMuPDF）、HTML
分块：结构化章节、ingest_document() → build_chunks()
质量：validation/chunk_validator.py，scripts/review_chunks.py 抽检
常用入口： scripts/build_chunks.py 里的 main()（读 PDF → 写 jsonl）。

### Phase 2：检索系统
阶段 2 · 混合检索
核心能力

1. 构建向量索引。 
2. 构建 BM25 索引。 
3. 实现统一 Retriever 接口。 
4. 实现 RRF。 
5. 命令行测试 Top-K 结果。

交付物： 输入问题，能够打印相关 Chunk 及其 metadata。

#### Embedding 抽象（src/embedding/）

* EmbeddingProvider + EmbeddingResult 
* FakeEmbeddingProvider（单测用） 
* BGEM3EmbeddingProvider（BAAI/bge-m3，1024 维，L2 归一化）

#### 本地向量检索原型（src/vector_store/local.py）

* 暴力扫描 + 点积（归一化后等价 cosine） 
* save / load（pickle） 
* 手工演示（在 test/ 里，不是正式脚本）
    ```
        test_cosine_similarity.py：语义相近句分数更高
        test_vector_index.py：3 条假文档检索 demo（if __name__ == "__main__"）
        test_bge_m3.py：打印向量维度
    ```
  
1. scripts/build_vector_index.py：读 chunks.jsonl → BGE-M3 批量 embed → LocalVectorIndex.save 
2. scripts/search_cli.py：输入问题 → Top-K，打印 section_title、页码、chunk_id

```bash
uv run python scripts/build_vector_index.py
uv run python scripts/search_cli.py "MCP 是什么" --k 5
```

#### 如何使用

```bash
# 1. 若还没有 chunks（Phase 1）
uv run python scripts/build_chunks.py

# 2. 建向量索引（首次会拉 BGE-M3，较慢）
uv run python scripts/build_vector_index.py

# 可选参数
uv run python scripts/build_vector_index.py \
  --chunks data/processed/chunks.jsonl \
  --output data/index/local.pkl \
  --batch-size 32 \
  --device cpu

# 3. 检索
uv run python scripts/search_cli.py "MCP 是什么" --k 5
uv run python scripts/search_cli.py "CLAUDE.md 做什么" --k 3 --snippet 300
```

### Phase 3：问答与引用
阶段 3 · RAG 问答
应用层
设计答案生成 Prompt。
生成答案和引用标记。
实现引用来源映射。
处理证据不足的情况。
测试 10 条真实问题。

交付物： 一个可用的单知识库问答命令行程序。

### Phase 4：Golden Dataset 与评估
阶段 4 · Evaluation
核心验收
建立 30～100 条 Golden Dataset。
实现 Recall@3、Recall@5、Recall@10。
统计答案正确率。
记录 Groundedness 和 Citation Correctness。
输出对比实验结果。

交付物： 可重复运行的评估脚本和指标报告。