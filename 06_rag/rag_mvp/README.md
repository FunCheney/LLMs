### Phase 1：文档处理
阶段 1 · 文档接入
基础设施
选择并下载真实文档。
解析 PDF / HTML。
保留标题、章节、页码和原始来源。
输出 chunks.jsonl。
人工检查至少 20 个 Chunk。

交付物： 可以被检索的结构化文档数据。

### Phase 2：检索系统
阶段 2 · 混合检索
核心能力
构建向量索引。
构建 BM25 索引。
实现统一 Retriever 接口。
实现 RRF。
命令行测试 Top-K 结果。

交付物： 输入问题，能够打印相关 Chunk 及其 metadata。

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