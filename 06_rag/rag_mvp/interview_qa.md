# RAG 面试准备：补充知识清单 + 本项目面试题

> 基于 `06_rag/rag_mvp` 当前实现梳理。项目已覆盖：文档解析(PyMuPDF/HTML)、结构化切分、BGE-M3 稠密 + BM25 稀疏、RRF 混合检索、BGE 重排、带引用的 LLM 生成、Golden 检索评测(Recall@k/MRR/nDCG)、local + Milvus 双后端、统一 CLI。以下按"面试高频"补齐缺口。

---

## 一、还需补充的知识（按优先级）

### P0 —— 面试高频，且项目明确留白

1. **生成侧评估与防幻觉**
   项目只做了检索侧指标，README 明确"生成侧待做"。补：Faithfulness（忠实度）、Answer Relevancy、Groundedness（接地性）、Citation Precision/Recall、LLM-as-Judge 的 prompt 设计与一致性校验、RAGAS 工具。这是"你怎么评估 RAG 质量"的必答部分。

2. **Query 改写与多轮对话**
   项目 `pre_query` 目前只有 NFKC 归一化 + 长度限制。补：multi-query 扩展、HyDE（假设文档）、子问题分解（sub-question）、多轮指代消解（"它"指什么）、query 复杂度路由。

3. **分块策略进阶**
   项目是「结构化章节 + 段落 + 字符 fallback」。补：语义分块（按 embedding 相似度找边界）、父子 chunk / small-to-big（小 chunk 检索 + 大 chunk 喂 LLM）、句子窗口检索、表格/代码/图片的特殊切分，以及 chunk_size/overlap 对召回-噪声的影响原理。

4. **ANN 检索底层原理**
   项目用 Milvus AUTOINDEX + local 暴力点积。补：HNSW、IVF、PQ/SQ 量化；为什么向量检索是"近似最近邻"、为什么可能漏召回；召回率 vs QPS vs 内存的三角权衡。

### P1 —— 进阶，让项目更有竞争力

5. **重排性能优化**：cross-encoder 慢，怎么两阶段 + 蒸馏 / 量化 / 早停。
6. **Agentic / 高级 RAG 范式**：Self-RAG、CRAG（检索置信度 + 纠错）、GraphRAG、RAPTOR 层次检索、Adaptive RAG。
7. **缓存体系**：exact cache、语义缓存、prompt cache，命中率与失效策略。
8. **增量更新与删除**：chunk 失效、文档版本、删除同步（项目 Milvus 做了 upsert 幂等，但没做 delete/版本）。
9. **可观测性**：Langfuse / LangSmith trace、线上反馈闭环、劣化告警。
10. **延迟优化**：流式生成（当前 `generate_answer` 未做 stream）、embedding 批处理、异步、GPU 推理。

### P2 —— 加分项 / 场景题

11. **Text2SQL 深化**：schema linking、NL2SQL 评测（Spider / BIRD）。
12. **多模态 RAG**：visual BGE 已引入但未落链路，补表格/图像解析、OCR。
13. **权限与多租户**：metadata 过滤 + ACL 隔离（谁只能检索谁的文档）。

---

## 二、本项目面试题（按模块）

### A. 文档解析与切分（ingestion）

1. 为什么选 PyMuPDF 解析 PDF？遇到扫描版 PDF（无文本层）怎么办？
2. 为什么要做"跨页段落感知切分"？`page_aware_splitter` 怎么把 chunk 映射回起止页码？它的遗留局限是什么（"空行=段落"启发式）？
3. `chunk_size` / `overlap` 怎么选？太大/太小的 trade-off？为什么约束 `0 <= overlap < chunk_size`？
4. 结构化 `chunker` 相比 `page_aware_splitter` 升级了什么？`section_title`/`section_path` 元数据对下游检索和引用有什么价值？
5. 你怎么验证切分质量？（`chunk_validator` + `review_chunks` 抽检 20 个）为什么人工抽检不可省略？

### B. Embedding 与向量

6. 为什么选 BGE-M3？它和 `bge-large`、OpenAI `text-embedding-3` 的区别？"稠密+稀疏+多向量一体"是什么含义？
7. 为什么对向量做 L2 归一化？归一化后点积为什么等价余弦相似度？
8. BM25 是什么？和 TF-IDF 的关系？中文为什么也需要分词（你的 `tokenizer`）？
9. 稠密检索和稀疏检索各擅长什么、各在什么场景失效？（专名/数字 vs 语义改写）

### C. 混合检索与 RRF

10. 为什么需要混合检索？dense 和 sparse 的分数为什么不能直接加权？
11. RRF 公式是什么？`rrf_k=60` 的作用？`candidate_multiplier` 为什么"先多召回再融合"？
12. 除了 RRF 还有哪些融合方式（分数加权、learned fusion）？各有什么优劣？
13. 混合检索一定比单路好吗？什么情况下 hybrid 反而变差？

### D. 重排（reranker）

14. 重排为什么用 cross-encoder 而不是 bi-encoder？两者精度/速度差异的根源？
15. 你的两阶段流程是什么？（BGE-M3 召回 → bge-reranker-v2-m3 精排）为什么不在全库上直接跑 reranker？
16. 重排为什么把 `prerank_score` 保留到 metadata？"稳定排序"为什么重要？

### E. 生成与引用（generation）

17. system prompt 怎么设计让模型"只用上下文、不足则拒答"？`temperature=0` 的目的是什么？
18. `[n]` 引用编号怎么保证上下文块与 citation 一一对应？`citations_referenced_in_answer` 如何从答案解析引用？答案无引用标记时怎么兜底？
19. "证据不足"怎么处理？（负例题 `expect_insufficient`）检索阈值和拒答策略怎么设计？
20. 为什么用 DeepSeek + LangChain 封装？`create_chat_model` 的惰性导入解决了什么实际问题？

### F. 评测（eval）

21. Golden Dataset 怎么构建？gold 为什么要精确到 `chunk_id` 而不是章节名？
22. Recall@k / MRR / nDCG@10 各自的含义与适用场景？为什么"Recall@10 未命中的用例"最值得看？
23. 负例题为什么不参与检索指标？它们留给哪一层评测？（生成层拒答）
24. 检索指标好 = RAG 效果好吗？还缺什么指标？（生成侧 groundedness / citation correctness）
25. `--rerank` 对比实验怎么设计的？它想证明什么？（量化重排收益，报告键 `<mode>+rerank`）

### G. 向量库与工程化

26. 你抽象了 `VectorStore` 接口（local / Milvus），好处是什么？`add/save/count/search` 语义怎么在两后端间对齐？
27. Milvus 为什么用 `chunk_id` 做业务主键 + upsert，而不是自增主键 + insert？（幂等，避免数据翻倍）
28. Milvus 为什么 `load_collection` 要显式调用？`search` 为什么不回取 embedding？（省带宽）
29. 惰性导入解决了什么？还有哪些工程化手段？（CLI `rag-mvp`、uv editable、`HF_ENDPOINT` 镜像）
30. 从 local pickle 切到 Milvus，检索/问答/评测怎么做到无感切换的？（`factory` + `--store-type`）

### H. 系统设计与开放题

31. 知识库从 1 万篇涨到 1000 万篇，你的系统哪一层先崩？怎么扩容？
32. 用户问题检索不到相关内容（召回失败），你会从哪几层逐层排查？
33. 怎么防止模型编造引用（幻觉引用）？怎么做到"宁可不答也不错答"？
34. 要支持"权限隔离"（不同用户只能检索自己有权限的文档），怎么设计？
35. Text2SQL：为什么需要先提取 DDL？schema 在 NL→SQL 里的作用？你的链路是什么？

---

## 三、高频难题回答要点速记

- **为什么 hybrid 用 RRF 而不是加权分数**：dense 是余弦相似度、sparse 是 BM25 分数，量纲/分布完全不同，直接加权需要调权重且不稳；RRF 只用名次，天然无量纲问题，`k=60` 平滑前排差异、避免单路垄断。
- **bi-encoder vs cross-encoder**：bi-encoder 把 query 和 doc 独立编码成向量后点积，可预计算索引、快（O(1) 检索）；cross-encoder 把 query+doc 拼接进 transformer 联合打分、有 token 级交互、更准但必须在线逐对计算，所以只能做精排。
- **怎么评估 RAG 质量**：分两层。检索层看 Recall@k/MRR/nDCG（离线 golden）；生成层看 Faithfulness/Groundedness/Citation Precision（判断答案是否忠于检索到的上下文、引用是否真实），用 LLM-as-Judge 或 RAGAS，再配线上指标（满意度、拒答率）闭环。
- **召回失败排查顺序**：query 本身是否歧义/需改写 → 归一化是否破坏语义 → 是否有对应文档（知识覆盖）→ embedding 是否对齐语料域 → 索引是否陈旧/漏数据 → rerank 是否误杀。每一层都有对应验证手段。
