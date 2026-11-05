### Retriever 要解决什么问题
上层（CLI、API、LLM）只关心：

* 输入：用户问题（+ 可选参数：top_k、过滤条件） 
* 输出：有序的相关 Chunk 列表，每条带分数、来源、完整 metadata

上层不应知道：

* 用的是 Qdrant 还是 pickle
* 是向量还是 BM25
* 有没有做 RRF

所以 Retriever 的核心是：抽象 + 统一结果结构 + 可组合（dense / sparse / hybrid）。

### 接口定义

统一检索结果（RetrievalResult）


### 需要完成的功能


```text
                  ┌─────────────────┐
                  │   Retriever     │  ← 对外接口
                  └────────┬────────┘
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
DenseRetriever      SparseRetriever      HybridRetriever
       │                   │                   │
       ▼                   ▼                   ├─ 调 dense + sparse
 VectorStore            BM25 Index             └─ RRF 融合
       │                   │
       ▼                   ▼
 Embedding            分词 / ES
```

### 存储层
主要完成：向量存储或者关键词索引

需要：

* 写入：upsert(chunks + vectors)（建索引脚本用） 
* 查询：search(query_vector, k) 或 search_keywords(query, k)
* 按 id 取回：融合、去重、补全 metadata 时用

#### DenseRetriever（向量检索）
要完成的功能：

* 加载索引（可懒加载，避免每次检索都读盘） 
* 对 query 做 与建库相同的 Embedding 
* 在 VectorStore 里 Top-K 搜索 
* 把底层记录 映射成 RetrievalResult 
* 填 rank、source=dense

边界：空 query、索引不存在、索引为空 → 明确行为（空列表或明确错误）

#### SparseRetriever（BM25 / 关键词)
要完成的功能：

* 建关键词索引（ES 或内存 BM25），文档字段与 chunk_id 一致 
* 中文分词（如 jieba）或 ES 分析器 
* retrieve：query 分词 → BM25 Top-K → 同样映射成 RetrievalResult，source=sparse 
* 分数语义与向量不同，不要和 dense 分数直接相加

#### 融合层（RRF）+ HybridRetriever
要完成的功能：

1. RRF 纯函数：输入多路 chunk_id 有序列表，输出融合后的 id + 融合分

2. HybridRetriever：

   * 对同一句 query 调 DenseRetriever 和 SparseRetriever（常各取 2*k 再融合） 
   * RRF 得到最终 id 顺序 
   * 用 id 回填完整 RetrievalResult（正文、metadata 以某一路或 chunk 表为准） 
   * source=hybrid，score 用 RRF 分 
   
3. 去重：同 chunk_id 只保留一条


#### 映射与元数据一致性
要完成的功能：

* 索引里的 payload 与 DocumentChunk 字段一致（建索引时写入，检索时不再依赖 jsonl） 
* chunk_id 全局唯一 
* 映射函数集中一处（避免 CLI 和问答各写一套）

#### 编排与工厂
要完成的功能：

* 根据配置创建 Retriever：mode=dense|sparse|hybrid
* 注入：index_path、embedder、ES 地址等
* CLI / 未来 API 只调 create_retriever(cfg).retrieve(...)



