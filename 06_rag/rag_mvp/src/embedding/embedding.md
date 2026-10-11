## Embedding 在 RAG 中的作用

### 语义检索的基础
RAG 的“检索”环节通常以基于 Embedding 的语义搜索为核心。通用流程如下： 

（1）离线索引构建：将知识库内文档切分后，使用 Embedding 模型将每个文档块（Chunk）转换为向量，存入专门的向量数据库中。

（2）在线查询检索：当用户提出问题时，使用同一个 Embedding 模型将用户的问题也转换为一个向量。

（3）相似度计算：在向量数据库中，计算“问题向量”与所有“文档块向量”的相似度。

（4）召回上下文：选取相似度最高的 Top-K 个文档块，作为补充的上下文信息，与原始问题一同送给大语言模型（LLM）生成最终答案。

### 决定检索质量的关键

Embedding 的质量直接决定了 RAG 检索召回内容的准确性与相关性。一个优秀的 Embedding 模型能够精准捕捉问题和文档之间的深层语义联系，
即使用户的提问和原文的表述不完全一致。反之，一个劣质的 Embedding 模型可能会因为无法理解语义而召回不相关或错误的信息，
从而“污染”提供给 LLM 的上下文，导致最终生成的答案质量低下。


### 实现一个真实的 Embedding Provider 
#### 问题 1：选什么模型？
需要考虑：

* 中文能力 
* 英文能力 
* 中英混合 
* 技术文档 
* 向量维度 
* 本地运行成本 
* Embedding 速度

#### 问题 2：Query 和 Document 是否需要不同处理？

某些 Embedding 模型采用：

```
query → query embedding
document → document embedding
```
两者可能有不同的 prompt/instruction。 这会直接影响检索效率。

#### 问题 3：如何批量 Embedding？
不能对 10,000 个 Chunk：
```
for chunk in chunks:
    embed(chunk)
```
然后完全不考虑：

* batch size 
* GPU/CPU 
* 内存 
* 缓存 
* 重复计算


#### 问题 4: 多模态 embedding？