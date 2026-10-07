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

