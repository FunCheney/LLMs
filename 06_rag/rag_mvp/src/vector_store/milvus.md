### Milvus 环境搭建


Milvus 提供了三种部署模式，适用于从简单应用到管理数百亿向量的大规模 Kubernetes 集群的数据规模。

* Milvus Lite 作为 Milvus 的轻量级版本，易于集成到应用程序中。
* Milvus Standalone 是 Milvus 的单机服务器版本，所有的组件都打包在一个 docker 镜像中，易于部署。
* Milvus Distributed 可以部署在 Kubernetes 集群上，支持十亿级向量甚至更大规模场景的云原生架构。

```shell
mkdir -p ~/services/milvus
cd ~/services/milvus

curl -L \
  https://github.com/milvus-io/milvus/releases/download/v2.6.24/milvus-standalone-docker-compose.yml \
  -o docker-compose.yml
  

# 启动
docker compose up -d
# 查看容器状态
docker compose ps
# 查看日志
docker compose logs -f standalone

# 安装依赖 milvus
uv add pymilvus

# 运行测试代码
uv run python example/test_milvus.py
```

### Milvus 实战

#### 数据库

#### 集合 (collection)

1. 集合的 schema

2. 集合的字段

   * 包含一个住建
   * 最多包含四个向量字段 
   * 多个标量字段  

3. 


#### 索引

Milvus 提供了多种向量索引算法，以适应不同的应用场景。以下是几种最核心的类型：

##### FLAT (精确查找)

* 原理：暴力搜索（Brute-force Search）。它会计算查询向量与集合中所有向量之间的实际距离，返回最精确的结果。 
* 优点：100% 的召回率，结果最准确。 
* 缺点：速度慢，内存占用大，不适合海量数据。 
* 适用场景：对精度要求极高，且数据规模较小（百万级以内）的场景。

##### IVF 系列 (倒排文件索引)

* 原理：类似于书籍的目录。它首先通过聚类将所有向量分成多个“桶”(nlist)，查询时，先找到最相似的几个“桶”，然后只在这几个桶内进行精确搜索。
   IVF_FLAT、IVF_SQ8、IVF_PQ 是其不同变体，主要区别在于是否对桶内向量进行了压缩（量化）。
* 优点：通过缩小搜索范围，极大地提升了检索速度，是性能和效果之间很好的平衡。
* 缺点：召回率不是100%，因为相关向量可能被分到了未被搜索的桶中。
* 适用场景：通用场景，尤其适合需要高吞吐量的大规模数据集。

##### HNSW (基于图的索引)

* 原理：构建一个多层的邻近图。查询时从最上层的稀疏图开始，快速定位到目标区域，然后在下层的密集图中进行精确搜索。
* 优点：检索速度极快，召回率高，尤其擅长处理高维数据和低延迟查询。
* 缺点：内存占用非常大，构建索引的时间也较长。
* 适用场景：对查询延迟有严格要求（如实时推荐、在线搜索）的场景。

##### DiskANN (基于磁盘的索引)

* 原理：一种为在 SSD 等高速磁盘上运行而优化的图索引。
* 优点：支持远超内存容量的海量数据集（十亿级甚至更多），同时保持较低的查询延迟。
* 缺点：相比纯内存索引，延迟稍高。
* 适用场景：数据规模巨大，无法全部加载到内存的场景。



#### 度量和搜索

##### 基础向量检索 (ANN Search)

##### 增强检索
1. 过滤检索 (Filtered Search)
2. 范围检索 (Range Search)
3. 多向量混合检索 (Hybrid Search)
4. 分组检索 (Grouping Search)

#### Milvus 混合检索实战


#### Milvus 多模态检索