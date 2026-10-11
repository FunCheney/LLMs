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

#### 集合

1. 集合的 schema

2. 集合的字段

   * 包含一个住建
   * 最多包含四个向量字段 
   * 多个标量字段  

3. 


#### 索引


#### 度量和搜索


#### Milvus 混合检索实战


#### Milvus 多模态检索