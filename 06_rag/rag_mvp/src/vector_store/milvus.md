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