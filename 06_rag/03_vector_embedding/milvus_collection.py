from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

print("Milvus connected:", client.list_collections())

# 创建一个 collection，需要
# 1. 创建一个 schema
# 2. 设置索引参数（可选）
# 3. 创建 collections

# 1. 创建 schema
schema = client.create_schema(auto_id=False, enable_automatic_fields=False)

# 2. 对 schema 添加字段
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)

# 3. 设置索引参数
index_params = client.prepare_index_params()
index_params.add_index(field_name="my_id", index_type="AUTOINDEX")
# AUTOINDEX 作为所有向量场的索引类型，并根据需要使用COSINE 、L2 和IP 中的一种作为度量类型
index_params.add_index(field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE")

# 4. 创建 collection
client.create_collection(
    collection_name="customized_setup_1",
    schema=schema,
    index_params=index_params,
)

state = client.get_load_state(collection_name="customized_setup_1")

# {'state': <LoadState: Loaded>}
print(state)


# 创建不带任何索引参数的 Collections，然后再添加索引参数。在这种情况下，Milvus 不会在创建时加载 Collection。.
client.create_collection(
    collection_name="customized_setup_2",
    schema=schema,
)

res = client.get_load_state(
    collection_name="customized_setup_2"
)

# {'state': <LoadState: NotLoad>}
print(res)

# 描述 collection
res = client.describe_collection(
    collection_name="customized_setup_1"
)

# {'collection_name': 'customized_setup_1', 'auto_id': False, 'num_shards': 1, 'description': '',
# 'fields': [
# {'field_id': 100, 'name': 'my_id', 'description': '', 'type': <DataType.INT64: 5>, 'params': {}, 'is_primary': True},
# {'field_id': 101, 'name': 'my_vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 5}},
# {'field_id': 102, 'name': 'my_varchar', 'description': '', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 512}}],
# 'functions': [], 'aliases': [], 'collection_id': 469773702419454841, 'consistency_level': 2, 'consistency_level_name': 'Bounded',
# 'properties': {'timezone': 'UTC'}, 'num_partitions': 1, 'enable_dynamic_field': False, 'enable_namespace': False,
# 'schema_version': 0, 'created_timestamp': 469774010617692178, 'update_timestamp': 469774010617692178}
print(res)


#  加载 collection。为创建索引，加载会报错
# client.load_collection(
#     collection_name="customized_setup_2"
# )
#
# res = client.get_load_state(
#     collection_name="customized_setup_2"
# )
#
# print(res)

# 释放
# client.release_collection(
#     collection_name="customized_setup_2"
# )

# 删除
client.drop_collection(
    collection_name="customized_setup_2"
)




