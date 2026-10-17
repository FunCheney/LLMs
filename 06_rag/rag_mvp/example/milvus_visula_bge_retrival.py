"""
多模态检索  milvus
"""

import os
from glob import glob
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from embedding.visual_bge import VisualBGEEmbeddingProvider

# 常量设置
DATA_DIR = "../data/example"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"


def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int = 300,
                      row_count: int = 3) -> np.ndarray:
    """从检索到的图像列表创建一个全景图用于可视化。"""
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width

        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved

        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)

    return np.hstack([query_display_area, panoramic_image])



def main():
    # 1.创建 向量数据库客户端
    client = MilvusClient(MILVUS_URI)
    # 1.出生嵌入模型（多模态）
    embedding_provider = VisualBGEEmbeddingProvider(
        model_name="BAAI/bge-base-en-v1.5",
        model_weight="../src/embedding/visual_bge/Visualized_base_en_v1.5.pth",
        device="cpu",
    )

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

    image_list = glob(os.path.join(DATA_DIR, "*.png"))
    if not image_list:
        raise FileNotFoundError(f"在 {DATA_DIR}/ 中未找到任何 .png 图像。")
    dim = len(embedding_provider.embed_image(image_list[0]).vector)

    fields = [
        # 主键字段，设置自增 (auto_id=True)
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 向量字段，维度与模型的输出向量维度一致
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        # 存储原图像路径的标量字段，最大长度限制 512 字符
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    ]

    # 创建集合 Schema
    schema = CollectionSchema(fields, description="多模态图文检索")
    print("Schema 结构:")
    print(schema)

    # 创建集合
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print(f"成功创建 Collection: '{COLLECTION_NAME}'")
    print("Collection 结构:")
    print(client.describe_collection(collection_name=COLLECTION_NAME))

    # 准备数据库中插入的数据
    print(f"正在向 Collection: '{COLLECTION_NAME}' 插入数据")
    data_to_insert = []
    for image_path in tqdm(image_list, desc="生成嵌入图像"):
        vector = embedding_provider.embed_image(image_path).vector
        data_to_insert.append({"vector": vector, "image_path": image_path})

    # 插入数据
    if data_to_insert:
        result = client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
        print(f"插入成功数据 {result["insert_count"]} 条")

    # 创建索引
    print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
    index_params = client.prepare_index_params(collection_name=COLLECTION_NAME)
    index_params.add_index(
        field_name="vector",
        index_type="HNSW", # 基于图的索引
        metric_type="COSINE",
        # M: '16' 和 efConstruction: '256' 是 HNSW 索引的两个关键参数，
        # 分别控制着图中每个节点的最大连接数和索引构建时的搜索范围，这些参数直接影响检索的性能和准确性
        params={"M": 16, "efConstruction": 256}
    )
    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
    print("成功为向量字段创建 HNSW 索引。")
    print("索引详情:")
    print(client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))
    client.load_collection(collection_name=COLLECTION_NAME)
    print("已加载 Collection 到内存中。")

    #  执行多模态检索
    print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
    query_image_path = os.path.join(DATA_DIR, "query.png")
    query_text = "一条龙"
    query_vector = embedding_provider.embed_multimodal(text=query_text, image=query_image_path).vector

    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        output_fields=["image_path"],
        limit=5,
        # ef: 搜索时的节点遍历深度，值越大召回率越高但耗时越长
        search_params={"metric_type": "COSINE", "params": {"ef": 128}}
    )[0]

    retrieved_images = []
    print("检索结果:")
    for i, hit in enumerate(search_results):
        print(f"  Top {i + 1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
        retrieved_images.append(hit['entity']['image_path'])

    # 8. 可视化与清理
    if not retrieved_images:
        print("没有检索到任何图像。")
    else:
        panoramic_image = visualize_results(query_image_path, retrieved_images)
        combined_image_path = os.path.join(DATA_DIR, "search_result.png")
        cv2.imwrite(combined_image_path, panoramic_image)
        print(f"结果图像已保存到: {combined_image_path}")
        Image.open(combined_image_path).show()

    client.release_collection(collection_name=COLLECTION_NAME)
    print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
    client.drop_collection(COLLECTION_NAME)
    print(f"已删除 Collection: '{COLLECTION_NAME}'")




if __name__ == "__main__":
    main()






