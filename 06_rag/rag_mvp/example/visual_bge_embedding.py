import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from visual_bge.modeling import Visualized_BGE

# 初始化模型
model = Visualized_BGE(
    model_name_bge = "BAAI/bge-base-en-v1.5",
    model_weight="../src/embedding/visual_bge/Visualized_base_en_v1.5.pth"
)
model.eval()

# 生成嵌入向量
with torch.no_grad():
    # 纯文本编码
    text_emb = model.encode(text="Contrastive Language-Image Pre-training")

    # 纯图像编码
    img_emb = model.encode(image="./imgs/img.png")

    # 图文组合编码（图像 + 文本）
    query_emb = model.encode(
        image="./imgs/img.png",
        text="Contrastive Language-Image Pre-training"
    )

# 计算相似度（向量已标准化，点积即余弦相似度）
sim = query_emb @ img_emb.T
print(sim)
