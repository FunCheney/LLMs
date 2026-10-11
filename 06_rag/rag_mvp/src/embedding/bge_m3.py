"""
BGE-M3 向量化实现（生产默认）。

选用 BAAI/bge-m3，输出 1024 维向量，并对向量做 L2 归一化。
归一化后，向量点积等价于余弦相似度，本地向量库可直接用点积打分。

注意：HF_ENDPOINT 必须在导入 transformers/huggingface_hub 之前设置才生效，
因此这里先设环境变量、再导入 sentence_transformers。
"""

from typing import List

# Hugging Face 的工具链（transformers、huggingface_hub）原生支持通过环境变量 HF_ENDPOINT 将请求重定向到国内镜像
# 需在 import transformers 或 huggingface_hub 之前生效
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from embedding.base import EmbeddingProvider
from embedding.embedding_types import EmbeddingResult


class BGEM3EmbeddingProvider(EmbeddingProvider):
    """BGE-M3 embedding provider."""
    MODEL_NAME = "BAAI/bge-m3"
    DIMENSION = 1024

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device  # 例如 "cpu" / "mps" / "cuda"，None 表示自动选择
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Embed text.
        :param text:
        :return:
        """
        # normalize_embeddings=True， Sentence Transformers 官方的 encode() 支持对输出进行归一化，使向量长度变成 1。
        vector = self.model.encode(text, normalize_embeddings=True).tolist()

        return EmbeddingResult(
            vector=vector,
            model=self.model_name,
            dimension=len(vector),
        )

    def embed_documents(self, documents: List[str]) -> List[EmbeddingResult]:
        """
        Embed documents.
        :param documents:
        :return:
        """
        # 批量编码，一次前向即可处理整个 batch，比逐条调用快得多
        vectors = self.model.encode(documents, normalize_embeddings=True).tolist()
        return [
            EmbeddingResult(
                vector=vector,
                model=self.model_name,
                dimension=len(vector),
            )
            for vector in vectors
        ]
