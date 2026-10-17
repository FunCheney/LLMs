from pathlib import Path
from typing import List, Optional

import torch

from embedding import EmbeddingProvider, EmbeddingResult
from visual_bge.modeling import Visualized_BGE


class VisualBGEEmbeddingProvider(EmbeddingProvider):
    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    MODEL_PATH = "../embedding/visual_bge/bge-base-en-v1.5.pth"
    DIMENSION = 768

    def __init__(
        self,
        model_name: str ,
        model_weight: str,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.weights = model_weight

        # 自动选择计算设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        # 注意：model_name_bge 是底层 BGE 模型名称
        # model_weight 才是 Visualized-BGE 的权重文件路径
        self.model = Visualized_BGE(
            model_name_bge=self.model_name,
            model_weight=self.weights,
        )

        self.model.to(self.device)
        self.model.eval()

    def _encode(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
    ) -> EmbeddingResult:
        """内部统一编码逻辑，支持文本、图片和图文组合。"""
        if text is None and image is None:
            raise ValueError("text 和 image 不能同时为空")

        if text is not None and not text.strip():
            raise ValueError("text 不能为空字符串")

        if image is not None and not Path(image).is_file():
            raise FileNotFoundError(f"图片文件不存在: {image}")

        with torch.no_grad():
            vector = self.model.encode(text=text, image=image)

        # 官方示例返回的是形如 [1, 768] 的 Tensor
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().float().cpu().reshape(-1).tolist()
        else:
            # 兼容返回 NumPy 数组的实现
            import numpy as np
            vector = np.asarray(vector, dtype=np.float32).reshape(-1).tolist()

        return EmbeddingResult(
            vector=vector,
            model=self.model_name,
            dimension=len(vector),
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        """单条文本编码。"""
        return self._encode(text=text)

    def embed_documents(
        self,
        documents: List[str],
    ) -> List[EmbeddingResult]:
        """批量文档编码。"""
        return [self.embed_text(text) for text in documents]

    def embed_image(self, image: str) -> EmbeddingResult:
        """单张图片编码。"""
        return self._encode(image=image)

    def embed_multimodal(
        self,
        text: str,
        image: str,
    ) -> EmbeddingResult:
        """图文组合编码。"""
        return self._encode(text=text, image=image)

