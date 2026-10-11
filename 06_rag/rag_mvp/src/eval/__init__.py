"""
eval 包：检索评测（数据集加载、命中判定、评测执行与报告）。

当前为 section 匹配形式的 smoke 评测，用于快速回归验证检索质量。
"""

from eval.smoke_runner import run_smoke_evaluation
from eval.smoke_schema import SmokeDataset, load_smoke_dataset

__all__ = ["SmokeDataset", "load_smoke_dataset", "run_smoke_evaluation"]
