"""
eval 包：检索评测（数据集加载、命中判定、评测执行与报告）。

包含两层评测：
- smoke：section/keyword 匹配的快速冒烟评测（每次改检索后快速回归）；
- golden：chunk_id 精确标注的正式评测（Recall@K / MRR / nDCG，Phase 4）。
"""

from eval.golden_runner import run_golden_evaluation
from eval.golden_schema import GoldenDataset, load_golden_dataset
from eval.smoke_runner import run_smoke_evaluation
from eval.smoke_schema import SmokeDataset, load_smoke_dataset

__all__ = [
    "SmokeDataset",
    "load_smoke_dataset",
    "run_smoke_evaluation",
    "GoldenDataset",
    "load_golden_dataset",
    "run_golden_evaluation",
]
