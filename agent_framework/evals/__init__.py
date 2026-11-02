"""评测资产（golden set、指标层、runner）的 Python 包根。

存在的原因：`evals/metrics.py` 要能被单测 `from evals.metrics import ...` 导入。
scripts/ 下的可执行入口仍然靠 `sys.path` 直连（它们要 `import bench_baseline`），
两种方式并存是刻意的：入口要能 `python evals/scripts/run_eval.py` 直接跑，
库要能被 import 测试——用 `python -m` 反而让 Makefile 和 CI 都更啰嗦。
"""
