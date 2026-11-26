"""评测资产（golden set、指标层、runner）的 Python 包根。

存在的原因：`evals/metrics.py` 要能被单测 `from evals.metrics import ...` 导入。
scripts/ 下每周一个子包（w1…w12），跨周复用改走 `evals.scripts.wN.xxx` 绝对导入；
入口仍可用 `python evals/scripts/w3/run_eval.py` 直接跑，库可被 import 测试。
两种入口并存是刻意的——用 `python -m` 反而让 Makefile 和 CI 都更啰嗦。
"""
