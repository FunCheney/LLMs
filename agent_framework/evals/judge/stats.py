"""一致性系数（W4 D1）：自己实现，不引 scipy。

为什么自己写
------------
三个函数一共 60 行，而 `scipy` 会给评测环境加 ~90MB 依赖与一次编译。
更实际的原因是**口径**：kappa 至少有名义（nominal）、线性加权、二次加权三种，
不同库默认值不同。自己写就能把"我们用的是二次加权 kappa"钉在代码里，
换了库也不会悄悄换口径。

三个数各有分工，报告里要同时给：
- `quadratic_weighted_kappa`：1–5 分是**有序**的，差 1 分和差 3 分不该同罪；
- `nominal_kappa`：3 档（低/中/高）上的纯一致性，排除"相邻档算接近"的宽容；
- `spearman`：对异常值不敏感的秩相关，看排序是否一致。
只报一个数容易被追问"换种算法还成立吗"。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

__all__ = [
    "agree_exact_rate",
    "bucket",
    "mean_absolute_error",
    "nominal_kappa",
    "quadratic_weighted_kappa",
    "spearman",
]

# 1–5 分压成三档：人工标注做不到区分 4 和 5 的细微差别，
# 而"低/中/高"这三档是两个人看同一个答案时能达成共识的粒度。
_BUCKETS = {"low": (1, 2), "mid": (3, 3), "high": (4, 5)}


def bucket(score: int) -> str:
    for name, (low, high) in _BUCKETS.items():
        if low <= score <= high:
            return name

    raise ValueError(f"分数 {score} 不在 1–5")


def _kappa(matrix: Sequence[Sequence[float]], weights: Sequence[Sequence[float]]) -> float:
    total = sum(sum(row) for row in matrix)

    if total == 0:
        raise ValueError("空样本无法计算 kappa")

    observed = sum(
        weights[i][j] * matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix))
    ) / total

    rows = [sum(row) for row in matrix]
    cols = [
        sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix))
    ]
    expected = sum(
        weights[i][j] * rows[i] * cols[j] / total
        for i in range(len(matrix))
        for j in range(len(matrix))
    ) / total

    # 1 - pe == 0 意味着两边都只用了同一个类别（例如全部 5 分）。
    # 此时"一致性"无从定义，返回 0 而不是 1：全给 5 分不是一致性，是没有区分度。
    if abs(1.0 - expected) < 1e-12:
        return 0.0

    return (observed - expected) / (1.0 - expected)


def quadratic_weighted_kappa(
    human: Sequence[int], judge: Sequence[int], *, low: int = 1, high: int = 5
) -> float:
    """二次加权 kappa。相邻 1 分的惩罚是 1/(k-1)²，差满档惩罚 1。"""

    if len(human) != len(judge):
        raise ValueError(f"长度不一致：human={len(human)} judge={len(judge)}")

    if not human:
        raise ValueError("空样本无法计算 kappa")

    size = high - low + 1
    matrix = [[0.0] * size for _ in range(size)]

    for h, j in zip(human, judge, strict=True):
        for value in (h, j):
            if not low <= value <= high:
                raise ValueError(f"分数 {value} 超出 {low}–{high}")

        matrix[h - low][j - low] += 1

    weights = [
        [1.0 - ((i - j) ** 2) / ((size - 1) ** 2) for j in range(size)]
        for i in range(size)
    ]

    return _kappa(matrix, weights)


def nominal_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    """名义 kappa（三档）。只认完全一致，不认"差一档"。"""

    if len(human) != len(judge):
        raise ValueError(f"长度不一致：human={len(human)} judge={len(judge)}")

    if not human:
        raise ValueError("空样本无法计算 kappa")

    labels = sorted(set(human) | set(judge))
    index = {label: position for position, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0.0] * size for _ in range(size)]

    for h, j in zip(human, judge, strict=True):
        matrix[index[h]][index[j]] += 1

    weights = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]

    return _kappa(matrix, weights)


def spearman(human: Sequence[int], judge: Sequence[float]) -> float:
    """秩相关。并列值取平均秩（人工标注里同一个分出现多次是常态）。"""

    if len(human) != len(judge):
        raise ValueError(f"长度不一致：human={len(human)} judge={len(judge)}")

    if len(human) < 2:
        raise ValueError("至少要 2 个样本")

    left = _ranks([float(item) for item in human])
    right = _ranks([float(item) for item in judge])

    return _pearson(left, right)


def agree_exact_rate(human: Sequence[int], judge: Sequence[int]) -> float:
    if not human:
        raise ValueError("空样本")

    return sum(1 for h, j in zip(human, judge, strict=True) if h == j) / len(human)


def mean_absolute_error(human: Sequence[int], judge: Sequence[int]) -> float:
    if not human:
        raise ValueError("空样本")

    return sum(abs(h - j) for h, j in zip(human, judge, strict=True)) / len(human)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0

    while position < len(order):
        end = position

        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1

        average = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            ranks[index] = average

        position = end + 1

    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    mean_left = sum(left) / n
    mean_right = sum(right) / n

    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    variance_left = sum((a - mean_left) ** 2 for a in left) ** 0.5
    variance_right = sum((b - mean_right) ** 2 for b in right) ** 0.5

    if variance_left == 0 or variance_right == 0:
        # 一侧方差为 0（例如 judge 全部同分）：相关系数无定义。
        return 0.0

    return covariance / (variance_left * variance_right)


def distribution(values: Sequence[int]) -> dict[int, int]:
    """分数分布。报告里必须给——它一眼能看出"是不是又全堆在 4 分"。"""

    return dict(sorted(Counter(values).items()))
