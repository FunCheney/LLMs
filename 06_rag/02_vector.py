import numpy as np

# 创建一个数组
v = np.array([3,4])
print(v)
# 计算大小
magnitude = np.linalg.norm(v)
print(magnitude)

# 计算单位向量
unit_vector = v / magnitude
print(unit_vector)

# 计算角度
angle = np.arctan2(v[1], v[0]) * 180 / np.pi
print(angle)

# 向量之间可以计算距离