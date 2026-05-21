'''
线性回归
有监督的机器学习
代码实战解析解求解模型的方法
'''

import numpy as np
import matplotlib.pyplot as plt

# 100 行 1 列的数据。100个样本，1 个纬度
X = 2 * np.random.rand(100, 1)
# print(len(X))
# print(X)
# y 是真实的数据，就是 y_hat + error (正太分布)
# y = b + w*x + error
y = 5 + 4 * X + np.random.randn(100, 1)

# 拼接一个 100 行 1 列的 全为 1 的数据。X_b 为 100 行两列的数据
# 为了求解 W0 截距项
X_b = np.c_[np.ones((100, 1)), X]

# 实现解析解的公式求解 θ=(X^T X)^-1 * X^T * y
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f'theta: {theta}')


# 使用模型去做预测
X_new = np.array([[0],
                  [2]])
print(X_new)
X_new_b = np.c_[np.ones((2, 1)), X_new]

print(X_new_b)

y_predict = X_new_b.dot(theta)
print(y_predict)

# 绘图展示
plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0.0, 2.0, 0.0, 15.0])
plt.show()