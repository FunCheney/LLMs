from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt

X_1 = 2 * np.random.rand(100, 1)
X_2 = 2 * np.random.rand(100, 1)

X = np.c_[X_1, X_2]

y = 4 + 3 * X_1 + 5 * X_2 + np.random.randn(100, 1)


reg = linear_model.LinearRegression()

# 计算模型的参数
reg.fit(X, y)

print(reg.intercept_, reg.coef_)
x_new = np.array([
    [0,0],
    [2,1],
    [2,4]
])

y_pred = reg.predict(x_new)

plt.plot(x_new[:, 0], y_pred, 'r-')
plt.plot(X_1, y, 'b.')
plt.axis([0,2,0,25])
plt.show()