import numpy as np
import matplotlib.pyplot as plt

# 一元线性回归算法

x = np.asarray([1, 2, 3, 4, 5])
y = np.asarray([1, 3, 2, 3, 5])

num = 0.0
d = 0.0
for x_i, y_i in zip(x, y):
    num += (x_i - np.mean(x)) * (y_i - np.mean(y))
    d += (x_i - np.mean(x)) ** 2

a = num / d
b = np.mean(y) - a * np.mean(x)

# 单一数据预测
y_hat = a * x + b

plt.scatter(x, y)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()


