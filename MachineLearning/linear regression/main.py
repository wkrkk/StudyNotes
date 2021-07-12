from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor

# 数据集划分具体实现
# def train_test_split(x, y, test_size=0.2, seed=None):
#     # 保证样本和标签个数一致
#     assert x.shape[0] == y.shape[0];
#     # 保证数据集划分有效
#     assert 0 < test_size < 1;
#     if seed:
#         np.random.seed(seed)
#
#     # 随机排列顺序
#     shuffled_indexes = np.random.permutation(len(x))
#     test = int(len(x) * test_size)
#     train_index = shuffled_indexes[test:]
#     test_index = shuffled_indexes[:test]
#
#     return x[train_index], x[test_index], y[train_index], y[test_index]


# 获取数据
boston = datasets.load_boston()

# 划分训练集和测试集
# x_train.shape(379,13)、y_train.shape(379,)
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25)

# 特征值和目标值进行标准化处理
# 特征值标准化
std1 = StandardScaler()
# 标准差标准化 （旧值-均值）/ 标准差
x_train = std1.fit_transform(x_train)
x_test = std1.transform(x_test)

# 目标值标准化
std2 = StandardScaler()
# print(y_train)
# y_train.reshape(-1,1)指定转变列为一列
y_train = std2.fit_transform(y_train.reshape(-1, 1))
y_test = std2.transform(y_test.reshape(-1, 1))

# 1.正规方程求解预测
lr = LinearRegression()
# 输入数据不断用训练数据建立模型
lr.fit(x_train, y_train)
print("lr回归系数为：", lr.coef_)
# 预测测试集房屋价格
y_predict = lr.predict(x_test)
# 得到标准化原来的真实值
y_predict = std2.inverse_transform(y_predict)
print("lr测试集每个样本的预测价格：", y_predict)
# 均方根误差（预测值，真实值为标准化之前的值）
print("lr均方根误差：", mean_squared_error(std2.inverse_transform(y_test), y_predict))

# 2.梯度下降求解预测
sgd = SGDRegressor()
sgd.fit(x_train, y_train)
print("sgd回归系数为：", sgd.coef_)
y_sgdPredict = sgd.predict(x_test)
y_sgdPredict = std2.inverse_transform(y_sgdPredict)
print("sgd测试集每个样本的预测价格：", y_sgdPredict)
print("sgd均方根误差：", mean_squared_error(std2.inverse_transform(y_test), y_sgdPredict))