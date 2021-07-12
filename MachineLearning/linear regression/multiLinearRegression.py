# 多元线性规划

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegression:

    def __init__(self):
        # 分别对应系数、截距、回归系数矩阵
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    # 正规方程
    def fit_normal(self, x_train, y_train):

        # 保证数据集得一致性
        assert x_train.shape[0] == y_train.shape[0]

        x_b = self._data_arrange(x_train)

        # 采用正规方程计算系数矩阵
        # np.linalg.inv()：矩阵求逆
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, x_train, y_train, eta=0.01, n_iters=1e4):
        '''

        批量梯度下降法
        使用归一化的数据训练线性回归算法
        :param x_train:
        :param y_train:
        :param eta:步幅
        :param n_iters:最大循环次数
        :return:
        '''

        assert x_train.shape[0] == y_train.shape[0], 'error'

        def J(theta, x_b, y):
            try:
                # 计算代价函数
                return np.sum((y - x_b.dot(theta)) ** 2) / len(x_b)
            except:
                return float('inf')

        #  求导数
        def dJ(theta, x_b, y):
            return x_b.T.dot(x_b.dot(theta) - y) * 2 / len(x_b)

        # 求梯度下降
        def gradient_descent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            i_iters = 0

            while i_iters < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theat = theta
                theta = theta - eta * gradient

                if (abs(J(theta, x_b, y) - J(last_theat, x_b, y)) < epsilon):
                    break

                i_iters += 1

            return theta

        x_b = self._data_arrange(x_train)

        # 初始系数矩阵
        initial_theta = np.zeros([x_b.shape[1], 1])

        self._theta = gradient_descent(x_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self, x_train, y_train, eta=0.01, n_iters=5):

        '''
        # 随机梯度下降法
        :param x_train:
        :param y_train:
        :param eta:步幅
        :param n_iters:样本整体看几遍
        :return:
        '''

        assert x_train.shape[0] == y_train.shape[0], "error"
        assert n_iters >= 1, "error"

        def dJ_sgd(theta, x_b_i, y_i):
            return x_b_i.T.dot(x_b_i.dot(theta) - y_i) * 2

        def sgd(x_b, y, initial_theta, n_iters):
            t0 = 5
            t1 = 50

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(len(x_b))
                x_b_new = x_b[indexes]
                y_new = y[indexes]
                for i in range(len(x_b)):
                    gradient = dJ_sgd(theta, x_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * len(x_b) + i) * gradient

            return theta

        x_b = self._data_arrange(x_train)
        initial_theta = np.zeros(x_b.shape[1])

        self._theta = sgd(x_b, y_train, initial_theta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def _data_arrange(self, data):
        # 进行数据整理--在数据矩阵第一列加一列均为 1 得列
        # 在水平方向上平铺
        return np.hstack([np.ones((len(data), 1)), data])

    def predict(self, x_predict):
        new_x_predict = self._data_arrange(x_predict)
        return new_x_predict.dot(self._theta)

    #  算法估计
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        mse = np.sum((y_predict - y_test) ** 2) / len(y_test)
        return 1 - mse / np.var(y_test)

    def __repr__(self):
        return "多元线性回归"


if __name__ == '__main__':
    # boston = datasets.load_boston()
    # # print(boston.feature_names)
    # x = boston.data
    # y = boston.target
    # x = x[y < 50.0]
    # y = y[y < 50.0]

    x = pd.read_csv(r"E:\Projects\PycharmProjects\regression\datasets\housing.csv", usecols=['RM', 'LSTAT', 'PTRATIO'])
    y = pd.read_csv(r"E:\Projects\PycharmProjects\regression\datasets\housing.csv", usecols=['MEDV'])

    x = np.asarray(x)
    y = np.asarray(y)

    # data = pd.read_csv(r"E:\Projects\PycharmProjects\regression\datasets\housing.csv", header=0)
    # y = data['MEDV']
    # x = data.drop('MEDV', axis=1)


    # print(x.shape)
    # print(y.shape)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    #
    # print(x_train)

    # print(x_train)
    reg = LinearRegression()
    # #
    # reg.fit_normal(x_train, y_train)
    reg.fit_gd(x_train, y_train, eta=0.001, n_iters=1e4)
    # reg.fit_sgd(x_train, y_train, eta=0.01, n_iters=5)
    # #
    print("对应系数：", reg.coef_)
    print("截距：", reg.interception_)

    print("误差估计：", reg.score(x_test, y_test))
