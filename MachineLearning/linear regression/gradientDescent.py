import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(666)
# 100个样本，每个样本有一个特征
x = 2 * np.random.random(size=100)
# 均值为0，方差为1
y = x * 3. + 4. + np.random.normal(size=100)
# 100行，一列数据
X = x.reshape(-1, 1)



class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
    #     """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
    #     assert X_train.shape[0] == y_train.shape[0], \
    #         "the size of X_train must be equal to the size of y_train"
    #
    #     def J(theta, X_b, y):
    #         try:
    #             # 计算代价损失函数——一个数值
    #             return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
    #         except:
    #             return float('inf')
    #
    #     def dJ(theta, X_b, y):
    #         # 随机产生数据，返回一个一维或多维数组
    #         res = np.empty(len(theta))
    #         res[0] = np.sum(X_b.dot(theta) - y)
    #         for i in range(1, len(theta)):
    #             res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
    #         return res * 2 / len(X_b)
    #
    #     def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    #
    #         theta = initial_theta
    #         cur_iter = 0
    #
    #         while cur_iter < n_iters:
    #             gradient = dJ(theta, X_b, y)
    #             last_theta = theta
    #             theta = theta - eta * gradient
    #             if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
    #                 break
    #
    #             cur_iter += 1
    #
    #         return theta
    #
    #     X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    #     initial_theta = np.zeros(X_b.shape[1])
    #     self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
    #     print(self._theta)
    #     self.intercept_ = self._theta[0]
    #     self.coef_ = self._theta[1:]
    #
    #     return self


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
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def _data_arrange(self, data):
        # 进行数据整理--在数据矩阵第一列加一列均为 1 得列
        # 在水平方向上平铺
        return np.hstack([np.ones((len(data), 1)), data])

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)


    def __repr__(self):
        return "LinearRegression()"

if __name__ == '__main__':


    lin_reg = LinearRegression()
    lin_reg.fit_sgd(X, y)
    print(lin_reg._theta)
    print(lin_reg.coef_)
    print(lin_reg.intercept_)
