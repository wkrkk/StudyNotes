# 一元线性规划
# 仅使用房间数预测房价

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class SimpleLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        '''

        :param x_train: 属性数据集
        :param y_train: 标签数据集
        :return: a 回归系数，b 截距
        '''

        assert x_train.ndim == 1;
        assert len(x_train) == len(y_train);

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # num = 0.0
        # d = 0.0
        # for x_i, y_i in zip(x_train, y_train):
        #     num += (x_i - x_mean) * (y_i - y_mean)
        #     d += (x_i - x_mean) **2

        # 向量运算
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        # 预测一组数据

        assert x_predict.ndim == 1;
        assert self.a_ is not None and self.b_ is not None;

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):

        return self.a_ * x_single + self.b_

    # 算法评估
    def score(self, y_test, y_predict):
        mes = np.sum((y_predict - y_test) ** 2) / len(y_test)
        return 1 - mes / np.var(y_test)

    def __repr__(self):

        return "一元线性规划"


if __name__ == '__main__':

    # x = np.asarray([1, 2, 3, 4, 5])
    # y = np.asarray([1, 3, 2, 3, 5])
    #
    # x_predict = np.asarray([1.2, 6])
    #
    # reg1 = SimpleLinearRegression()
    # reg1.fit(x, y)
    # print(reg1.predict(x_predict))

    # !!! 只使用房间个数来预测房价
    boston = datasets.load_boston()
    # print(boston.feature_names)
    x = boston.data[:, 5]
    y = boston.target
    x = x[y < 50.0]
    y = y[y < 50.0]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    reg = SimpleLinearRegression()
    reg.fit(x_train, y_train)

    plt.scatter(x_train, y_train)
    plt.scatter(x_test, y_test, color='y')
    plt.plot(x_train, reg.predict(x_train), color='r')
    plt.scatter(x_test, reg.predict(x_test), color='pink')
    plt.show()

    print(reg.score(y_test, reg.predict(x_test)))

