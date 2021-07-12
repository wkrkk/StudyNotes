# k近邻法回归算法实现通过花萼长度、宽度和花瓣长度三个特征来预测花瓣的宽度。

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv(r"E:\Projects\PycharmProjects\knn\dataset\iris.arff.csv", header=0)
data.drop(["class"], axis=1, inplace=True)


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def predict(self, x):
        x = np.asarray(x)
        result = []
        for x in x:
            dis = np.sqrt(np.sum((x - self.x)**2, axis=1))
            index = dis.argsort()
            index = index[:self.k]
            result.append(np.mean(self.y[index]))

        return np.asarray(result)

    def predict2(self, x):
        x = np.asarray(x)
        result = []
        for x in x:
            dis = np.sqrt(np.sum((x - self.x) ** 2, axis=1))
            index = dis.argsort()
            index = index[:self.k]
            s = np.sum(1 / (dis[index] + 0.001))
            weight = (1 / (dis[index] + 0.001)) / s
            result.append(np.sum(self.y[index] * weight))

        return np.asarray(result)


if __name__ == '__main__':

    t = data.sample(len(data), random_state=0)
    train_X = t.iloc[:120, :-1]
    train_y = t.iloc[:120, -1]
    test_X = t.iloc[120:, :-1]
    test_y = t.iloc[120:, -1]
    # print(np.array(train_X))

    knn = KNN(k=3)
    knn.fit(train_X, train_y)
    result = knn.predict(test_X)
    print(result)
    # result2 = knn.predict2(test_X)
    # print(result2)
    # print(test_y.values)
    #
    # print(np.mean((result - test_y) ** 2))
    # print(np.mean((result2 - test_y) ** 2))

    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(train_X,train_y)
    print(neigh.predict(test_X))


