import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# 读取数据
data = pd.read_csv(r"E:\Projects\PycharmProjects\knn\dataset\iris.arff.csv", header=0)
# print(data)

# 将原本的类被名称映射为数字
data["class"] = data["class"].map({"Iris-versicolor": 0, "Iris-setosa": 1, "Iris-virginica": 2})
# print(data)

# 删除其中的重复数据
if data.duplicated().any():
    data.drop_duplicates(inplace=True)
    # print(len(data))


# 查看各个类别的鸢尾花的个数
# print(data["class"].value_counts())


# KNN实现分类
class KNN:

    # 初始化邻居数
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        '''

        :param x:训练数据，类型为[[特征向量],[]...]
        :param y:类别，类型为 [标签（标签）]

        '''

        '''
        x：[[6.  2.9 4.5 1.5]
            [5.9 3.  4.2 1.5]
            [5.  2.  3.5 1. ]
            ...
            []]
        y：[0 0 0 0 0 0 0 0 0....]
        '''
        self.x = np.asarray(x)
        self.y = np.asarray(y)


    def predict(self, x):
        '''

        :param x:测试数据，类型为[[特征向量],[]...]
        :return:预测结果，类型：数组
        '''

        # x为（27，4）
        '''
        [[6.7 3.1 4.7 1.5]
        [6.1 2.8 4.0 1.3]
        ...
        [5.6 2.5 3.9 1.1]]
        '''
        x = np.asarray(x)
        # 存放预测结果
        result = []
        # 对narray数据进行遍历，每次取数组中的一行
        for x in x:
            # 对于测试集中的每个样本，一次与训练集中的所有数据求欧式距离
            dis = np.sqrt(np.sum((x - self.x) ** 2, axis=1))
            # 返回数组排序后，每个元素在原数组（排序之前的数组）中的索引,并取出前k个最近的下标索引
            index = dis.argsort()
            index = index[:self.k]
            # 查找y的标签（投票法）。返回数组中每个整数元素出现次数，元素必须是非负整数
            count = np.bincount(self.y[index])
            # 返回ndarray中值最大的元素所对应的索引，就是出现次数最多的索引，也就是我们判定的类别
            result.append(count.argmax())

        return np.asarray(result)

    def predict2(self, x):
        '''

        加入权重计算
        :param x:测试数据，类型为[[特征向量],[]...]
        :return:预测结果
        '''

        x = np.asarray(x)
        result = []
        for x in x:
            dis = np.sqrt(np.sum((x - self.x) ** 2, axis=1))
            index = dis.argsort()
            index = index[:self.k]
            # 考虑权重，使用距离的倒数作为权重，对位的权重进行相加求和
            count = np.bincount(self.y[index], weights=1 / dis[index])
            # result:[0,0,0,...]
            result.append(count.argmax())
        # return:[0 0 0 ...]
        return np.asarray(result)


if __name__ == '__main__':
    # 提取每个类中鸢尾花的数据
    t0 = data[data["class"] == 0]
    t1 = data[data["class"] == 1]
    t2 = data[data["class"] == 2]

    # 打乱每个类别中的数据
    t0 = t0.sample(len(t0), random_state=0)
    t1 = t1.sample(len(t1), random_state=0)
    t2 = t2.sample(len(t2), random_state=0)

    '''
    划分训练集和测试集，每类中的前40条用于训练，
    并取出除class类别的数据作为x,最后一个class类别数据作为y
    '''
    train_x = pd.concat([t0.iloc[:40, :-1], t1.iloc[:40, :-1], t2.iloc[:40, :-1]], axis=0)
    train_y = pd.concat([t0.iloc[:40, -1], t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
    test_x = pd.concat([t0.iloc[40:, :-1], t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)
    test_y = pd.concat([t0.iloc[40:, -1], t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)

    knn = KNN(3)
    knn.fit(x=train_x, y=train_y)
    result = knn.predict(test_x)
    # result = knn.predict2(test_x)
    print(result)
    # 分类准确率
    print(np.sum(result == test_y) / len(result))

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_x, train_y.astype('int'))
    print(neigh.predict(test_x))