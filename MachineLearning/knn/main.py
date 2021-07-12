import numpy as np
from math import sqrt
from collections import Counter

# 训练数据、标签
film_data = [[100, 5],
             [95, 3],
             [105, 31],
             [2, 59],
             [3, 60],
             [10, 80]]

film_labels = [0, 0, 0, 1, 1, 1]

film_train_data = np.array(film_data)
film_train_labels = np.array(film_labels)

# 测试数据
# film_data_A = np.array([5, 70])

film_data_A = np.array([[5, 70],
                        [100, 2],
                        [4, 60]])


# print(film_train_data)
# print(film_train_labels)

def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)


class KNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equal to the size of y_train"
        assert self.k <= x_train.shape[0], \
            "the size of x_train must be at least k."

        self._x_train = x_train
        self._y_train = y_train
        return self

    def predict(self, x_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._x_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert x_predict.shape[1] == self._x_train.shape[1], \
            "the feature number of x_predict must be equal to x_train"

        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert x.shape[0] == self._x_train.shape[1], \
            "the feature number of x must be equal to x_train"

        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._x_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, x_test, y_test):
        """根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k


if __name__ == '__main__':
    knn_clf = KNNClassifier(k=3)
    knn_clf.fit(film_train_data, film_train_labels)
    # 将其转换为二维数据
    print(film_data_A)
    film_data_A = film_data_A.reshape(3, -1)
    print(film_data_A)
    predict_labels = knn_clf.predict(film_data_A)
    print(predict_labels)
