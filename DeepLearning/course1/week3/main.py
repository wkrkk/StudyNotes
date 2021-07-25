import numpy as np
import matplotlib.pyplot as plt
from week3.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from week3.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# 设置固定随机种子
np.random.seed(1)

# 加载数据
X, Y = load_planar_dataset()
# # cmap = plt.cm.Spectral实现的功能是给label为1的点一种颜色，给label为0的点另一种颜色
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# # plt.show()

'''
X的维度为: (2, 400)
Y的维度为: (1, 400)
数据集里面的数据有：400 个
'''
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]

'''
逻辑回归查看分类效果
'''
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# plt.show()
# LR_predictions = clf.predict(X.T)
# print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
#                                np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
#       "% " + "(正确标记的数据点所占的百分比)")

'''
构建神经网络一般方法：
1.定义神经网络结构
2.初始化模型参数
3.循环：前向传播、计算损失、后向传播、更新参数
'''


# 定义神经网络结构
def layer_sizes(X, Y):
    '''

    :param X:输入数据集,维度为（输入的数量，训练/测试的数量）
    :param Y:标签，维度为（输出的数量，训练/测试数量）
    :return:
        n_x:输入层的数量
        n_h:隐藏层的数量
        n_y:输出层的数量
    '''
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

# 初始化模型参数
def initialize_parameters(n_x, n_h, n_y):
    '''

    :param n_x:输入层节点的数量
    :param n_h:隐藏层节点的数量
    :param n_y:输出层节点的数量
    :return:
        parameters:包含参数的字典：
            W1:权重矩阵（n_h, n_x）
            b1:偏向量（n_h, 1）
            W2:权重矩阵（n_y, n_h）
            b2:偏向量（n_y, 1）
    '''

    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 使用断言确保我的数据格式是正确的
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
    }

    return parameters

# 前向传播
def forward_propagation(X, parameters):
    '''

    :param X:维度为（n_x，m）的输入数据
    :param parameters:初始化函数（initialize_parameters）的输出
    :return:
        A2:使用sigmoid()函数计算的第二次激活后的数值
        cache:包含“Z1”，“A1”，“Z2”和“A2”的字典类型变量
    '''

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 前向传播计算A2
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # 使用断言确保格式正确
    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2
    }

    return (A2, cache)

# 计算成本函数
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # 计算成本
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))

    assert (isinstance(cost, float))

    return cost

# 向后传播
def backward_propagation(parameters, cache, X, Y):
    '''

    :param parameters:包含参数的一个字典类型的变量
    :param cache:包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量
    :param X:输入数据，维度为（2，数量）
    :param Y:标签，维度为（1，数量）
    :return:
        grads:包含W和b的导数一个字典类型的变量
    '''

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2
    }

    return grads

# 更新参数
def update_parameters(parameters, grads, lr=1.2):
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
    }

    return parameters

# 整合功能
def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    '''

    :param X:数据集,维度为（2，示例数）
    :param Y:标签，维度为（1，示例数）
    :param n_h: 隐藏层的数量
    :param num_iterations:梯度下降循环中的迭代次数
    :param print_cost:如果为True，则每1000次迭代打印一次成本数值
    :return:
        parameters:模型学习的参数，它们可以用来进行预测
    '''

    np.random.randn(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, lr=0.5)

        if print_cost:
            if i % 1000 == 0:
                print("第 ", i, " 次循环，成本为：" + str(cost))

    return parameters

# 预测
def predict(parameters, X):
    '''

    :param parameters:包含参数的字典类型的变量
    :param X:输入数据（n_x，m）
    :return:
        predictions:我们模型预测的向量（红色：0 /蓝色：1）
    '''

    A2, cache = forward_propagation(X, parameters)
    # 用于单隐藏层的二分类神经网络，大于0.5输出为1，小于等于0.5输出0
    predictions = np.round(A2)

    return predictions

if __name__ == '__main__':
    # parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    #
    # # 绘制边界
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(4))
    # plt.show()
    #
    # predictions = predict(parameters, X)
    # print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    # 更改隐藏层节点的数量
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量
    for i, n_h in enumerate(hidden_layer_sizes):
        # 画子图，规定5行2列及画图序号
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))

    plt.show()
