'''
2. 正则化模型：
	2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
	2.2：使用随机删除节点的方法精简模型，同样是为了尝试避免过拟合。
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import week1.init_utils
import week1.reg_utils
import week1.gc_utils

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = week1.reg_utils.load_2D_dataset(is_plot=True)
plt.show()


# 随机舍弃节点的前向传播
def forward_propapagation_with_dropout(X, parameters, keep_prob):
    '''
    :param X:输入数据集，维度为（2，示例数
    :param parameters:包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
                        W1:权重矩阵，维度为（20,2）
                        b1:偏向量，维度为（20,1）
                        W2:权重矩阵，维度为（3,20）
                        b2:偏向量，维度为（3,1）
                        W3:权重矩阵，维度为（1,3）
                        b3:偏向量，维度为（1,1）
    :param keep_prob:随机删除的概率，实数
    :return:
        A3:最后的激活值，维度为（1,1），正向传播的输出
        cache:存储了一些用于计算反向传播的数值的元组
    '''

    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = week1.reg_utils.relu(Z1)

    # 第二层
    # 步骤1：初始化矩阵D1 = np.random.rand(..., ...)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    # 步骤2：将D1的值转换为0或1（使用keep_prob作为阈值）
    D1 = D1 < keep_prob
    # 步骤3：舍弃A1的一些节点（将它的值变为0或False）
    A1 = A1 * D1
    # 步骤4：缩放未舍弃的节点(不为0)的值
    A1 = A1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = week1.reg_utils.relu(Z2)

    # 第三层
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = week1.reg_utils.relu(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

# 正则化向前传播计算成本
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    '''

    :param A3:正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
    :param Y:标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
    :param parameters:包含模型学习后的参数的字典
    :param lambd:
    :return:
        cost:计算正则化损失值
    '''

    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = week1.reg_utils.compute_cost(A3, Y)
    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost
    return cost


# 正则化向后传播
def backward_propagation_with_regularization(X, Y, cache, lambd):
    '''

    :param X:输入数据集，维度为（输入节点数量，数据集里面的数量）
    :param Y:标签，维度为（输出节点数量，数据集里面的数量
    :param cache:来自forward_propagation（）的cache输出
    :param lambd:regularization超参数，实数
    :return:
        gradients:一个包含了每个参数、激活值和预激活值变量的梯度的字典
    '''

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = (1/m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

# 随机删除节点的后向传播
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    '''

    :param X:输入数据集，维度为（2，示例数）
    :param Y:标签，维度为（输出节点数量，示例数量）
    :param cache:来自forward_propagation_with_dropout（）的cache输出
    :param keep_prob:随机删除的概率，实数
    :return:
        gradients:一个关于每个参数、激活值和预激活变量的梯度值的字典
    '''

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA2 = dA2 * D2
    # 步骤2：缩放未舍弃的节点(不为0)的值
    dA2 = dA2 / keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def model(X, Y, lr=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):
    '''

    :param X:输入的数据，维度为(2, 要训练/测试的数量)
    :param Y:标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
    :param lr:学习速率
    :param num_iterations:迭代的次数
    :param print_cost:是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
    :param is_plot:是否绘制梯度下降的曲线图
    :param lambd:正则化的超参数，实数
    :param keep_prob:随机删除节点的概率
    :return:
        parameters:学习后的参数
    '''

    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    # 初始化参数
    parameters = week1.reg_utils.initialize_parameters(layers_dims)

    # 开始学习
    for i in range(0, num_iterations):
        # 不随机删除节点
        if keep_prob == 1:
            a3, cache = week1.reg_utils.forward_propagation(X, parameters)
            # 随机删除节点
        elif keep_prob < 1:
            a3, cache = forward_propapagation_with_dropout(X, parameters, keep_prob)
        else:
            print("Keep_prob参数错误！程序退出。")
            exit

        # 计算成本
        # 是否使用二范数
        if lambd == 0:
            # 不使用L2正则化
            cost = week1.reg_utils.compute_cost(a3, Y)
        else:
            # 使用L2正则化
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # 不同时使用正则化和随机删除节点
        assert (lambd == 0 or keep_prob == 1)

        if (lambd == 0 and keep_prob == 1):
            # 不使用L2正则化和不使用随机删除节点
            grads = week1.reg_utils.backward_propagation(X, Y, cache)
        elif lambd != 0:
            # 使用L2正则化，不使用随机删除节点
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            # 使用随机删除节点，不使用L2正则化
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        parameters = week1.reg_utils.update_parameters(parameters, grads, lr)

        # 记录并打印成本
        if i % 1000 == 0:
            ## 记录成本
            costs.append(cost)
            if (print_cost and i % 10000 == 0):
                # 打印成本
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

        # 是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(lr))
        plt.show()

    return parameters


parameters = model(train_X, train_Y, keep_prob=0.86, lr=0.3, is_plot=True)
print("训练集:")
predictions_train = week1.reg_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = week1.reg_utils.predict(test_X, test_Y, parameters)

plt.title("Model without dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
week1.reg_utils.plot_decision_boundary(lambda x: week1.reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
