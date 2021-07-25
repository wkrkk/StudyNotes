import numpy as np
import h5py
import matplotlib.pyplot as plt
import week4.testCases
from week4.dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import week4.lr_utils

np.random.seed(1)


# 多层神经网络
def initialize_parameters_deep(layers_dims):
    '''

    :param layers_dims:网络中每个层中的节点数量的列表
    :return:
        parameters - 包含参数“W1”，“b1”，...，“WL”，“bL”的字典：
                     Wl - 权重矩阵，维度为（layers_dims [l]，layers_dims [l-1]）
                     bl - 偏向量，维度为（layers_dims [l]，1）

    '''

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters


# 前向传播中线性部分
def linear_forward(A, W, b):
    '''

    :param A:来自上一层（或输入数据）的激活，维度为(上一层的节点数量，示例的数量）
    :param W:权重矩阵，numpy数组，维度为（当前图层的节点数量，前一图层的节点数量）
    :param b:偏向量，numpy向量，维度为（当前图层节点数量，1）
    :return:
        Z:激活功能的输入，也称为预激活参数
        cache:一个包含“A”，“W”和“b”的字典，存储这些变量以有效地计算后向传递
    '''

    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


# 线性激活部分函数
def linear_activation_forward(A_prev, W, b, activation):
    '''

    :param A_prev:来自上一层（或输入层）的激活，维度为(上一层的节点数量，示例数）
    :param W:权重矩阵，numpy数组，维度为（当前层的节点数量，前一层的大小）
    :param b:偏向量，numpy阵列，维度为（当前层的节点数量，1）
    :param activation:选择在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    :return:
         A:激活函数的输出，也称为激活后的值
         cache - 一个包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效地计算后向传递
    '''

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# 多层模型的前向传播
def L_model_forward(X, parameters):
    '''

    :param X:数据，numpy数组，维度为（输入节点数量，示例数）
    :param parameters:initialize_parameters_deep（）的输出
    :return:
        AL:最后的激活值
        caches:包含以下内容的缓存列表：
                 linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）
                 linear_sigmoid_forward（）的cache（只有一个，索引为L-1）
    '''

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


# 计算成本
def compute_cost(AL, Y):
    '''

    :param AL:与标签预测相对应的概率向量，维度为（1，示例数量）
    :param Y:标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）
    :return:
        cost:交叉熵成本
    '''

    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


# 向后传播线性部分
def linear_backward(dZ, cache):
    '''

    :param dZ:相对于（当前第l层的）线性输出的成本梯度
    :param cache:来自当前层前向传播的值的元组（A_prev，W，b）
    :return:
        dA_prev:相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
        dW:相对于W（当前层l）的成本梯度，与W的维度相同
        db:相对于b（当前层l）的成本梯度，与b维度相同
    '''

    A_pre, W, b = cache
    m = A_pre.shape[1]
    dW = np.dot(dZ, A_pre.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_pre.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# 向后传播线性激活部分
def linear_activation_backward(dA, cache, activation="relu"):
    '''

    :param dA:当前层l的激活后的梯度值
    :param cache:存储的用于有效计算反向传播的值的元组(值为linear_cache，activation_cache)
    :param activation:要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    :return:
        dA_prev:相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
        dW:相对于W（当前层l）的成本梯度，与W的维度相同
        db:相对于b（当前层l）的成本梯度，与b维度相同
    '''

    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# 多层模型的向后传播函数
def L_model_backward(AL, Y, caches):
    '''

    :param AL:概率向量，正向传播的输出（L_model_forward（））
    :param Y:标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）
    :param caches:包含以下内容的cache列表：
                 linear_activation_forward（"relu"）的cache，不包含输出层
                 linear_activation_forward（"sigmoid"）的cache
    :return:
        grads:具有梯度值的字典
              grads [“dA”+ str（l）] = ...
              grads [“dW”+ str（l）] = ...
              grads [“db”+ str（l）] = ...
    '''

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# 更新参数
def update_parameters(parameters, grads, lr):
    '''

    :param parameters:参数的字典
    :param grads:梯度值的字典
    :param lr:学习率
    :return:
        parameters:包含更新参数的字典
                   参数[“W”+ str（l）] = ...
                   参数[“b”+ str（l）] = ...
    '''

    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - lr * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - lr * grads["db" + str(l + 1)]

    return parameters


# 搭建多层神经网络
def L_layer_model(X, Y, layer_dims, lr=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    '''

    :param X:输入的数据，维度为(n_x，例子数)
    :param Y:标签，向量，0为非猫，1为猫，维度为(1,数量)
    :param layer_dims:层数的向量，维度为(n_y,n_h,···,n_h,n_y)
    :param lr:学习率
    :param num_iterations:迭代的次数
    :param print_cost:是否打印成本值，每100次打印一次
    :param isPlot:是否绘制出误差值的图谱
    :return:
        parameters:模型学习的参数。 然后他们可以用来预测
    '''

    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layer_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, lr)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(cost))

    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(lr))
        plt.show()

    return parameters


# 预测
def predict(X, y, parameters):
    '''

    :param X:测试集
    :param y:标签
    :param parameters:训练模型的参数
    :return:
        p：给定数据集X的预测
    '''

    m = X.shape[1]
    # 神经网络的层数
    n = len(parameters) // 2
    p = np.zeros((1, m))

    #根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("准确度为: " + str(float(np.sum((p == y)) / m)))

    return p


if __name__ == '__main__':
    # 加载数据
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = week4.lr_utils.load_dataset()

    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    train_y = train_set_y
    test_x = test_x_flatten / 255
    test_y = test_set_y

    layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True, isPlot=True)

    # 预测
    pred_train = predict(train_x, train_y, parameters)  # 训练集
    pred_test = predict(test_x, test_y, parameters)  # 测试集
