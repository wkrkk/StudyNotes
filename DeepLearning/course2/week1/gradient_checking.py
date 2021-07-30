'''
3. 梯度校验：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况。
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import week1.init_utils
import week1.reg_utils
import week1.gc_utils
import week1.testCases

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
一维线性
'''


def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J


def backward_propagation(x, theta):
    dtheta = x
    return dtheta


# 梯度检验
def gradient_check(x, theta, epsilon=1e-7):
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x, thetaplus)
    J_minus = forward_propagation(x, thetaminus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    grad = backward_propagation(x, theta)

    # 计算误差
    # np.linalg.norm求范数
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")

    return difference


# # 测试gradient_check
# print("-----------------测试gradient_check-----------------")
# x, theta = 2, 4
# difference = gradient_check(x, theta)
# print("difference = " + str(difference))

'''
高维梯度检验
'''


def forward_propagation_n(X, Y, parameters):
    '''

    :param X:训练集为m个例子
    :param Y:m个示例的标签
    :param parameters:包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
                        W1:权重矩阵，维度为（5,4）
                        b1:偏向量，维度为（5,1）
                        W2:权重矩阵，维度为（3,5）
                        b2:偏向量，维度为（3,1）
                        W3:权重矩阵，维度为（1,3）
                        b3:偏向量，维度为（1,1）

    :return:
        cost:成本函数
    '''

    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = week1.gc_utils.relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = week1.gc_utils.relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = week1.gc_utils.sigmoid(Z3)

    # 计算成本
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = (1 / m) * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backward_propagation_n(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    # dW2 = 1. / m * np.dot(dZ2, A1.T) * 2
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    # db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    # 初始化参数
    parameters_values, keys = week1.gc_utils.dictionary_to_vector(parameters)
    grad = week1.gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # 计算gradapprox
    for i in range(num_parameters):
        # 计算J_plus [i]。输入：“parameters_values，epsilon”。输出=“J_plus [i]”
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        J_plus[i], cache = forward_propagation_n(X, Y, week1.gc_utils.vector_to_dictionary(thetaplus))

        # 计算J_minus [i]。输入：“parameters_values，epsilon”。输出=“J_minus [i]”。
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        J_minus[i], cache = forward_propagation_n(X, Y, week1.gc_utils.vector_to_dictionary(thetaminus))

        # 计算gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # 通过计算差异比较gradapprox和后向传播梯度。
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'

    if difference < 1e-7:
        print("梯度检查：梯度正常!")
    else:
        print("梯度检查：梯度超出阈值!")

    return difference

X, Y, parameters = week1.testCases.gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)
print(difference)

