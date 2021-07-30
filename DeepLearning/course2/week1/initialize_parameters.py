'''
1. 初始化参数：
	1.1：使用0来初始化参数。
	1.2：使用随机数来初始化参数。
	1.3：使用抑梯度异常初始化参数（参见梯度消失和梯度爆炸）。

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


train_X, train_Y, test_X, test_Y = week1.init_utils.load_dataset(is_plot=True)
plt.show()


# 初始化参数为0
def initialize_parameters_zeros(layers_dims):
    parameters = {}

    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# 随机初始化
def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# 抑梯度异常初始化
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l -1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters


# 定义一个三层网络模型
def model(X, Y, lr=0.01, num_iterations=15000, print_cost=True, initialization="he", is_polt=True):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    # 选择初始化参数的类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else:
        print("错误的初始化参数！程序退出")
        exit

    # 开始学习
    for i in range(0, num_iterations):
        a3, cache = week1.init_utils.forward_propagation(X, parameters)
        cost = week1.init_utils.compute_loss(a3, Y)
        grads = week1.init_utils.backward_propagation(X, Y, cache)
        parameters = week1.init_utils.update_parameters(parameters, grads, lr)

        if i % 1000 == 0:
            costs.append(cost)

            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    # 学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(lr))
        plt.show()

    return parameters

parameters = initialize_parameters_he([2,4,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization = "he",is_polt=True)
plt.show()

print ("训练集:")
predictions_train = week1.init_utils.predict(train_X, train_Y, parameters)
print ("测试集:")
predictions_test = week1.init_utils.predict(test_X, test_Y, parameters)

print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
week1.init_utils.plot_decision_boundary(lambda x: week1.init_utils.predict_dec(parameters, x.T), train_X, train_Y)
