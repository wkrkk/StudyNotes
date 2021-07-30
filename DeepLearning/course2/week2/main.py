'''
1. 分割数据集
2. 优化梯度下降算法：
    2.1 不使用任何优化算法
    2.2 mini-batch梯度下降法
    2.3 使用具有动量的梯度下降算法
    2.4 使用Adam算法
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import week2.opt_utils
import week2.testCase

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
梯度下降
'''


def update_parameters_with_gd(parameters, grads, lr):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - lr * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - lr * grads["db" + str(l + 1)]

    return parameters


# #测试update_parameters_with_gd
# print("-------------测试update_parameters_with_gd-------------")
# parameters , grads , learning_rate = week2.testCase.update_parameters_with_gd_test_case()
# parameters = update_parameters_with_gd(parameters,grads,learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

'''
获取小批量数据
'''


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    '''

    :param X:输入数据，维度为(输入节点数量，样本的数量)
    :param Y:对应的是X的标签，【1 | 0】（蓝|红），维度为(1,样本的数量)
    :param mini_batch_size:每个mini-batch的样本数量
    :param seed:
    :return:
        mini-bacthes:一个同步列表，维度为（mini_batch_X,mini_batch_Y）
    '''

    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 打乱顺序
    # 它会返回一个长度为m的随机数组，且里面的数是0到m-1
    permutation = list(np.random.permutation(m))
    # 将每一列的数据按permutation的顺序来重新排列
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # 分割
    # match.floor:返回小于或等于一个给定数字的最大整数
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 获取剩余部分
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# #测试random_mini_batches
# print("-------------测试random_mini_batches-------------")
# X_assess,Y_assess,mini_batch_size = week2.testCase.random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess,Y_assess,mini_batch_size)
#
# print("第1个mini_batch_X 的维度为：",mini_batches[0][0].shape)
# print("第1个mini_batch_Y 的维度为：",mini_batches[0][1].shape)
# print("第2个mini_batch_X 的维度为：",mini_batches[1][0].shape)
# print("第2个mini_batch_Y 的维度为：",mini_batches[1][1].shape)
# print("第3个mini_batch_X 的维度为：",mini_batches[2][0].shape)
# print("第3个mini_batch_Y 的维度为：",mini_batches[2][1].shape)

'''
包含动量的梯度下降
'''


# 初始化速度
def initialize_velocity(parameters):
    '''

    :param parameters:一个字典，包含了以下参数：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    :return:
        v:一个字典变量，包含了以下参数：
            v["dW" + str(l)] = dWl的速度
            v["db" + str(l)] = dbl的速度
    '''

    L = len(parameters) // 2
    v = {}

    for l in range(L):
        # 生成一个和所给数组一样大小的全零数组
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v


# 使用动量更新参数
def update_parameters_with_momentun(parameters, grads, v, beta, lr):
    '''

    :param parameters:一个字典类型的变量，包含了以下字段：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    :param grads:一个包含梯度值的字典变量，具有以下字段：
            grads["dW" + str(l)] = dWl
            grads["db" + str(l)] = dbl
    :param v:包含当前速度的字典变量，具有以下字段：
            v["dW" + str(l)] = ...
            v["db" + str(l)] = ...
    :param beta:超参数，动量，实数
    :param lr:学习率，实数
    :return:
        parameters:更新后的参数字典
        v:包含了更新后的速度变量
    '''

    L = len(parameters) // 2
    for l in range(L):
        # 计算速度
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

        # 更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - lr * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - lr * v["db" + str(l + 1)]

    return parameters, v


# #测试update_parameters_with_momentun
# print("-------------测试update_parameters_with_momentun-------------")
# parameters,grads,v = week2.testCase.update_parameters_with_momentum_test_case()
# update_parameters_with_momentun(parameters,grads,v,beta=0.9,lr=0.01)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))

'''
Adam算法
'''

# 参数初始化
def initialize_adam(parameters):
    '''

    :param parameters:包含了以下参数的字典变量：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    :return:v:包含梯度的指数加权平均值，字段如下：
                v["dW" + str(l)] = ...
                v["db" + str(l)] = ...
            s:包含平方梯度的指数加权平均值，字段如下：
                s["dW" + str(l)] = ...
                s["db" + str(l)] = ...

    '''

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return (v, s)

# Adam更新参数
def update_parameters_with_adam(parameters, grads, v, s, t, lr=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8):
    '''

    :param parameters:包含了以下字段的字典：
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
    :param grads:包含了梯度值的字典，有以下key值：
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
    :param v:Adam的变量，第一个梯度的移动平均值，是一个字典类型的变量
    :param s:Adam的变量，平方梯度的移动平均值，是一个字典类型的变量
    :param t:当前迭代的次数
    :param lr:学习率
    :param beta1:动量，超参数
    :param beta2:RMSprop的一个参数，超参数
    :param epsilon:防止除零操作（分母为0
    :return:
        parameters:更新后的参数
        v:第一个梯度的移动平均值，是一个字典类型的变量
        s:平方梯度的移动平均值，是一个字典类型的变量
    '''

    L = len(parameters) // 2
    # 偏差修正后的值
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        # 梯度的移动平均值
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # 计算第一阶段的偏差修正后的估计值
        v_corrected["dW"+str(l+1)] = v["dW" + str(l + 1)] / (1-np.power(beta1,t))
        v_corrected["db" + str(l + 1)]=v["db" + str(l + 1)] / (1-np.power(beta1,t))

        # 计算平均梯度的移动平均值
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])

        # 计算第二阶段的偏差修正后的估计值
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        # 更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - lr * (
                    v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - lr * (
                    v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))

    return (parameters, v, s)

# #测试update_with_parameters_with_adam
# print("-------------测试update_with_parameters_with_adam-------------")
# parameters , grads , v , s = week2.testCase.update_parameters_with_adam_test_case()
# update_parameters_with_adam(parameters,grads,v,s,t=2)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
# print('s["dW1"] = ' + str(s["dW1"]))
# print('s["db1"] = ' + str(s["db1"]))
# print('s["dW2"] = ' + str(s["dW2"]))
# print('s["db2"] = ' + str(s["db2"]))

def model(X, Y, layers_dims, optimizer, lr=0.0007,
          mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):
    """
    可以运行在不同优化器模式下的3层神经网络模型。

    参数：
        X - 输入数据，维度为（2，输入的数据集里面样本数量）
        Y - 与X对应的标签
        layers_dims - 包含层数和节点数量的列表
        optimizer - 字符串类型的参数，用于选择优化类型，【 "gd" | "momentum" | "adam" 】
        lr - 学习率
        mini_batch_size - 每个小批量数据集的大小
        beta - 用于动量优化的一个超参数
        beta1 - 用于计算梯度后的指数衰减的估计的超参数
        beta1 - 用于计算平方梯度后的指数衰减的估计的超参数
        epsilon - 用于在Adam中避免除零操作的超参数，一般不更改
        num_epochs - 整个训练集的遍历次数，（视频2.9学习率衰减，1分55秒处，视频中称作“代”）,相当于之前的num_iteration
        print_cost - 是否打印误差值，每遍历1000次数据集打印一次，但是每100次记录一个误差值，又称每1000代打印一次
        is_plot - 是否绘制出曲线图

    返回：
        parameters - 包含了学习后的参数

    """
    L = len(layers_dims)
    costs = []
    t = 0  # 每学习完一个minibatch就增加1
    seed = 10  # 随机种子

    # 初始化参数
    parameters = week2.opt_utils.initialize_parameters(layers_dims)

    # 选择优化器
    if optimizer == "gd":
        pass  # 不使用任何优化器，直接使用梯度下降法
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)  # 使用动量
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)  # 使用Adam优化
    else:
        print("optimizer参数错误，程序退出。")
        exit(1)

    # 开始学习
    for i in range(num_epochs):
        # 定义随机 minibatches,我们在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            # 选择一个minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # 前向传播
            A3, cache = week2.opt_utils.forward_propagation(minibatch_X, parameters)

            # 计算误差
            cost = week2.opt_utils.compute_cost(A3, minibatch_Y)

            # 反向传播
            grads = week2.opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)

            # 更新参数
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, lr)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentun(parameters, grads, v, beta, lr)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, lr, beta1, beta2,
                                                               epsilon)
        # 记录误差值
        if i % 100 == 0:
            costs.append(cost)
            # 是否打印误差值
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))

    # 是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(lr))
        plt.show()

    return parameters

if __name__ == '__main__':

    # 加载数据集
    train_X, train_Y = week2.opt_utils.load_dataset(is_plot=True)

    layers_dims = [train_X.shape[0], 5, 2, 1]
    # 使用普通的梯度下降
    # parameters = model(train_X, train_Y, layers_dims, optimizer="gd", is_plot=True)
    # 使用动量的梯度下降
    # parameters = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum", is_plot=True)
    # 使用Adam优化的梯度下降
    parameters = model(train_X, train_Y, layers_dims, optimizer="adam", is_plot=True)

    # 预测
    preditions = week2.opt_utils.predict(train_X, train_Y, parameters)

    # 绘制分类图
    # plt.title("Model with Gradient Descent optimization")
    # plt.title("Model with Momentum optimization")
    plt.title("Model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    week2.opt_utils.plot_decision_boundary(lambda x: week2.opt_utils.predict_dec(parameters, x.T), train_X, train_Y)



