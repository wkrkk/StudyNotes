import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import week3.tf_utils
import time


# 线性函数
def linear_function():
    '''
        初始化W，类型为tensor的随机变量，维度为(4,3)
        初始化X，类型为tensor的随机变量，维度为(3,1)
        初始化b，类型为tensor的随机变量，维度为(4,1)
    :return:
        result:运行了session后的结果，运行的是Y = WX + b
    '''

    np.random.seed(1)

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    # tf.matmul矩阵乘法
    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)

    sess.close()

    return result


def sigmoid(z):
    # 创建一个占位符x
    x = tf.placeholder(tf.float32, name="x")

    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})

    return result


# 独热编码，将标签和分类对应
def one_hot_matrix(lables, C):
    '''

    :param lables:标签向量
    :param C:分类数
    :return:
        one_hot:独热矩阵
    '''

    C = tf.constant(C, name="C")
    one_hot_matrix = tf.one_hot(indices=lables, depth=C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot


# # 测试
# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels,C=4)
# print(str(one_hot))

# 创建占位符
def create_placeholders(n_x, n_y):
    '''

    :param n_x:一个实数，图片向量的大小（64*64*3 = 12288）
    :param n_y:一个实数，分类数（从0到5，所以n_y = 6）
    :return:
        X:一个数据输入的占位符，维度为[n_x, None]，dtype = "float"
        Y:一个对应输入的标签的占位符，维度为[n_Y,None]，dtype = "float"
    '''

    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y


# 初始化参数
def initialize_parameters():
    '''
    初始化神经网络的参数，参数的维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]
    :return:
        parameters:包含了W和b的字典
    '''

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


# 前向传播
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


# 计算成本
def comput_cost(Z3, Y):
    '''

    :param Z3:前向传播的结果
    :param Y:标签，一个占位符，和Z3的维度相同
    :return:
        cost:成本值
    '''
    logits = tf.transpose(Z3)  # 转置
    labels = tf.transpose(Y)  # 转置

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


# 构建模型
def model(X_train, Y_train, X_test, Y_test,
          lr=0.0001, num_epochs=1500, minibatch_size=32,
          print_cost=True, is_plot=True):
    # 重新运行模型而不覆盖tf变量
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    # 获取输入节点数量和样本数
    (n_x, m) = X_train.shape
    # 获取输出节点数量
    n_y = Y_train.shape[0]
    costs = []

    # 给X和Y创建placeholder
    X, Y = create_placeholders(n_x, n_y)
    # 初始化参数
    parameters = initialize_parameters()
    # 前向传播
    Z3 = forward_propagation(X, parameters)
    # 计算成本
    cost = comput_cost(Z3, Y)
    # 反向传播，使用Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # 初始化所有变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 初始化
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = week3.tf_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches

            # 记录并打印成本
            ## 记录成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                # 是否打印：
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(lr))
            plt.show()

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        # tf.cast将张量数据类型转换
        # tf.reduce_mean求平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # 在一个Seesion里面“评估”tensor的值（其实就是计算）

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


if __name__ == '__main__':
    # X_train_orig.shape:(1080, 64, 64, 3)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = week3.tf_utils.load_dataset()

    # 显示训练集中某个数据
    # index = 24
    # plt.imshow(X_train_orig[index])
    # plt.show()
    # print("Y = " + str(np.squeeze(Y_train_orig[:, index])))

    # 1.对数据进行扁平化+归一化
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255

    # 2.独热矩阵
    Y_train = week3.tf_utils.convert_to_one_hot(Y_train_orig, 6)
    Y_test = week3.tf_utils.convert_to_one_hot(Y_test_orig, 6)

    # print("训练集样本数 = " + str(X_train.shape[1]))
    # print("测试集样本数 = " + str(X_test.shape[1]))
    # print("X_train.shape: " + str(X_train.shape))
    # print("Y_train.shape: " + str(Y_train.shape))
    # print("X_test.shape: " + str(X_test.shape))
    # print("Y_test.shape: " + str(Y_test.shape))

    # 开始时间
    start_time = time.clock()
    # 开始训练
    parameters = model(X_train, Y_train, X_test, Y_test)
    # 结束时间
    end_time = time.clock()
    # 计算时差
    print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")
