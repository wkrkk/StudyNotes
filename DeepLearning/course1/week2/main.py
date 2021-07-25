import matplotlib.pyplot as plt
from week2.lr_utils import load_dataset
import numpy as np

# train_set_x_orig.shape：(209, 64, 64, 3)
# train_set_y.shape：(1, 209)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 查看数据集中的某个数据
# index = 25
# plt.imshow(train_set_x_orig[index])
# plt.show()
#
# print("y=" + str(train_set_y[:, index]) + ", it's a " + classes[np.squeeze(train_set_y[:, index])].decode(
#     "utf-8") + "' picture")

# 训练集里图片的数量
m_train = train_set_y.shape[1]
# 测试集里图片的数量
m_test = test_set_y.shape[1]
# 训练、测试集里面的图片的宽度和高度（均为64x64）
num_px = train_set_x_orig.shape[1]

# 加载数据的具体情况
print("训练集的数量: m_train = " + str(m_train))
print("测试集的数量 : m_test = " + str(m_test))
print("每张图片的宽/高 : num_px = " + str(num_px))
print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集_图片的维数: " + str(test_set_x_orig.shape))
print("测试集_标签的维数: " + str(test_set_y.shape))

# 将维度（64，64，3）重新构造为（64*64*3，1）
# 3的原因是每张图片是由64x64像素构成的，而每个像素点由（R，G，B）三原色构成的
# 将数据维度降低并转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print("测试集_标签的维数 : " + str(test_set_y.shape))

# 将数据集进行标准化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# 初始化参数w,b
def initialize_with_zeros(dim):
    '''

     此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0

    :param dim:我们想要的w矢量的大小（或者这种情况下的参数数量）
    :return:
          w  - 维度为（dim，1）的初始化向量
          b  - 初始化的标量（对应于偏差）
    '''

    w = np.zeros(shape=(dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return (w, b)


def propagate(w, b, X, Y):
    '''

    实现前向和后向传播的成本函数及其梯度。

    :param w:权重，大小不等的数组（num_px * num_px * 3，1）
    :param b:偏差，一个标量
    :param X:矩阵类型为（num_px * num_px * 3，训练数量）
    :param Y:真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)
    :return:
        cost:逻辑回归的负对数似然成本
        dw:相对于w的损失梯度，与w的形状相同
        db:相对于b的损失梯度，与b的形状相同
    '''

    m = X.shape[1]

    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    # 使用断言判断数据是否正确
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {
        "dw": dw,
        "db": db
    }

    return (grads, cost)


def optimize(w, b, X, Y, num_iterations, lr, print_cost=False):
    '''
    通过梯度下降优化参数
    :param w:权重，大小不等的数组（num_px * num_px * 3，1）
    :param b:偏差，一个标量
    :param X:维度为（num_px * num_px * 3，训练数据的数量）的数组
    :param Y:真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量）
    :param num_iterations:优化循环的迭代次数
    :param lr:梯度下降更新规则的学习率
    :param print_cost:每100步打印一次损失值
    :return:
        params:包含权重和偏差的字典
        grads:包含权重和偏差相对于成本函数的梯度的字典
        cost:优化期间计算的所有成本列表，将用于绘制学习曲线
    '''

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - lr * dw
        b = b - lr * db

        # 计算成本
        if i % 100 == 0:
            costs.append(cost)

        # 打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数：%i, 误差值：%f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return (params, grads, costs)


def predict(w, b, X):
    '''
    使用学习逻辑回归参数logistic （w，b）预测标签是0还是1
    :param w:权重，大小不等的数组（num_px * num_px * 3，1）
    :param b:偏差，一个标量
    :param X:维度为（num_px * num_px * 3，训练数据的数量）的数据
    :return:
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, lr=0.5, print_cost=False):
    '''
    通过调用之前实现的函数来构建逻辑回归模型
    :param X_train:  numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
    :param Y_train:  numpy的数组,维度为（1，m_train）（矢量）的训练标签集
    :param X_test:   numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
    :param Y_test:   numpy的数组,维度为（1，m_test）的（向量）的测试标签集
    :param num_iterations:  表示用于优化参数的迭代次数的超参数
    :param lr:  学习速率的超参数
    :param print_cost:  设置为true以每100次迭代打印成本
    :return:
        d：包含有关模型信息的字典
    '''

    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, lr, print_cost)

    # 从字典参数中检索w,b
    w, b = parameters["w"], parameters["b"]

    # 预测
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": lr,
        "num_iterations": num_iterations}

    return d

if __name__ == '__main__':

    # lr=0.005
    # d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, lr=0.005,
    #           print_cost=True)
    #
    # # 绘制图
    # costs = np.squeeze(d['costs'])
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(d["learning_rate"]))
    # plt.show()

    #尝试三种不同的学习率
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, lr=i,
                               print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

