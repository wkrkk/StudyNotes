import numpy as np
import tensorflow as tf


np.random.seed(1)

# # 定义y_hat为固定值36
# y_hat = tf.constant(36, name="y_hat")
# # 定义y为固定值39
# y = tf.constant(39, name="y")
#
# # 为损失函数创建一个变量
# loss = tf.Variable((y - y_hat)**2, name="loss")
# # 运行之后的初始化
# init = tf.global_variables_initializer()
# # 损失变量被初始化并准备计算
# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))

# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a, b)
#
# sess = tf.Session()
# print(sess.run(c))

# 创建一个占位符
x = tf.placeholder(tf.int64, name="x")
# 利用feed_dict改变x的值
sess = tf.Session()
print(sess.run(2 * x, feed_dict={x:3}))
sess.close()