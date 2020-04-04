import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# 训练数据
point_num = 200
p_x = np.linspace(0, np.pi, point_num, dtype='float32')
p_y = np.sin(p_x)

# 设置网络结构
n_hidden = 10
n_input = 1
n_output = 1

x = tf.placeholder("float32", [None, 1])
y = tf.placeholder("float32", [None, 1])
stddev = 0.01
weights = {
        'w1': tf.Variable(tf.random.normal([n_input, n_hidden], stddev=stddev)),
        'out': tf.Variable(tf.random.normal([n_hidden, n_output], stddev=stddev))}
biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden])),
        'out': tf.Variable(tf.random.normal([n_output]))}


# 前向传播算法
def multilayer(x, weights, biases):
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    return tf.add(tf.matmul(layer, weights['out']), biases['out'])


if __name__ == '__main__':

    # 设置训练参数
    yi = multilayer(x, weights, biases)
    loss = tf.reduce_mean(tf.square(yi - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    # 训练
    costs = []
    epoch = 2000
    eps = 1e-4
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoch):
            cost = 0
            for j in range(point_num):
                feeds = {x: np.reshape(p_x[j], [1, 1]), y: np.reshape(p_y[j], [1, 1])}
                sess.run(train, feed_dict=feeds)
                cost += sess.run(loss, feed_dict=feeds) / point_num
            costs.append(cost)
            print(i + 1, 'loss: %.6f' % cost)
            if costs[i] <= eps:
                print('convergence!!!')
                break
            pass

        # 绘制loss曲线
        plt.figure()
        plt.plot(range(len(costs)), costs)

        # 测试
        feeds = {x: np.reshape(p_x, [point_num, 1])}
        plt.figure()
        plt.plot(p_x, p_y, '.r')
        plt.plot(p_x, sess.run(yi, feed_dict=feeds), '.b')
        plt.show()
