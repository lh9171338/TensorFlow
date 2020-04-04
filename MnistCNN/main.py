import tensorflow as tf
import os
import matplotlib.pyplot as plt
import tensorflow.contrib.learn.python.learn.datasets.mnist as input_data
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 加载图片
mnist = input_data.read_data_sets('data/', one_hot=True)

# 设置网络结构
n_input = 784
n_classes = 10
weights = {
    'wc1': tf.Variable(tf.random.normal([3, 3, 1, 64], stddev=0.1)),
    'wc2': tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.1)),
    'wd1': tf.Variable(tf.random.normal([7 * 7 * 128, 1024], stddev=0.1)),
    'wd2': tf.Variable(tf.random.normal([1024, n_classes], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.random.normal([64], stddev=0.1)),
    'bc2': tf.Variable(tf.random.normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random.normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random.normal([n_classes], stddev=0.1))
}
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keepratic = tf.Variable(0.7)

# 前向传播算法
def cnnff(input, w, b, keepratic):
    data = tf.reshape(input, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.conv2d(data, w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b['bc1']))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool1_drop = tf.nn.dropout(pool1, keepratic)

    conv2 = tf.nn.conv2d(pool1_drop, w['wc2'], strides=[1, 1, 1, 1],padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b['bc2']))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2_drop = tf.nn.dropout(pool2, keepratic)

    fc_data = tf.reshape(pool2_drop, [-1, w['wd1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc_data, w['wd1']), b['bd1']))
    fc1_drop = tf.nn.dropout(fc1, keepratic)
    out = tf.add(tf.matmul(fc1_drop, w['wd2']), b['bd2'])
    return out


if __name__ == '__main__':
    # 设置训练参数
    preds = cnnff(x, weights, biases, keepratic)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y))  # 交叉熵函数
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)
    corr = tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1))
    accr = tf.reduce_mean(tf.cast(corr, tf.float32))
    init = tf.global_variables_initializer()

    # 训练
    bacth_size = 100
    costs = []
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for n in range(100):
            cost = 0
            # num_batch = int(mnist.train.num_examples / bacth_size)
            num_batch = 10  # 全部训练的话耗时太长...
            for i in range(num_batch):
                train_x, train_y = mnist.train.next_batch(bacth_size)
                feeds = {x: train_x, y: train_y}
                sess.run(train, feed_dict=feeds)
                cost += sess.run(loss, feed_dict=feeds) / num_batch
            costs.append(cost)
            print(n + 1, ': %.6f' % cost)

        # 绘制loss曲线
        plt.plot(range(len(costs)), costs)
        plt.show()

        # 测试
        test_x, test_y = mnist.test.next_batch(bacth_size)
        test = {x: test_x, y: test_y}
        print('test accuracy: ', sess.run(accr, feed_dict=test))
