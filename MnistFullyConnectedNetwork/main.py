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
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
stddev = 0.1
weights = {'w1': tf.Variable(tf.random.normal([n_input, n_hidden_1], stddev=stddev)),
        'w2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes], stddev=stddev))}
biases = {'b1': tf.Variable(tf.random.normal([n_hidden_1])),
        'b2': tf.Variable(tf.random.normal([n_hidden_2])),
        'out': tf.Variable(tf.random.normal([n_classes]))}


# 前向传播算法
def multilayer(x, weights, biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    out = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out


if __name__ == '__main__':
    # 设置训练参数
    preds = multilayer(x, weights, biases)
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
        for n in range(50):
            cost = 0
            num_batch = int(mnist.train.num_examples / bacth_size)
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
        test_x, test_y = mnist.test.next_batch(mnist.test.num_examples)
        test = {x: test_x, y: test_y}
        print('test accuracy: ', sess.run(accr, feed_dict=test))
