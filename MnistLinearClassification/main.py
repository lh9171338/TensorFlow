import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.learn.python.learn.datasets.mnist as input_data
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 加载图片
mnist = input_data.read_data_sets('data/', one_hot=True)
training_image = mnist.train.images
training_label = mnist.train.labels

# 显示一张图片
img = np.reshape(training_image[0, :], (28, 28))
plt.matshow(img, cmap=plt.get_cmap('gray'))
plt.show()

# 训练线性分类器
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


if __name__ == '__main__':
    output = tf.nn.softmax(tf.matmul(x, w) + b)  # 激活函数
    loss = tf.reduce_mean(tf.reduce_sum(- y * tf.log(output), reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    preds = tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1))
    accr = tf.reduce_mean(tf.cast(preds, tf.float32))
    bacth_size = 100
    costs = []
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            cost = 0
            num_batch = int(mnist.train.num_examples/bacth_size)
            for n in range(num_batch):
                train_x, train_y = mnist.train.next_batch(bacth_size)
                feeds = {x: train_x, y: train_y}
                sess.run(train, feed_dict=feeds)
                cost += sess.run(loss, feed_dict=feeds)/num_batch
            costs.append(cost)
            print(i + 1, ': %.6f' % cost)

        # 绘制loss曲线
        plt.plot(range(100), costs)
        plt.show()

        # 测试
        test_x, test_y = mnist.test.next_batch(bacth_size)
        test = {x: test_x, y: test_y}
        print('test accuracy: ', sess.run(accr, feed_dict=test))

