import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

point_num = 1000
vector_set = []
for i in range(point_num):
    x = np.random.normal(0.0, 0.55)
    y = 0.1 * x + 0.3 + np.random.normal(0.0, 0.05)
    vector_set.append([x, y])
x = [v[0] for v in vector_set]
y = [v[1] for v in vector_set]

w = tf.Variable(tf.random.uniform([1], -1.0, 1.0), name='w')
b = tf.Variable(tf.zeros([1]), name='b')


if __name__ == '__main__':
    yi = w * x + b
    loss = tf.reduce_mean(tf.square(yi - y), name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss, var_list=[w, b], name='train')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            sess.run(train)
            print('loss = ', sess.run(loss), sess.run(w))

        plt.plot(x, sess.run(w) * x + sess.run(b))
        plt.scatter(x, y, c='r')
        plt.show()


