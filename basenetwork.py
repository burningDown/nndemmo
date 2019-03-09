import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
import random


class Nn:
    x = None
    y = None
    y_ = None
    keep_prob = None
    loss = None

    def linear_build_and_train(self, mnist, sess):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        w = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        y = tf.nn.softmax(tf.matmul(x, w)+b)
        loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), axis=[1]))
        self.x, self.y, self.y_, self.loss = x, y, y_, loss

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)
        sess.run(tf.initialize_all_variables())
        for i in range(10000):
            batch_x, batch_y = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={self.x: batch_x, self.y_: batch_y})
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy, feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels})

    def cnn_build_and_train(self, mnist, sess):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
        b = tf.Variable(tf.constant(value=0.1, shape=[32]))
        x2 = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv = tf.nn.relu(tf.nn.conv2d(x2, w, [1, 1, 1, 1], 'SAME') + b)
        pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        w = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1))
        b = tf.Variable(tf.constant(value=0.1, shape=[64]))
        conv = tf.nn.relu(tf.nn.conv2d(pool, w, [1, 1, 1, 1], 'SAME') + b)
        pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        w = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=0.1))
        b = tf.Variable(tf.constant(value=0.1, shape=[1024]))
        pool = tf.reshape(pool, shape=[-1, 7*7*64])
        fc = tf.nn.relu(tf.matmul(pool, w) + b)

        keep_prob = tf.placeholder(dtype=tf.float32)
        dp = tf.nn.dropout(fc, keep_prob)

        w = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1))
        b = tf.Variable(tf.constant(value=0.1, shape=[10]))
        y = tf.nn.softmax(tf.matmul(dp, w) + b)

        loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), axis=[1]))

        self.x, self.y, self.y_, self.keep_prob, self.loss = x, y, y_, keep_prob, loss

        train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        sess.run(tf.initialize_all_variables())

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for i in range(10000):
            batch_x, batch_y = mnist.train.next_batch(100)
            if i % 100 == 0:
                index = random.randint(0, 9899)
                print('episode: ', i, 'accuracy: ', sess.run(accuracy, feed_dict={self.x: mnist.test.images[index:index+1000], self.y_: mnist.test.labels[index:index+1000], self.keep_prob: 1.}))
            sess.run(train_step, feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 0.5})
        return sess.run(accuracy, feed_dict={self.x: mnist.test.images[500:1500], self.y_: mnist.test.labels[500:1500], self.keep_prob: 1.})

    def reco(self, img, sess):
        res = sess.run(self.y, feed_dict={self.x: img})
        return np.argmax(res)


def get_img():
    return transform.resize(io.imread('E:/data_input/six.bmp'), (28, 28))


def main():
    mnist = input_data.read_data_sets('/root/workspace/data_input/mnist', one_hot=True)
    nn = Nn()
    sess = tf.Session()
    accuracy = nn.cnn_build_and_train(mnist, sess)
    print(accuracy)

    img = get_img()
    temp = []
    temp.append(img.flatten())
    print(nn.reco(temp, sess))
    plt.imshow(img)
    plt.show()


def test():
    img = io.imread('E:/data_input/six.bmp')
    img = transform.resize(img, (28, 28))
    plt.imshow(img)
    plt.show()
    '''
    temp = [[]]
    img = img.flatten()
    for i in range(img.size):
        temp[0].append(img[i])
    return temp
    '''


if __name__ == "__main__":
    main()
