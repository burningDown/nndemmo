import tensorflow as tf


FILES = ["/root/workspace/data_input/cifar-10-batches-bin/data_batch_1.bin",
         "/root/workspace/data_input/cifar-10-batches-bin/data_batch_2.bin",
         "/root/workspace/data_input/cifar-10-batches-bin/data_batch_3.bin",
         "/root/workspace/data_input/cifar-10-batches-bin/data_batch_4.bin",
         "/root/workspace/data_input/cifar-10-batches-bin/data_batch_5.bin"]
class AlexNet:
    network = None
    keep_prob = tf.placeholder(dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    label = tf.placeholder(dtype=tf.int32, shape=[None])
    weights = {
        "w1": tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 48], stddev=0.01)),
        "w2": tf.Variable(tf.truncated_normal(shape=[3, 3, 48, 128], stddev=0.01)),
        "w3": tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 192], stddev=0.01)),
        "w4": tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 192], stddev=0.01)),
        "w5": tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 128], stddev=0.01)),
        "w6": None,
        "w7": tf.Variable(tf.truncated_normal(shape=[1024, 1024], stddev=0.01)),
        "w8": tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.01))
    }
    bias = {
        "b1": tf.Variable(tf.constant(value=0.1, shape=[48])),
        "b2": tf.Variable(tf.constant(value=0.1, shape=[128])),
        "b3": tf.Variable(tf.constant(value=0.1, shape=[192])),
        "b4": tf.Variable(tf.constant(value=0.1, shape=[192])),
        "b5": tf.Variable(tf.constant(value=0.1, shape=[128])),
        "b6": tf.Variable(tf.constant(value=0.1, shape=[1024])),
        "b7": tf.Variable(tf.constant(value=0.1, shape=[1024])),
        "b8": tf.Variable(tf.constant(value=0.1, shape=[10]))
    }
    layers = {}

    def get_images(self, mini_batch):
        def map_reshape(value):
            record_bytes = tf.decode_raw(value, tf.uint8)
            label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)
            img = tf.cast(tf.transpose(tf.reshape(tf.strided_slice(record_bytes, [1], [3073]), shape=[3, 32, 32]), [1, 2, 0]), tf.float32)
            img = tf.image.per_image_standardization(img)
            return img, label[0]
        filenames = tf.constant(FILES)
        file_dataset = tf.data.FixedLengthRecordDataset(filenames, 3073)\
            .map(map_reshape, num_parallel_calls=100).batch(mini_batch)
        return file_dataset.make_one_shot_iterator().get_next()

    def inference(self):
        with tf.name_scope("conv1"):
            # self.convs["conv1"]
            conv = tf.nn.relu(tf.nn.conv2d(self.x, self.weights["w1"], [1, 1, 1, 1], 'SAME') + self.bias["b1"])
            self.layers["conv1"] = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        with tf.name_scope("conv2"):
            conv = tf.nn.relu(tf.nn.conv2d(self.layers["conv1"],
                                           self.weights["w2"], [1, 1, 1, 1], 'SAME') + self.bias["b2"])
            self.layers["conv2"] = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        with tf.name_scope("conv3"):
            self.layers["conv3"] = tf.nn.relu(tf.nn.conv2d(self.layers["conv2"],
                                                           self.weights["w3"], [1, 1, 1, 1], 'SAME') + self.bias["b3"])
        with tf.name_scope("conv4"):
            self.layers["conv4"] = tf.nn.relu(tf.nn.conv2d(self.layers["conv3"],
                                                           self.weights["w4"], [1, 1, 1, 1], 'SAME') + self.bias["b4"])
        with tf.name_scope("conv5"):
            conv = tf.nn.relu(tf.nn.conv2d(self.layers["conv4"],
                                           self.weights["w5"], [1, 1, 1, 1], 'SAME') + self.bias["b5"])
            self.layers["conv5"] = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        with tf.name_scope("fc1"):
            conv5_shape = self.layers["conv5"].shape[1] * self.layers["conv5"].shape[2] * self.layers["conv5"].shape[3]
            reshaped_conv5 = tf.reshape(self.layers["conv5"], shape=[-1, conv5_shape.value])
            self.weights["w6"] = tf.Variable(tf.truncated_normal(shape=[conv5_shape.value, 1024], stddev=0.1))
            self.layers["fc1"] = tf.nn.dropout(tf.nn.relu(tf.matmul(reshaped_conv5, self.weights["w6"])
                                                          + self.bias["b6"]), self.keep_prob)
        with tf.name_scope("fc2"):
            self.layers["fc2"] = tf.nn.dropout(tf.nn.relu(tf.matmul(self.layers["fc1"], self.weights["w7"])
                                                          + self.bias["b7"]), self.keep_prob)
        with tf.name_scope("softmax"):
            self.layers["softmax"] = tf.nn.softmax(tf.matmul(self.layers["fc2"], self.weights["w8"]) + self.bias["b8"])
        return self.layers["softmax"]

    def loss(self, y):
        one_hot_labels = tf.one_hot(self.label, 10, dtype=tf.float32)
        return tf.reduce_mean(-tf.reduce_sum(one_hot_labels*tf.log(y), axis=[1]))

    def train_step(self, loss, alpha=0.0001):
        return tf.train.AdamOptimizer(alpha).minimize(loss)

    def accuracy(self, y):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.cast(self.label, dtype=tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():
    alex_net = AlexNet()
    network = alex_net.inference()
    loss = alex_net.loss(network)
    train_step = alex_net.train_step(loss)
    image_batch = alex_net.get_images(100)
    acc = alex_net.accuracy(network)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 10000):
            if i % 500 == 0 and i != 0:
                image_batch = alex_net.get_images(100)
            images, labels = sess.run(image_batch)
            # print(sess.run(alex_net.layers["softmax"], feed_dict={alex_net.keep_prob: 1, alex_net.x: images, alex_net.label: labels}))
            sess.run(train_step, feed_dict={alex_net.keep_prob: 0.5, alex_net.x: images, alex_net.label: labels})
            if i % 1 == 0:
                print(i)
                if i % 1 == 0:
                    print(sess.run(acc, feed_dict={alex_net.keep_prob: 1, alex_net.x: images, alex_net.label: labels}))


if __name__ == "__main__":
    main()
