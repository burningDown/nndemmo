import tensorflow as tf

IMAGE_INFO = {
    "data_path": [
        "/root/workspace/data_input/cifar-10-batches-bin/data_batch_1.bin",
        "/root/workspace/data_input/cifar-10-batches-bin/data_batch_2.bin",
        "/root/workspace/data_input/cifar-10-batches-bin/data_batch_3.bin",
        "/root/workspace/data_input/cifar-10-batches-bin/data_batch_4.bin",
        "/root/workspace/data_input/cifar-10-batches-bin/data_batch_5.bin"],
    "test_path": ["/root/workspace/data_input/cifar-10-batches-bin/test_batch.bin"],
    "size": 32,
    "depth": 3
}


class AlexNet:
    network = None,
    loss = None,
    train_step = None,
    acc = None,
    keep_prob = tf.placeholder(dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_INFO["size"], IMAGE_INFO["size"], IMAGE_INFO["depth"]])
    label = tf.placeholder(dtype=tf.int32, shape=[None])
    weights = [
        [3, 48],
        [3, 128],
        [3, 192],
        [3, 192],
        [3, 128],
        [2048],
        [2048]
    ]
    layers = {}

    def get_images(self, files, mini_batch):
        def map_reshape(value):
            record_bytes = tf.decode_raw(value, tf.uint8)
            label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)
            img = tf.cast(
                tf.transpose(
                    tf.reshape(
                        tf.strided_slice(record_bytes, [1],
                                         [IMAGE_INFO["size"]*IMAGE_INFO["size"]*IMAGE_INFO["depth"] + 1]),
                        shape=[3, 32, 32]), [1, 2, 0]), tf.float32)
            img = tf.image.per_image_standardization(img)
            return img, label[0]
        filenames = tf.constant(files)
        file_dataset = tf.data.FixedLengthRecordDataset(
            filenames,
            IMAGE_INFO["size"]*IMAGE_INFO["size"]*IMAGE_INFO["depth"] + 1)\
            .map(map_reshape, num_parallel_calls=100).batch(mini_batch)
        return file_dataset.make_one_shot_iterator().get_next()

    def inference(self):
        with tf.name_scope("conv1"):
            w = tf.Variable(tf.truncated_normal(shape=[self.weights[0][0],
                                                       self.weights[0][0],
                                                       3,
                                                       self.weights[0][1]], stddev=0.01))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.weights[0][1]])),
            conv = tf.nn.relu(tf.nn.conv2d(self.x, w, [1, 1, 1, 1], 'SAME') + b)
            lay = self.layers["conv1"] = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        with tf.name_scope("conv2"):
            w = tf.Variable(tf.truncated_normal(shape=[self.weights[1][0],
                                                       self.weights[1][0],
                                                       self.weights[0][1],
                                                       self.weights[1][1]], stddev=0.01))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.weights[1][1]]))
            conv = tf.nn.relu(tf.nn.conv2d(lay, w, [1, 1, 1, 1], 'SAME') + b)
            lay = self.layers["conv2"] = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        with tf.name_scope("conv3"):
            w = tf.Variable(tf.truncated_normal(shape=[self.weights[2][0],
                                                       self.weights[2][0],
                                                       self.weights[1][1],
                                                       self.weights[2][1]], stddev=0.01))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.weights[2][1]]))
            lay = self.layers["conv3"] = tf.nn.relu(tf.nn.conv2d(lay, w, [1, 1, 1, 1], 'SAME') + b)
        with tf.name_scope("conv4"):
            w = tf.Variable(tf.truncated_normal(shape=[self.weights[3][0],
                                                       self.weights[3][0],
                                                       self.weights[2][1],
                                                       self.weights[3][1]], stddev=0.01))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.weights[3][1]]))
            lay = self.layers["conv4"] = tf.nn.relu(tf.nn.conv2d(lay, w, [1, 1, 1, 1], 'SAME') + b)
        with tf.name_scope("conv5"):
            w = tf.Variable(tf.truncated_normal(shape=[self.weights[4][0],
                                                       self.weights[4][0],
                                                       self.weights[3][1],
                                                       self.weights[4][1]], stddev=0.01))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.weights[4][1]]))
            conv = tf.nn.relu(tf.nn.conv2d(lay, w, [1, 1, 1, 1], 'SAME') + b)
            lay = self.layers["conv5"] = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        with tf.name_scope("fc1"):
            conv5_shape = lay.shape[1] * lay.shape[2] * lay.shape[3]
            reshaped_conv5 = tf.reshape(lay, shape=[-1, conv5_shape.value])
            w = tf.Variable(tf.truncated_normal(shape=[conv5_shape.value, self.weights[5][0]], stddev=0.1))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.weights[5][0]]))
            lay = self.layers["fc1"] = tf.nn.dropout(tf.nn.relu(tf.matmul(reshaped_conv5, w) + b), self.keep_prob)
        with tf.name_scope("fc2"):
            w = tf.Variable(tf.truncated_normal(shape=[self.weights[5][0], self.weights[6][0]], stddev=0.01))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.weights[6][0]]))
            lay = self.layers["fc2"] = tf.nn.dropout(tf.nn.relu(tf.matmul(lay, w) + b), self.keep_prob)
        with tf.name_scope("softmax"):
            w = tf.Variable(tf.truncated_normal(shape=[self.weights[6][0], 10], stddev=0.01))
            b = tf.Variable(tf.constant(value=0.1, shape=[10]))
            lay = self.layers["softmax"] = tf.nn.softmax(tf.matmul(lay, w) + b)
        self.network = lay
        self.loss = None
        self.acc = None
        return lay

    def loss(self):
        if self.loss is None:
            if self.network is None:
                self.inference()
            one_hot_labels = tf.one_hot(self.label, 10, dtype=tf.float32)
            self.loss = tf.reduce_mean(-tf.reduce_sum(one_hot_labels*tf.log(self.network), axis=[1]))
        return self.loss

    def train_step(self, alpha=0.0005):
        if self.train_step is None:
            if self.loss is None:
                self.loss()
            self.train_step = tf.train.AdamOptimizer(alpha).minimize(self.loss)
        return self.train_step

    def accuracy(self):
        if self.acc is None:
            if self.network is None:
                self.inference()
            correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.cast(self.label, dtype=tf.int64))
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.acc


def main():
    alpha = 0.0005
    batch = 100
    alex_net = AlexNet()
    loss = alex_net.loss()
    train_step = alex_net.train_step(alpha)
    image_batch = alex_net.get_images(IMAGE_INFO["data_path"], batch)
    acc = alex_net.accuracy()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0,5000):
            if i % (50000 / batch) == 0 and i != 0:
                image_batch = alex_net.get_images(IMAGE_INFO["data_path"], batch)
            images, labels = sess.run(image_batch)
            result = sess.run((train_step, acc, loss), feed_dict={alex_net.keep_prob: 0.5, alex_net.x: images, alex_net.label: labels})
            print("%d: acc=%f, loss=%f" % (i, result[1], result[2]))
        image_batch = alex_net.get_images(IMAGE_INFO["test_path"], 10000)
        images, labels = sess.run(image_batch)
        print("test:", sess.run(acc, feed_dict={alex_net.keep_prob: 1, alex_net.x: images, alex_net.label: labels}))


if __name__ == "__main__":
    main()
