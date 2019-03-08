import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


# For cifar-py input and display testing
def test1():
    file = "E:/data_input/cifar-10-batches-py/data_batch_1"
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    a = dict[b'data'][5300].reshape((3, 32, 32)).transpose((1, 2, 0))
    img = Image.fromarray(a.astype('uint8')).convert('RGB')
    plt.figure("Image")
    plt.imshow(img)
    plt.show()


# For cifar-bin input and display using numpy testing
def test2():
    with open("E:/data_input/cifar-10-batches-bin/data_batch_1.bin", "rb") as f:
        for i in range(0, 289):
            label1 = f.read(1)
            img1 = f.read(3072)
        a = np.ndarray(shape=[3072], dtype=int)
        for i in range(0, 3072):
            a[i] = img1[i]
    a = a.reshape([3, 32, 32]).transpose([1, 2, 0])
    img = Image.fromarray(a.astype('uint8')).convert('RGB')
    plt.imshow(img)
    plt.show()
    print(label1[0])


# For cifar-bin input and display using tf's FixedLengthRecordReader() testing
def test3():
    filename_queue = tf.train.string_input_producer(["E:/data_input/cifar-10-batches-bin/data_batch_1.bin"])
    reader = tf.FixedLengthRecordReader(record_bytes=3073)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.strided_slice(record_bytes, [0], [1])
    img = tf.reshape(tf.strided_slice(record_bytes, [1], [3073]), shape=[3, 32, 32])
    img = tf.transpose(img, [1, 2, 0])
    images = tf.train.batch([img], batch_size=10)
    sess = tf.Session()
    threads = tf.train.start_queue_runners(sess=sess)
    sess.run(tf.global_variables_initializer())
    imgs = sess.run(images)
    print(imgs)
    img2 = Image.fromarray(imgs[0].astype('uint8')).convert('RGB')
    plt.imshow(img2)
    plt.show()


def test4():
    def map_reshape(value):
        record_bytes = tf.decode_raw(value, tf.uint8)
        label = tf.strided_slice(record_bytes, [0], [1])
        img = tf.transpose(tf.reshape(tf.strided_slice(record_bytes, [1], [3073]), shape=[3, 32, 32]), [1, 2, 0])
        return img, label
    filenames = tf.constant(["E:/data_input/cifar-10-batches-bin/data_batch_1.bin"])
    file_dataset = tf.data.FixedLengthRecordDataset(filenames, 3073).map(map_reshape, num_parallel_calls=100).batch(10)
    it = file_dataset.make_one_shot_iterator()
    el = it.get_next()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img, label = sess.run(el)
    print(img.shape)
    for i in range(0, 9):
        img2 = Image.fromarray(img[i].astype('uint8')).convert('RGB')
        plt.imshow(img2)
        plt.show()


if __name__ == "__main__":
    test4()
