import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow.examples.tutorials.mnist
from PIL import Image
import random
# import tensorflow.contrib.learn

# the convolutional is a example for me.
# from tensorflow.models.image.mnist import convolutional

"""
write on Feb 11, 2017.
"""

DEBUG = True
DATA_DIR = r'./MNIST_data'


def debug_only(func):
    if DEBUG:
        return func

    def dummy(*args, **kwargs):
        pass

    return dummy


@debug_only
def debug_print(*item):
    print(*item)


def print_type(item):
    if DEBUG:
        print(type(item))


def show_pic(index=None):
    if not index:
        index = random.randint(0, 10000)
    print('index: ', index)
    print_type(input_data)
    data = input_data.read_data_sets(DATA_DIR)
    print('train images:', data.train.num_examples)
    print('test images:', data.test.num_examples)
    print('validation images:', data.validation.num_examples)
    print_type(data)
    print_type(data.test)
    images = data.test.images
    print_type(images)
    print(images.ndim, images.shape, images.size, images.dtype)
    arr = np.array(images[index])
    print(arr.ndim, arr.shape, arr.size)
    pic = arr.reshape((28, 28))
    print(pic.ndim, pic.shape, pic.size)
    # print(pic)
    img = Image.fromarray(pic, 'I')
    # print(img)
    print(img.height, img.width, img.info)
    img.show()
    print(data.test.labels[index])


def softmax_train():
    train_images = input_data.read_data_sets(DATA_DIR).train.images
    debug_print(train_images.shape)


def train():
    data_sets = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train_vec = data_sets.train.images
    train_label = data_sets.train.labels
    train_images = data_sets.train.images.reshape((55000, 28, 28))
    assert train_images.shape == (55000, 28, 28)
    test_vec = data_sets.test.images
    test_label = data_sets.test.labels
    print('shape of train_vec:', train_vec.shape)
    print('shape of test labels', test_label.shape)
    # print(test_label)

    with tf.name_scope('source'):
        # data_in = tf.placeholder(tf.float32, [None, 28, 28])
        data_in = tf.placeholder(tf.float32, [None, 784], name='data_in')
        y_mark = tf.placeholder(tf.float32, [None, 10], name='y_mark')
        # y_learned = tf.Variable(tf.zeros([10]))

    # use a N1-node full link layer.
    n1 = 25
    with tf.name_scope('layer1'):
        weight = tf.Variable(tf.random_normal([784, n1]), name='weight1')
        biases = tf.Variable(tf.zeros([n1]), name='biases1')
        layer1 = tf.matmul(data_in, weight) + biases
        layer1 = tf.nn.relu(layer1, name='layer1_relu')
        tf.summary.histogram('layer1', layer1)

    with tf.name_scope('layer_output'):
        weight2 = tf.Variable(tf.random_normal([n1, 10]), name='weight2')
        biases2 = tf.Variable(tf.zeros([10]), name='biases2')
        y_learned = tf.add(tf.matmul(layer1, weight2), biases2, name='output')
        tf.summary.histogram('output', y_learned)
        debug_print(y_learned)

    # tf v1.0.0 do not support the function below.
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_learned, y_mark))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_learned, labels=y_mark))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_learned, 1), tf.argmax(y_mark, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('=== start train ===')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('./logs/mnist_logs', sess.graph)
        merge_summary_op = tf.summary.merge_all()
        saver.restore(sess, './ckpt/mnist.ckpt')
        for _ in range(8001):
            sess.run(train_step, feed_dict={data_in: train_vec, y_mark: train_label})
            if _ % 100 == 0:
                print(_)
                print(sess.run(accuracy, feed_dict={data_in: test_vec, y_mark: test_label}))
                summary_writer.add_summary(
                    sess.run(merge_summary_op, feed_dict={data_in: train_vec, y_mark: train_label}))

                '''
                print('weight', weight)
                print(sess.run(weight))
                print('biases', biases)
                print(sess.run(biases))

                print('weight2', weight2)
                print(sess.run(weight2))
                print('biases2', biases2)
                print(sess.run(biases2))
                '''
        saver.save(sess, './ckpt/mnist.ckpt')


def main():
    # show_pic()
    train()


if __name__ == '__main__':
    main()
    # show_pic()
