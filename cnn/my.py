import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow.examples.tutorials.mnist
from PIL import Image
import random

# the convolutional is a example for me.
# from tensorflow.models.image.mnist import convolutional

"""
write on Feb 11, 2017.
"""

DEBUG = False
DATA_DIR = r'./MNIST_data'


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
    # img.show()


def train():
    data_sets = input_data.read_data_sets(DATA_DIR)
    train_images = data_sets.train.images.reshape((55000, 28, 28))
    assert train_images.shape == (55000, 28, 28)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


def main():
    show_pic()
    train()


if __name__ == '__main__':
    main()
