import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import PIL
from PIL import Image
import random

# the convolutional is a example for me.
from tensorflow.models.image.mnist import convolutional

"""
write on Feb 11, 2017.
"""

DEBUG = True
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
    print_type(data)
    print_type(data.test)
    images = data.test.images
    print_type(images)
    print(images.ndim, images.shape, images.size, images.dtype)
    arr = np.array(images[index])
    print(arr.ndim, arr.shape, arr.size)
    pic = arr.reshape((28, 28))
    print(pic.ndim, pic.shape, pic.size)
    print(pic)
    img = Image.fromarray(pic, 'I')
    print(img)
    print(img.height, img.width, img.info)
    img.show()


def main():
    show_pic()


if __name__ == '__main__':
    main()
