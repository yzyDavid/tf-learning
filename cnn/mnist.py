#!/usr/bin/python
'''
ALL THE CODES in this file should be wrote by myself.
learning purpose.

figure out how the functions exchange data, and i can konw how it works.
'''

from __future__ import absolute_import

import copy

import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

def check_type():
    '''
    This function is aiming to learn the type of TF variables.
    not a function to use in training or something useful.
    '''
    print(type(mnist))
    print(type(mnist.train.images))
    print(type(mnist.train.labels))

def print_meta():
    images = mnist.train.images
    print(images)
    labels = mnist.train.labels
    print(labels)

def do_training():
    '''
    this function is aiming to train the classifier using the softmax algorithm.
    provided by the tf as the tutorial.
    NOT knowing return what as the result of training. 
    '''
    #Create Model:

    # x is the tensor of the input images.
    x = tf.placeholder(tf.float32, [None, 784])

    # W and b are the parameters, the result to be trained.
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # y is the result. haven't understand it yet.
    y = tf.matmul(x, W) + b

    # why should i block the statement below?
    #y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])
    y_origin = copy.copy(y_)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    pass

def make_cnn_layer(in_nodes, activation_func = None):
    '''
    a part for the do cnn training function.
    create a cnn layer and return it.
    '''
    out_nodes = None
    if activation_func == None:
        activation_func = tf.nn.sigmoid()
    return out_nodes

def do_cnn_training():
    '''
    i want to implement this function by myself, after reognizing all the concepts in CNN.
    '''
    #Create model:
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.matmul(x, W) + b

def do_estimate():
    '''
    maybe i should output the result firstly.
    '''
    pass

def main():
    print_meta()
    check_type()

    do_training()
    do_estimate()


if __name__ == '__main__':
    main()
    
