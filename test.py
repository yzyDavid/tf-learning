#!/usr/bin/python

# A test file to try out ML concepts and tf usages.

import tensorflow as tf
import numpy as np

def reduce_try():
    x = tf.constant([
            [0., 1., 2.],
            [1., 2., 3.],
            [3., 5., 7.]
        ])
    y = tf.constant([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]
        ])

    x_reduced = tf.reduce_mean(x, 1)
    mul_mat = tf.matmul(x, y)

    #init = tf.initialize_all_variables()
    #sess.run(init)

    sess = tf.Session()
    result_mat_mul = sess.run(mul_mat)
    result_reduced = sess.run(x_reduced)

    print(x)
    print(x_reduced)
    print(mul_mat)
    print()
    print(result_mat_mul)
    print(result_reduced)

def main():
    reduce_try()

if __name__ == '__main__':
    main()
