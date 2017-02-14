#!/usr/bin/python

import tensorflow as tf
import numpy as np

# create 100 numpy phony x, y data points in Numpy, y = x * 0.1 + 0.3

x_data = np.random.rand(100).astype(np.float32)
# x_data = np.array([np.random.random().astype(np.float32) * 10000 for _ in range(100)])
y_data = x_data * 0.1 + 0.3

x= tf.constant(x_data, name='x')
y = tf.constant(y_data, name='y')

'''
print(x_data)
print(y_data)

print(x)
print(y)
'''

# Try to find W and b

# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y = x * W + b

saver =  tf.train.Saver()

# minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        # print(sess.run(x), '\n', sess.run(y))


