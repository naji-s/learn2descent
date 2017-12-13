
# coding: utf-8

# In[10]:
# from meta import MetaOptimizer
from matplotlib import use
use('Qt4Agg')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from learning_to_learn.meta import MetaOptimizer
from time import time



DIMS=10
TRAINING_STEPS = 40 # This is 100 in the paper
initial_pos = tf.random_normal([DIMS])


sess = tf.Session()


t = time()
def pre_processor(input, p=10.):
    big_input_condition = tf.greater_equal(tf.abs(input), tf.exp(-p))
    output = tf.where(big_input_condition,
                           tf.stack([tf.log(tf.abs(input)) / p, tf.sign(input)], axis=1),
                           tf.stack([-tf.ones(input.shape[0]), tf.exp(p) * input], axis=1))

    return output

#
DIMS = 10  # Dimensions of the parabola
# scale = tf.random_uniform([DIMS], 0.5, 1.5)
W = tf.random_normal([DIMS, DIMS])
y = tf.random_normal([DIMS])

#  This represents the network/function we are trying to optimize,
def f(x):
    if True:
        x = tf.sin(W*tf.log(tf.abs(x+1)) - y)
    return tf.reduce_sum(x*x)

# Gradient Descent
def g_sgd(gradients, state, learning_rate=0.01):
    return -learning_rate*gradients, state



# RMSProp
def g_rms(gradients, state, learning_rate=0.1, decay_rate=0.99):
    if state is None:
        state = tf.zeros(DIMS)
    state = decay_rate * state + (1-decay_rate)*tf.pow(gradients, 2)
    update = -learning_rate*gradients / (tf.sqrt(state)+1e-5)
    return update, state


def learn_2(optimizer):
    losses = []
    x = initial_pos
    state = None
    for _ in range(TRAINING_STEPS):
        loss = f(x)
        losses.append(loss)
        grads, = tf.gradients(loss, x)
        new_grads = pre_processor(grads)
        update, state = optimizer(new_grads, state)
        x += update
    return losses

def learn(optimizer):
    losses = []
    x = initial_pos
    state = None
    for _ in range(TRAINING_STEPS):
        loss = f(x)
        losses.append(loss)
        grads, = tf.gradients(loss, x)
        # new_grads = pre_processor(grads)
        new_grads = grads
        update, state = optimizer(new_grads, state)
        x += update
    return losses

print ("The time to execute: " + str(time() - t))
sess.run(tf.global_variables_initializer())
x = np.arange(TRAINING_STEPS)

sgd_losses = learn(g_sgd)
rms_losses = learn(g_rms)


for _ in range(3):
    sgd_l, rms_l= sess.run(
        [sgd_losses, rms_losses])
    p1, = plt.semilogy(x, sgd_l, label='SGD')
    p2, = plt.semilogy(x, rms_l, label='RMS')

    plt.legend(handles=[p1, p2])
    plt.title('Losses')
    plt.show()


meta_opt = MetaOptimizer()

