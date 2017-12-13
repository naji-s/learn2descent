
# coding: utf-8

# In[10]:
# from learn_to_learn.meta import MetaOptimizer
from matplotlib import use
use('Qt4Agg')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from time import time

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

def learn_quadratic(qaudratic_optimizer):
    losses = []
    x = initial_pos
    state=None
    for _ in range(TRAINING_STEPS):
        loss = f(x)
        losses.append(loss)
        grads, = tf.gradients(loss, x)
        hessians, = tf.hessians(loss, x)
        hessian_diag = tf.diag_part(hessians)
        normalized_grads = pre_processor(grads)
        normalized_hess = pre_processor(hessian_diag)
        derivatives = tf.concat([normalized_grads, normalized_hess], 1)
        update, state= qaudratic_optimizer(derivatives, state)
        x += update
    return losses


LAYERS_1 = 2
STATE_SIZE_1 = 20


########################
# code for new TF
########################

# cell_1= tf.nn.rnn_cell
# cell_1 = tf.nn.rnn_cell.MultiRNNCell(
#     [tf.nn.rnn_cell.LSTMCell(STATE_SIZE_1) for _ in range(LAYERS_1)])
# cell_1 = tf.contrib.rnn.InputProjectionWrapper(cell_1, STATE_SIZE_1)
# cell_1 = tf.contrib.rnn.OutputProjectionWrapper(cell_1, 1)
# cell_1 = tf.make_template('cell', cell_1)

############################
# code for old TF
############################

cell_1 = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.LSTMCell(STATE_SIZE_1) for _ in range(LAYERS_1)])
cell_1 = tf.contrib.rnn.InputProjectionWrapper(cell_1, STATE_SIZE_1)
cell_1 = tf.contrib.rnn.OutputProjectionWrapper(cell_1, 1)
cell_1 = tf.make_template('cell', cell_1)


def g_rnn(gradients, state):
    # Make a `batch' of single gradients to create a
    # "coordinate-wise" RNN as the paper describes.

    # gradients = tf.expand_dims(gradients, axis=1)

    if state is None:
        state = [[tf.zeros([DIMS, STATE_SIZE_1])] * 2] * LAYERS_1
    update, state = cell_1(gradients, state)
    # Squeeze to make it a single batch again.
    return tf.squeeze(update, axis=[1]), state


## Our LSTM based on hessians
derivativeLSTM_cell_LAYERS = 2
derivativeLSTM_cell_STATE_SIZE = 20

############################
# code for new TF
############################
# derivativeLSTM_cell = tf.nn.rnn_cell.MultiRNNCell(
#     [tf.nn.rnn_cell.LSTMCell(derivativeLSTM_cell_STATE_SIZE) for _ in range(derivativeLSTM_cell_LAYERS)])

############################
# code for old TF
############################
derivativeLSTM_cell = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.LSTMCell(derivativeLSTM_cell_STATE_SIZE) for _ in range(derivativeLSTM_cell_LAYERS)])


derivativeLSTM_cell = tf.contrib.rnn.InputProjectionWrapper(derivativeLSTM_cell, derivativeLSTM_cell_STATE_SIZE)
derivativeLSTM_cell = tf.contrib.rnn.OutputProjectionWrapper(derivativeLSTM_cell, 1)
derivativeLSTM_cell = tf.make_template('cell', derivativeLSTM_cell)


def g_h_rnn_2(derivatives, state):
    # Make a `batch' of single gradients to create a
    # "coordinate-wise" RNN as the paper describes.
    # derivatives = tf.expand_dims(derivatives, axis=1)
    # derivatives=tf.reshape(derivatives, [-1, 2])

    if state is None:
        state = [[tf.zeros([DIMS, derivativeLSTM_cell_STATE_SIZE])] * 2] * derivativeLSTM_cell_LAYERS
    update, state = derivativeLSTM_cell(derivatives, state)

    # Squeeze to make it a single batch again.
    return tf.squeeze(update, axis=[1]), state

def optimize(loss):
    optimizer = tf.train.AdamOptimizer(0.001)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.)

    # gradients=gradients / tf.norm(gradients)
    return optimizer.apply_gradients(zip(gradients, v))









TRAINING_STEPS = 40 # This is 100 in the paper
initial_pos = tf.random_normal([DIMS])



initial_pos = tf.random_normal([DIMS])



sgd_losses = learn(g_sgd)
rms_losses = learn(g_rms)


###########################
### Using CPU/GPU
###########################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction=1.0 # don't hog all vRAM
# config.operation_timeout_in_ms=50000   # terminate on long hangs
# sess = tf.Session("", config=config)
sess = tf.Session()


# session_conf = tf.ConfigProto(
#           device_count={'CPU': 8, 'GPU': 0},
#           allow_soft_placement=True,
#           log_device_placement=False
#           )


# sess = tf.Session()
sess.run(tf.global_variables_initializer())
x = np.arange(TRAINING_STEPS)












rnn_losses = learn_2(g_rnn)
sum_losses = tf.reduce_sum(rnn_losses[-20:])
apply_update = optimize(sum_losses)

new_rnn_losses = learn_quadratic(g_h_rnn_2)
new_sum_losses = tf.reduce_sum(new_rnn_losses[-20:])
new_apply_update = optimize(new_sum_losses)

sess.run(tf.global_variables_initializer())
print ("reporting the error change for Google's LSTM...")
ave = 0
for i in range(4000):
    err, _ = sess.run([sum_losses, apply_update])
    ave += err
    if i % 1000 == 0:
        print(ave / 1000 if i!=0 else ave)
        ave = 0
print(ave / 1000)

print ("reporting the error change for our LSTM...")
new_ave = 0
for i in range(4000):
    new_err, _ = sess.run([new_sum_losses, new_apply_update])
    new_ave += new_err
    if i % 1000 == 0:
        print(new_ave / 1000 if i!=0 else new_ave)
        new_ave = 0
print(new_ave / 1000)


# for _ in range(3):
#     sgd_l, rms_l, rnn_l, new_rnn_l = sess.run(
#         [sgd_losses, rms_losses, rnn_losses, new_rnn_losses])
#     p1, = plt.plot(x, sgd_l, label='SGD')
#     p2, = plt.plot(x, rms_l, label='RMS')
#     p3, = plt.plot(x, rnn_l, label='RNN')
#     p4, = plt.plot(x, new_rnn_l, label='new_RNN')
#
#     plt.legend(handles=[p1, p2, p3, p4])
#     plt.title('Losses')
#     plt.show()

print ("The time to execute: " + str(time() - t))

for _ in range(3):
    sgd_l, rms_l, rnn_l, new_rnn_l= sess.run(
        [sgd_losses, rms_losses, rnn_losses, new_rnn_losses])
    p1, = plt.semilogy(x, sgd_l, label='SGD')
    p2, = plt.semilogy(x, rms_l, label='RMS')
    p3, = plt.semilogy(x, rnn_l, label='RNN')
    p4, = plt.semilogy(x, new_rnn_l, label='new_RNN')

    plt.legend(handles=[p1, p2, p3, p4])
    plt.title('Losses')
    plt.show()

