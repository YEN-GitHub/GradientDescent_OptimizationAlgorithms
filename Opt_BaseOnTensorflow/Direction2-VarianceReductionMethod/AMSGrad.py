# Date: 2018-08-10 21:27
# Author: Enneng Yang
# Abstract：simple linear regression problem: DNN, optimization is AddSign

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data

from Opt_BaseOnTensorflow.OptimizerImplementation.AMSGrad import AMSGradOptimizer

mnist = input_data.read_data_sets("Data/MNIST_data/", one_hot=True)

# training Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 256
display_step = 1
learning_momentum = 0.9

# Network Parameters
n_input = 784     # MNIST data input (img shape: 28*28)
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 512  # 2nd layer number of features
n_classes = 10    # MNIST total classes (0-9 digits)

# tf Graph input
with tf.name_scope('inputs'):
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])


# Store layers weight & biases
with tf.name_scope('weights'):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
with tf.name_scope('biases'):
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    with tf.name_scope('layer_1'):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    with tf.name_scope('layer_2'):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    with tf.name_scope('out_layer'):
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

plt.title('Optimizer:AMSGrad')
plt.xlabel('training_epochs')
plt.ylabel('loss')

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# optimizer setting
with tf.name_scope('optimizer'):
    optimizer = AMSGradOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

all_loss = []
all_step = []


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):

        avg_cost = 0.
        epoch_cost = 0.

        total_batch = 256
        # writer = tf.summary.FileWriter("logs/", sess.graph)
        # Loop over all batches
        for i in range(total_batch):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c_ = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            # Compute average loss
            epoch_cost += c_

        avg_cost = epoch_cost / total_batch

        # opt loss
        all_loss.append(avg_cost)
        all_step.append(epoch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

plt.plot(all_step, all_loss, color='red', label='AMSGrad')
plt.legend(loc='best')

plt.show()
plt.pause(1000)
