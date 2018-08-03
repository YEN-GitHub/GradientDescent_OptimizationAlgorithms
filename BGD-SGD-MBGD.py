# Date: 2018-08-02 17:09
# Author: Enneng Yang
# Abstract：

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pylab
from sklearn.datasets.samples_generator import make_regression


def bgd(alpha, x, y, numIterations):
    """Copied from Internet"""
    m = x.shape[0]  # number of samples
    theta = np.ones(2)
    J_list = []

    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (iter, J))

        gradient = np.dot(x_transpose, loss) / m
        theta += alpha * gradient  # update

    pylab.plot(range(numIterations), J_list, "k-")
    return theta


def sgd(alpha, x, y, num_iter):
    """Writtern by kissg"""
    m = x.shape[0]  # number of samples
    theta = np.ones(2)
    J_list = []

    # 随机化序列
    idx = np.random.permutation(y.shape[0])
    x, y = x[idx], y[idx]

    for j in range(num_iter):

        for i in idx:
            single_hypothesis = np.dot(x[i], theta)
            single_loss = y[i] - single_hypothesis
            gradient = np.dot(x[i].transpose(), single_loss)
            theta += alpha * gradient  # update

        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (j, J))

    pylab.plot(range(num_iter), J_list, "r-")
    return theta


def mbgd(alpha, x, y, num_iter, minibatches):
    """Writtern by kissg"""
    m = x.shape[0]  # number of samples
    theta = np.ones(2)
    J_list = []

    for j in range(num_iter):

        idx = np.random.permutation(y.shape[0])
        x, y = x[idx], y[idx]
        mini = np.array_split(range(y.shape[0]), minibatches)

        for i in mini:
            mb_hypothesis = np.dot(x[i], theta)
            mb_loss = y[i] - mb_hypothesis
            gradient = np.dot(x[i].transpose(), mb_loss) / minibatches
            theta += alpha * gradient  # update

        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (j, J))

    pylab.plot(range(num_iter), J_list, "y-")
    return theta


if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                           random_state=0, noise=35)
    m, n = np.shape(x)
    x = np.c_[np.ones(m), x]  # insert column, bias
    alpha = 0.01  # learning rate

    pylab.plot(x[:, 1], y, 'o')

    print("\n#***BGD***#\n")
    theta_bgd = bgd(alpha, x, y, 800)
    for i in range(x.shape[1]):
        y_bgd_predict = theta_bgd * x
    pylab.plot(x, y_bgd_predict, 'b--')

    print("\n#***SGD***#\n")
    theta_sgd = sgd(alpha, x, y, 10)
    for i in range(x.shape[1]):
        y_sgd_predict = theta_sgd * x
    pylab.plot(x, y_sgd_predict, 'r--')

    print("\n#***MBGD***#\n")
    theta_mbgd = mbgd(alpha, x, y, 50, 10)
    for i in range(x.shape[1]):
        y_mbgd_predict = theta_mbgd * x
    pylab.plot(x, y_mbgd_predict, 'g--')

    pylab.show()
    print("Done!")