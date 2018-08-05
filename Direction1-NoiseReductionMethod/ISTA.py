# Date: 2018-08-01 21:47
# Author: Enneng Yang
# Abstract：
# Solve LASSO regression problem with ISTA
# iterative solvers.

# Author : Alexandre Gramfort, first.last@telecom-paristech.fr
# License BSD

import time
from math import sqrt
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)
m, n = 15, 20

# random design
A = rng.randn(m, n)  # random design

x0 = rng.rand(n)
x0[x0 < 0.9] = 0
b = np.dot(A, x0)
lambda_ = 0.5  # regularization parameter


def soft_thresh(x, lambda_ ):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.)


def ista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    L = linalg.norm(A) ** 2  # Lipschitz constant

    for itera in range(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((itera, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times



maxit = 100
x_ista, pobj_ista, times_ista = ista(A, b, lambda_, maxit)


plt.close('all')

plt.figure()
plt.plot(times_ista, pobj_ista, label='ista')
plt.xlabel('Time')
plt.ylabel('Primal')
plt.legend()
plt.show()