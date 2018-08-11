# Date: 2018-08-08 08:27
# Author: Enneng Yang
# Abstract：simple linear regression problem: DNN, optimization is Momentum

import numpy as np


C = 3
all_f = []

def f(x_t, grad_t):
    if(grad_t > 0 ):
        return C * x_t
    else:
        return -1 * x_t

def f_grad(t):
    if t % 3 == 1:
        return C
    else:
        return -1

def SGD(x_t,step1):
    grad = f(x_t, f_grad(step1))
    x_t = x_t - 0.001 * grad
    all_f.append(f(x_t, grad))

if __name__ == '__main__':
    x_arr = np.arange(-1, 1, 0.1)
    step = 0
    for i in x_arr:
        SGD(i, step)
        step += 1

    print(all_f)

    print(np.array(all_f).min())

