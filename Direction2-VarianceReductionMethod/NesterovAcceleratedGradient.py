# Date: 2018-08-03 18:12
# Author: Enneng Yang
# Abstract：simple linear regression problem:Y=AX+B, optimization is Nesterov accelerated gradient(NAG)

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import random

rate = 0.2 # learning rate
def da(y, y_p,x):
    return (y-y_p)*(-x)

def db(y, y_p):
    return (y-y_p)*(-1)

def calc_loss(a,b,x,y):
    tmp = y - (a * x + b)
    tmp = tmp ** 2  # Square every element in the matrix
    SSE = sum(tmp) / 2*len(x)# Take the average
    return SSE

#draw all curve point
def draw_hill(x,y):
    a = np.linspace(-20,20,100)
    print(a)
    b = np.linspace(-20,20,100)
    x = np.array(x)
    y = np.array(y)

    allSSE = np.zeros(shape=(len(a),len(b)))
    for ai in range(0,len(a)):
        for bi in range(0,len(b)):
            a0 = a[ai]
            b0 = b[bi]
            SSE = calc_loss(a=a0, b=b0, x=x, y=y)
            allSSE[ai][bi] = SSE

    a,b = np.meshgrid(a, b)

    return [a,b,allSSE]

def shuffle_data(x,y):
    # random disturb x，y，while keep x_i corresponding to y_i
    seed = random.random()
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)


# create simulated data
x = [30, 35, 37, 59, 70, 76, 88, 100]
y = [1100,	1423,	1377,	1800,	2304,	2588,	3495,	4839]

# Data normalization
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)

for i in range(0, len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)


[ha, hb, hallSSE] = draw_hill(x, y)

# init a,b value
a = 10.0
b = -20.0
fig = plt.figure(1, figsize=(12, 8))

# draw fig.1 contour line
plt.subplot(1, 2, 1)
plt.contourf(ha, hb, hallSSE, 15, alpha=0.5, cmap=plt.cm.hot)
C = plt.contour(ha, hb, hallSSE, 15, colors='blue')
plt.clabel(C,inline=True)
plt.title('NAG')
plt.xlabel('opt param: a')
plt.ylabel('opt param: b')


plt.ion() # iteration on


a_nag = a
b_nag = b

all_loss = []
all_step = []
all_loss_nag = []
all_step_nag = []

last_a = a
last_b = b
last_a_nag = a_nag
last_b_nag = b_nag

# momentum
va = 0
vb = 0
va_nag = 0
vb_nag = 0
gamma = 0.9

for step in range(1, 100):
    loss = 0
    loss_nag = 0

    all_da = 0
    all_db = 0
    all_da_nag = 0
    all_db_nag = 0

    a_ahead = a_nag - gamma * va_nag
    b_ahead = b_nag - gamma * vb_nag

    shuffle_data(x,y)
    for i in range(0,1):
        y_p = a*x[i] + b

        y_p_nag = a_ahead * x[i] + b_ahead

        loss = (y[i] - y_p)*(y[i] - y_p)/2
        loss_nag = (y[i] - y_p_nag)*(y[i] - y_p_nag)/2

        all_da = da(y[i], y_p, x[i])
        all_db = db(y[i], y_p)
        all_da_nag = da(y[i], y_p_nag, x[i])
        all_db_nag = db(y[i], y_p_nag)

    # draw fig.1 contour line
    plt.subplot(1, 2, 1)
    plt.scatter(a, b, s=5, color='black')
    plt.plot([last_a, a], [last_b, b], color='red', label="momentum sgd")
    plt.scatter(a_nag, b_nag, s=5, color='black')
    plt.plot([last_a_nag, a_nag], [last_b_nag, b_nag], color='orange', label="nag")

    # draw fig.2 loss line
    all_loss.append(loss)
    all_loss_nag.append(loss_nag)

    all_step.append(step)
    all_step_nag.append(step)

    plt.subplot(1, 2, 2)
    plt.plot(all_step, all_loss, color='red', label='momentum sgd')
    plt.plot(all_step_nag, all_loss_nag, color='orange', label='nag')

    plt.title('NAG')
    plt.xlabel("step")
    plt.ylabel("loss")

    last_a = a
    last_b = b
    last_a_sgd = a_nag
    last_b_sgd = b_nag

    # update param
    va = gamma * va + rate * all_da
    vb = gamma * vb + rate * all_db
    a = a - va
    b = b - vb

    va_nag = gamma * va_nag + rate * all_da_nag
    vb_nag = gamma * vb_nag + rate * all_db_nag
    a_nag = a_nag - va_nag
    b_nag = b_nag - vb_nag


    if step % 5 == 0:
        print("step: ", step, " loss: ", loss, " a: ", a, " b: ", b)
        plt.show()
        plt.pause(0.01)



plt.show()
plt.pause(1000)
