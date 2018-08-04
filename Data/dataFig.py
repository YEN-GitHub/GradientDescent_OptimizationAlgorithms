# Date: 2018-08-03 22:41
# Author: Enneng Yang
# Abstract：

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import tensorflow as tf


a = np.loadtxt('LRdata.txt')
x = a[:,1]
y = a[:,2]

plt.plot(x, y, color='ro')
plt.show()
