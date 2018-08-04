import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import tensorflow as tf
from scipy.linalg.misc import norm

# all_d = np.matrix([-28.264843 -61.20483 ])
all_d = np.array([-28.264843 -61.20483 ]).astype(np.float32)
print(norm(all_d,np.inf))