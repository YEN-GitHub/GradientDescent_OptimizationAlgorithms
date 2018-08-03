#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D


x = np.matrix([15,16,18,20,23,33,46,56,58,65,84,95])
a = 0.837
b=  -0.7
y = a*x +b
print(y)

#
# def calc_loss(a,b,x,y):
#     tmp = y - (a * x + b)
#     tmp = tmp ** 2  # 对矩阵内的每一个元素平方
#     SSE = sum(tmp) / (2 * len(x))
#     return SSE
#
# # 计算x,y坐标对应的高度值
# def draw_hill(x, y):
#     a = np.linspace(-1, 1, 2)
#     print(a)
#     b = np.linspace(-1, 1, 2)
#     x = np.array(x)
#     y = np.array(y)
#
#     allSSE = np.zeros(shape=(len(a),len(b))) #Height是个 3*3 的数组，记录地图上 9 个点的高度汇总
#
#     for ai in range(0,len(a)):
#         for bi in range(0,len(b)):
#             a0 = a[ai]
#             b0 = b[bi]
#             SSE = calc_loss(a=a0, b=b0, x=x, y=y)
#             allSSE[ai][bi] = SSE
#
#     a,b = np.meshgrid(a, b)
#     return [a, b, allSSE]
#
#
# x = [0.1, 0.5]
# y = [0.2, 0.9]
#
# [ha, hb, hallSSE] = draw_hill(x, y)
#
# hallSSE = np.matrix([[10,20],[40,50]])
#
# hallSSE = hallSSE.T # 重要，将所有的losses做一个转置。原因是矩阵是以左上角至右下角顺序排列元素，而绘图是以左下角为原点。
#
# fig = plt.figure(1, figsize=(12, 8))
# # 绘制图1的曲面
# ax = fig.add_subplot(2, 1, 1, projection='3d')
# ax.set_top_view()
# ax.plot_surface(ha, hb, hallSSE, rstride=2, cstride=2, cmap='rainbow')
#
# # 绘制图2的等高线图
# plt.subplot(2,1,2)
# plt.contourf(ha, hb, hallSSE, 15, alpha=0.5, cmap=plt.cm.hot)
# C = plt.contour(ha, hb, hallSSE, 15, colors='black')
# plt.clabel(C, inline=True)
# plt.xlabel('opt param: a')
# plt.ylabel('opt param: b')
#
# plt.show()
#
#
#

# # 生成x,y的数据
# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
#
# # 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
# X, Y = np.meshgrid(x, y)
#
# # 填充等高线
# plt.contourf(X, Y, f(X, Y), 20, cmap=plt.cm.hot)
# # 添加等高线
# C = plt.contour(X, Y, f(X, Y), 20)
# plt.clabel(C, inline=True, fontsize=12)
# # 显示图表
# plt.show()

# 绘制两点间直线
# plt.plot([10, 15], [-20, -55], 'r-')
# plt.plot([15, 25], [-55, -90], 'r-')
# plt.show()

