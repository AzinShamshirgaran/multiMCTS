#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 17:23:03 2022

@author: azin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:01:48 2022

@author: azin
"""

import math, random
import os, sys


import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from matplotlib import cm, animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import graph as gp

def get_gp(visited_sample_locations, current_vertex):
    #print(current_vertex)

    #print("visited sample locations {}".format(visited_sample_locations))

    sample_locations = [[0, 0], [1, 3], [2, 6], [4, 5], [3, 7], [6, 2], [7, 2], [8, 3], [9, 7], [9, 9]]
    sensory_readings = [0.25, 0.028, 0.549, 0.435, 0.41, 0.330, 0.204, 0.619,
                        0.299, 0.266]

    NUM = 10  # len(visited_sample_locations)
    NUMz=10

    x_train = []
    for i in range(NUM):
        for j in range(NUM):
            x_train.append([i, j])

    Xx = []
    Yy = []
    for jj in x_train:
        Xx.append([jj[0]])
        Yy.append([jj[1]])
    z = []
    for x in range(10):
        for y in range(10):
            if [x, y] in visited_sample_locations:

                z.append(sensory_readings[sample_locations.index([x, y])])
            else:
                z.append(0)

    X_train = np.asarray(Xx)  # ([Xx for _ in range(NUM)])
    Y_train = np.asarray(Yy)  # ([Yy for _ in range(NUM)])
    pos = np.vstack((X_train.flatten(), Y_train.flatten())).T

    kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e4)) + WhiteKernel(
        noise_level=1, noise_level_bounds=(1e-5, 1e1)
    )

    GP_train = np.stack((X_train.flatten(), Y_train.flatten())).T

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gpr.fit(GP_train, z)

    Xxx = []
    Yyy = []
    for jj in sample_locations:
        Xxx.append(jj[0])
        Yyy.append([jj[1]] * 10)

    X = np.asarray([Xxx for _ in range(NUMz)])
    Y = np.asarray(Yyy)  # ([Yyy for _ in range(NUMz)])
    xx = np.vstack((X.flatten(), Y.flatten())).T

    z_mean, z_std = gpr.predict(xx, return_std=True)
    z = z_mean.reshape(-1, NUMz)
    #print("updated value for sensory reading {}".format(z[0]))
    zstd = z_std.reshape(-1, NUMz)
    # plt.contourf(X, Y, z, 5, cmap='RdGy') #, cmap='RdGy' cmap=mpl.cm.viridis

    # fig1,ax1=plt.subplots()
    # cf=ax1.contourf(X,Y,z, 10, cmap='RdGy' )
    plt.contourf(X, Y, z, 50, cmap='RdGy')  # , cmap='RdGy' cmap=mpl.cm.viridis cmap='RdGy'
    plt.plot([i[0] for i in visited_sample_locations], [j[1] for j in visited_sample_locations], "ob")
    # ax1.scatter([Xx[i] for i in range(len(Xx))], [Yy[i] for i in range(len(Yy))])
    # plt.contourf(X1,Y1,z1, 50, cmap='RdGy')
    plt.colorbar();
    plt.xlim([0, 9])
    plt.ylim([0, 9])
    #plt.show()
    #print("reward gp {}".format(zstd[0]))
    return  zstd[0]
