#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:01:48 2022

@author: azin
"""
import scipy.io as sio
import math, random
import os, sys

import warnings
import numpy as np
#from itertools import product
#from matplotlib import pyplot as plt
#from matplotlib import cm, animation
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
#from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
#import graph as gp
warnings.filterwarnings("ignore")

def get_gp(visited_sample_locations):
    NUM = 51  # len(visited_sample_locations)
    NUMz = 50

    sample_locations = [[0, 0], [32, 13], [23, 39], [43, 18], [16, 38], [41, 21], [26, 26],
              [7, 42], [2, 46], [32, 43], [18, 18], [48, 8], [17, 23], [41, 31], [39, 33],
              [27, 42], [13, 44], [43, 3], [46, 42], [12, 16], [23, 1], [9, 7], [17, 1], [31, 17],
              [36, 38], [13, 48], [23, 20], [12, 14], [34, 40], [50, 50]]
    sensory_readings = [0.25, 0.028, 0.549, 0.435, 0.41, 0.330, 0.204, 0.119,
                        0.299, 0.0766, 0.25, 0.028, 0.549, 0.435, 0.41, 0.130, 0.204, 0.619,
                        0.099, 0.1266, 0.45, 0.028, 0.549, 0.435, 0.41, 0.530, 0.04, 0.219,
                        0.219, 0.16]
    """
    sample_locations =[[0, 0], [32, 13], [23, 39], [43, 18], [16, 38], [41, 21], [26, 26], [7, 42], [2, 46], [32, 43],
     [18, 18], [48, 8], [17, 23], [41, 31], [39, 33], [27, 42], [13, 44], [43, 3], [46, 42], [12, 16],
     [23, 1], [9, 7], [17, 1], [31, 17], [36, 38], [13, 48], [23, 20], [12, 14], [34, 40], [7, 10], [28, 10], [37, 8],
     [22, 17], [37, 21], [38, 26], [12, 33], [1, 27], [30, 44], [20, 28], [9, 28], [18, 20], [39, 14],
     [3, 14], [10, 22], [43, 20], [47, 43], [12, 1], [10, 49], [15, 35], [8, 4], [3, 40], [31, 8], [17, 45],
     [45, 50], [33, 38], [37, 36], [34, 15], [11, 44], [43, 41], [5, 41], [38, 19], [18, 39], [28, 49], [36, 18],
     [32, 21], [36, 15], [48, 40], [45, 31], [15, 23], [14, 18], [35, 2], [37, 11], [1, 3], [35, 19], [50, 26],
     [43, 25], [31, 46], [39, 38], [11, 38], [46, 50], [35, 45], [35, 48], [44, 9], [30, 3], [5, 45], [2, 12], [12, 26], [3, 10],
     [50, 29], [47, 16], [46, 28], [17, 7], [25, 47], [46, 46], [40, 27], [22, 11], [8, 45], [20, 44], [46, 24], [50, 50]]
    """
    #content = sio.loadmat('2018-06-21_ripperdan.mat')
    #sensor_reading = content['krig_val']
    #sensory_readings= []
    #for i in sample_locations:
     #   sensory_readings.append(sensor_reading[i[0], i[1]])
    #sensory_readings = [0.25, 0.028, 0.549, 0.435, 0.41, 0.330, 0.204, 0.119,
       #                      0.299, 0.0766, 0.25, 0.028, 0.549, 0.435, 0.41, 0.130, 0.204, 0.619,
       #                      0.099, 0.1266, 0.45, 0.028, 0.549, 0.435, 0.41, 0.530, 0.04, 0.219,
       #                      0.219, 0.16, 0.25, 0.028, 0.549, 0.435, 0.41, 0.330, 0.204, 0.119,
        #                     0.299, 0.0766, 0.25, 0.028, 0.549, 0.435, 0.41, 0.130, 0.204, 0.619,
        #                     0.099, 0.1266, 0.45, 0.028, 0.549, 0.435, 0.41, 0.530, 0.04, 0.219,
        #                     0.219, 0.16, 0.25, 0.028, 0.549, 0.435, 0.41, 0.330, 0.204, 0.119,
        #                     0.299, 0.0766, 0.25, 0.028, 0.549, 0.435, 0.41, 0.130, 0.204, 0.619,
         #                    0.099, 0.1266, 0.45, 0.028, 0.549, 0.435, 0.41, 0.530, 0.04, 0.219,
         #                    0.219, 0.16, 0.45, 0.35, 0.17, 0.20, 0.28, 0.19, 0.16, 0.26, 0.23, 0.16]



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
    for x in range(NUM):
        for y in range(NUM):
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
        Yyy.append([jj[1]] * NUM)

    X = np.asarray([Xxx for _ in range(NUMz)])
    Y = np.asarray(Yyy)  # ([Yyy for _ in range(NUMz)])
    xx = np.vstack((X.flatten(), Y.flatten())).T

    z_mean, z_std = gpr.predict(xx, return_std=True)
    z_m = z_mean.reshape(-1, NUMz)
    #print("updated value for sensory reading {}".format(z_m[0][0]))
    zstd = z_std.reshape(-1, NUMz)
    # plt.contourf(X, Y, z, 5, cmap='RdGy') #, cmap='RdGy' cmap=mpl.cm.viridis

    # fig1,ax1=plt.subplots()
    # cf=ax1.contourf(X,Y,z, 10, cmap='RdGy' )
    #plt.contourf(X, Y, z_m, 50, cmap='RdGy')  # , cmap='RdGy' cmap=mpl.cm.viridis cmap='RdGy'
    #plt.plot([i[0] for i in visited_sample_locations], [j[1] for j in visited_sample_locations], "ob")
    # ax1.scatter([Xx[i] for i in range(len(Xx))], [Yy[i] for i in range(len(Yy))])
    # plt.contourf(X1,Y1,z1, 50, cmap='RdGy')
    #plt.colorbar();
    #plt.xlim([0, 9])
    #plt.ylim([0, 9])
    #plt.show()
    #print("reward gp {}".format(zstd[0]))
    z_pred=[]
    n=0
    for x in range(NUM):
        for y in range(NUM):
            if [x, y] in sample_locations:

                z_pred.append(z_m[0][n])
                n += 1
            else:
                z_pred.append(0)
    #print(z_pred)
    print("MSE: {}".format(mean_squared_error(z,z_pred, squared=False)))
    return  zstd[0]
