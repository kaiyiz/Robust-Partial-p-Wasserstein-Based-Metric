import argparse
from operator import index
import string
from tkinter import NONE
import numpy as np
import ot
import os
import sys
import time
import tensorflow as tf
import scipy

from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

import jpype
import jpype.imports
from jpype.types import *
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx10000m", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping

from kneed import KneeLocator


"""
Relevant parts of the code have been adapted from :
https://github.com/debarghya-mukherjee/Robust-Optimal-Transport/blob/main/ROBOT_mnist_outlier_detection.py
"""

def c_dist(b, a):
    return cdist(a, b, metric='minkowski', p=1)

def e_dist(A, B):
    A_n = (A**2).sum(axis=1).reshape(-1,1)
    B_n = (B**2).sum(axis=1).reshape(1,-1)
    inner = np.matmul(A, B.T)
    return A_n - 2*inner + B_n

def computeDistMatrixGrid(m,metric='euclidean'):
    A = np.zeros((m**2,2))
    iter = 0
    for i in range(m):
        for j in range(m):
            A[iter,0] = i
            A[iter,1] = j
            iter += 1
    dist = cdist(A, A, metric)
    return dist

def rand_pick(data, data_labels, n=1000, seed = 1):
    # eps = Contamination proportion
    # n = number of samples
    ############ Creating pure and contaminated dataset ############

    np.random.seed(seed)
    p = np.random.permutation(len(data_labels))
    data = data[p,:]
    data_labels = data_labels[p]
    # all_index = np.arange(len(data_labels))
    # index_perm = np.random.permutation(all_index)

    ind_all = np.array([])
    for i in range(10):
        ind = np.nonzero(data_labels == i)[0][:int(n/10)]
        ind_all = np.append(ind_all, ind)

    ind_all = ind_all.astype(int)
    data_pick, data_pick_label = data[ind_all, :], data_labels[ind_all]
    data_pick = data_pick/255.0
    data_shape = data_pick.shape
    data_pick = data_pick.reshape(-1,data_shape[1]*data_shape[2],3)
    data_pick = data_pick / data_pick.sum(axis=1, keepdims=1)
    data_pick[np.nonzero(data_pick==0)] = 0.000001
    data_pick = data_pick / data_pick.sum(axis=1, keepdims=1)

    return data_pick, data_pick_label


def LP_metric(X=None, Y=None, dist=None, delta=0.1, all_res = None, i = 0, j = 0, channel = 0, time_start = NONE):
    # delta : acceptable additive error
    # q_idx : index to get returned values
    nz = len(X)
    gtSolver = Mapping(nz, list(X), list(Y), dist, delta)
    APinfo = np.array(gtSolver.getAPinfo())

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]
    totalFlow = sum(APinfo_cleaned[:,2])
    cost_AP = APinfo_cleaned[:,4] * APinfo_cleaned[:,2]
    cumCost = np.cumsum(cost_AP)
    totalCost = cumCost[-1]
    cumFlow = np.cumsum((APinfo_cleaned[:,2]).astype(int))

    flowProgress = (cumFlow)/(1.0 * totalFlow)
    costProgress = (cumCost)/(1.0 * totalCost)
    dualProgress = APinfo_cleaned[:,4]/(1.0 * APinfo_cleaned[-1,4])
    
    d_cost = (1 - flowProgress) - costProgress
    d_ind = np.nonzero(d_cost<0)[0][0]-1
    d_1st = (1 - flowProgress) - dualProgress
    d_1st_ind = np.nonzero(d_1st<0)[0][0]-1
    
    realtotalCost = gtSolver.getTotalCost()

    if j == 0:
        size = all_res.shape
        time_cur = time.time()
        amount_of_work_done = i*size[0] + j + 1
        time_spent = time_cur - time_start
        total_time = time_spent/(amount_of_work_done/(size[0]*size[1]))
        print("estimate to finish the job in {}s".format(total_time-time_spent))

    all_res[i,j,channel] = np.array([1-flowProgress[d_ind], cumCost[d_ind], 1-flowProgress[d_1st_ind], APinfo_cleaned[d_1st_ind,4], totalCost, realtotalCost])
    # return 1-flowProgress[d_ind], cumCost[d_ind], 1-flowProgress[d_1st_ind], APinfo_cleaned[d_1st_ind,4], totalCost, realtotalCost

if __name__ == "__main__":
    # LOAD Data
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--delta', type=float, default=0.01)
    args = parser.parse_args()
    print(args)

    n = int(args.n)
    delta = args.delta

    if os.path.exists('./cifar10.npy'):
        cifar10 = np.load('./cifar10.npy') # 60k x 28 x 28
        cifar10_labels = np.load('./cifar10_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8
    else:
        (cifar10, cifar10_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
        np.save('cifar10.npy', cifar10)
        np.save('cifar10_labels.npy', cifar10_labels)

    cifar10_pick, cifar10_pick_label = rand_pick(cifar10, cifar10_labels, n, 1)
    # mnist_pick, mnist_pick_label = rand_pick_mnist_09(mnist, mnist_labels, 1)

    all_res = np.zeros((n,n,3,6))
    dist = computeDistMatrixGrid(32)
    dist = dist/np.max(dist)

    start_time = time.time()
    Parallel(n_jobs=-1, prefer="threads")(delayed(LP_metric)(cifar10_pick[i,:,c], cifar10_pick[j,:,c], dist, delta, all_res, i, j, c, start_time) for i in range(n) for j in range(n) for c in range(3))
    end_time = time.time()

    print("finish all job in {}s".format(end_time-start_time))
    all_res_ = np.mean(all_res,2)
    np.savez('cifar_OTP_lp_metric_n_{}'.format(n), all_res=all_res_, data_pick=cifar10_pick, data_pick_label=cifar10_pick_label)