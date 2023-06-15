import argparse
from operator import index
import numpy as np
import ot
import os
import sys
import time
import tensorflow as tf
import scipy
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

def rand_pick_mnist(mnist, mnist_labels, n=1000, seed = 1):
    # eps = Contamination proportion
    # n = number of samples
    ############ Creating pure and contaminated mnist dataset ############

    np.random.seed(seed)
    all_index = np.arange(len(mnist_labels))
    index_pick = np.random.permutation(all_index)[:n]

    mnist_pick, mnist_pick_label = mnist[index_pick, :, :], mnist_labels[index_pick]
    mnist_pick = mnist_pick/255.0
    mnist_pick = mnist_pick.reshape(-1, 784)
    mnist_pick = mnist_pick / mnist_pick.sum(axis=1, keepdims=1)
    mnist_pick[np.nonzero(mnist_pick==0)] = 0.000001
    mnist_pick = mnist_pick / mnist_pick.sum(axis=1, keepdims=1)

    return mnist_pick, mnist_pick_label

def rand_pick_mnist_09(mnist, mnist_labels, seed=1):
    np.random.seed(seed)
    all_index = np.arange(len(mnist_labels))
    rand_index = np.random.permutation(all_index)
    mnist, mnist_labels = mnist[rand_index, :, :], mnist_labels[rand_index]

    mnist_pick_ind = []
    for i in range(10):
        cur_index = 0
        while True:
            if mnist_labels[cur_index] == i:
                mnist_pick_ind.append(cur_index)
                break
            else:
                cur_index += 1

    mnist_pick, mnist_pick_label = mnist[mnist_pick_ind, :, :], mnist_labels[mnist_pick_ind]
    mnist_pick = mnist_pick/255.0
    mnist_pick = mnist_pick.reshape(-1, 784)
    mnist_pick = mnist_pick / mnist_pick.sum(axis=1, keepdims=1)
    mnist_pick[np.nonzero(mnist_pick==0)] = 0.000001
    mnist_pick = mnist_pick / mnist_pick.sum(axis=1, keepdims=1)
    
    return mnist_pick, mnist_pick_label


def LP_metric(X=None, Y=None, dist=None, delta=0.1):
    # delta : acceptable additive error
    # q_idx : index to get returned values
    nz = len(X)
    gtSolver = Mapping(nz, list(X), list(Y), dist, delta)
    APinfo = np.array(gtSolver.getAPinfo())

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]
    totalFlow = sum(APinfo_cleaned[:,2])
    totalCost = sum(APinfo_cleaned[:,4])
    cumFlow = np.cumsum((APinfo_cleaned[:,2]).astype(int))
    cumCost = np.cumsum(APinfo_cleaned[:,4])

    flowProgress = (cumFlow)/(1.0 * totalFlow)
    costProgress = (cumCost)/(1.0 * totalCost)
    dualProgress = APinfo_cleaned[:,4]/(1.0 * APinfo_cleaned[-1,4])
    
    d_cost = (1 - flowProgress) - costProgress
    d_ind = np.nonzero(d_cost<0)[0][0]-1
    d_1st = (1 - flowProgress) - dualProgress
    d_1st_ind = np.nonzero(d_1st<0)[0][0]-1

    return 1-flowProgress[d_ind], cumCost[d_ind], 1-flowProgress[d_1st_ind], APinfo_cleaned[d_1st_ind,4]

if __name__ == "__main__":
    # LOAD Data
    if os.path.exists('./mnist.npy'):
        mnist = np.load('./mnist.npy') # 60k x 28 x 28
        mnist_labels = np.load('./mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8
    else:
        (mnist, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        np.save('mnist.npy', mnist)
        np.save('mnist_labels.npy', mnist_labels)

    # mnist_pick, mnist_pick_label = rand_pick_mnist(mnist, mnist_labels, n, 1)
    num_exp = 30
    n = 10
    mnist_pick, mnist_pick_label = rand_pick_mnist_09(mnist, mnist_labels, 1)

    metric_alpha = np.zeros((n,n))
    metric_cost_alpha = np.zeros((n,n))
    metric_alpha_dual = np.zeros((n,n))
    metric_dual_alpha = np.zeros((n,n))
    dist = computeDistMatrixGrid(28)
    dist = dist/np.max(dist)

    for k in range(num_exp):
        mnist_pick_a, mnist_pick_label_a = rand_pick_mnist_09(mnist, mnist_labels, k)
        mnist_pick_b, mnist_pick_label_b = rand_pick_mnist_09(mnist, mnist_labels, k+n)
        for i in range(n):
            for j in range(n):
                alpha, cost_at_alpha, alpha_dual, dual_at_alpha = LP_metric(X=mnist_pick_a[i,:], Y=mnist_pick_b[j,:], dist=dist, delta=0.01)
                metric_alpha[i,j] += alpha
                metric_cost_alpha[i,j] += cost_at_alpha
                metric_alpha_dual[i,j] += alpha_dual
                metric_dual_alpha[i,j] += dual_at_alpha

    metric_alpha /= num_exp
    metric_cost_alpha /= num_exp
    metric_alpha_dual /= num_exp
    metric_dual_alpha /= num_exp
    np.savetxt("alpha.csv", metric_alpha, delimiter=",")
    np.savetxt("cost_alpha.csv", metric_cost_alpha, delimiter=",")
    np.savetxt("alpha_dual.csv", metric_alpha_dual, delimiter=",")
    np.savetxt("dual_alpha.csv", metric_dual_alpha, delimiter=",")