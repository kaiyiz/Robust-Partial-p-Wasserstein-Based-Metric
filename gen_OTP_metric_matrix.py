import argparse
from operator import index
import string
from tkinter import NONE
import numpy as np
import ot
import os
import sys
import time
# import tensorflow as tf
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
from utils import add_niose


"""
Relevant parts of the code have been adapted from :
https://github.com/debarghya-mukherjee/Robust-Optimal-Transport/blob/main/ROBOT_mnist_outlier_detection.py
"""

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
    p = np.random.permutation(len(mnist_labels))
    mnist = mnist[p,:,:]
    mnist_labels = mnist_labels[p]
    # all_index = np.arange(len(mnist_labels))
    # index_perm = np.random.permutation(all_index)

    ind_all = np.array([])
    for i in range(10):
        ind = np.nonzero(mnist_labels == i)[0][:int(n/10)]
        ind_all = np.append(ind_all, ind)

    ind_all = ind_all.astype(int)
    mnist_pick, mnist_pick_label = mnist[ind_all, :, :], mnist_labels[ind_all]
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

def OTP_metric(X=None, Y=None, dist=None, delta=0.1, metric_scaler=1, all_res=None, i=0, j=0, time_start=NONE):
    # delta : acceptable additive error
    # q_idx : index to get returned values
    nz = len(X)
    gtSolver = Mapping(nz, list(X), list(Y), dist, delta)
    APinfo = np.array(gtSolver.getAPinfo())

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]

    cost_AP = APinfo_cleaned[:,4] * APinfo_cleaned[:,2]
    cumCost = np.cumsum(cost_AP)
    real_total_cost = gtSolver.getTotalCost()
    if real_total_cost == 0:
        cumCost = cumCost * 0.0
    else:
        cumCost = cumCost / (cumCost[-1] / real_total_cost)
    cumCost *= metric_scaler
    totalCost = cumCost[-1]
    if totalCost == 0:
        normalized_cumcost = (cumCost) * 0.0
    else:
        normalized_cumcost = (cumCost)/(1.0 * totalCost)

    alphaa = 4.0*np.max(dist)/delta
    maxdual = APinfo_cleaned[:,4]/alphaa
    final_dual = maxdual[-1]
    if final_dual == 0:
        normalized_maxdual = maxdual * 0.0
    else:
        normalized_maxdual = maxdual/final_dual

    cumFlow = np.cumsum((APinfo_cleaned[:,2]).astype(int))
    totalFlow = cumFlow[-1]
    flowProgress = (cumFlow)/(1.0 * totalFlow)

    d_cost = (1 - flowProgress) - cumCost
    d_ind = np.nonzero(d_cost<=0)[0][0]-1
    alpha = 1-flowProgress[d_ind]
    alpha_OT = cumCost[d_ind]
    d_cost = (1 - flowProgress) - normalized_cumcost
    d_ind = np.nonzero(d_cost<=0)[0][0]-1
    alpha_normalized = 1-flowProgress[d_ind]
    alpha_normalized_OT = normalized_cumcost[d_ind]
    
    d_dual = (1 - flowProgress) - maxdual
    d_ind = np.nonzero(d_dual<=0)[0][0]-1
    beta = 1-flowProgress[d_ind]
    beta_maxdual = maxdual[d_ind]
    d_dual = (1 - flowProgress) - normalized_maxdual
    d_ind = np.nonzero(d_dual<=0)[0][0]-1
    beta_normalized = 1-flowProgress[d_ind]
    beta_normalized_maxdual = normalized_maxdual[d_ind]
    
    realtotalCost = gtSolver.getTotalCost()

    if j == 0:
        size = all_res.shape
        time_cur = time.time()
        amount_of_work_done = i*size[0] + j + 1
        time_spent = time_cur - time_start
        total_time = time_spent/(amount_of_work_done/(size[0]*size[1]))
        print("estimate to finish the job in {}s".format(total_time-time_spent))

    all_res[i,j,:9] = np.array([alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost])

if __name__ == "__main__":
    # LOAD Data
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--metric_scaler', type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    n = int(args.n)
    delta = args.delta
    data_name = args.data_name
    noise = args.noise
    metric_scaler = args.metric_scaler
    argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}".format(n, delta, data_name, noise, metric_scaler)

    if os.path.exists('./data/mnist.npy'):
        mnist = np.load('./data/mnist.npy') # 60k x 28 x 28
        mnist_labels = np.load('./data/mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8
    else:
        (mnist, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        np.save('mnist.npy', mnist)
        np.save('mnist_labels.npy', mnist_labels)

    mnist_pick_a, mnist_pick_label = rand_pick_mnist(mnist, mnist_labels, n, 0)
    mnist_pick_b, mnist_pick_label = rand_pick_mnist(mnist, mnist_labels, n, 1)

    mnist_pick_b_noise = add_niose(mnist_pick_b, noise_level=noise)
    # mnist_pick, mnist_pick_label = rand_pick_mnist_09(mnist, mnist_labels, 1)

    all_res = np.zeros((n,n,10))
    dist = computeDistMatrixGrid(28)
    dist = dist/np.max(dist)
    start_time = time.time()
    Parallel(n_jobs=-1, prefer="threads")(delayed(OTP_metric)(mnist_pick_a[i,:], mnist_pick_b_noise[j,:], dist, delta, metric_scaler, all_res, i, j, start_time) for i in range(n) for j in range(n))
    end_time = time.time()

    L1_metric = cdist(mnist_pick_a.reshape(int(n),-1), mnist_pick_b_noise.reshape(int(n),-1), metric='minkowski', p=1)
    all_res[:,:,9] = L1_metric

    print("finish all job in {}s".format(end_time-start_time))
    np.savez('./results/OTP_lp_metric_{}'.format(argparse), all_res=all_res, data_a=mnist_pick_a, data_b=mnist_pick_b_noise, mnist_pick_label=mnist_pick_label)