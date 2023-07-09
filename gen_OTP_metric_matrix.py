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
from utils import load_data, add_noise, get_ground_dist, rand_pick_mnist, rand_pick_cifar10, add_noise_3d_matching


"""
Relevant parts of the code have been adapted from :
https://github.com/debarghya-mukherjee/Robust-Optimal-Transport/blob/main/ROBOT_mnist_outlier_detection.py
"""

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
    parser.add_argument('--n', type=int, default=20)
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

    data, data_labels = load_data(data_name)
    all_res = np.zeros((n,n,10))

    if data_name == "mnist":
        data_pick_a, data_pick_label = rand_pick_mnist(data, data_labels, n, 0)
        data_pick_b, data_pick_label = rand_pick_mnist(data, data_labels, n, 1)
        data_pick_b_noise = add_noise(data_pick_b, noise_type = 'geo_normal', noise_level=noise)
        dist = get_ground_dist(data_pick_a[0,:], data_pick_b_noise[1,:], 'geo_transport')
        start_time = time.time()
        Parallel(n_jobs=-1, prefer="threads")(delayed(OTP_metric)(data_pick_a[i,:], data_pick_b_noise[j,:], dist, delta, metric_scaler, all_res, i, j, start_time) for i in range(n) for j in range(n))
        end_time = time.time()
    elif data_name == "cifar10":
        start_time = time.time()
        data_pick_a, data_pick_label = rand_pick_cifar10(data, data_labels, n, 0)
        data_pick_b, data_pick_label = rand_pick_cifar10(data, data_labels, n, 1)
        data_pick_b_noise = add_noise_3d_matching(data_pick_b, noise_type = 'uniform', noise_level=noise)
        m = data_pick_a.shape[1]
        a = np.ones(m)/m
        b = np.ones(m)/m
        Parallel(n_jobs=-1, prefer="threads")(delayed(OTP_metric)(a, b, get_ground_dist(data_pick_a[i,:], data_pick_b_noise[j,:], transport_type="matching"), delta, metric_scaler, all_res, i, j, start_time) for i in range(n) for j in range(n))
        end_time = time.time()
    else:
        raise ValueError("data not found")

    L1_metric = cdist(data_pick_a.reshape(int(n),-1), data_pick_b_noise.reshape(int(n),-1), metric='minkowski', p=1)
    all_res[:,:,9] = L1_metric

    print("finish all job in {}s".format(end_time-start_time))
    np.savez('./results/OTP_lp_metric_{}'.format(argparse), all_res=all_res, data_a=data_pick_a, data_b=data_pick_b_noise, mnist_pick_label=data_pick_label)