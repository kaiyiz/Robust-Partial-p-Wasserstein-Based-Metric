import numpy as np
import ot

import jpype
import jpype.imports
from jpype.types import *
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx128g", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping

from utils import *

def get_threshold(a, dist, w_ab, p):
    shifting_pixel = 0
    while shifting_pixel < 15:
        # add one empty dimension to a_2, no need to reshape a_2
        a_shifted = shift_image(a[np.newaxis, :], shifting_pixel)[0,:]
        w_ab_shifted = (ot.emd2(a, a_shifted, dist))**(1/p)
        if w_ab_shifted > w_ab:
            return shifting_pixel
        shifting_pixel += 1
    return shifting_pixel

def shifting_compare_wass(a, b, dist, p):
    dist_p = dist**p
    wp_ab = (ot.emd2(a, b, dist_p))**(1/p)
    threshold = get_threshold(a, dist_p, wp_ab, p)
    return wp_ab, threshold

def shifting_compare_RPW(a, b, dist, delta, k, p):
    RPW_ab = RPW(a, b, dist, delta, k, p)
    threshold = get_threshold_RPW(a, dist, RPW_ab, delta, k, p)
    return RPW_ab, threshold

def get_threshold_RPW(a, dist, RPW_ab, delta, k, p):
    shifting_pixel = 0
    while shifting_pixel < 15:
        # add one empty dimension to a_2, no need to reshape a_2
        a_shifted = shift_image(a[np.newaxis, :], shifting_pixel)[0,:]
        RPW_ab_shifted = RPW(a, a_shifted, dist, delta, k, p)
        if RPW_ab_shifted > RPW_ab:
            return shifting_pixel
        shifting_pixel += 1
    return shifting_pixel

def RPW(X=None, Y=None, dist=None, delta=0.1, k=1, p=1):
    # delta : acceptable additive error
    # q_idx : index to get returned values
    nz = len(X)
    dist = dist**p
    alphaa = 4.0*np.max(dist)/delta
    gtSolver = Mapping(nz, list(X), list(Y), dist, delta)
    APinfo = np.array(gtSolver.getAPinfo())

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]

    cost_AP = APinfo_cleaned[:,4] * APinfo_cleaned[:,2]
    cumCost = (np.cumsum(cost_AP)/(alphaa*alphaa*nz))**(1/p)
    # cumCost = np.cumsum(cost_AP)/(alphaa*alphaa*nz)

    cumCost *= 1/k
    totalCost = cumCost[-1]
    if totalCost == 0:
        normalized_cumcost = (cumCost) * 0.0
    else:
        normalized_cumcost = (cumCost)/(1.0 * totalCost)

    maxdual = APinfo_cleaned[:,4]/alphaa*1/k
    final_dual = maxdual[-1]
    if final_dual == 0:
        normalized_maxdual = maxdual * 0.0
    else:
        normalized_maxdual = maxdual/final_dual

    cumFlow = np.cumsum((APinfo_cleaned[:,2]).astype(int))
    totalFlow = cumFlow[-1]
    flowProgress = (cumFlow)/(1.0 * totalFlow)

    d_cost = (1 - flowProgress) - cumCost
    d_ind_a = np.nonzero(d_cost<=0)[0][0]-1
    d_ind_b = d_ind_a + 1
    alpha = find_intersection_point(flowProgress[d_ind_a], d_cost[d_ind_a], flowProgress[d_ind_b], d_cost[d_ind_b])
    RPW = 1 - alpha
    return RPW

def find_intersection_point(x1, y1, x2, y2):
    # x1 < x2
    # y1 > 0
    # y2 < 0
    # y = ax + b
    # find x when y = 0
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    x = -b/a
    return x

N = 10
k = 1
X, y = load_data("mnist")
w1_ab_thresholds = []
w1_a1a2_thresholds = []
w2_ab_thresholds = []
w2_a1a2_thresholds = []
w3_ab_thresholds = []
w3_a1a2_thresholds = []

w1_ab_res = []
w1_a1a2_res = []
w2_ab_res = []
w2_a1a2_res = []
w3_ab_res = []
w3_a1a2_res = []

RPW1_ab_thresholds = []
RPW1_a1a2_thresholds = []
RPW2_ab_thresholds = []
RPW2_a1a2_thresholds = []
RPW3_ab_thresholds = []
RPW3_a1a2_thresholds = []

RPW1_ab_res = []
RPW1_a1a2_res = []
RPW2_ab_res = []
RPW2_a1a2_res = []
RPW3_ab_res = []
RPW3_a1a2_res = []

scaler = 28*np.sqrt(2)

for k in range(N):
    X_pick, y_pick = rand_pick_mnist(X, y, 20, k)
    dist = get_ground_dist(X_pick[0,:], X_pick[1,:], 'fixed_bins_2d', metric='euclidean')
    # Experiment
    for i in range(10):
        a_1, a_2 = X_pick[i*2, :], X_pick[i*2+1, :]
        for j in range(10):
            if j != i:
                b = X_pick[j*2, :]
                w1_ab, w1_threshold = shifting_compare_wass(a_1, b, dist, 1)
                RPW1_ab, RPW1_threshold = shifting_compare_RPW(a_1, b, dist, delta=0.01, k=1, p=1)
                w1_a1a2, w1_threshold_a1a2 = shifting_compare_wass(a_1, a_2, dist, 1)
                RPW1_a1a2, RPW1_threshold_a1a2 = shifting_compare_RPW(a_1, a_2, dist, delta=0.01, k=1, p=1)
                w2_ab, w2_threshold = shifting_compare_wass(a_1, b, dist, 2)
                RPW2_ab, RPW2_threshold = shifting_compare_RPW(a_1, b, dist, delta=0.0001, k=1, p=2)
                w2_a1a2, w2_threshold_a1a2 = shifting_compare_wass(a_1, a_2, dist, 2)
                RPW2_a1a2, RPW2_threshold_a1a2 = shifting_compare_RPW(a_1, a_2, dist, delta=0.0001, k=1, p=2)
                w3_ab, w3_threshold = shifting_compare_wass(a_1, b, dist, 3)
                RPW3_ab, RPW3_threshold = shifting_compare_RPW(a_1, b, dist, delta=0.00001, k=1, p=3)
                w3_a1a2, w3_threshold_a1a2 = shifting_compare_wass(a_1, a_2, dist, 3)
                RPW3_a1a2, RPW3_threshold_a1a2 = shifting_compare_RPW(a_1, a_2, dist, delta=0.00001, k=1, p=3)
                w1_ab_thresholds.append(w1_threshold/scaler)
                w1_a1a2_thresholds.append(w1_threshold_a1a2/scaler)
                RPW1_ab_thresholds.append(RPW1_threshold/scaler)
                RPW1_a1a2_thresholds.append(RPW1_threshold_a1a2/scaler)
                w2_ab_thresholds.append(w2_threshold/scaler)
                w2_a1a2_thresholds.append(w2_threshold_a1a2/scaler)
                RPW2_ab_thresholds.append(RPW2_threshold/scaler)
                RPW2_a1a2_thresholds.append(RPW2_threshold_a1a2/scaler)
                w3_ab_thresholds.append(w3_threshold/scaler)
                w3_a1a2_thresholds.append(w3_threshold_a1a2/scaler)
                RPW3_ab_thresholds.append(RPW3_threshold/scaler)
                RPW3_a1a2_thresholds.append(RPW3_threshold_a1a2/scaler)
                w1_ab_res.append(w1_ab)
                w1_a1a2_res.append(w1_a1a2)
                RPW1_ab_res.append(RPW1_ab)
                RPW1_a1a2_res.append(RPW1_a1a2)
                w2_ab_res.append(w2_ab)
                w2_a1a2_res.append(w2_a1a2)
                RPW2_ab_res.append(RPW2_ab)
                RPW2_a1a2_res.append(RPW2_a1a2)
                w3_ab_res.append(w3_ab)
                w3_a1a2_res.append(w3_a1a2)
                RPW3_ab_res.append(RPW3_ab)
                RPW3_a1a2_res.append(RPW3_a1a2)


res = np.array([w1_ab_thresholds, w1_a1a2_thresholds, w2_ab_thresholds, w2_a1a2_thresholds, w3_ab_thresholds, w3_a1a2_thresholds, w1_ab_res, w1_a1a2_res, w2_ab_res, w2_a1a2_res, w3_ab_res, w3_a1a2_res]).T
res_RPW = np.array([RPW1_ab_thresholds, RPW1_a1a2_thresholds, RPW2_ab_thresholds, RPW2_a1a2_thresholds, RPW3_ab_thresholds, RPW3_a1a2_thresholds, RPW1_ab_res, RPW1_a1a2_res, RPW2_ab_res, RPW2_a1a2_res, RPW3_ab_res, RPW3_a1a2_res]).T

# Print mean values
print("w1_ab_thresholds: ", np.mean(w1_ab_thresholds))
print("w1_ab: ", np.mean(w1_ab_res))
print("w1_a1a2_thresholds: ", np.mean(w1_a1a2_thresholds))
print("w1_a1a2: ", np.mean(w1_a1a2_res))
print("w2_ab_thresholds: ", np.mean(w2_ab_thresholds))
print("w2_ab: ", np.mean(w2_ab))
print("w2_a1a2_thresholds: ", np.mean(w2_a1a2_thresholds))
print("w2_a1a2: ", np.mean(w2_a1a2_res))
print("w3_ab_thresholds: ", np.mean(w3_ab_thresholds))
print("w3_ab: ", np.mean(w3_ab_res))
print("w3_a1a2_thresholds: ", np.mean(w3_a1a2_thresholds))
print("w3_a1a2: ", np.mean(w3_a1a2_res))
# Print mean values for RPWs
print("RPW1_ab_thresholds: ", np.mean(RPW1_ab_thresholds))
print("RPW1_ab: ", np.mean(RPW1_ab_res))
print("RPW1_a1a2_thresholds: ", np.mean(RPW1_a1a2_thresholds))
print("RPW1_a1a2: ", np.mean(RPW1_a1a2_res))
print("RPW2_ab_thresholds: ", np.mean(RPW2_ab_thresholds))
print("RPW2_ab: ", np.mean(RPW2_ab_res))
print("RPW2_a1a2_thresholds: ", np.mean(RPW2_a1a2_thresholds))
print("RPW2_a1a2: ", np.mean(RPW2_a1a2_res))
print("RPW3_ab_thresholds: ", np.mean(RPW3_ab_thresholds))
print("RPW3_ab: ", np.mean(RPW3_ab_res))
print("RPW3_a1a2_thresholds: ", np.mean(RPW3_a1a2_thresholds))
print("RPW3_a1a2: ", np.mean(RPW3_a1a2_res))
# save the w1 w2 thresholds results together as csv
np.savetxt("./results/shifting_threshold.csv", res, delimiter=",")
np.savetxt("./results/RPW_threshold.csv", res_RPW, delimiter=",")