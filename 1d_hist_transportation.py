# Visulize the transportation plan between two 1d histograms.
# Histograms A are the mixture of two normal distributions.
# Histograms B are the mixture of three normal distributions, with one of them being outliers.
# The transportation plan is computed by OT.
# We visulize the transport plan by transfer the 2d transportation matrix into a grey scale image, where the intensity of each pixel is proportional to the transportation amount.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import ot

import jpype
import jpype.imports
from jpype.types import *
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx128g", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping

def normal_fun(x, mu, sigma):
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*((x - mu)/sigma)**2)

def OTP_metric(X=None, Y=None, dist=None, delta=0.1, metric_scaler=1):
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
    maxdual = APinfo_cleaned[:,4]/alphaa*metric_scaler
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
    alpha_OT = cumCost[d_ind_a] + (cumCost[d_ind_b]-cumCost[d_ind_a])*(alpha-flowProgress[d_ind_a])/(flowProgress[d_ind_b]-flowProgress[d_ind_a])
    alpha = 1 - alpha

    d_cost = (1 - flowProgress) - normalized_cumcost
    d_ind_a = np.nonzero(d_cost<=0)[0][0]-1
    d_ind_b = d_ind_a + 1
    alpha_normalized = find_intersection_point(flowProgress[d_ind_a], d_cost[d_ind_a], flowProgress[d_ind_b], d_cost[d_ind_b])
    alpha_normalized_OT = normalized_cumcost[d_ind_a] + (normalized_cumcost[d_ind_b]-normalized_cumcost[d_ind_a])*(alpha_normalized-flowProgress[d_ind_a])/(flowProgress[d_ind_b]-flowProgress[d_ind_a])
    alpha_normalized = 1 - alpha_normalized
    
    d_dual = (1 - flowProgress) - maxdual
    d_ind_a = np.nonzero(d_dual<=0)[0][0]-1
    d_ind_b = d_ind_a + 1
    beta = find_intersection_point(flowProgress[d_ind_a], d_dual[d_ind_a], flowProgress[d_ind_b], d_dual[d_ind_b])
    beta_maxdual = maxdual[d_ind_a] + (maxdual[d_ind_b]-maxdual[d_ind_a])*(beta-flowProgress[d_ind_a])/(flowProgress[d_ind_b]-flowProgress[d_ind_a])
    beta = 1 - beta

    d_dual = (1 - flowProgress) - normalized_maxdual
    d_ind_a = np.nonzero(d_dual<=0)[0][0]-1
    d_ind_b = d_ind_a + 1
    beta_normalized = find_intersection_point(flowProgress[d_ind_a], d_dual[d_ind_a], flowProgress[d_ind_b], d_dual[d_ind_b])
    beta_normalized_maxdual = normalized_maxdual[d_ind_a] + (normalized_maxdual[d_ind_b]-normalized_maxdual[d_ind_a])*(beta_normalized-flowProgress[d_ind_a])/(flowProgress[d_ind_b]-flowProgress[d_ind_a])
    beta_normalized = 1 - beta_normalized
    
    realtotalCost = gtSolver.getTotalCost()

    return alpha, alpha_normalized, beta, beta_normalized, alpha_OT, alpha_normalized_OT, beta_maxdual, beta_normalized_maxdual, realtotalCost
    
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

# Parameters
n_bins = 200
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Histogram A
mu1 = 0.3
sigma1 = 0.1
pdf_A_1 = normal_fun(bin_centers, mu1, sigma1)
mu2 = 0.65
sigma2 = 0.1
pdf_A_2 = normal_fun(bin_centers, mu2, sigma2)
mu3 = 0.04
sigma3 = 0.02
pdf_A_3 = normal_fun(bin_centers, mu3, sigma3)
wt1 = 0.3
wt2 = 0.55
wt3 = 0.15
hist_A = wt1*pdf_A_1 + wt2*pdf_A_2 + wt3*pdf_A_3

# Histogram B
mu1 = 0.25 
sigma1 = 0.1
pdf_B_1 = normal_fun(bin_centers, mu1, sigma1)
mu2 = 0.6
sigma2 = 0.1
pdf_B_2 = normal_fun(bin_centers, mu2, sigma2)
mu3 = 0.96
sigma3 = 0.02
pdf_B_3 = normal_fun(bin_centers, mu3, sigma3)
wt1 = 0.2
wt2 = 0.65
wt3 = 0.15
hist_B = wt1*pdf_B_1 + wt2*pdf_B_2 + wt3*pdf_B_3

# Normalize histograms
hist_A /= np.sum(hist_A)
hist_B /= np.sum(hist_B)

# Compute the ground distance
dist = cdist(bin_centers.reshape(-1, 1), bin_centers.reshape(-1, 1), metric = 'sqeuclidean')
# dist /= np.max(dist)

gtSolver = Mapping(n_bins, list(hist_B), list(hist_A), dist, 0.001)
F_ours = gtSolver.getFlow()
F_ours = np.array(F_ours)

F_emd = ot.emd(hist_A, hist_B, dist)

F_sink = ot.sinkhorn(hist_A, hist_B, dist, 0.001)

ks = [1, 0.1, 0.01]
fig, ax = plt.subplots(2+len(ks), 3, figsize=(10, 15))
plt.subplots_adjust(hspace=0.5)
ax[0, 0].axis('off')
ax[0, 2].axis('off')
# merge the first one row
ax[0, 1].plot(bin_centers, hist_A, label='A')
ax[0, 1].plot(bin_centers, hist_B, label='B')
ax[0, 1].set_title('Histograms')
ax[0, 1].legend()

ax[1, 0].imshow(F_ours, cmap='gray')
ax[1, 0].set_title('OT (neupris 19), cost = {:.4f}'.format(np.sum(F_ours*dist)))
ax[1, 0].set_xlabel('B')
ax[1, 0].set_ylabel('A')
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])

ax[1, 1].imshow(F_emd, cmap='gray')
ax[1, 1].set_title('OT (emd), cost = {:.4f}'.format(np.sum(F_emd*dist)))
ax[1, 1].set_xlabel('B')
ax[1, 1].set_ylabel('A')
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])

ax[1, 2].imshow(F_sink, cmap='gray')
ax[1, 2].set_title('OT (sinkhorn), cost = {:.4f}'.format(np.sum(F_sink*dist)))
ax[1, 2].set_xlabel('B')
ax[1, 2].set_ylabel('A')
ax[1, 2].set_xticks([])
ax[1, 2].set_yticks([])

i = 2
for k in ks:
    # k = np.sum(F_ours*dist)
    # calculate OTP metric
    alpha, alpha_normalized, beta, beta_normalized, alpha_OT, alpha_normalized_OT, beta_maxdual, beta_normalized_maxdual, realtotalCost = OTP_metric(hist_B, hist_A, dist, delta=0.01, metric_scaler=1/k)

    # add one dummy node to both histograms and corresponding cost matrix
    # the weight of the dummy node is alpha, and normalized weight to 1 (scale by 1/(1+alpha))
    # the cost between the dummy node and other nodes are 0
    # the cost between the dummy nodes is 1e6
    # after calculate the transportation plan, we get ride of the dummy node from the transport plan, and rescale the remaining transportation plan by 1+alpha

    dummy_mass = beta_maxdual
    # add dummy node
    hist_A_ = np.append(hist_A, dummy_mass)
    hist_B_ = np.append(hist_B, dummy_mass)
    # normalize the histograms
    hist_A_ /= (1+dummy_mass)
    hist_B_ /= (1+dummy_mass)

    # add dummy node to cost matrix
    dist_ = np.append(dist, np.zeros((1, dist.shape[1])), axis=0)
    dist_ = np.append(dist_, np.zeros((dist_.shape[0], 1)), axis=1)
    dist_[-1, -1] = 100

    # calculate the transportation plan
    gtSolver_ = Mapping(n_bins+1, list(hist_B_), list(hist_A_), dist_, 0.001)
    F_ours_ = gtSolver_.getFlow()
    F_ours_ = np.array(F_ours_)
    F_1_minus_dummy_mass = F_ours_[:-1, :-1]
    F_1_minus_dummy_mass *= (1+dummy_mass)

    F_emd_ = ot.emd(hist_A_, hist_B_, dist_)
    F_emd_1_minus_dummy_mass = F_emd_[:-1, :-1]
    F_emd_1_minus_dummy_mass *= (1+dummy_mass)

    F_sink_ = ot.sinkhorn(hist_A_, hist_B_, dist_, 0.001)
    F_sink_1_minus_dummy_mass = F_sink_[:-1, :-1]
    F_sink_1_minus_dummy_mass *= (1+dummy_mass)

    # Plot the histograms and three transport plans with different methods (together with cost vale)

    ax[i, 0].imshow(F_1_minus_dummy_mass, cmap='gray')
    ax[i, 0].set_title('neupris19, k={:.3f}, \n alpha={:.4f}, cost={:.4f}'.format(k, dummy_mass, np.sum(F_1_minus_dummy_mass*dist)))
    ax[i, 0].set_xlabel('B')
    ax[i, 0].set_ylabel('A')

    ax[i, 1].imshow(F_emd_1_minus_dummy_mass, cmap='gray')
    ax[i, 1].set_title('emd, k={:.3f}, \n alpha={:.4f}, cost={:.4f}'.format(k, dummy_mass, np.sum(F_emd_1_minus_dummy_mass*dist)))
    ax[i, 1].set_xlabel('B')
    ax[i, 1].set_ylabel('A')

    ax[i, 2].imshow(F_sink_1_minus_dummy_mass, cmap='gray')
    ax[i, 2].set_title('sinkhorn, k={:.3f}, \n alpha={:.4f}, cost={:.4f}'.format(k, dummy_mass, np.sum(F_sink_1_minus_dummy_mass*dist)))
    ax[i, 2].set_xlabel('B')
    ax[i, 2].set_ylabel('A')

    i += 1

plt.show()