import numpy as np
import matplotlib

from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

# npzfiles = np.load('OTP_lp_variables_n_100.npz')
# metric_alpha = npzfiles['metric_alpha']
# metric_cost_alpha = npzfiles['metric_cost_alpha']
# metric_alpha_dual = npzfiles['metric_alpha_dual']
# metric_dual_alpha = npzfiles['metric_dual_alpha']
# metric_scaled_total_cost = npzfiles['metric_scaled_total_cost']
# metric_real_total_cost = npzfiles['metric_real_total_cost']
# mnist = npzfiles['mnist_pick']
# mnist_label = npzfiles['mnist_pick_label']
# 
n = 100

npzfiles = np.load('OTP_lp_metric_n_{}.npz'.format(n))
all_res = npzfiles['all_res']
metric_alpha = all_res[:,:,0]
metric_cost_alpha = all_res[:,:,1]
metric_alpha_dual = all_res[:,:,2]
metric_dual_alpha = all_res[:,:,3]
metric_scaled_total_cost = all_res[:,:,4]
metric_real_total_cost = all_res[:,:,5]
mnist = npzfiles['mnist_pick']
mnist_label = npzfiles['mnist_pick_label']

metric_alpha_mean = np.zeros((10,10))
metric_cost_alpha_mean = np.zeros((10,10))
metric_alpha_dual_mean = np.zeros((10,10))
metric_dual_alpha_mean = np.zeros((10,10))
metric_scaled_total_cost_mean = np.zeros((10,10))
metric_real_total_cost_mean = np.zeros((10,10))

for i in range(10):
    for j in range(10):
        ind_i = np.nonzero(mnist_label==i)[0]
        ind_j = np.nonzero(mnist_label==j)[0]

        metric_alpha_temp = metric_alpha[ind_i,:]
        metric_alpha_temp = metric_alpha_temp[:,ind_j]
        metric_alpha_mean[i,j] = np.mean(metric_alpha_temp)

        metric_cost_alpha_temp = metric_cost_alpha[ind_i,:]
        metric_cost_alpha_temp = metric_cost_alpha_temp[:,ind_j]
        metric_cost_alpha_mean[i,j] = np.mean(metric_cost_alpha_temp)

        metric_alpha_dual_temp = metric_alpha_dual[ind_i,:]
        metric_alpha_dual_temp = metric_alpha_dual_temp[:,ind_j]
        metric_alpha_dual_mean[i,j] = np.mean(metric_alpha_dual_temp)

        metric_dual_alpha_temp = metric_dual_alpha[ind_i,:]
        metric_dual_alpha_temp = metric_dual_alpha_temp[:,ind_j]
        metric_dual_alpha_mean[i,j] = np.mean(metric_dual_alpha_temp)

        metric_scaled_total_cost_temp = metric_scaled_total_cost[ind_i,:]
        metric_scaled_total_cost_temp = metric_scaled_total_cost_temp[:,ind_j]
        metric_scaled_total_cost_mean[i,j] = np.mean(metric_scaled_total_cost_temp)

        metric_real_total_cost_temp = metric_real_total_cost[ind_i,:]
        metric_real_total_cost_temp = metric_real_total_cost_temp[:,ind_j]
        metric_real_total_cost_mean[i,j] = np.mean(metric_real_total_cost_temp)

np.savetxt("metric_alpha_mean_{}_cumdual.csv".format(n), metric_alpha_mean, delimiter=",")
np.savetxt("metric_cost_alpha_mean_{}_cumdual.csv".format(n), metric_cost_alpha_mean, delimiter=",")
np.savetxt("metric_alpha_dual_mean_{}_cumdual.csv".format(n), metric_alpha_dual_mean, delimiter=",")
np.savetxt("metric_dual_alpha_mean_{}_cumdual.csv".format(n), metric_dual_alpha_mean, delimiter=",")
np.savetxt("metric_scaled_total_cost_mean_{}_cumdual.csv".format(n), metric_scaled_total_cost_mean, delimiter=",")
np.savetxt("metric_real_total_cost_mean_{}_cumdual.csv".format(n), metric_real_total_cost_mean, delimiter=",")