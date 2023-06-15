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
n = 100
channel = 2

# npzfiles = np.load('OTP_lp_metric_n_{}.npz'.format(n))
npzfiles = np.load('cifar_OTP_lp_metric_n_{}_grayscale.npz'.format(n))
# all_res = npzfiles['all_res'][:,:,channel,:]
all_res = npzfiles['all_res']
metric_alpha = all_res[:,:,0]
metric_cost_alpha = all_res[:,:,1]
metric_alpha_dual = all_res[:,:,2]
metric_dual_alpha = all_res[:,:,3]
metric_scaled_total_cost = all_res[:,:,4]
metric_real_total_cost = all_res[:,:,5]
data = npzfiles['data_pick']
# all_res = npzfiles['all_res'][:,:,channel,:]
data_label = npzfiles['data_pick_label']

print("Clustering with L1 metric:")
L1_metric = cdist(data.reshape(int(n),-1), data.reshape(int(n),-1), metric='minkowski', p=1)

np.savetxt('L1.csv', L1_metric, delimiter=',')
cur_metric = L1_metric
kmedoids_L1 = KMedoids(n_clusters=10, metric='precomputed', method  ='pam').fit(cur_metric)
nmi_L1 = normalized_mutual_info_score(data_label, kmedoids_L1.labels_)
print("nmi={}".format(nmi_L1))

cur_metric = metric_scaled_total_cost
print("Clustering with metric_scaled_total_cost:")
kmedoids = KMedoids(n_clusters=10, metric='precomputed', method  ='pam').fit(cur_metric)
nmi = normalized_mutual_info_score(data_label, kmedoids.labels_)
print("nmi={}".format(nmi))

cur_metric = metric_real_total_cost
print("Clustering with metric_real_total_cost:")
kmedoids = KMedoids(n_clusters=10, metric='precomputed', method  ='pam').fit(cur_metric)
nmi = normalized_mutual_info_score(data_label, kmedoids.labels_)
print("nmi={}".format(nmi))

cur_metric = metric_alpha
print("Clustering with OTP alpha metric:")
kmedoids = KMedoids(n_clusters=10, metric='precomputed', method  ='pam').fit(cur_metric)
nmi = normalized_mutual_info_score(data_label, kmedoids.labels_)
print("nmi={}".format(nmi))

cur_metric = metric_cost_alpha
print("Clustering with OT cost at OTP alpha metric:")
kmedoids = KMedoids(n_clusters=10, metric='precomputed', method  ='pam').fit(cur_metric)
nmi = normalized_mutual_info_score(data_label, kmedoids.labels_)
print("nmi={}".format(nmi))

cur_metric = metric_alpha_dual
print("Clustering with OTP alpha on dual metric:")
kmedoids = KMedoids(n_clusters=10, metric='precomputed', method  ='pam').fit(cur_metric)
nmi = normalized_mutual_info_score(data_label, kmedoids.labels_)
print("nmi={}".format(nmi))

cur_metric = metric_dual_alpha
print("Clustering with OT dual at OTP alpha metric:")
kmedoids = KMedoids(n_clusters=10, metric='precomputed', method  ='pam').fit(cur_metric)
nmi = normalized_mutual_info_score(data_label, kmedoids.labels_)
print("nmi={}".format(nmi))