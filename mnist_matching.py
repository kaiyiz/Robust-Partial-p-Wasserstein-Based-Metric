import numpy as np
import torch
import matplotlib
import ot

from scipy.spatial.distance import cdist

def split_metric(metric):
    metric_ab = metric[::2,:]
    metric_ab = metric_ab[:,1::2]
    return metric_ab

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
channel = 0

# npzfiles = np.load('cifar_OTP_lp_metric_n_{}_grayscale.npz'.format(n))
npzfiles = np.load('OTP_lp_metric_n_{}.npz'.format(n))
all_res = npzfiles['all_res']
# all_res = npzfiles['all_res'][:,:,channel,:]
metric_alpha = all_res[:,:,0]
metric_cost_alpha = all_res[:,:,1]
metric_alpha_dual = all_res[:,:,2]
metric_dual_alpha = all_res[:,:,3]
metric_scaled_total_cost = all_res[:,:,4]
metric_real_total_cost = all_res[:,:,5]
# data = npzfiles['data_pick'][:,:,channel]
data = npzfiles['mnist_pick']
data_label = npzfiles['mnist_pick_label']
# data = npzfiles['data_pick']
# data_label = npzfiles['data_pick_label']

data_a = data[::2,:]
data_b = data[1::2,:]
data_label_a = data_label[::2]
data_label_b = data_label[1::2]

metric_alpha_ab = split_metric(metric_alpha)
metric_cost_alpha_ab = split_metric(metric_cost_alpha)
metric_alpha_dual_ab = split_metric(metric_alpha_dual)
metric_dual_alpha_ab = split_metric(metric_dual_alpha)
metric_scaled_total_cost_ab = split_metric(metric_scaled_total_cost)
metric_real_total_cost_ab = split_metric(metric_real_total_cost)
# metric_alpha_ab = metric_alpha[::2,:]
# metric_alpha_ab = metric_alpha_ab[:,1::2]
L1_metric = cdist(data_a.reshape(int(n/2),-1), data_b.reshape(int(n/2),-1), metric='minkowski', p=1)

n = len(data_label_a)
a = np.ones(n)/n
b = np.ones(n)/n

G = ot.emd(a, b, L1_metric, numItermax=100000000)
b_match = data_label_b[np.nonzero(G)[1]]
acc = 1-len(np.nonzero(data_label_b - data_label_b[np.nonzero(G)[1]])[0])/n
print("acc_L1={}".format(acc))

G = ot.emd(a, b, metric_scaled_total_cost_ab, numItermax=100000000)
b_match = data_label_b[np.nonzero(G)[1]]
acc = 1-len(np.nonzero(data_label_b - data_label_b[np.nonzero(G)[1]])[0])/n
print("acc_metric_scaled_total_cost={}".format(acc))

G = ot.emd(a, b, metric_real_total_cost_ab, numItermax=100000000)
b_match = data_label_b[np.nonzero(G)[1]]
acc = 1-len(np.nonzero(data_label_b - data_label_b[np.nonzero(G)[1]])[0])/n
print("acc_metric_real_total_cost={}".format(acc))

G = ot.emd(a, b, metric_alpha_ab, numItermax=100000000)
b_match = data_label_b[np.nonzero(G)[1]]
acc = 1-len(np.nonzero(data_label_b - data_label_b[np.nonzero(G)[1]])[0])/n
print("acc_OTP alpha={}".format(acc))

G = ot.emd(a, b, metric_cost_alpha_ab, numItermax=100000000)
b_match = data_label_b[np.nonzero(G)[1]]
acc = 1-len(np.nonzero(data_label_b - data_label_b[np.nonzero(G)[1]])[0])/n
print("acc_OT cost at OTP alpha={}".format(acc))

G = ot.emd(a, b, metric_alpha_dual_ab, numItermax=100000000)
b_match = data_label_b[np.nonzero(G)[1]]
acc = 1-len(np.nonzero(data_label_b - data_label_b[np.nonzero(G)[1]])[0])/n
print("acc_OTP alpha on dual={}".format(acc))

G = ot.emd(a, b, metric_dual_alpha_ab, numItermax=100000000)
b_match = data_label_b[np.nonzero(G)[1]]
acc = 1-len(np.nonzero(data_label_b - data_label_b[np.nonzero(G)[1]])[0])/n
print("acc_OT dual at OTP alpha={}".format(acc))