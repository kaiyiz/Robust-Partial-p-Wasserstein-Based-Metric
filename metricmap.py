import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

def split_metric(metric):
    metric_ab = metric[::2,:]
    metric_ab = metric_ab[:,1::2]
    return metric_ab

n = 100
num_labels = 10
# npzfiles = np.load('OTP_lp_metric_n_{}.npz'.format(n))
npzfiles = np.load('cifar_OTP_lp_metric_n_{}_grayscale.npz'.format(n))
all_res = npzfiles['all_res']
metric_alpha = all_res[:,:,0]
np.fill_diagonal(metric_alpha, 0)
metric_cost_alpha = all_res[:,:,1]
metric_alpha_dual = all_res[:,:,2]
metric_dual_alpha = all_res[:,:,3]
metric_scaled_total_cost = all_res[:,:,4]
metric_real_total_cost = all_res[:,:,5]
# data = npzfiles['mnist_pick']
# data_label = npzfiles['mnist_pick_label']
data = npzfiles['data_pick']
data_label = npzfiles['data_pick_label']

L1_metric = cdist(data.reshape(int(n),-1), data.reshape(int(n),-1), metric='minkowski', p=1)
labels = list('0123456789')

def format_fn(tick_val, tick_pos):
    return labels[int(tick_val/num_labels)]

# fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)

ax = sns.heatmap(metric_real_total_cost)
plt.xticks(np.arange(0, n, int(n/num_labels)))
plt.yticks(np.arange(0, n, int(n/num_labels)))
ax.xaxis.set_major_formatter(format_fn)
ax.yaxis.set_major_formatter(format_fn)
ax2 = sns.heatmap(L1_metric)
# ax3 = sns.heatmap(L1_metric, ax=axes[2])
# ax4 = sns.heatmap(L1_metric, ax=axes[3])
# ax5 = sns.heatmap(L1_metric, ax=axes[4])
# ax6 = sns.heatmap(L1_metric, ax=axes[5])

