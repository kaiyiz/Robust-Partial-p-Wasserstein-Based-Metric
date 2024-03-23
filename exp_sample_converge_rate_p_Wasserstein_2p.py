# This experiment is to test the convergence rate of our PRW

import argparse
import numpy as np
import time
import ot
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist, pdist

import jpype
import jpype.imports
from jpype.types import *
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx128g", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping

from utils import *

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

nn = np.linspace(1, 6, 10)
sample_size = [int(10**i) for i in nn]
N = 10
ms = [0.1, 1, 10]
p = 2
d_OTP_metric = np.zeros((len(sample_size), len(ms), N))
d_emd = np.zeros((len(sample_size), N))
d_l1 = np.zeros((len(sample_size), N))
mu_a_d = 0.0
mu_b_d = 1.0
argparse = "converge_exp_N_{}_2points_p_{}_ms_{}".format(N, p, ms)

# Generate random data
for i in range(len(sample_size)):
    for j in range(N):
        cur_sample_size = sample_size[i]
        np.random.seed(j)
        choices_X = np.random.randint(0, 2, size=cur_sample_size)
        choices_Y = np.random.randint(0, 2, size=cur_sample_size)
        X_num_a = np.sum(choices_X)
        Y_num_b = np.sum(choices_Y)
        discreapency_mass = np.abs((X_num_a - Y_num_b) / cur_sample_size)
        d_emd[i,j] = (discreapency_mass * (mu_a_d - mu_b_d)**p)**(1/p)
        d_l1[i,j] = discreapency_mass * np.abs(mu_a_d - mu_b_d)

        for m in range(len(ms)):
            if discreapency_mass == 0:
                d_OTP_metric[i,m,j] = 0
            else:
                d_OTP_metric[i,m,j] = 1-find_intersection_point(1-discreapency_mass, discreapency_mass, 1, -ms[m]*d_emd[i,j])
        # print('sample size: {}, k: {}, emd: {}, OTP: {}, j: {}'.format(cur_sample_size, float(1/ms[m]), d_emd[i,j], d_OTP_metric[i,m,j], j))

# save the results
np.savez('./results/exp_sample_converge_rate_p_Wasserstein_2p_{}'.format(argparse), d_OTP_metric=d_OTP_metric, d_emd=d_emd, sample_size=sample_size, ms=ms)

# plot the cost curve with error band (a shaded region), x-axis: sample size (log scale), y-axis: average distance (log scale)
fig = plt.figure()
fig.set_figwidth(5)
ax = fig.add_subplot(111)
# make figure wide enough to show all subplots
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Sample Size')
ax.set_ylabel('Cost')
color_codes = ['g', 'b', 'c', 'm', 'y']
col_i = 0
for m in range(len(ms)):
    d_OTP_metric_cur = d_OTP_metric[:,m,:]
    y = np.mean(d_OTP_metric_cur, axis=1)
    error = np.std(d_OTP_metric_cur, axis=1)
    ax.plot(sample_size, y, label='({},{})-RPW'.format(p,float(1/ms[m])), color=color_codes[col_i])
    ax.fill_between(sample_size, y-error, y+error, color=color_codes[col_i], alpha=0.2)
    print('PRW, y: {}, error: {}'.format(y, error))
    print('slope: {}'.format(np.polyfit(np.log(sample_size), np.log(y), 1)[0]))
    col_i += 1
    # fill zeros with 1e-9 to avoid dividing by zero
    d_emd_ = np.where(d_emd==0, 1e-9, d_emd) 
    OTP_metric_vs_d_emd = d_OTP_metric_cur/d_emd_
    # y = np.mean(OTP_metric_vs_d_emd, axis=1)
    # ax.plot(sample_size, y, label='EMD/PRW mean, k = {}'.format(float(1/ms[m])), color='red')


y = np.mean(d_emd, axis=1)
error = np.std(d_emd, axis=1)
ax.plot(sample_size, y, label='{}-Wasserstein'.format(p), color='r')
ax.fill_between(sample_size, y-error, y+error, color='r', alpha=0.2)
print('Wasserstein, y: {}, error: {}'.format(y, error))
print('slope: {}'.format(np.polyfit(np.log(sample_size), np.log(y), 1)[0]))

y = np.mean(d_l1, axis=1)
error = np.std(d_l1, axis=1)
ax.plot(sample_size, y, label='TV', color='orange')
ax.fill_between(sample_size, y-error, y+error, color='orange', alpha=0.2)
print('L1, y: {}, error: {}'.format(y, error))
print('slope: {}'.format(np.polyfit(np.log(sample_size), np.log(y), 1)[0]))

ax.legend()
# make legend small enough to fit in the figure, upper right corner
ax.legend(loc='lower left', prop={'size': 8})

# save figure with a name that contains all the parameters, so that we can compare different experiments
fig.savefig('./results/exp_sample_converge_rate_p_Wasserstein_2p_{}.png'.format(argparse), dpi=fig.dpi, transparent=True)