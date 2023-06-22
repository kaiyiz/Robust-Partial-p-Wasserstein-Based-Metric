'''
This experiment visualing the OTP_metric againest OT and other
distances with mnist data. We used multidimensional scaling (MDS) method.
'''
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.spatial.distance import cdist
from utils import load_data

import warnings
warnings.filterwarnings('ignore')

def mds(data_label, metric, argparse, title):
    metric = (metric + metric.T) / 2
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=5, max_iter=1000)
    projected = mds.fit_transform(metric)
    # project the data into 2D with mnist digits as labels and show legend
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=data_label, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend(*scatter.legend_elements())
    plt.title(title)
    # if there is no folder, create one
    if not os.path.exists('./results/mds_{}/'.format(argparse)):
        os.makedirs('./results/mds_{}/'.format(argparse))
    plt.savefig('./results/mds_'+argparse+'/'+title+'.png')
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--noise', type=float, default=0.1)
    args = parser.parse_args()

    n = int(args.n)
    delta = args.delta
    data_name = args.data_name
    noise = args.noise
    argparse = "n_{}_delta_{}_data_{}_noise_{}".format(n, delta, data_name, noise)
    '''
    alpha: maximum transported mass where partial OT cost less than 1-eps
    alpha_OT: partial OT cost at alpha
    alpha_normalized: maximum transported mass where normalized_OT(alpha) less than 1-alpha
    alpha_normalized_OT: normalized_OT at alpha
    beta: maximum transported mass where maxdual_OT(beta) less than 1-beta
    beta_maxdual: maxdual_OT at beta
    beta_normalized: maximum transported mass where normalized_maxdual_OT(beta) less than 1-beta
    beta_normalized_maxdual: normalized_maxdual_OT at beta
    realtotalCost: real total OT cost
    '''

    data_name_ = 'OTP_lp_metric_{}'.format(argparse)
    data_a, data_b, data_label, alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost = load_data(n, data_name_)


    print("MDS with L1 metric:")
    L1_metric = cdist(data_a.reshape(int(n),-1), data_b.reshape(int(n),-1), metric='minkowski', p=1)
    cur_metric = L1_metric
    mds(data_label, cur_metric, argparse, 'MDS_L1')

    print("MDS with alpha:")
    cur_metric = alpha
    mds(data_label, cur_metric, argparse, 'MDS_alpha')

    print("MDS with OT at alpha:")
    cur_metric = alpha_OT
    mds(data_label, cur_metric, argparse, 'MDST_alpha_OT')

    print("MDS with alpha_normalized:")
    cur_metric = alpha_normalized
    mds(data_label, cur_metric, argparse, 'MDS_alpha_normalized')

    print("MDS with alpha_normalized_OT:")
    cur_metric = alpha_normalized_OT
    mds(data_label, cur_metric, argparse, 'MDS_alpha_normalized_OT')

    print("MDS with beta:")
    cur_metric = beta
    mds(data_label, cur_metric, argparse, 'MDS_beta')

    print("MDS with maxdual at beta:")
    cur_metric = beta_maxdual
    mds(data_label, cur_metric, argparse, 'MDS_beta_maxdual')

    print("MDS with beta_normalized:")
    cur_metric = beta_normalized
    mds(data_label, cur_metric, argparse, 'MDS_beta_normalized')

    print("MDS with beta_normalized_maxdual:")
    cur_metric = beta_normalized_maxdual
    mds(data_label, cur_metric, argparse, 'MDS_beta_normalized_maxdual')

    print("MDS with real OT cost:")
    cur_metric = realtotalCost
    mds(data_label, cur_metric, argparse, 'MDS_real_OT_cost')