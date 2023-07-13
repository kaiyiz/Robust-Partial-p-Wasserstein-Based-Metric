'''
This experiment compare the preformance of different metric on image matching.
We used a precalculated metric matrix for mnist digits and corresponding labels, and 
extract two sets of images from the dataset and do image matching using the metric.
'''

import numpy as np
import argparse
import torch
import matplotlib
import ot

from scipy.spatial.distance import cdist
from utils import load_data, load_computed_matrix

def split_metric(metric, data_label, seed=0):
    np.random.seed(seed)
    n = metric.shape[0]
    n_half = int(n / 2)
    metric_ab = np.zeros((n_half, n_half))
    data_label_a = np.zeros(n_half)
    data_label_b = np.zeros(n_half)
    label_ind_a = np.zeros(n_half)
    label_ind_b = np.zeros(n_half)
    n_half_digit = int(n_half/10)
    for i in range(10):
        cur_label_ind = np.where(data_label == i)[0]
        cur_label_ind_a = np.random.choice(cur_label_ind, n_half_digit, replace=False)
        cur_label_ind_b = np.array(list(set(cur_label_ind) - set(cur_label_ind_a)))
        label_ind_a[i * n_half_digit: (i + 1) * n_half_digit] = cur_label_ind_a
        label_ind_b[i * n_half_digit: (i + 1) * n_half_digit] = cur_label_ind_b
        data_label_a[i * n_half_digit: (i + 1) * n_half_digit] = data_label[cur_label_ind_a]
        data_label_b[i * n_half_digit: (i + 1) * n_half_digit] = data_label[cur_label_ind_b]
    metric_ab = metric[label_ind_a.astype(int), :][:, label_ind_b.astype(int)]
    return metric_ab, data_label_a, data_label_b

def match_images(metric, data_label, n_exp=10):
    # random split the metric for 10 times, do minimum bipartite matching and calculate the precision
    # of label alignment.
    acc = np.zeros(n_exp)
    for i in range(n_exp):
        metric_ab, data_label_a, data_label_b = split_metric(metric, data_label, seed=i)
        n = len(data_label_a)
        a = np.ones(n)/n
        b = np.ones(n)/n
        G = ot.emd(a, b, metric_ab, numItermax=100000000)
        acc[i] = 1-len(np.nonzero(data_label_b - data_label_b[np.nonzero(G)[1]])[0])/n
    return np.mean(acc), np.std(acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--noise_st', type=float, default=0.0)
    parser.add_argument('--noise_ed', type=float, default=1.0)
    parser.add_argument('--noise_d', type=float, default=0.1)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--metric_scaler', type=float, default=1.0)
    parser.add_argument('--shift_pixel', type=int, default=0)
    args = parser.parse_args()
    print(args)

    n = int(args.n)
    delta = args.delta
    data_name = args.data_name
    noise_st = args.noise_st
    noise_ed = args.noise_ed
    noise_d = args.noise_d
    verbose = args.verbose
    metric_scaler = args.metric_scaler
    shift_pixel = args.shift_pixel
    noise_rates = np.arange(noise_st, noise_ed+noise_d, noise_d)

    matching_acc_res = np.zeros((len(noise_rates), 10))
    matching_std_res = np.zeros((len(noise_rates), 10))
    cur_ind = 0
    for noise in noise_rates:
        noise = round(noise, 2)
        argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}_sp_{}".format(n, delta, data_name, noise, metric_scaler, shift_pixel)
        print(argparse)
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
        try:
            data_a, data_b, data_label, alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost, L1_metric = load_computed_matrix(n, data_name_)
        except:
            print("data {} not found, run gen_OTP_metric_matrix.py first".format(data_name_))
            exit(0)

        L1_acc, L1_std = match_images(L1_metric, data_label)
        matching_acc_res[cur_ind, 0] = L1_acc
        matching_std_res[cur_ind, 0] = L1_std

        alpha_acc, alpha_std = match_images(alpha, data_label)
        matching_acc_res[cur_ind, 1] = alpha_acc
        matching_std_res[cur_ind, 1] = alpha_std

        alpha_OT_acc, alpha_OT_std = match_images(alpha_OT, data_label)
        matching_acc_res[cur_ind, 2] = alpha_OT_acc
        matching_std_res[cur_ind, 2] = alpha_OT_std

        alpha_normalized_acc, alpha_normalized_std = match_images(alpha_normalized, data_label)
        matching_acc_res[cur_ind, 3] = alpha_normalized_acc
        matching_std_res[cur_ind, 3] = alpha_normalized_std

        alpha_normalized_OT_acc, alpha_normalized_OT_std = match_images(alpha_normalized_OT, data_label)
        matching_acc_res[cur_ind, 4] = alpha_normalized_OT_acc
        matching_std_res[cur_ind, 4] = alpha_normalized_OT_std

        beta_acc, beta_std = match_images(beta, data_label)
        matching_acc_res[cur_ind, 5] = beta_acc
        matching_std_res[cur_ind, 5] = beta_std

        beta_maxdual_acc, beta_maxdual_std = match_images(beta_maxdual, data_label)
        matching_acc_res[cur_ind, 6] = beta_maxdual_acc
        matching_std_res[cur_ind, 6] = beta_maxdual_std

        beta_normalized_acc, beta_normalized_std = match_images(beta_normalized, data_label)
        matching_acc_res[cur_ind, 7] = beta_normalized_acc
        matching_std_res[cur_ind, 7] = beta_normalized_std

        beta_normalized_maxdual_acc, beta_normalized_maxdual_std = match_images(beta_normalized_maxdual, data_label)
        matching_acc_res[cur_ind, 8] = beta_normalized_maxdual_acc
        matching_std_res[cur_ind, 8] = beta_normalized_maxdual_std
        
        realtotalCost_acc, realtotalCost_std = match_images(realtotalCost, data_label)
        matching_acc_res[cur_ind, 9] = realtotalCost_acc
        matching_std_res[cur_ind, 9] = realtotalCost_std

        cur_ind += 1
        if verbose:
            print('L1_metric: acc = {}, std = {}'.format(L1_acc, L1_std))
            print('alpha: acc = {}, std = {}'.format(alpha_acc, alpha_std))
            print('alpha_OT: acc = {}, std = {}'.format(alpha_OT_acc, alpha_OT_std))
            print('alpha_normalized: acc = {}, std = {}'.format(alpha_normalized_acc, alpha_normalized_std))
            print('alpha_normalized_OT: acc = {}, std = {}'.format(alpha_normalized_OT_acc, alpha_normalized_OT_std))
            print('beta: acc = {}, std = {}'.format(beta_acc, beta_std))
            print('beta_maxdual: acc = {}, std = {}'.format(beta_maxdual_acc, beta_maxdual_std))
            print('beta_normalized: acc = {}, std = {}'.format(beta_normalized_acc, beta_normalized_std))
            print('beta_normalized_maxdual: acc = {}, std = {}'.format(beta_normalized_maxdual_acc, beta_normalized_maxdual_std))
            print('realtotalCost: acc = {}, std = {}'.format(realtotalCost_acc, realtotalCost_std))

    np.savetxt("./results/matching_acc_res_{}.csv".format(argparse), matching_acc_res, delimiter=",")