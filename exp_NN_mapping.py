'''
This experiment compare the preformance of different metric on nearest neighbor mapping.
We used a precalculated metric for mnist digits/cifar10 images and corresponding labels, and
compute the nearest neighbor between n centroids of images and ublabeled images.
'''
import numpy as np
import argparse
import matplotlib.pyplot as plts
import pandas as pd
from scipy.spatial.distance import cdist
from utils import load_data, load_computed_matrix
import os

import warnings
warnings.filterwarnings('ignore')

def nearest_neighbor(data_label_a, data_label_b, cur_metric, metric_name, verbose=False):
    # compute the nearest neighbor for each image in data_a
    # data_label_a: labels of data_a
    # data_label_b: labels of data_b
    # cur_metric: metric to compute the distance between two images
    # metric_name: name of the metric
    # verbose: print the progress
    if verbose:
        print("computing nearest neighbor for {}...".format(metric_name))
    n = cur_metric.shape[0]
    m = cur_metric.shape[1]
    print("n: {}, m: {}".format(n, m))
    print("label_a: {}, label_b: {}".format(data_label_a.shape, data_label_b.shape))
    nn_images_ind = np.zeros(m)
    for i in range(m):
        if verbose:
            if i % 100 == 0:
                print("progress: {}/{}".format(i, m))
        cur_metric_i = cur_metric[:, i]
        cur_metric_i_ind = np.argsort(cur_metric_i)
        nn_images_ind[i] = cur_metric_i_ind[0]
    nn_images_mapping_label = data_label_a[nn_images_ind.astype(int)]
    acc = np.sum(nn_images_mapping_label == data_label_b) / m
    return acc

# print the average retrival image label composition for digit 0 to 9
def print_retrival_comp(top_k_images, data_label_b):
    top_k_images_label = data_label_b[top_k_images.astype(int)]
    top_k_images_label_comp = np.zeros((10, 10))
    for i in range(10):
        cur_label_ind = np.where(data_label_b == i)[0]
        cur_top_k_images_label = top_k_images_label[cur_label_ind, :]
        # count the number of each label in the top_k images
        cur_comp = np.zeros(10)
        for j in range(10):
            cur_comp[j] = np.sum(cur_top_k_images_label == j)
        cur_comp = cur_comp / np.sum(cur_comp)
        top_k_images_label_comp[i, :] = cur_comp
    # print a table of average retrival image label composition
    print("average retrival image label composition:")
    print(pd.DataFrame(top_k_images_label_comp, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--noise_st', type=float, default=0.0)
    parser.add_argument('--noise_ed', type=float, default=1.0)
    parser.add_argument('--noise_d', type=float, default=0.1)
    parser.add_argument('--shift_st', type=int, default=0)
    parser.add_argument('--shift_ed', type=int, default=0)
    parser.add_argument('--shift_d', type=int, default=1)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--metric_scaler', type=float, default=1.0)
    parser.add_argument('--noise_type', type=str, default='blackout')
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
    shift_pixel_st = args.shift_st
    shift_pixel_ed = args.shift_ed
    shift_pixel_d = args.shift_d
    noise_type = args.noise_type
    noise_rates = np.arange(noise_st, noise_ed+noise_d, noise_d)
    shift_pixels = np.arange(shift_pixel_st, shift_pixel_ed+1, shift_pixel_d)

    nn_res = np.zeros((len(shift_pixels), len(noise_rates), 10))
    noise_ind = 0
    for noise in noise_rates:
        shift_ind = 0
        for shift_pixel in shift_pixels:
            noise = round(noise, 2)
            argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}_sp_{}_nt_{}".format(n, delta, data_name, noise, metric_scaler, shift_pixel, noise_type)
            print(argparse)
            data_name_ = 'OTP_lp_metric_{}'.format(argparse)
            try:
                data_a, data_b, data_label_a, data_label_b, alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost, L1_metric = load_computed_matrix(n, data_name_)
            except:
                print("data {} not found, run gen_OTP_metric_matrix.py first".format(data_name_))
                exit(0)
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
            # compute the nearest neighbor for each image in data_b using L1 metric
            L1_acc = nearest_neighbor(data_label_a, data_label_b, L1_metric, "L1", verbose=verbose)
            nn_res[shift_ind, noise_ind, 0] = L1_acc
            print("L1 acc: {}".format(L1_acc))

            # compute the nearest neighbor for each image in data_b using OTP metric
            OTP_acc = nearest_neighbor(data_label_a, data_label_b, alpha, "OTP", verbose=verbose)
            nn_res[shift_ind, noise_ind, 1] = OTP_acc
            print("OTP acc: {}".format(OTP_acc))

            # compute the nearest neighbor for each image in data_b using OT_at_alpha metric    
            OT_at_alpha_acc = nearest_neighbor(data_label_a, data_label_b, alpha_OT, "OT_at_alpha", verbose=verbose)
            nn_res[shift_ind, noise_ind, 2] = OT_at_alpha_acc
            print("OT_at_alpha acc: {}".format(OT_at_alpha_acc))

            # compute the nearest neighbor for each image in data_b using alpha_normalized metric
            alpha_normalized_acc = nearest_neighbor(data_label_a, data_label_b, alpha_normalized, "alpha_normalized", verbose=verbose)
            nn_res[shift_ind, noise_ind, 3] = alpha_normalized_acc
            print("alpha_normalized acc: {}".format(alpha_normalized_acc))

            # compute the nearest neighbor for each image in data_b using alpha_normalized_OT metric
            alpha_normalized_OT_acc = nearest_neighbor(data_label_a, data_label_b, alpha_normalized_OT, "alpha_normalized_OT", verbose=verbose)
            nn_res[shift_ind, noise_ind, 4] = alpha_normalized_OT_acc
            print("alpha_normalized_OT acc: {}".format(alpha_normalized_OT_acc))

            # compute the nearest neighbor for each image in data_b using beta metric
            beta_acc = nearest_neighbor(data_label_a, data_label_b, beta, "beta", verbose=verbose)
            nn_res[shift_ind, noise_ind, 5] = beta_acc
            print("beta acc: {}".format(beta_acc))

            # compute the nearest neighbor for each image in data_b using beta_maxdual metric
            beta_maxdual_acc = nearest_neighbor(data_label_a, data_label_b, beta_maxdual, "beta_maxdual", verbose=verbose)
            nn_res[shift_ind, noise_ind, 6] = beta_maxdual_acc
            print("beta_maxdual acc: {}".format(beta_maxdual_acc))

            # compute the nearest neighbor for each image in data_b using beta_normalized metric
            beta_normalized_acc = nearest_neighbor(data_label_a, data_label_b, beta_normalized, "beta_normalized", verbose=verbose)
            nn_res[shift_ind, noise_ind, 7] = beta_normalized_acc
            print("beta_normalized acc: {}".format(beta_normalized_acc))

            # compute the nearest neighbor for each image in data_b using beta_normalized_maxdual metric
            beta_normalized_maxdual_acc = nearest_neighbor(data_label_a, data_label_b, beta_normalized_maxdual, "beta_normalized_maxdual", verbose=verbose)
            nn_res[shift_ind, noise_ind, 8] = beta_normalized_maxdual_acc
            print("beta_normalized_maxdual acc: {}".format(beta_normalized_maxdual_acc))

            # compute the nearest neighbor for each image in data_b using realtotalCost metric
            realtotalCost_acc = nearest_neighbor(data_label_a, data_label_b, realtotalCost, "realtotalCost", verbose=verbose)
            nn_res[shift_ind, noise_ind, 9] = realtotalCost_acc
            print("realtotalCost acc: {}".format(realtotalCost_acc))

            shift_ind += 1
        noise_ind += 1

np.savetxt("./results/nn_mapping_res_{}_topk_vs_acc.csv".format(argparse), np.squeeze(nn_res), delimiter=",")
