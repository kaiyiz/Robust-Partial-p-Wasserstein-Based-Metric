'''
This experiment compare the preformance of different metric in task image retrival.
We used a precalculated metric matrix for mnist digits and corresponding labels, and
retrieve the top 10 images for each image in the dataset and calculate the precision.
'''
import numpy as np
import argparse
import matplotlib.pyplot as plts
import pandas as pd
from scipy.spatial.distance import cdist
from utils import load_data, load_computed_matrix

import warnings
warnings.filterwarnings('ignore')

def retrive_images(data_label, cur_metric, metric_name, top_k=10, verbose=False):
    n = data_label.shape[0]
    top_k_images = np.zeros((n, top_k))
    for i in range(n):
        cur_dist = cur_metric[i, :]
        cur_dist[i] = np.inf
        top_k_images[i, :] = np.argsort(cur_dist)[:top_k]
    correct = 0
    # precision is the percentage of images in the top_k images are in the same class
    for i in range(n):
        cur_label = data_label[i]
        cur_top_k = top_k_images[i, :]
        cur_top_k_label = data_label[cur_top_k.astype(int)]
        correct += np.sum(cur_top_k_label == cur_label) 
    precision = correct / (n * top_k)
    if verbose:
        print("{}_top_{} precision={}".format(metric_name, top_k, precision))
    return top_k_images, precision

# print the average retrival image label composition for digit 0 to 9
def print_retrival_comp(top_k_images, data_label):
    top_k_images_label = data_label[top_k_images.astype(int)]
    top_k_images_label_comp = np.zeros((10, 10))
    for i in range(10):
        cur_label_ind = np.where(data_label == i)[0]
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
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--noise_st', type=float, default=0.0)
    parser.add_argument('--noise_ed', type=float, default=1.0)
    parser.add_argument('--noise_d', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--metric_scaler', type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    n = int(args.n)
    delta = args.delta
    data_name = args.data_name
    noise_st = args.noise_st
    noise_ed = args.noise_ed
    noise_d = args.noise_d
    top_k = args.top_k
    verbose = args.verbose
    metric_scaler = args.metric_scaler
    noise_rates = np.arange(noise_st, noise_ed+noise_d, noise_d)

    img_retrival_res = np.zeros((len(noise_rates), 10))
    cur_ind = 0
    for noise in noise_rates:
        noise = round(noise, 2)
        argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}".format(n, delta, data_name, noise, metric_scaler)
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

        _, L1_precision = retrive_images(data_label, L1_metric, 'L1_metric', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 0] = L1_precision
        # print_retrival_comp(top_k_images, data_label)

        _, alpha_precision = retrive_images(data_label, alpha, 'distance_alpha', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 1] = alpha_precision
        # print_retrival_comp(top_k_images, data_label)

        _, alpha_OT_precision = retrive_images(data_label, alpha_OT, 'OT_at_alpha', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 2] = alpha_OT_precision
        # print_retrival_comp(top_k_images, data_label)

        _, alpha_normalized_precision = retrive_images(data_label, alpha_normalized, 'distance_alpha_normalized', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 3] = alpha_normalized_precision
        # print_retrival_comp(top_k_images, data_label)

        _, alpha_normalized_OT_precision = retrive_images(data_label, alpha_normalized_OT, 'OT_at_alpha_normalized', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 4] = alpha_normalized_OT_precision
        # print_retrival_comp(top_k_images, data_label)

        _, beta_precision = retrive_images(data_label, beta, 'distance_beta', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 5] = beta_precision
        # print_retrival_comp(top_k_images, data_label)

        _, beta_maxdual_precision = retrive_images(data_label, beta_maxdual, 'maxdual_at_beta', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 6] = beta_maxdual_precision
        # print_retrival_comp(top_k_images, data_label)

        _, beta_normalized_precision = retrive_images(data_label, beta_normalized, 'distance_beta_normalized', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 7] = beta_normalized_precision
        # print_retrival_comp(top_k_images, data_label)

        _, beta_normalized_maxdual_precision = retrive_images(data_label, beta_normalized_maxdual, 'maxdual_at_beta_normalized', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 8] = beta_normalized_maxdual_precision
        # print_retrival_comp(top_k_images, data_label)

        _, realtotalCost_precision = retrive_images(data_label, realtotalCost, 'real_total_OT_cost', top_k=top_k, verbose=verbose)
        img_retrival_res[cur_ind, 9] = realtotalCost_precision
        # print_retrival_comp(top_k_images, data_label)

        cur_ind += 1
    
    argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}".format(n, delta, data_name, noise, metric_scaler)
    np.savetxt("./results/img_retrival_res_{}.csv".format(argparse), img_retrival_res, delimiter=",")
