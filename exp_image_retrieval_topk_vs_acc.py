'''
This experiment compare the preformance of different metric on image retrival.
We used a precalculated metric matrix for mnist digits and corresponding labels, and
retrieve the top 10 images for each image in the dataset and calculate the precision.
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

def retrive_images(data_label_a, data_label_b, cur_metric, metric_name, top_k=10, verbose=False):
    n = len(data_label_a)
    top_k_images = np.zeros((n, top_k))
    for i in range(n):
        cur_dist = cur_metric[i, :]
        top_k_images[i, :] = np.argsort(cur_dist)[:top_k]
    correct = 0
    # precision is the percentage of images in the top_k images are in the same class
    for i in range(n):
        cur_label = data_label_a[i]
        cur_top_k = top_k_images[i, :]
        cur_top_k_label = data_label_b[cur_top_k.astype(int)]
        correct += np.sum(cur_top_k_label == cur_label) 
    precision = correct / (n * top_k)
    if verbose:
        print("{}_top_{} precision={}".format(metric_name, top_k, precision))
    return top_k_images, precision

def save_images(data_name, data_a, data_b, top_k_images, metric_name, argparse):
    # save the top_k images for each image in the dataset
    save_path = "./results/retrival_{}".format(argparse)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    top_k = top_k_images.shape[1]
    ind = 0
    n_request_per_class = data_a.shape[0] // 10
    for k in range(10):
        if data_name == "mnist":
            nn = 28
            final_image = np.zeros((n_request_per_class*(top_k+1), data_a.shape[1]))
        elif data_name == "cifar10":
            nn = 32
            final_image = np.zeros((n_request_per_class*(top_k+1), data_a.shape[1],3))
        else:
            raise ValueError("data not found")
        cur_data_a = data_a[k*n_request_per_class:(k+1)*n_request_per_class, :]
        for i in range(n_request_per_class):
            cur_image_a = data_a[ind, :]
            final_image[i*(top_k+1), :] = cur_image_a
            cur_top_k = top_k_images[ind, :]
            cur_top_k_images = data_b[cur_top_k.astype(int), :]
            final_image[i*(top_k+1)+1:(i+1)*(top_k+1), :] = cur_top_k_images
            ind += 1
        if data_name == "mnist":
            nn = 28
            # final_image = final_image.reshape(n_request_per_class*nn, (top_k+1)*nn)
            final_image = final_image.reshape(-1,nn,nn) # data shape 2n_request_per_class*32*32*3
            final_image = final_image.reshape(n_request_per_class,top_k+1,nn,nn)
            final_image = final_image.transpose(0,2,1,3)
            final_image = final_image.reshape(n_request_per_class*nn,(top_k+1)*nn)
            cur_image_a = cur_data_a.reshape(n_request_per_class,nn,nn)
            cur_image_a = cur_image_a.transpose(0,2,1)
            cur_image_a = cur_image_a.reshape(n_request_per_class*nn,nn)
            plts.imsave("{}/{}_label_{}_top_{}.png".format(save_path, metric_name, k, top_k), final_image, cmap='gray')
            plts.imsave("{}/label_{}.png".format(save_path, k), cur_image_a, cmap='gray')
        elif data_name == "cifar10":
            nn = 32
            final_image = final_image.reshape(-1,nn,nn,3) # data shape 220*32*32*3
            final_image = final_image.reshape(n_request_per_class,top_k+1,nn,nn,3)
            final_image = final_image.transpose(0,2,1,3,4)
            final_image = final_image.reshape(n_request_per_class*nn,(top_k+1)*nn,3)
            cur_image_a = cur_data_a.reshape(n_request_per_class,nn,nn,3)
            cur_image_a = cur_image_a.transpose(0,2,1,3)
            cur_image_a = cur_image_a.reshape(n_request_per_class*nn,nn,3)
            plts.imsave("{}/{}_label_{}_top_{}.png".format(save_path, metric_name, k, top_k), final_image)
            plts.imsave("{}/label_{}.png".format(save_path, k), cur_image_a)
        else:
            raise ValueError("data not found")

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
    parser.add_argument('--topk_st', type=int, default=1)
    parser.add_argument('--topk_ed', type=int, default=10)
    parser.add_argument('--topk_d', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--metric_scaler', type=float, default=1.0)
    parser.add_argument('--noise_type', type=str, default='blackout')
    args = parser.parse_args()
    print(args)

    n = int(args.n)
    delta = args.delta
    data_name = args.data_name
    top_k_st = args.topk_st
    top_k_ed = args.topk_ed
    top_k_d = args.topk_d
    noise = args.noise
    verbose = args.verbose
    metric_scaler = args.metric_scaler
    shift_pixel = args.shift
    noise_type = args.noise_type
    top_ks = np.arange(top_k_st, top_k_ed+top_k_d, top_k_d)

    img_retrival_res = np.zeros((len(top_ks), 10))
    noise_ind = 0
    argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}_sp_{}_nt_{}".format(n, delta, data_name, noise, metric_scaler, shift_pixel, noise_type)
    print(argparse)
    data_name_ = 'OTP_lp_metric_{}'.format(argparse)
    try:
        data_a, data_b, data_label_a, data_label_b, alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost, L1_metric = load_computed_matrix(n, data_name_)
    except:
        print("data {} not found, run gen_OTP_metric_matrix.py first".format(data_name_))
        exit(0)

    top_ks_ind = 0
    for top_k in top_ks:
        noise = round(noise, 2)
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
        top_k_images, L1_precision = retrive_images(data_label_a, data_label_b, L1_metric, 'L1_metric', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 0] = L1_precision
        # save_images(data_name, data_a, data_b, top_k_images, 'L1_metric', argparse)
        # print_retrival_comp(top_k_images, data_label_b)

        top_k_images, alpha_precision = retrive_images(data_label_a, data_label_b, alpha, 'distance_alpha', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 1] = alpha_precision
        # save_images(data_name, data_a, data_b, top_k_images, 'distance_alpha', argparse)
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, alpha_OT_precision = retrive_images(data_label_a, data_label_b, alpha_OT, 'OT_at_alpha', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 2] = alpha_OT_precision
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, alpha_normalized_precision = retrive_images(data_label_a, data_label_b, alpha_normalized, 'distance_alpha_normalized', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 3] = alpha_normalized_precision
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, alpha_normalized_OT_precision = retrive_images(data_label_a, data_label_b, alpha_normalized_OT, 'OT_at_alpha_normalized', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 4] = alpha_normalized_OT_precision
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, beta_precision = retrive_images(data_label_a, data_label_b, beta, 'distance_beta', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 5] = beta_precision
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, beta_maxdual_precision = retrive_images(data_label_a, data_label_b, beta_maxdual, 'maxdual_at_beta', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 6] = beta_maxdual_precision
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, beta_normalized_precision = retrive_images(data_label_a, data_label_b, beta_normalized, 'distance_beta_normalized', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 7] = beta_normalized_precision
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, beta_normalized_maxdual_precision = retrive_images(data_label_a, data_label_b, beta_normalized_maxdual, 'maxdual_at_beta_normalized', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 8] = beta_normalized_maxdual_precision
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, realtotalCost_precision = retrive_images(data_label_a, data_label_b, realtotalCost, 'real_total_OT_cost', top_k=top_k, verbose=verbose)
        img_retrival_res[top_ks_ind, 9] = realtotalCost_precision
        # save_images(data_name, data_a, data_b, top_k_images, 'real_total_OT_cost', argparse)
        # print_retrival_comp(top_k_images, data_label)
        top_ks_ind += 1

    # plot the precision vs top_k with different metrics, two subplots 
    # one for alpha beta ..., one for OT at alpha, maxdual at beta, etc
    # plt width 10, height 5
    plts.figure()
    plts.figure(figsize=(10,5))
    plts.subplot(1,2,1)
    plts.plot(top_ks, img_retrival_res[:, 0], label='L1_metric')
    plts.plot(top_ks, img_retrival_res[:, 1], label='distance_alpha')
    # plts.plot(top_ks, img_retrival_res[:, 3], label='distance_alpha_normalized')
    # plts.plot(top_ks, img_retrival_res[:, 5], label='distance_beta')
    # plts.plot(top_ks, img_retrival_res[:, 7], label='distance_beta_normalized')
    plts.plot(top_ks, img_retrival_res[:, 9], label='real_total_OT_cost')
    plts.xlabel('image retrived')
    plts.ylabel('precision')
    plts.legend()
    plts.subplot(1,2,2)
    plts.plot(top_ks, img_retrival_res[:, 0], label='L1_metric')
    plts.plot(top_ks, img_retrival_res[:, 2], label='OT_at_alpha')
    # plts.plot(top_ks, img_retrival_res[:, 4], label='OT_at_alpha_normalized')
    # plts.plot(top_ks, img_retrival_res[:, 6], label='maxdual_at_beta')
    # plts.plot(top_ks, img_retrival_res[:, 8], label='maxdual_at_beta_normalized')
    plts.plot(top_ks, img_retrival_res[:, 9], label='real_total_OT_cost')
    plts.xlabel('image retrived')
    plts.ylabel('precision')
    plts.legend()
    plts.savefig("./results/img_retrival_res_{}_topk_vs_acc.png".format(argparse))
    plts.close()

    np.savetxt("./results/img_retrival_res_{}_topk_vs_acc.csv".format(argparse), img_retrival_res, delimiter=",")