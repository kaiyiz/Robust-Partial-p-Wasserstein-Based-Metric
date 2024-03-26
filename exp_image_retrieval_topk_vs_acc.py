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
from utils import load_data, load_computed_matrix, load_computed_matrix_p1p2
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
    acc_query = np.zeros(n)
    for i in range(n):
        cur_label = data_label_a[i]
        cur_top_k = top_k_images[i, :]
        cur_top_k_label = data_label_b[cur_top_k.astype(int)]
        acc_query[i] = np.sum(cur_top_k_label == cur_label)/top_k
        # correct += np.sum(cur_top_k_label == cur_label)
    # acc = correct / (n * top_k)
    acc = np.mean(acc_query)
    acc_std = np.std(acc_query)
    if verbose:
        print("{}_top_{} acc={}".format(metric_name, top_k, acc))
    return top_k_images, acc, acc_std

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
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--delta', type=float, default=0.0001)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--topk_st', type=int, default=1)
    parser.add_argument('--topk_ed', type=int, default=100)
    parser.add_argument('--topk_d', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--shift', type=int, default=2)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--metric_scaler', type=float, default=10.0)
    parser.add_argument('--noise_type', type=str, default='rand1pxl')
    parser.add_argument('--range_noise', type=int, default=0)
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
    range_noise = args.range_noise
    top_ks = np.arange(top_k_st, top_k_ed+top_k_d, top_k_d)

    img_retrival_acc = np.zeros((len(top_ks), 13))
    img_retrival_acc_std = np.zeros((len(top_ks), 13))
    noise_ind = 0
    argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}_sp_{}_nt_{}_rangenoise_{}".format(n, delta, data_name, noise, metric_scaler, shift_pixel, noise_type, range_noise)
    argparse_w1w2 = "n_{}_data_{}_noise_{}_sp_{}_nt_{}_rangenoise_{}_w1w2".format(n, data_name, noise, shift_pixel, noise_type, range_noise)
    argparse_ROBOT12 = "n_{}_data_{}_noise_{}_sp_{}_nt_{}_rangenoise_{}_ROBOT".format(n, data_name, noise, shift_pixel, noise_type,range_noise)
    print(argparse)
    print(argparse_w1w2)
    print(argparse_ROBOT12)
    data_name_ = 'OTP_lp_metric_{}'.format(argparse)
    data_name_w1w2 = 'OTP_lp_metric_{}'.format(argparse_w1w2)
    data_name_ROBOT12 = 'OTP_lp_metric_{}'.format(argparse_ROBOT12)
    try:
        data_a, data_b, data_label_a, data_label_b, alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost, L1_metric = load_computed_matrix(n, data_name_)
    except:
        print("data {} not found, run gen_OTP_metric_matrix.py first".format(data_name_))
        exit(0)
    try: 
        data_a_, data_b_, data_label_a_, data_label_b_, w1, w2 = load_computed_matrix_p1p2(n, data_name_w1w2)
    except:
        print("data {} not found, run gen_w1w2_metric_matrix.py first".format(data_name_w1w2))
        exit(0)
    try:
        data_a_ROBOT, data_b_ROBOT, data_label_a_ROBOT, data_label_b_ROBOT, ROBOT1, ROBOT2 = load_computed_matrix_p1p2(n, data_name_ROBOT12)
    except:
        print("data {} not found, run gen_ROBOT_metric_matrix_IR.py first".format(data_name_ROBOT12))
        exit(0)
    # test if label size is the same, if not, print the size
    if data_label_a.shape != data_label_a_.shape:
        print("data size not matched, RPW_metric shape: {}, w1w2_metric shape: {}".format(data_label_a.shape, data_label_a_.shape))
        exit(0)
    if data_label_b.shape != data_label_b_.shape:
        print("shape size not matched, RPW_metric shape: {}, w1w2_metric shape: {}".format(data_label_b.shape, data_label_b_.shape))
        exit(0)
    if data_label_a.shape != data_label_a_ROBOT.shape:
        print("data size not matched, RPW_metric shape: {}, ROBOT12_metric shape: {}".format(data_label_a.shape, data_label_a_ROBOT.shape))
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
        top_k_images, L1_acc, L1_std = retrive_images(data_label_a, data_label_b, L1_metric, 'L1_metric', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 0] = L1_acc
        img_retrival_acc_std[top_ks_ind, 0] = L1_std
        # save_images(data_name, data_a, data_b, top_k_images, 'L1_metric', argparse)
        # print_retrival_comp(top_k_images, data_label_b)

        top_k_images, alpha_acc, alpha_std = retrive_images(data_label_a, data_label_b, alpha, 'distance_alpha', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 1] = alpha_acc
        img_retrival_acc_std[top_ks_ind, 1] = alpha_std
        # save_images(data_name, data_a, data_b, top_k_images, 'distance_alpha', argparse)
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, alpha_OT_acc, _ = retrive_images(data_label_a, data_label_b, alpha_OT, 'OT_at_alpha', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 2] = alpha_OT_acc
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, alpha_normalized_acc, _ = retrive_images(data_label_a, data_label_b, alpha_normalized, 'distance_alpha_normalized', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 3] = alpha_normalized_acc
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, alpha_normalized_OT_acc, _ = retrive_images(data_label_a, data_label_b, alpha_normalized_OT, 'OT_at_alpha_normalized', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 4] = alpha_normalized_OT_acc
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, beta_acc, _ = retrive_images(data_label_a, data_label_b, beta, 'distance_beta', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 5] = beta_acc
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, beta_maxdual_acc, _ = retrive_images(data_label_a, data_label_b, beta_maxdual, 'maxdual_at_beta', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 6] = beta_maxdual_acc
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, beta_normalized_acc, _ = retrive_images(data_label_a, data_label_b, beta_normalized, 'distance_beta_normalized', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 7] = beta_normalized_acc
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, beta_normalized_maxdual_acc, _ = retrive_images(data_label_a, data_label_b, beta_normalized_maxdual, 'maxdual_at_beta_normalized', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 8] = beta_normalized_maxdual_acc
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, w1_acc, w1_std = retrive_images(data_label_a, data_label_b, w1, 'w1', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 9] = w1_acc
        img_retrival_acc_std[top_ks_ind, 9] = w1_std
        # print("w1_top_{} acc={}".format(top_k, w1_acc))
        # save_images(data_name, data_a, data_b, top_k_images, 'real_total_OT_cost', argparse)
        # print_retrival_comp(top_k_images, data_label)

        top_k_images, w2_acc, w2_std = retrive_images(data_label_a, data_label_b, w2, 'w2', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 10] = w2_acc
        img_retrival_acc_std[top_ks_ind, 10] = w2_std
        # print("w2_top_{} acc={}".format(top_k, w2_acc))

        top_k_images, ROBOT1_acc, ROBOT1_std = retrive_images(data_label_a, data_label_b, ROBOT1, 'ROBOT1', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 11] = ROBOT1_acc
        img_retrival_acc_std[top_ks_ind, 11] = ROBOT1_std

        top_k_images, ROBOT2_acc, ROBOT2_std = retrive_images(data_label_a, data_label_b, ROBOT2, 'ROBOT2', top_k=top_k, verbose=verbose)
        img_retrival_acc[top_ks_ind, 12] = ROBOT2_acc
        img_retrival_acc_std[top_ks_ind, 12] = ROBOT2_std

        top_ks_ind += 1

    # plot the precision vs top_k with different metrics, two subplots 
    # one for alpha beta ..., one for OT at alpha, maxdual at beta, etc
    # plt width 10, height 5
    color_codes = ['b', 'g', 'c', 'm', 'y', 'k', 'w']
    plts.figure()
    plts.figure(figsize=(6,6))
    plts.plot(top_ks, img_retrival_acc[:, 0], label='TV', color='orange')
    # plts.fill_between(top_ks, img_retrival_acc[:, 0]-img_retrival_acc_std[:, 0], img_retrival_acc[:, 0]+img_retrival_acc_std[:, 0], color='g', alpha=0.2)
    plts.plot(top_ks, img_retrival_acc[:, 1], label='(2,{})-RPW'.format(float(1/metric_scaler)), color='b')
    # plts.fill_between(top_ks, img_retrival_acc[:, 1]-img_retrival_acc_std[:, 1], img_retrival_acc[:, 1]+img_retrival_acc_std[:, 1], alpha=0.2, color='b')
    # plts.plot(top_ks, img_retrival_acc[:, 3], label='distance_alpha_normalized')
    # plts.plot(top_ks, img_retrival_acc[:, 5], label='distance_beta')
    # plts.plot(top_ks, img_retrival_acc[:, 7], label='distance_beta_normalized')
    plts.plot(top_ks, img_retrival_acc[:, 9], label='1-Wasserstein', color='purple')
    plts.plot(top_ks, img_retrival_acc[:, 10], label='2-Wasserstein', color='r')
    plts.plot(top_ks, img_retrival_acc[:, 11], label='1-ROBOT', color='g')
    plts.plot(top_ks, img_retrival_acc[:, 12], label='2-ROBOT', color='c')
    # plts.fill_between(top_ks, img_retrival_acc[:, 9]-img_retrival_acc_std[:, 9], img_retrival_acc[:, 9]+img_retrival_acc_std[:, 9], alpha=0.2, color='r')
    plts.xlabel('Number of images retrieved',fontsize=14)    
    plts.ylabel('Accuracy',fontsize=14)
    # plts.title("Image retrival, {}, delta={}".format(title, delta))
    plts.legend(fontsize=14)
    # plts.subplot(1,2,2)
    # plts.plot(top_ks, img_retrival_acc[:, 0], label='L1_metric')
    # plts.plot(top_ks, img_retrival_acc[:, 2], label='OT_at_alpha')
    # # plts.plot(top_ks, img_retrival_acc[:, 4], label='OT_at_alpha_normalized')
    # # plts.plot(top_ks, img_retrival_acc[:, 6], label='maxdual_at_beta')
    # # plts.plot(top_ks, img_retrival_acc[:, 8], label='maxdual_at_beta_normalized')
    # plts.plot(top_ks, img_retrival_acc[:, 9], label='real_total_OT_cost')
    # plts.xlabel('image retrived')
    # plts.ylabel('precision')
    # plts.legend()
    plts.savefig("./results/img_retrival_acc_{}_topk_vs_acc.png".format(argparse), transparent=True)
    plts.savefig("./img_retrival_acc_{}_topk_vs_acc.png".format(argparse), transparent=True)
    plts.close()

    np.savetxt("./results/img_retrival_acc_{}_topk_vs_acc.csv".format(argparse), img_retrival_acc, delimiter=",")