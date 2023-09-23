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

def save_images(data_name, data_a, data_b, data_label, top_k_images, metric_name, argparse):
    # save the top_k images for each image in the dataset
    save_path = "./results/retrival_{}".format(argparse)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    n = data_label.shape[0]
    top_k = top_k_images.shape[1]
    ind = 0
    for k in range(10):
        if data_name == "mnist":
            nn = 28
            final_image = np.zeros((20*(top_k+1), data_a.shape[1]))
        elif data_name == "cifar10":
            nn = 32
            final_image = np.zeros((20*(top_k+1), data_a.shape[1],3))
        else:
            raise ValueError("data not found")
        cur_data_a = data_a[k*20:(k+1)*20, :]
        for i in range(20):
            cur_image_a = data_a[ind, :]
            final_image[i*(top_k+1), :] = cur_image_a
            # cur_label = data_label[i]
            cur_top_k = top_k_images[ind, :]
            cur_top_k_images = data_b[cur_top_k.astype(int), :]
            final_image[i*(top_k+1)+1:(i+1)*(top_k+1), :] = cur_top_k_images
            ind += 1
        if data_name == "mnist":
            nn = 28
            # final_image = final_image.reshape(20*nn, (top_k+1)*nn)
            final_image = final_image.reshape(-1,nn,nn) # data shape 220*32*32*3
            final_image = final_image.reshape(20,11,nn,nn)
            final_image = final_image.transpose(0,2,1,3)
            final_image = final_image.reshape(20*nn,11*nn)
            cur_image_a = cur_data_a.reshape(20,nn,nn)
            cur_image_a = cur_image_a.transpose(0,2,1)
            cur_image_a = cur_image_a.reshape(20*nn,nn)
            plts.imsave("{}/{}_label_{}_top_{}.png".format(save_path, metric_name, k, top_k), final_image, cmap='gray')
            plts.imsave("{}/label_{}.png".format(save_path, k), cur_image_a, cmap='gray')
        elif data_name == "cifar10":
            nn = 32
            final_image = final_image.reshape(-1,nn,nn,3) # data shape 220*32*32*3
            final_image = final_image.reshape(20,11,nn,nn,3)
            final_image = final_image.transpose(0,2,1,3,4)
            final_image = final_image.reshape(20*nn,11*nn,3)
            cur_image_a = cur_data_a.reshape(20,nn,nn,3)
            cur_image_a = cur_image_a.transpose(0,2,1,3)
            cur_image_a = cur_image_a.reshape(20*nn,nn,3)
            plts.imsave("{}/{}_label_{}_top_{}.png".format(save_path, metric_name, k, top_k), final_image)
            plts.imsave("{}/label_{}.png".format(save_path, k), cur_image_a)
        else:
            raise ValueError("data not found")

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
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--noise_st', type=float, default=0.0)
    parser.add_argument('--noise_ed', type=float, default=1.0)
    parser.add_argument('--noise_d', type=float, default=0.1)
    parser.add_argument('--shift_st', type=int, default=0)
    parser.add_argument('--shift_ed', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--noise_type', type=str, default='geo_normal')
    args = parser.parse_args()
    print(args)

    n = int(args.n)
    data_name = args.data_name
    noise_st = args.noise_st
    noise_ed = args.noise_ed
    noise_d = args.noise_d
    top_k = args.top_k
    verbose = args.verbose
    shift_pixel_st = args.shift_st
    shift_pixel_ed = args.shift_ed
    noise_type = args.noise_type
    noise_rates = np.arange(noise_st, noise_ed+noise_d, noise_d)
    shift_pixels = np.arange(shift_pixel_st, shift_pixel_ed+1, 1)

    img_retrival_res = np.zeros((len(shift_pixels), len(noise_rates)))
    noise_ind = 0
    for noise in noise_rates:
        shift_ind = 0
        for shift_pixel in shift_pixels:
            noise = round(noise, 2)
            argparse = "n_{}_data_{}_noise_{}_sp_{}_nt_{}".format(n, data_name, noise, shift_pixel, noise_type)
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

            data_name_ = 'LP_metric_{}'.format(argparse)
            try:
                npzfiles = np.load('./results/{}.npz'.format(data_name_))
                LP_metric = npzfiles['all_res']
                data_label = npzfiles['mnist_pick_label']
                data_a = npzfiles['data_a']
                data_b = npzfiles['data_b']
            except:
                print("data {} not found, run gen_OTP_metric_matrix.py first".format(data_name_))
                exit(0)

            top_k_images, LP_precision = retrive_images(data_label, LP_metric, 'LP_metric', top_k=top_k, verbose=verbose)
            img_retrival_res[shift_ind, noise_ind] = LP_precision
            save_images(data_name, data_a, data_b, data_label, top_k_images, 'LP_metric', argparse)

            shift_ind += 1
        noise_ind += 1

    img_retrival_res = np.squeeze(img_retrival_res)
    np.savetxt("./results/img_retrival_res_{}_LP.csv".format(argparse), img_retrival_res, delimiter=",")
