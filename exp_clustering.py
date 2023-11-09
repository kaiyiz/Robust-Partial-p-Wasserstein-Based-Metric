'''
This experiment compare the effective of OTP_metric againest OT and other
distances in clustering task. We used a precalculated metric matrix
for mnist digits and corresponding labels to evaluate the effective.
'''
import numpy as np
import argparse

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import load_computed_matrix
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--noise_st', type=float, default=0)
    parser.add_argument('--noise_ed', type=float, default=1)
    parser.add_argument('--noise_d', type=float, default=0.1)
    parser.add_argument('--shift_st', type=int, default=0)
    parser.add_argument('--shift_ed', type=int, default=0)
    parser.add_argument('--shift_d', type=int, default=1)
    parser.add_argument('--maxiter', type=int, default=1000)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--metric_scaler', type=float, default=1.0)
    parser.add_argument('--noise_type', type=str, default="blackout")
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
    maxiter = args.maxiter
    noise_rates = np.arange(noise_st, noise_ed+noise_d, noise_d)
    shift_pixels = np.arange(shift_pixel_st, shift_pixel_ed+1, shift_pixel_d)

    nmi_res = np.zeros((len(shift_pixels), len(noise_rates), 10))
    noise_ind = 0
    for noise in noise_rates:
        shift_ind = 0
        for shift_pixel in shift_pixels:
            noise = round(noise, 2)
            argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}_sp_{}_nt_{}".format(n, delta, data_name, noise, metric_scaler, shift_pixel, noise_type)
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
            # try:
            data_a, data_b, data_label_a, data_label_b, alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost, L1_metric = load_computed_matrix(n, data_name_)
            # except:
            #     print("data {} not found, run gen_OTP_metric_matrix.py first".format(data_name_))

            print("Clustering with L1 metric:")
            kmedoids_L1 = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(L1_metric)
            nmi_L1 = normalized_mutual_info_score(data_label_a, kmedoids_L1.labels_)
            nmi_res[shift_ind, noise_ind, 0] = nmi_L1

            print("Clustering with distance alpha:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(alpha)
            nmi_alpha = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 1] = nmi_alpha

            print("Clustering with distance OT at alpha:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(alpha_OT)
            nmi_alpha_OT = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 2] = nmi_alpha_OT

            print("Clustering with distance alpha_normalized:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(alpha_normalized)
            nmi_alpha_normalized = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 3] = nmi_alpha_normalized

            print("Clustering with distance normalized_OT at alpha_normalized:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(alpha_normalized_OT)
            nmi_normalized_OT = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 4] = nmi_normalized_OT

            print("Clustering with distance beta:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(beta)
            nmi_beta = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 5] = nmi_beta

            print("Clustering with distance maxdual_OT at beta:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(beta_maxdual)
            nmi_beta_maxdual = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 6] = nmi_beta_maxdual

            print("Clustering with distance beta_normalized:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(beta_normalized)
            nmi_beta_normalized = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 7] = nmi_beta_normalized

            print("Clustering with distance normalized_maxdual_OT at beta_normalized:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(beta_normalized_maxdual)
            nmi_beta_normalized_maxdual = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 8] = nmi_beta_normalized_maxdual

            print("Clustering with distance realtotalCost:")
            kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(realtotalCost)
            nmi_realtotalCost = normalized_mutual_info_score(data_label_a, kmedoids.labels_)
            nmi_res[shift_ind, noise_ind, 9] = nmi_realtotalCost
            shift_ind += 1
        noise_ind += 1

    # make a plot of noise vs accuracy or shift vs accuracy
    nmi_res = np.squeeze(nmi_res)
    x_axis = noise_rates if len(shift_pixels) == 1 else shift_pixels
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    # plot the result of L1, alpha, and OT
    plt.plot(x_axis, nmi_res[:, 0], label="L1")
    plt.plot(x_axis, nmi_res[:, 1], label="alpha")
    plt.plot(x_axis, nmi_res[:, 3], label="alpha_normalized")
    plt.plot(x_axis, nmi_res[:, 5], label="beta")
    plt.plot(x_axis, nmi_res[:, 7], label="beta_normalized")
    plt.plot(x_axis, nmi_res[:, 9], label="OT")
    plt.legend()
    plt.xlabel("noise rate" if len(shift_pixels) == 1 else "shift pixels")
    plt.ylabel("NMI")
    plt.title("Clustering result on {}".format(argparse))
    # plot the result of OT at alpha and beta
    plt.subplot(1,2,2)
    plt.plot(x_axis, nmi_res[:, 0], label="L1")
    plt.plot(x_axis, nmi_res[:, 2], label="alpha_OT")
    plt.plot(x_axis, nmi_res[:, 4], label="normalized_OT")
    plt.plot(x_axis, nmi_res[:, 6], label="maxdual_OT")
    plt.plot(x_axis, nmi_res[:, 8], label="normalized_maxdual_OT")
    plt.plot(x_axis, nmi_res[:, 9], label="OT")
    plt.legend()
    plt.xlabel("noise rate" if len(shift_pixels) == 1 else "shift pixels")
    plt.ylabel("NMI")
    plt.title("Clustering result on {}".format(argparse))
    plt.savefig("./results/clustering_res_{}_noise_vs_acc.png".format(argparse))
    # save the result as csv file
    np.savetxt("./results/clustering_res_{}_noise_vs_acc.csv".format(argparse), nmi_res, delimiter=",")
