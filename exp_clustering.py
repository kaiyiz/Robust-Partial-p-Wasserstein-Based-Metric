'''
This experiment compare the effective of OTP_metric againest OT and other
distances in clustering task. We used a precalculated metric matrix
for mnist digits and corresponding labels to evaluate the effective.
'''
import numpy as np
import argparse

from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import load_data

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
    parser.add_argument('--maxiter', type=int, default=1000)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    n = int(args.n)
    delta = args.delta
    data_name = args.data_name
    noise_st = args.noise_st
    noise_ed = args.noise_ed
    noise_d = args.noise_d
    maxiter = args.maxiter
    verbose = args.verbose
    noise_rates = np.arange(noise_st, noise_ed+noise_d, noise_d)

    nmi_res = np.zeros((len(noise_rates), 10))
    cur_ind = 0
    for noise in noise_rates:
        noise = round(noise, 2)
        argparse = "n_{}_delta_{}_data_{}_noise_{}".format(n, delta, data_name, noise)
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
            data_a, data_b, data_label, alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost, L1_metric = load_data(n, data_name_)
        except:
            print("data {} not found, run gen_OTP_metric_matrix.py first".format(data_name_))
            exit(0)

        print("Clustering with L1 metric:")
        kmedoids_L1 = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(L1_metric)
        nmi_L1 = normalized_mutual_info_score(data_label, kmedoids_L1.labels_)
        nmi_res[cur_ind, 0] = nmi_L1

        print("Clustering with distance alpha:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(alpha)
        nmi_alpha = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 1] = nmi_alpha

        print("Clustering with distance OT at alpha:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(alpha_OT)
        nmi_alpha_OT = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 2] = nmi_alpha_OT

        print("Clustering with distance alpha_normalized:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(alpha_normalized)
        nmi_alpha_normalized = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 3] = nmi_alpha_normalized

        print("Clustering with distance normalized_OT at alpha_normalized:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(alpha_normalized_OT)
        nmi_normalized_OT = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 4] = nmi_normalized_OT

        print("Clustering with distance beta:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(beta)
        nmi_beta = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 5] = nmi_beta

        print("Clustering with distance maxdual_OT at beta:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(beta_maxdual)
        nmi_beta_maxdual = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 6] = nmi_beta_maxdual

        print("Clustering with distance beta_normalized:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(beta_normalized)
        nmi_beta_normalized = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 7] = nmi_beta_normalized

        print("Clustering with distance normalized_maxdual_OT at beta_normalized:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(beta_normalized_maxdual)
        nmi_beta_normalized_maxdual = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 8] = nmi_beta_normalized_maxdual

        print("Clustering with distance realtotalCost:")
        kmedoids = KMedoids(n_clusters=10, metric='precomputed', method='pam', max_iter=maxiter).fit(realtotalCost)
        nmi_realtotalCost = normalized_mutual_info_score(data_label, kmedoids.labels_)
        nmi_res[cur_ind, 9] = nmi_realtotalCost

        cur_ind += 1
        if verbose:
            print("nmi_L1: {}".format(nmi_L1))
            print("nmi_alpha: {}".format(nmi_alpha))
            print("nmi_alpha_OT: {}".format(nmi_alpha_OT))
            print("nmi_alpha_normalized: {}".format(nmi_alpha_normalized))
            print("nmi_normalized_OT: {}".format(nmi_normalized_OT))
            print("nmi_beta: {}".format(nmi_beta))
            print("nmi_beta_maxdual: {}".format(nmi_beta_maxdual))
            print("nmi_beta_normalized: {}".format(nmi_beta_normalized))
            print("nmi_beta_normalized_maxdual: {}".format(nmi_beta_normalized_maxdual))
            print("nmi_realtotalCost: {}".format(nmi_realtotalCost))

    # save the result as csv file
    argparse = "n_{}_delta_{}_data_{}_noise_st_{}_noise_ed_{}_noise_d_{}".format(n, delta, data_name, noise_st, noise_ed, noise_d)
    np.savetxt("./results/nmi_res_{}.csv".format(argparse), nmi_res, delimiter=",")
