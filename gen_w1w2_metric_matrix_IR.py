import argparse
import numpy as np
import time
import ot

from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

from utils import *


"""
Relevant parts of the code have been adapted from :
https://github.com/debarghya-mukherjee/Robust-Optimal-Transport/blob/main/ROBOT_mnist_outlier_detection.py
"""

def w1w2metric(X=None, Y=None, dist=None, all_res=None, i=0, j=0, time_start=None):
    # dist : ground distance, eucledian distance
    w1 = ot.emd2(X, Y, dist)
    w2 =np.sqrt(ot.emd2(X, Y, dist**2))
    # realtotalCost = gtSolver.getTotalCost()

    if j == 0:
        size = all_res.shape
        time_cur = time.time()
        amount_of_work_done = i*size[1] + 1
        time_spent = time_cur - time_start
        total_time = time_spent/(amount_of_work_done/(size[0]*size[1]))
        print("estimate to finish the job in {}s".format(total_time-time_spent))

    all_res[i,j,:] = np.array([w1, w2])

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

if __name__ == "__main__":
    # LOAD Data
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--shift_pixel', type=int, default=0)
    parser.add_argument('--noise_type', type=str, default='rand1pxl')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--range_noise', type=int, default=0)

    args = parser.parse_args()
    print(args)

    n = int(args.n)
    data_name = args.data_name
    noise = args.noise
    shift_pixel = args.shift_pixel
    noise_type = args.noise_type
    k = args.k
    range_noise = args.range_noise
    argparse = "n_{}_data_{}_noise_{}_sp_{}_nt_{}_rangenoise_{}_w1w2".format(n, data_name, noise, shift_pixel, noise_type,range_noise)

    data, data_labels = load_data(data_name)
    data_size = int(data.shape[0]/k)
    all_res = np.zeros((n,data_size,2))
    print("data size: {}".format(all_res.shape))

    if data_name == "mnist":
        data_pick_a, data_pick_label_a = rand_pick_mnist(data, data_labels, n, 0)
        data_pick_b, data_pick_label_b = rand_pick_mnist(data, data_labels, data_size, 1)
        data_pick_b = add_noise(data_pick_b, noise_type = noise_type, noise_level=noise, range_noise=range_noise)
        data_pick_b = shift_image(data_pick_b, shift_pixel)
        # data_pick_a = add_noise(data_pick_a, noise_type = noise_type, noise_level=noise)
        # data_pick_a = shift_image(data_pick_a, shift_pixel)
        dist = get_ground_dist(data_pick_a[0,:], data_pick_b[1,:], 'fixed_bins_2d', metric='euclidean')
        start_time = time.time()
        Parallel(n_jobs=-1, prefer="threads")(delayed(w1w2metric)(data_pick_a[i,:], data_pick_b[j,:], dist, all_res, i, j, start_time) for i in range(n) for j in range(data_size))
        end_time = time.time()
    elif data_name == "cifar10":
        start_time = time.time()
        data_pick_a, data_pick_label_a = rand_pick_cifar10(data, data_labels, n, 0)
        data_pick_b, data_pick_label_b = rand_pick_cifar10(data, data_labels, data_size, 1)
        data_pick_b = add_noise_3d_matching(data_pick_b, noise_type = noise_type, noise_level=noise)
        data_pick_b = shift_image_color(data_pick_b, shift_pixel)
        # data_pick_a = shift_image_3d(data_pick_a, shift_pixel)
        geo_dist = get_ground_dist(data_pick_a[0,:], data_pick_b[1,:], 'fixed_bins_2d', metric='euclidean')
        m = data_pick_a.shape[1]
        a = np.ones(m)/m
        b = np.ones(m)/m
        diam_color = 3
        lamda = 0.5
        Parallel(n_jobs=-1, prefer="threads")(delayed(w1w2metric)(a, b, get_ground_dist(data_pick_a[i,:], data_pick_b[j,:], transport_type="high_dim", metric='euclidean', diam=diam_color) + lamda*geo_dist, all_res, i, j, start_time) for i in range(n) for j in range(data_size))
        end_time = time.time()
    else:
        raise ValueError("data not found")
    

    print("finish all job in {}s".format(end_time-start_time))
    np.savez('./results/OTP_lp_metric_{}'.format(argparse), all_res=all_res, data_a=data_pick_a, data_b=data_pick_b, data_pick_label_a=data_pick_label_a, data_pick_label_b=data_pick_label_b)