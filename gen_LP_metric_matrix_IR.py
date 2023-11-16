import argparse
import numpy as np
import networkx as nx
import time

from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

from utils import *


"""
Relevant parts of the code have been adapted from :
https://github.com/debarghya-mukherjee/Robust-Optimal-Transport/blob/main/ROBOT_mnist_outlier_detection.py
"""

def levy_prokhorov_metric(X, Y, dist, all_res=None, i=0, j=0, time_start=None):
    # Create the directed graph
    delta = 1e-6
    G = nx.DiGraph()
    for ind_x, mass in enumerate(X):
        G.add_edge('source', 'A{}'.format(ind_x), capacity=mass)
    for ind_y, mass in enumerate(Y):
        G.add_edge('B{}'.format(ind_y), 'sink', capacity=mass)

    # Create a list of tuples (cost, edges), where each tuple contains a the cost and corresponding edges with that cost, sorted by cost
    costs_edges = [(dist[ind_i,ind_j], ind_i, ind_j)for ind_i in range(dist.shape[0]) for ind_j in range(dist.shape[1])]
    costs_edges = sorted(costs_edges, key=lambda x: x[0])
    # Do binary search to find the largest cost such that the maximum flow plus that cost is less than 1
    low = 0
    high = len(costs_edges) - 1
    last_mid = 0
    while low < high:
        mid = (low + high) // 2
        if mid == low:
            break
        elif mid > last_mid:
            # add edges from low to mid
            for ind in range(low, mid):
                cost, ind_x, ind_j = costs_edges[ind]
                G.add_edge('A{}'.format(ind_x), 'B{}'.format(ind_j), cost=cost)
        else:
            # remove edges from mid to high
            for ind in range(mid, high):
                cost, ind_x, ind_j = costs_edges[ind]
                G.remove_edge('A{}'.format(ind_x), 'B{}'.format(ind_j))
        # Compute the maximum flow
        flow_value, flow_dict = nx.maximum_flow(G, 'source', 'sink')
        if np.abs(flow_value + cost - 1) < delta:
            break
        # Check if the maximum flow plus δ is less than 1
        if flow_value + cost < 1:
            low = mid
        else:
            high = mid
        last_mid = mid

    if j == 0:
        size = all_res.shape
        time_cur = time.time()
        amount_of_work_done = i*size[1] + 1
        time_spent = time_cur - time_start
        total_time = time_spent/(amount_of_work_done/(size[0]*size[1]))
        print("estimate to finish the job in {}s".format(total_time-time_spent))

    cost, ind_x, ind_j = costs_edges[(low + high) // 2]
    print("i: {}, j: {}, cost: {}".format(i, j, cost))
    all_res[i,j] = cost

if __name__ == "__main__":
    # LOAD Data
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--shift_pixel', type=int, default=1)
    parser.add_argument('--noise_type', type=str, default='whiteout')
    parser.add_argument('--k', type=int, default=1000)

    args = parser.parse_args()
    print(args)

    n = int(args.n)
    delta = args.delta
    data_name = args.data_name
    noise = args.noise
    shift_pixel = args.shift_pixel
    noise_type = args.noise_type
    k = args.k
    argparse = "n_{}_delta_{}_data_{}_noise_{}_sp_{}_nt_{}".format(n, delta, data_name, noise, shift_pixel, noise_type)

    data, data_labels = load_data(data_name)
    data_size = int(data.shape[0]/k)
    all_res = np.zeros((n,data_size))
    print("data size: {}".format(all_res.shape))

    if data_name == "mnist":
        data_pick_a, data_pick_label_a = rand_pick_mnist(data, data_labels, n, 0)
        data_pick_b, data_pick_label_b = rand_pick_mnist(data, data_labels, data_size, 1)
        data_pick_a_noise = add_noise(data_pick_a, noise_type = noise_type, noise_level=noise)
        data_pick_a_noise = shift_image(data_pick_a_noise, shift_pixel)
        dist = get_ground_dist(data_pick_a_noise[0,:], data_pick_b[1,:], 'fixed_bins_2d')
        start_time = time.time()
        Parallel(n_jobs=-1, prefer="threads")(delayed(levy_prokhorov_metric)(extract_mnist_mass(data_pick_a, i), extract_mnist_mass(data_pick_b, j), get_ground_dist(extract_mnist_loc(data_pick_a, i), extract_mnist_loc(data_pick_b, j), 'mnist_extract', 'minkowski'), all_res, i, j, start_time) for i in range(n) for j in range(n))
        end_time = time.time()
    elif data_name == "cifar10":
        start_time = time.time()
        data_pick_a, data_pick_label_a = rand_pick_cifar10(data, data_labels, n, 0)
        data_pick_b, data_pick_label_b = rand_pick_cifar10(data, data_labels, data_size, 1)
        data_pick_a_noise = add_noise_3d_matching(data_pick_a, noise_type = noise_type, noise_level=noise)
        data_pick_a_noise = shift_image_color(data_pick_a_noise, shift_pixel)
        # data_pick_a_noise = shift_image_3d(data_pick_a_noise, shift_pixel)
        geo_dist = get_ground_dist(data_pick_a_noise[0,:], data_pick_b[1,:], 'fixed_bins_2d')
        m = data_pick_a.shape[1]
        a = np.ones(m)/m
        b = np.ones(m)/m
        diam_color = 3
        lamda = 0.5
        Parallel(n_jobs=-1, prefer="threads")(delayed(levy_prokhorov_metric)(a, b, np.sqrt(get_ground_dist(data_pick_a_noise[i,:], data_pick_b[j,:], transport_type="high_dim", metric='sqeuclidean', diam=diam_color) + lamda*geo_dist), all_res, i, j, start_time) for i in range(n) for j in range(n))
        end_time = time.time()
    else:
        raise ValueError("data not found")

    print("finish all job in {}s".format(end_time-start_time))
    np.savez('./results/LP_metric_{}'.format(argparse), all_res=all_res, data_a=data_pick_a_noise, data_b=data_pick_b, data_pick_label_a=data_pick_label_a, data_pick_label_b=data_pick_label_b)