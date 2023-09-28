import argparse
import numpy as np
import networkx as nx
import time

from joblib import Parallel, delayed

from utils import load_data, add_noise, get_ground_dist, rand_pick_mnist, rand_pick_cifar10, add_noise_3d_matching, shift_image, extract_mnist_mass, extract_mnist_loc


"""
Relevant parts of the code have been adapted from :
https://github.com/debarghya-mukherjee/Robust-Optimal-Transport/blob/main/ROBOT_mnist_outlier_detection.py
"""

def levy_prokhorov_metric(X, Y, dist, all_res=None, i=0, j=0, time_start=None):
    # Create the directed graph
    G = nx.DiGraph()
    for ind_x, mass in enumerate(X):
        G.add_edge('source', 'A{}'.format(ind_x), capacity=mass)
    for ind_y, mass in enumerate(Y):
        G.add_edge('B{}'.format(ind_y), 'sink', capacity=mass)

    # Create a list of tuples (cost, edges), where each tuple contains a unique cost and the edges with that cost
    costs_edges = [(cost, list(zip(*np.where(dist == cost)))) for cost in np.unique(dist)]

    # Start with the second largest cost
    for cost, edges in costs_edges:
        # Create the δ-disc graph
        for ind_x, ind_j in edges:
            G.add_edge('A{}'.format(ind_x), 'B{}'.format(ind_j), cost=cost)

        # Compute the maximum flow
        flow_value, flow_dict = nx.maximum_flow(G, 'source', 'sink')

        # Check if the maximum flow plus δ is less than 1
        if flow_value + cost >= 1:
            break

    if j == 0:
        size = all_res.shape
        time_cur = time.time()
        amount_of_work_done = i*size[0] + j + 1
        time_spent = time_cur - time_start
        total_time = time_spent/(amount_of_work_done/(size[0]*size[1]))
        print("estimate time finish computing Lévy_Prokhorov distance in {}s".format(total_time-time_spent))

    all_res[i,j] = cost

if __name__ == "__main__":
    # LOAD Data
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--shift_pixel', type=int, default=0)
    parser.add_argument('--noise_type', type=str, default='geo_normal')

    args = parser.parse_args()
    print(args)

    n = int(args.n)
    data_name = args.data_name
    noise = args.noise
    shift_pixel = args.shift_pixel
    noise_type = args.noise_type
    argparse = "n_{}_data_{}_noise_{}_sp_{}_nt_{}".format(n, data_name, noise, shift_pixel, noise_type)

    data, data_labels = load_data(data_name)
    all_res = np.zeros((n,n))

    if data_name == "mnist":
        data_pick_a, data_pick_label = rand_pick_mnist(data, data_labels, n, 0)
        data_pick_b, data_pick_label = rand_pick_mnist(data, data_labels, n, 1)
        data_pick_b_noise = add_noise(data_pick_b, noise_type = noise_type, noise_level=noise)
        data_pick_b_noise = shift_image(data_pick_b_noise, shift_pixel)
        start_time = time.time()
        Parallel(n_jobs=1, prefer="threads")(delayed(levy_prokhorov_metric)(extract_mnist_mass(data_pick_a, i), extract_mnist_mass(data_pick_b_noise, j), get_ground_dist(extract_mnist_loc(data_pick_a, i), extract_mnist_loc(data_pick_b_noise, j), 'mnist_extract', 'minkowski'), all_res, i, j, start_time) for i in range(n) for j in range(n))
        end_time = time.time()
    elif data_name == "cifar10":
        start_time = time.time()
        data_pick_a, data_pick_label = rand_pick_cifar10(data, data_labels, n, 0)
        data_pick_b, data_pick_label = rand_pick_cifar10(data, data_labels, n, 1)
        data_pick_b_noise = add_noise_3d_matching(data_pick_b, noise_type = noise_type, noise_level=noise)
        geo_dist = get_ground_dist(data_pick_a[0,:], data_pick_b_noise[1,:], 'fixed_bins_2d')
        m = data_pick_a.shape[1]
        a = np.ones(m)/m
        b = np.ones(m)/m
        diam_color = 3
        lamda = 0.5
        Parallel(n_jobs=-1, prefer="threads")(delayed(levy_prokhorov_metric)(a, b, (1-lamda)*get_ground_dist(data_pick_a[i,:], data_pick_b_noise[j,:], transport_type="high_dim", metric='sqeuclidean', diam=diam_color) + lamda*geo_dist, all_res, i, j, start_time) for i in range(n) for j in range(n))
        end_time = time.time()
    else:
        raise ValueError("data not found")

    print("finish all job in {}s".format(end_time-start_time))
    np.savez('./results/LP_metric_{}'.format(argparse), all_res=all_res, data_a=data_pick_a, data_b=data_pick_b_noise, mnist_pick_label=data_pick_label)