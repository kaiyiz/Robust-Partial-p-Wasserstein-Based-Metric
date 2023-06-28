import numpy as np
from scipy.stats import norm

def load_data(n, dataset_name):
    npzfiles = np.load('./results/{}.npz'.format(dataset_name))
    all_res = npzfiles['all_res']
    data_label = npzfiles['mnist_pick_label']
    data_a = npzfiles['data_a']
    data_b = npzfiles['data_b']
    alpha = all_res[:,:,0]
    alpha_OT = all_res[:,:,1]
    alpha_normalized = all_res[:,:,2]
    alpha_normalized_OT = all_res[:,:,3]
    beta = all_res[:,:,4]
    beta_maxdual = all_res[:,:,5]
    beta_normalized = all_res[:,:,6]
    beta_normalized_maxdual = all_res[:,:,7]
    realtotalCost = all_res[:,:,8]
    L1_metric = all_res[:,:,9]
    return  data_a, data_b, data_label, alpha, alpha_OT, alpha_normalized, alpha_normalized_OT, beta, beta_maxdual, beta_normalized, beta_normalized_maxdual, realtotalCost, L1_metric

def add_niose(data, noise_level=0.1):
    m = data.shape[0]
    n = data.shape[1]
    loc = np.repeat(np.arange(n)[np.newaxis,:], m, axis=0)
    mu = np.random.randint(0, n, m)[:,np.newaxis]
    noise = norm.pdf(loc, mu, n*0.1)
    noise = noise / np.sum(noise, axis=1).reshape(-1,1) * data.sum(axis=1).reshape(-1,1)
    # noise = np.random.rand(data.shape[0], data.shape[1])
    data = (1-noise_level)*data + noise_level * noise
    return data

def rand_pick_mnist(mnist, mnist_labels, n=1000, seed = 1):
    # eps = Contamination proportion
    # n = number of samples
    ############ Creating pure and contaminated mnist dataset ############

    np.random.seed(seed)
    p = np.random.permutation(len(mnist_labels))
    mnist = mnist[p,:,:]
    mnist_labels = mnist_labels[p]
    # all_index = np.arange(len(mnist_labels))
    # index_perm = np.random.permutation(all_index)

    ind_all = np.array([])
    for i in range(10):
        ind = np.nonzero(mnist_labels == i)[0][:int(n/10)]
        ind_all = np.append(ind_all, ind)

    ind_all = ind_all.astype(int)
    mnist_pick, mnist_pick_label = mnist[ind_all, :, :], mnist_labels[ind_all]
    mnist_pick = mnist_pick/255.0
    mnist_pick = mnist_pick.reshape(-1, 784)
    mnist_pick = mnist_pick / mnist_pick.sum(axis=1, keepdims=1)
    mnist_pick[np.nonzero(mnist_pick==0)] = 0.000001
    mnist_pick = mnist_pick / mnist_pick.sum(axis=1, keepdims=1)

    return mnist_pick, mnist_pick_label