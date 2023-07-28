import numpy as np
import os
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import cdist
import tensorflow as tf

def load_computed_matrix(n, dataset_name):
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

def add_noise(data, noise_type = "uniform", noise_level=0.1):
    m = data.shape[0]
    n = data.shape[1]
    if noise_type == "uniform":
        noise = np.random.rand(m, n)
    elif noise_type == "normal":
        loc = np.repeat(np.arange(n)[np.newaxis,:], m, axis=0)
        mu = np.random.randint(0, n, m)[:,np.newaxis]
        noise = norm.pdf(loc, mu, n*0.1)
        noise = noise / np.sum(noise, axis=1).reshape(-1,1) * data.sum(axis=1).reshape(-1,1)
    elif noise_type == "geo_normal":
        nn = int(np.sqrt(n))
        mu_loc = [(nn/8, nn/8), (nn/8, 3*nn/8), (nn/8, 5*nn/8), (nn/8, 7*nn/8), (3*nn/8, nn/8), (3*nn/8, 7*nn/8), (5*nn/8, nn/8), (5*nn/8, 7*nn/8), (7*nn/8, nn/8), (7*nn/8, 3*nn/8), (7*nn/8, 5*nn/8), (7*nn/8, 7*nn/8)]
        noise = np.zeros((m, n))
        for i in range(m):
            mu = np.random.choice(len(mu_loc))
            mu = mu_loc[mu]
            x = np.arange(nn)
            y = np.arange(nn)
            X, Y = np.meshgrid(x, y)
            pos = np.dstack((X, Y))
            rv = multivariate_normal(mu, [[nn*0.1, 0], [0, nn*0.1]])
            noise[i,:] = rv.pdf(pos).reshape(-1)
        noise = noise / np.sum(noise, axis=1).reshape(-1,1) * data.sum(axis=1).reshape(-1,1)
    else:
        raise ValueError("noise type not found")
    data = (1 - noise_level) * data + noise_level * noise
    return data

def add_noise_3d_matching(data, noise_type = "uniform", noise_level=0.1):
    # add noise to 3d data
    m = data.shape[0]
    n = data.shape[1] # number of pixels/voxels
    nn = data.shape[2]
    # pick a number of pixels/voxels to add noise based on noise_level
    noise_ind = np.random.choice(n, int(n*noise_level), replace=False)
    if noise_type == "uniform":
        noise = np.random.rand(m, len(noise_ind), nn)
    elif noise_type == "normal3d":
        mu_positions = np.array([(x, y, z) for x in [1/4, 1/2, 3/4] for y in [1/4, 1/2, 3/4] for z in [1/4, 1/2, 3/4]])
        noise = np.zeros((m, len(noise_ind), nn))
        for i in range(m):
            # Randomly select one of these positions as the mean
            np.random.seed(i) 
            mu = mu_positions[np.random.choice(mu_positions.shape[0])]
            # Define the covariance matrix
            cov = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
            # Generate data
            pts = np.random.multivariate_normal(mu, cov, len(noise_ind))
            noise[i,:,:] = pts
    else:
        raise ValueError("noise type not found")
    data[:,noise_ind,:] = noise
    return data

def shift_image(data, shift=1):
    # shift image by shift pixels
    m = data.shape[0]
    n = data.shape[1]
    nn = int(np.sqrt(n))
    data = data.reshape(m, nn, nn)
    data_shift = np.zeros((m, nn, nn))
    for i in range(m):
        data_shift[i,:,:] = np.roll(data[i,:,:], shift, axis=0)
    data_shift = data_shift.reshape(m, n)
    return data_shift

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

def get_ground_dist(a, b, transport_type="geo_transport", metric='euclidean'):
    m = a.shape[0]
    if len(a.shape) == 1:
        d = 1
    else:
        d = a.shape[1]
    if transport_type == "geo_transport":
        dist = computeDistMatrixGrid2d(int(np.sqrt(m)), metric)
        dist = dist / np.max(dist) 
    elif transport_type == "color_matching":
        dist = cdist(a, b, metric)
        one = np.ones((1, d))
        zero = np.zeros((1, d))
        dist_max = cdist(one, zero, metric)
        dist = dist / dist_max[0][0]
    elif transport_type == "mnist_extract":
        dist = cdist(a, b, metric, p=1)
        dist = dist / (np.sqrt(2)*28)
    else:
        raise ValueError("transport type not found")
    return dist

def computeDistMatrixGrid2d(n,metric='euclidean'):
    A = np.zeros((n**2,2))
    iter = 0
    for i in range(n):
        for j in range(n):
            A[iter,0] = i
            A[iter,1] = j
            iter += 1
    dist = cdist(A, A, metric)
    return dist

def load_data(data_name):
    if data_name == 'mnist':
        if os.path.exists('./data/mnist.npy'):
            data = np.load('./data/mnist.npy') # 60k x 28 x 28
            data_labels = np.load('./data/mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8
        else:
            (data, data_labels), (_, _) = tf.keras.datasets.mnist.load_data()
            np.save('mnist.npy', data)
            np.save('mnist_labels.npy', data_labels)
    elif data_name == 'cifar10':
        if os.path.exists('./data/cifar10.npy'):
            data = np.load('./data/cifar10.npy')
            data_labels = np.load('./data/cifar10_labels.npy').ravel().astype(int)
        else:
            (data, data_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
            np.save('./data/cifar10.npy', data)
            np.save('./data/cifar10_labels.npy', data_labels)
    else:
        raise ValueError("data_name not found")

    return data, data_labels

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

def extract_mnist_mass(data_pick, ind, n=28):
    threshold = 1/(n*n*n)
    data_pick = data_pick[ind, :]
    data_pick_mass = data_pick[np.where(data_pick>threshold)]
    data_pick_mass = data_pick_mass/np.sum(data_pick_mass)

    return data_pick_mass

def extract_mnist_loc(data_pick, ind, n=28):
    threshold = 1/(n*n*n)
    data_pick = data_pick[ind, :]
    data_pick_sq = data_pick.reshape(n, n)
    data_pick_sq_mass_ind = np.argwhere(data_pick_sq>threshold)

    return data_pick_sq_mass_ind

def rand_pick_cifar10(data, data_labels, n=200, seed = 0):
    np.random.seed(seed)
    p = np.random.permutation(len(data_labels))
    data = data[p,:]
    data_labels = data_labels[p]
    n_unique_labels = len(np.unique(data_labels))
    nn = data.shape[1]
    mm = data.shape[2]

    ind_all = np.array([])
    for i in range(n_unique_labels):
        ind = np.nonzero(data_labels == i)[0][:int(n/10)]
        ind_all = np.append(ind_all, ind)

    ind_all = ind_all.astype(int)
    data_pick, data_pick_label = data[ind_all, :], data_labels[ind_all]
    data_pick = data_pick/255.0
    data_pick = data_pick.reshape(-1, nn*mm, 3)

    return data_pick, data_pick_label

def e_dist(A, B):
    A_n = (A**2).sum(axis=1).reshape(-1,1)
    B_n = (B**2).sum(axis=1).reshape(1,-1)
    inner = np.matmul(A, B.T)
    return A_n - 2*inner + B_n

def rand_pick_mnist_09(mnist, mnist_labels, seed=1):
    np.random.seed(seed)
    all_index = np.arange(len(mnist_labels))
    rand_index = np.random.permutation(all_index)
    mnist, mnist_labels = mnist[rand_index, :, :], mnist_labels[rand_index]

    mnist_pick_ind = []
    for i in range(10):
        cur_index = 0
        while True:
            if mnist_labels[cur_index] == i:
                mnist_pick_ind.append(cur_index)
                break
            else:
                cur_index += 1

    mnist_pick, mnist_pick_label = mnist[mnist_pick_ind, :, :], mnist_labels[mnist_pick_ind]
    mnist_pick = mnist_pick/255.0
    mnist_pick = mnist_pick.reshape(-1, 784)
    mnist_pick = mnist_pick / mnist_pick.sum(axis=1, keepdims=1)
    mnist_pick[np.nonzero(mnist_pick==0)] = 0.000001
    mnist_pick = mnist_pick / mnist_pick.sum(axis=1, keepdims=1)
    
    return mnist_pick, mnist_pick_label