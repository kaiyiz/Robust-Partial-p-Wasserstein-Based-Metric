import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import cdist

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

def add_niose(data, noise_type = "uniform", noise_level=0.1):
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
            rv = multivariate_normal(mu, [[n*0.05, 0], [0, n*0.05]])
            noise[i,:] = rv.pdf(pos).reshape(-1)
        noise = noise / np.sum(noise, axis=1).reshape(-1,1) * data.sum(axis=1).reshape(-1,1)
    else:
        raise ValueError("noise type not found")
    data = (1 - noise_level) * data + noise_level * noise
    return data

def add_geometric_noise(data, noise_level=0.1):
    # each row is a flattened image, I want to add a 2d normal noise to each image at a random locations
    # noise_level is the percentage of noise added to the image
    # mnist is 28x28 images
    # create a 28x28 grid, each cell is a pixel
    # for each image, add a 2d normal noise to a random location
    m = data.shape[0]
    n = data.shape[1]
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
        rv = multivariate_normal(mu, [[n*0.05, 0], [0, n*0.05]])
        noise[i,:] = rv.pdf(pos).reshape(-1)
    noise = noise / np.sum(noise, axis=1).reshape(-1,1) * data.sum(axis=1).reshape(-1,1)
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

def get_ground_dist(a, b, transport_type="geo"):
    m = a.shape[0]
    if transport_type == "geo":
        dist = computeDistMatrixGrid2d(int(np.sqrt(m)))
        dist = dist / np.max(dist) 
    elif transport_type == "hist":
        dist = cdist(a, b, metric='minkowski', p=1)
        dist = dist
    else:
        raise ValueError("dist_type not found")
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