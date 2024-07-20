import ot
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from skimage.transform import resize

from joblib import Parallel, delayed

import jpype
import jpype.imports
from jpype.types import *
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx128g", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping

from utils import *

def project_simplex(x):
    """Project Simplex

    Projects an arbitrary vector :math:`\mathbf{x}` into the probability simplex, such that,

    .. math:: \tilde{\mathbf{x}}_{i} = \dfrac{\mathbf{x}_{i}}{\sum_{j=1}^{n}\mathbf{x}_{j}}

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Numpy array of shape (n,)

    Returns
    -------
    y : :class:`numpy.ndarray`
        numpy array lying on the probability simplex of shape (n,)
    """
    x[x < 0] = 0
    if np.isclose(sum(x), 0):
        y = np.zeros_like(x)
    else:
        y = x.copy() / sum(x)
    return y

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

def OTP_metric(X=None, Y=None, dist=None, delta=0.1, metric_scaler=1, i=0, sqrt_cost=False):
    # delta : acceptable additive error
    # q_idx : index to get returned values
    nz = len(X)
    alphaa = 4.0*np.max(dist)/delta
    gtSolver = Mapping(nz, list(X), list(Y), dist, delta)
    APinfo = np.array(gtSolver.getAPinfo())

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1)
    APinfo_cleaned = APinfo[clean_mask]

    # APinfo_cleaned[:,4] = APinfo_cleaned[:,4] - 1
    cost_AP = APinfo_cleaned[:,4] * APinfo_cleaned[:,2]
    cumCost = np.cumsum(cost_AP)
    real_total_cost = gtSolver.getTotalCost()
    if real_total_cost == 0:
        cumCost = cumCost * 0.0
    else:
        cumCost = cumCost / (cumCost[-1] / real_total_cost)
    if sqrt_cost:
        cumCost = np.sqrt(cumCost)

    cumCost *= metric_scaler

    cumFlow = np.cumsum((APinfo_cleaned[:,2]).astype(int))
    totalFlow = cumFlow[-1]
    flowProgress = (cumFlow)/(1.0 * totalFlow)

    d_cost = (1 - flowProgress) - cumCost
    d_ind_a = np.nonzero(d_cost<=0)[0][0]-1
    d_ind_b = d_ind_a + 1
    alpha = find_intersection_point(flowProgress[d_ind_a], d_cost[d_ind_a], flowProgress[d_ind_b], d_cost[d_ind_b])
    alpha = 1 - alpha

    # dual weights at alpha, dual_weights_at_alpha_X > 0, dual_weights_at_alpha_X < 0, late in the matching means higher value
    iter_idx = APinfo_cleaned[d_ind_b, :][0]
    dual_weights_at_alpha_X, dual_weights_at_alpha_Y = np.split((np.array(gtSolver.getDual(iter_idx))+np.array(gtSolver.getDual(iter_idx-1)))/(2*alphaa), 2)
    dual_weights_at_alpha_X = dual_weights_at_alpha_X/(1+dual_weights_at_alpha_X.max()) # late in the matching means higher value
    # dual_weights_at_alpha_Y = dual_weights_at_alpha_Y/dual_weights_at_alpha_Y.min() # late in the matching means lower value
    potentials[i,:] = dual_weights_at_alpha_X

def create_digits_image(images, labels, digit=0, grid_size=64, n_digits=15, original_size=28, is_distribution=True):
    batch = images[np.where(labels==digit)[0]]
    images = []
    for i in range(n_digits):
        grid = np.zeros([grid_size, grid_size])
        final_size = np.random.randint(original_size // 2, original_size * 2) 
        grid_cells = np.arange(0, grid_size - final_size)
        img = resize(batch[np.random.randint(0, len(batch))], (final_size, final_size)) 
        p = [10] + [1] * (len(grid_cells) - 2) + [10]
        center_x = np.random.choice(grid_cells, size=1, p=np.array(p) / sum(p))[0] 
        center_y = np.random.choice(grid_cells, size=1, p=np.array(p) / sum(p))[0] 
        grid[center_x:center_x+final_size, center_y:center_y+final_size] = img.copy()
        if is_distribution:
            grid = grid / np.sum(grid)
        images.append(grid.reshape(1, grid_size, grid_size))
    images = np.array(images).reshape(-1, grid_size ** 2)

    return images



def fixed_support_barycenter_OTP_metric(B, M, weights=None, eta=10, numItermax=100, stopThr=1e-9, verbose=False, norm='max', ms = 1, argparse = None, sqrt_cost = False):
    """Fixed Support Wasserstein Barycenter

    We follow the Algorithm 1. of [1], into calculating the Wasserstein barycenter of N measures over a pre-defined
    grid :math:`\mathbf{X}`. These measures, of course, have variable sample weights :math:`\mathbf{b}_{i}`.
    
    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Numpy array of shape (N, d), for N histograms, and d dimensions.
    M : :class:`numpy.ndarray`
        Numpy array of shape (d, d), containing the pairwise distances for the support of B
    weights : :class:`numpy.ndarray`
        Numpy array or None. If None, weights are uniform. Otherwise, weights each measure in the barycenter
    eta : float
        Mirror descent step size
    numItermax : integer
        Maximum number of descent steps
    stopThr : float
        Threshold for stopping mirror descent iterations
    verbose : bool
        If true, display information about each descent step
    norm : str
        Either 'max', 'median' or 'none'. If not 'none', normalizes pairwise distances.

    Returns
    -------
    a : :class:`numpy.ndarray`
        Array of shape (d,) containing the barycenter of the N measures.
    """
    a = ot.unif(B.shape[1])
    a_prev = a.copy()
    weights = ot.unif(B.shape[0]) if weights is None else weights
    if norm == "max":
        _M = M / np.max(M)
    elif norm == "median":
        _M = M / np.median(M)
    else:
        _M = M

    for k in range(numItermax):
        # potentials = np.zeros(B.shape)
        # potentials = []
        time_st = time.time()
        Parallel(n_jobs=-1, prefer="threads")(delayed(OTP_metric)(X=a, Y=B[i], dist=_M, delta=0.001, metric_scaler=ms, i=i, sqrt_cost=sqrt_cost) for i in range(B.shape[0]))
        time_end = time.time()
        print("OTP time: {}".format(time_end-time_st))
        # for i in range(B.shape[0]):
            # _, ret = ot.emd(a, B[i], _M, log=True)
            # potentials.append(ret['u'])
            # Parallel(n_jobs=-1, prefer="threads")(delayed(OTP_metric)(X=a, Y=B[i], dist=_M, delta=0.01, metric_scaler=1, i=0, sqrt_cost=True)for i in B.shape[0])
            # alpha, duo_a, duo_B = OTP_metric(X=a, Y=B[i], dist=_M, delta=0.01, metric_scaler=1, i=0, sqrt_cost=True)
            # potentials.append(duo_a)

        potentials_ = np.nan_to_num(potentials)
        # Calculates the gradient
        f_star = sum(potentials_) / len(potentials_)

        # Mirror Descent
        a = a * np.exp(- eta * f_star)

        # Projection
        a = project_simplex(a)

        # Calculate change in a
        da = sum(np.abs(a - a_prev))
        if da < stopThr: return a
        if verbose: print('[{}, {}] |da|: {}'.format(k, numItermax, da))

        # Update previous a
        a_prev = a.copy()
        # save the image every 10 iterations
        if k % 50 == 0:
            plt.imshow(a.reshape(grid_size, grid_size), cmap='gray')
            plt.axis('off')
            plt.savefig('./results/figs/barycenter_{}_iter{}.png'.format(argparse,k), bbox_inches='tight')
    return a

def fixed_support_barycenter_Wasserstein(B, M, weights=None, eta=10, numItermax=100, stopThr=1e-9, verbose=False, norm='max'):
    """Fixed Support Wasserstein Barycenter

    We follow the Algorithm 1. of [1], into calculating the Wasserstein barycenter of N measures over a pre-defined
    grid :math:`\mathbf{X}`. These measures, of course, have variable sample weights :math:`\mathbf{b}_{i}`.
    
    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Numpy array of shape (N, d), for N histograms, and d dimensions.
    M : :class:`numpy.ndarray`
        Numpy array of shape (d, d), containing the pairwise distances for the support of B
    weights : :class:`numpy.ndarray`
        Numpy array or None. If None, weights are uniform. Otherwise, weights each measure in the barycenter
    eta : float
        Mirror descent step size
    numItermax : integer
        Maximum number of descent steps
    stopThr : float
        Threshold for stopping mirror descent iterations
    verbose : bool
        If true, display information about each descent step
    norm : str
        Either 'max', 'median' or 'none'. If not 'none', normalizes pairwise distances.

    Returns
    -------
    a : :class:`numpy.ndarray`
        Array of shape (d,) containing the barycenter of the N measures.
    """
    a = ot.unif(B.shape[1])
    a_prev = a.copy()
    weights = ot.unif(B.shape[0]) if weights is None else weights
    if norm == "max":
        _M = M / np.max(M)
    elif norm == "median":
        _M = M / np.median(M)
    else:
        _M = M

    for k in range(numItermax):
        potentials = []
        time_st = time.time()
        for i in range(B.shape[0]):
            _, ret = ot.emd(a, B[i], _M, log=True)
            potentials.append(ret['u'])
        time_ed = time.time()
        print("emd time: {}".format(time_ed-time_st))
        
        # Calculates the gradient
        f_star = sum(potentials) / len(potentials)

        # Mirror Descent
        a = a * np.exp(- eta * f_star)

        # Projection
        a = project_simplex(a)

        # Calculate change in a
        da = sum(np.abs(a - a_prev))
        if da < stopThr: return a
        if verbose: print('[{}, {}] |da|: {}'.format(k, numItermax, da))

        # Update previous a
        a_prev = a.copy()
    return a

parser = argparse.ArgumentParser()
parser.add_argument('--digit', type=int, default=3)
parser.add_argument('--grid_size', type=int, default=64)
parser.add_argument('--n_digits', type=int, default=15)
parser.add_argument('--original_size', type=int, default=20)
parser.add_argument('--is_distribution', type=bool, default=True)
parser.add_argument('--metric', type=str, default='sqeuclidean')
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--ms', type=float, default=10)
parser.add_argument('--sqrt_cost', type=bool, default=True)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--max_iter', type=int, default=2000)

args = parser.parse_args()

digit = args.digit
grid_size = args.grid_size
n_digits = args.n_digits
original_size = args.original_size
is_distribution = args.is_distribution
metric = args.metric
noise = args.noise
ms = args.ms
sqrt_cost = args.sqrt_cost
eta = args.eta
max_iter = args.max_iter

argparse = "digit{}_grid_size_{}_n_digits_{}_metric_{}_noise_{}_ms_{}_sqrt_cost_{}".format(digit, grid_size, n_digits, metric, noise, ms, sqrt_cost)
print(argparse)

images, labels = load_data('mnist')
images = images.astype(float).reshape(-1, 28, 28) / 255

np.random.seed(0)
B = create_digits_image(images, labels,
                        digit=digit,
                        grid_size=grid_size,
                        n_digits=n_digits,
                        original_size=original_size,
                        is_distribution=is_distribution,
                        )
B = add_noise(B, "rand1pxl", noise)
potentials = np.zeros(B.shape)

M = get_ground_dist(B[0,:], B[0,:], 'fixed_bins_2d', metric=metric)

a_OTP_metric = fixed_support_barycenter_OTP_metric(B, M, eta=eta, numItermax=max_iter, stopThr=1e-9, verbose=True, ms=ms, argparse=argparse, sqrt_cost=sqrt_cost)
# a_OTP_metric = B
a_wasserstein = fixed_support_barycenter_Wasserstein(B, M, verbose=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(sum(B).reshape(grid_size, grid_size), cmap='gray')
axes[0].axis('off')
axes[1].imshow(a_wasserstein.reshape(grid_size, grid_size), cmap='gray')
axes[1].axis('off')
axes[2].imshow(a_OTP_metric.reshape(grid_size, grid_size), cmap='gray')
axes[2].axis('off')

# save
plt.savefig('./results/figs/barycenter_{}.png'.format(argparse), bbox_inches='tight')