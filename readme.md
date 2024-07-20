# A New Robust Partial $p$-Wasserstein-Based Metric for Comparing Distributions

This repository contains the implementation of our paper ([OpenReview](https://openreview.net/forum?id=opieUcKjPa) | [ICML](https://icml.cc/virtual/2024/poster/33067) | [arxiv](https://arxiv.org/abs/2405.03664)) and the experiment code.

The $2$-Wasserstein distance is sensitive to minor geometric differences between distributions, making it a very powerful dissimilarity metric. However, due to this sensitivity, a small outlier mass can also cause a significant increase in the $2$-Wasserstein distance between two similar distributions. Similarly, sampling discrepancy can cause the empirical $2$-Wasserstein distance on $n$ samples in $\mathbb{R}^2$ to converge to the true distance at a rate of $n^{-1/4}$, which is significantly slower than the rate of $n^{-1/2}$ for $1$-Wasserstein distance.
We introduce a new family of distances parameterized by $k \ge 0$, called $k$-RPW that is based on computing the partial $2$-Wasserstein distance. We show that (1) $k$-RPW satisfies the metric properties, (2) $k$-RPW is robust to small outlier mass while retaining the sensitivity of $2$-Wasserstein distance to minor geometric differences, and (3) when $k$ is a constant, $k$-RPW distance between empirical distributions on $n$ samples in $\mathbb{R}^2$ converges to the true distance at a rate of $n^{-1/3}$, which is faster than the convergence rate of $n^{-1/4}$ for the $2$-Wasserstein distance.
Using the partial $p$-Wasserstein distance, we extend our distance to any $p \in [1,\infty]$.
By setting parameters $k$ or $p$ appropriately, we can reduce our distance to the total variation, $p$-Wasserstein, and the L\'evy-Prokhorov distances. Experiments show that our distance function achieves higher accuracy in comparison to the $1$-Wasserstein, $2$-Wasserstein, and TV distances for image retrieval tasks on noisy real-world data sets.

## Citation
If you find this work helpful, please consider citing our paper:

```
@inproceedings{raghvendranew,
  title={A New Robust Partial p-Wasserstein-Based Metric for Comparing Distributions},
  author={Raghvendra, Sharath and Shirzadian, Pouyan and Zhang, Kaiyi},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

## Environments

This code is written in Python 3 and relies on the following libraries:
```
numpy
scipy
pot
tensorflow
jpype
joblib
```

## To repeat our experiment, simply run the following command:
### Image Retrieval
MNIST with noise 

```sh IR_noise_mnist_final.sh```

MNIST with shifting

```sh IR_shift_mnist_final.sh```

MNIST with noise and shifting

```sh IR_shift_noise_mnist_final.sh```

CIFAR-10 with noise

```sh IR_noise_cifar10_final.sh```

CIFAR-10 with shifting

```sh IR_shift_cifar10_final.sh```

CIFAR-10 with noise and shifting

```sh IR_shift_noise_cifar10_final.sh```

### Rate of Convergence
Two Points Distribution

```python exp_sample_converge_rate_p_Wasserstein_2p.py```

Grid Distribution

```python exp_sample_converge_rate_fix_dim_2_Wasserstein.py```