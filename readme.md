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