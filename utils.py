import numpy as np


def gaussian_vec(mu, sigma, n):
    return np.random.normal(mu, sigma, n)


def gaussian_mat(mu, sigma, n, d):
    return np.random.normal(mu, sigma, (n, d))