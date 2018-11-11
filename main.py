import numpy as np
import importlib

import utils
import barrier_method as bm

importlib.reload(utils)
importlib.reload(bm)


# Dimensions of the problem
d = 50
n = 500

# Lambda
lamb = 10

# Generate Gaussian matrix and Gaussian vector y
mean = 0
sigma = 1
X = utils.gaussian_mat(mean, sigma, n, d)
y = utils.gaussian_vec(mean, sigma, n)

# Pose the problem as QP
A = np.concatenate((X.T, -X.T))
b = lamb * np.ones((2 * d, ))
Q = 0.5 * np.eye(n)
p = - y

# Feasible starting point
v0 = np.zeros((n, ))

# Initial value of t
t0 = 10

# Value of mu for tests
mu = 2

# Precision
eps = 0.001

# Test central path function
central_path = bm.centering_step(Q, p, A, b, t0, v0)

# Test of barrier method
barrier_iterates = bm.barr_method(Q, p, A, b, v0, eps, t0, mu)

# Test different values for mu
mus = [2, 15, 30, 50, 100, 200, 500]
dict_iterates, dict_f0_iterates = utils.compare_mus(mus, Q, p, A, b, v0, eps, t0)

# Plot surrogate of optimality gap for the different mus
utils.compare_opti_gaps(dict_iterates, dict_f0_iterates)
