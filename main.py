import numpy as np
import importlib

import utils
import barrier_method as bm

importlib.reload(utils)
importlib.reload(bm)


d = 10
n = 100
lamb = 2

mean = 0
sigma = 1

X = utils.gaussian_mat(mean, sigma, n, d)

A = X.T

b = lamb * np.ones((d, ))

y = utils.gaussian_vec(mean, sigma, n)

v0 = np.zeros((n, ))

Q = 0.5 * np.eye(n)

p = - y

t0 = 5
mu = 2

eps = 0.001

central_path = bm.centering_step(Q, p, A, b, t, v0, eps=0.001)

barrier_test = bm.barr_method(Q, p, A, b, v0, eps, t0, mu)
