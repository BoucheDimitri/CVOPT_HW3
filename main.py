import numpy as np
import importlib

import utils
import barrier_method as bm

importlib.reload(utils)
importlib.reload(bm)


d = 10
n = 100
lamb = 10

mean = 0
sigma = 1

X = utils.gaussian_mat(mean, sigma, n, d)

A = np.concatenate((X.T, -X.T))

b = lamb * np.ones((2 * d, ))

y = utils.gaussian_vec(mean, sigma, n)

v0 = np.zeros((n, ))

Q = 0.5 * np.eye(n)

p = - y

t0 = 10
mu = 2

eps = 0.001
#
# delta, lamb_sqr = bm.newton_update(Q, p, A, b, t0, v0)
# eta = bm.bktk_line_search(Q, p, A, b, t0, v0, delta, alpha=0.1, beta=0.5, maxit=1000)
# c = bm.logbarrier_objective(Q, p, A, b, t0, v0 + eta * delta)

# central_path = bm.centering_step(Q, p, A, b, t0, v0, eps=0.001)
#
# barrier_iterates = bm.barr_method(Q, p, A, b, v0, eps, t0, mu)
#
# f0_iterates = utils.objective_iterates(barrier_iterates, Q, p)


mus = [2, 15, 30, 50, 100, 200]
dict_iterates, dict_f0_iterates = utils.compare_mus(mus, Q, p, A, b, v0, eps, t0)

utils.compare_opti_gaps(dict_iterates, dict_f0_iterates)

utils.compare_precisions(dict_iterates, 2 * d, t0)