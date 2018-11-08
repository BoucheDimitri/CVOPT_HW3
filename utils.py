import numpy as np
import matplotlib.pyplot as plt

import barrier_method as bm


def gaussian_vec(mu, sigma, n):
    return np.random.normal(mu, sigma, n)


def gaussian_mat(mu, sigma, n, d):
    return np.random.normal(mu, sigma, (n, d))


def objective_iterates(iterates, Q, p):
    f0_iterates = np.zeros((len(iterates), ))
    i = 0
    for v in iterates:
        f0_iterates[i] = bm.objective(Q, p, v)
        i += 1
    return f0_iterates


def compare_mus(mus, Q, p, A, b, v0, eps, t0, maxit=1000):
    dict_iterates = {}
    dict_f0_iterates = {}
    for mu in mus:
        print("Mu is equal to " + str(mu))
        dict_iterates[mu] = bm.barr_method(Q, p, A, b, v0, eps, t0, mu, maxit)
        dict_f0_iterates[mu] = objective_iterates(dict_iterates[mu], Q, p)
    return dict_iterates, dict_f0_iterates


def plot_opti_gap(f0_iterates, ax, label):
    opti_gap = f0_iterates - np.min(f0_iterates)
    ax.semilogy(opti_gap, label=label)


def compare_opti_gaps(dict_f0_iterates):
    fig, ax = plt.subplots()
    for key in dict_f0_iterates.keys():
        plot_opti_gap(dict_f0_iterates[key], ax, "mu = " + str(key))
    ax.legend()
    ax.set_ylabel("$f(v_t) - f^*$")
    ax.set_xlabel("t")
    ax.set_title("Convergence of log barrier")