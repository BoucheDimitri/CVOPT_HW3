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
        # print(dict_iterates[mu][0])
        dict_f0_iterates[mu] = objective_iterates(dict_iterates[mu][0], Q, p)
    return dict_iterates, dict_f0_iterates


def plot_opti_gap(f0_iterates, cum_nits, ax, label):
    print(cum_nits)
    opti_gap = f0_iterates - np.min(f0_iterates)
    ax.semilogy(cum_nits, opti_gap, label=label, marker="o")


def compare_opti_gaps(dict_iterates, dict_f0_iterates):
    fig, ax = plt.subplots()
    for key in dict_f0_iterates.keys():
        cum_nits = np.cumsum(dict_iterates[key][1])
        plot_opti_gap(dict_f0_iterates[key], cum_nits, ax, "mu = " + str(key))
    ax.legend()
    ax.set_ylabel("$f(v_t) - f^*$")
    ax.set_xlabel("Cumulative number of Newton iterations")
    ax.set_title("Convergence of log barrier")