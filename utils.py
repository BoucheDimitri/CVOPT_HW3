import numpy as np
import matplotlib.pyplot as plt

import barrier_method as bm


def gaussian_vec(mu, sigma, n):
    """
    Generate vector of Gaussians
    """
    return np.random.normal(mu, sigma, n)


def gaussian_mat(mu, sigma, n, d):
    """
    Generate matrix of Gaussians
    """
    return np.random.normal(mu, sigma, (n, d))


def objective_iterates(iterates, Q, p):
    """
    From a list of iterates compute the value of the objective at each iteration
    """
    f0_iterates = np.zeros((len(iterates), ))
    i = 0
    for v in iterates:
        f0_iterates[i] = bm.objective(Q, p, v)
        i += 1
    return f0_iterates


def compare_mus(mus, Q, p, A, b, v0, eps, t0, maxit=1000):
    """
    Run barrier method for several mus and store results in dictionnaries (iterates and objective value at iterates)
    """
    dict_iterates = {}
    dict_f0_iterates = {}
    for mu in mus:
        print("Mu is equal to " + str(mu))
        dict_iterates[mu] = bm.barr_method(Q, p, A, b, v0, eps, t0, mu, maxit)
        dict_f0_iterates[mu] = objective_iterates(dict_iterates[mu][0], Q, p)
    return dict_iterates, dict_f0_iterates


def plot_opti_gap(f0_iterates, cum_nits, ax, label):
    """
    Plot f(v_t) - f* as a function of cumulative number of Newton iterates
    """
    print(cum_nits)
    opti_gap = f0_iterates - np.min(f0_iterates)
    ax.semilogy(cum_nits, opti_gap, label=label, marker="o")


def plot_precision_criterion(nits, d, mu, t0, ax, label):
    """
    Plot m/t as a function of barrier iterations
    """
    ts = np.zeros((nits, ))
    ts[0] = t0
    for i in range(0, nits-1):
        ts[i+1] = ts[i] * mu
    ax.semilogy(d / ts, label=label, marker="o")


def compare_opti_gaps(dict_iterates, dict_f0_iterates):
    """
    plot_opti_gap for several mus on same figure
    """
    fig, ax = plt.subplots()
    for key in dict_f0_iterates.keys():
        cum_nits = np.cumsum(dict_iterates[key][1])
        plot_opti_gap(dict_f0_iterates[key], cum_nits, ax, "mu = " + str(key))
    ax.legend()
    ax.set_ylabel("$f(v_t) - f^*$")
    ax.set_xlabel("Cumulative number of Newton iterations")
    ax.set_title("Convergence of log barrier")


def compare_precisions(dict_iterates, d, t0):
    """
    Compare precision criterio m/t for several mus
    """
    fig, ax = plt.subplots()
    for key in dict_iterates.keys():
        nits = len(dict_iterates[key][1])
        plot_precision_criterion(nits, d, key, t0, ax, "mu = " + str(key))
    ax.legend()
    ax.set_ylabel("2d/t")
    ax.set_xlabel("Log barrier iterations")
    ax.set_title("Decrease of duality gap bound")