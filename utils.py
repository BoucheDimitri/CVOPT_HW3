import numpy as np
import matplotlib.pyplot as plt

import barrier_method as bm

# Plot parameters
plt.rcParams.update({"font.size": 30})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})



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


def plot_opti_gap(f0_iterates, cum_nits, label):
    """
    Plot f(v_t) - f* as a function of cumulative number of Newton iterates
    """
    print(cum_nits)
    opti_gap = f0_iterates - np.min(f0_iterates)
    plt.semilogy(cum_nits, opti_gap, label=label, marker="o")


def compare_opti_gaps(dict_iterates, dict_f0_iterates):
    """
    plot_opti_gap for several mus on same figure
    """
    # fig, ax = plt.subplots(1)
    for key in dict_f0_iterates.keys():
        cum_nits = np.cumsum(dict_iterates[key][1])
        plot_opti_gap(dict_f0_iterates[key], cum_nits, "mu = " + str(key))
    plt.legend()
    plt.ylabel("$f(v_t) - f^*$")
    plt.xlabel("Cumulative number of Newton iterations")
    plt.title("Convergence of log barrier")