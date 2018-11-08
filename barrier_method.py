import numpy as np
import warnings

warnings.filterwarnings("error")


def phi(A, b, v):
    """
    Log barrier function for linear inequality constraints
    """
    try:
        g = - np.sum(np.log(- np.dot(A, v) + b))
    # Necessary sometimes for the first step of linesearch when v + delta is not feasible
    except RuntimeWarning:
        g = np.inf
    return g


def objective(Q, p, v):
    """
    QP objective
    """
    return np.dot(np.dot(v.T, Q), v) + np.dot(p.T, v)


def logbarrier_objective(Q, p, A, b, t, v):
    """
    Objective for the centering step (linear objective penalized by log barrier of constraints)
    """
    obj = t * objective(Q, p, v) + phi(A, b, v)
    return obj


def gradient(Q, p, A, b, t, v):
    """
    Gradient of logbarrier_objective in v
    """
    d = A.shape[0]
    n = A.shape[1]
    term1 = 2 * t * np.dot(Q, v) + t * p
    Av_minus_b = np.dot(A, v) - b
    grad = np.zeros((n, ))
    for k in range(0, d):
        grad -= A[k, :] / Av_minus_b[k]
    return term1 + grad


def hessian(Q, A, b, t, v):
    """
    Hessian of logbarrier_objective in v
    """
    n = A.shape[1]
    d = A.shape[0]
    Av_minus_b = np.dot(A, v) - b
    Av_minus_b_sqr = Av_minus_b ** 2
    hess = np.zeros((n, n))
    for i in range(0, d):
        hess += (1 / Av_minus_b_sqr[i]) * np.dot(A[i, :].reshape(n, 1),
                                                 A[i, :].reshape(n, 1).T)
    return 2 * t * Q + hess


def newton_update(Q, p, A, b, t, v):
    """
    Compute Newton update for centering step
    """
    hess = hessian(Q, A, b, t, v)
    grad = gradient(Q, p, A, b, t, v)
    hess_inv = np.linalg.inv(hess)
    delta = - np.dot(hess_inv, grad)
    lamb_sqr = - np.dot(grad.T, delta)
    return delta, lamb_sqr


def bktk_line_search(Q, p, A, b, t, v, delta, alpha, beta, maxit):
    """
    Backtracking line search for centering step
    """
    eta = 1
    grad = gradient(Q, p, A, b, t, v)
    fv = logbarrier_objective(Q, p, A, b, t, v)
    for k in range(0, maxit):
        c = logbarrier_objective(Q, p, A, b, t, v + eta * delta)
        d = fv + alpha * eta * np.dot(grad, delta)
        if c > d:
            eta *= beta
        else:
            return eta
    warnings.warn("Line search: Maxit attained, no garanties of convergence")
    return eta


def centering_step(Q, p, A, b, t, v0,
                   eps=0.001,
                   maxit_newton=1000,
                   maxit_search=100,
                   alpha=0.1,
                   beta=0.5):
    v = v0.copy()
    iterates = [v.copy()]
    for m in range(0, maxit_newton):
        # Compute direction of descent delta
        delta, lamb_sqr = newton_update(Q, p, A, b, t, v)
        # Perform backtracking linesearch to find learning rate
        eta = bktk_line_search(Q, p, A, b, t, v, delta,
                               alpha, beta, maxit_search)
        # Stopping criterion
        if lamb_sqr / 2 <= eps:
            return iterates
        # Update v
        v += eta * delta
        print(objective(Q, p, v))
        iterates.append(v.copy())
    warnings.warn("Newton: Maxit attained, no garanties of convergence")
    return iterates


def barr_method(Q, p, A, b, v0, eps, t0, mu, maxit=1000):
    d = A.shape[0]
    vt = v0.copy()
    t = t0
    iterates = [vt.copy()]
    nits_newton_list = [0]
    for k in range(0, maxit):
        # Perform centering step
        newton_iterates = centering_step(Q, p, A, b, t, vt, eps)
        vtplus1 = newton_iterates[-1]
        nits_newton_list.append(len(newton_iterates))
        iterates.append(vtplus1.copy())
        # Stopping criterion
        if d / t < eps:
            return iterates, nits_newton_list
        # Update t
        t *= mu
        vt = vtplus1
    warnings.warn("Barrier method: Maxit attained, no garanties of convergence")
    return iterates, nits_newton_list




