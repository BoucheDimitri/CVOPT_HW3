import numpy as np
import warnings


def phi(A, b, v):
    return - np.sum(np.log(- np.dot(A, v) + b))


def objective(Q, p, v):
    return np.dot(np.dot(v.T, Q), v) + np.dot(p.T, v)


def logbarrier_objective(Q, p, A, b, t, v):
    return t * objective(Q, p, v) + phi(A, b, v)


def gradient(Q, p, A, b, t, v):
    d = A.shape[0]
    n = A.shape[1]
    term1 = 2 * t * np.dot(Q, v) + t * p
    Av_minus_b = np.dot(A, v) - b
    grad = np.zeros((n, ))
    for k in range(0, d):
        grad -= A[k, :] / Av_minus_b[k]
    # term2 = np.sum(A.T * (- Av_minus_b).reshape(()), axis=1)
    # return term1 + term2
    return term1 + grad


def hessian(Q, A, b, t, v):
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
    hess = hessian(Q, A, b, t, v)
    grad = gradient(Q, p, A, b, t, v)
    hess_inv = np.linalg.inv(hess)
    delta = - np.dot(hess_inv, grad)
    lamb_sqr = - np.dot(grad.T, delta)
    return delta, lamb_sqr


def bktk_line_search(Q, p, A, b, t, v, delta, alpha, beta, maxit):
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
                   alpha=0.2,
                   beta=0.7):
    v = v0.copy()
    iterates = [v.copy()]
    for m in range(0, maxit_newton):
        delta, lamb_sqr = newton_update(Q, p, A, b, t, v)
        eta = bktk_line_search(Q, p, A, b, t, v, delta,
                               alpha, beta, maxit_search)
        if lamb_sqr / 2 <= eps:
            return iterates
        v += eta * delta
        print(logbarrier_objective(Q, p, A, b, t, v))
        iterates.append(v.copy())
    warnings.warn("Newton: Maxit attained, no garanties of convergence")
    return iterates


def barr_method(Q, p, A, b, v0, eps, t0, mu, maxit=1000):
    d = A.shape[0]
    v = v0.copy()
    t = t0
    iterates = [v.copy()]
    for k in range(0, maxit):
        v = centering_step(Q, p, A, b, t, v, eps)[-1]
        iterates.append(v.copy())
        if d / t < eps:
            return iterates
        t *= mu
    warnings.warn("Barrier method: Maxit attained, no garanties of convergence")
    return iterates




