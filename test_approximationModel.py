# testing ApproximationModel

from ApproximationModel import ApproximationModel
from Sample import Sample
import numpy as np

def sphere(x):
    return np.linalg.norm(x)**2

n = 3
model = ApproximationModel(3)

def fit(self, samp):
    m, p = samp.size
    Q = samp.Q                     # n × p
    Yc = samp.Ycentered(self.center)  # m × n
    fY = samp.fY                   # (m,)

    # project samples onto subspace
    Z = Yc @ Q                     # m × p

    n_quad = p * (p + 1) // 2
    A = np.zeros((m, 1 + p + n_quad))
    A[:, 0] = 1
    A[:, 1:1+p] = Z
    idx = 1 + p
    for i in range(p):
        for j in range(i, p):
            A[:, idx] = Z[:, i] * Z[:, j]
            idx += 1
    theta, *_ = np.linalg.lstsq(A, fY, rcond=None)

    # extract c, gradient, Hessian in subspace
    self.c = theta[0]
    self.g_sub = theta[1:1+p]
    Hsub = np.zeros((p, p))
    k = 1 + p
    for i in range(p):
        for j in range(i, p):
            Hsub[i, j] = Hsub[j, i] = theta[k]
            k += 1

    # lift to full space
    self.g = Q @ self.g_sub
    self.H = Q @ Hsub @ Q.T
