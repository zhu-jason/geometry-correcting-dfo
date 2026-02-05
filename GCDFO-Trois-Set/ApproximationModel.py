import numpy as np
from trust_sub import *
import copy
import cvxpy as cp

class ApproximationModel:
    """Quadratic or linear model in full space or subspace."""

    def __init__(self, n, options):
        self.n = n
        self.type = {
            'model': options['alg_model'],
            'TR': options['alg_TR'],
            'TRsub': options['alg_TRsub']
        }

        self.g = np.zeros(n)
        self.H = np.zeros((n, n))
        self.delta = 0.0

    # -------------------------------------------
    # FIT MODEL BASED ON SAMPLE
    # -------------------------------------------
    def fit(self, samp):
        m, p = samp.mTotal, samp.p

        # Fit the gradient approximation
        g, *_ = np.linalg.lstsq(samp.Y.points, samp.Y.values - samp.fc, rcond=None)

        # Fit Hessian approximation
        n_quad = p * (p + 1) // 2
        A = np.zeros((n_quad, n_quad))
        idx = 0
        for i in range(p):
            for j in range(i, p):
                if i == j:
                    A[:, idx] = samp.Z.points[:, i] * samp.Z.points[:, j]
                else:
                    A[:, idx] = 0.5 * samp.Z.points[:, i] * samp.Z.points[:, j]
                idx += 1

        theta, *_ = np.linalg.lstsq(A, samp.Z.values - samp.fc - samp.Z.points @ g, rcond=None)

        # extract Hessian in subspace
        H = np.zeros((p, p))
        k = 0
        for i in range(p):
            for j in range(i, p):
                H[i, j] = H[j, i] = theta[k]
                k += 1
        # lift to full space
        self.g = g
        self.H = H

    def fit_full_quadratic(self, samp):
        # Fits a possibly underdetermined quadratic
        p = samp.p
        num_points_without_c = samp.mTotal - 1
        if samp.Z.points.size == 0:
            X = samp.Y.points
        else:
            X = np.vstack([samp.Y.points, samp.Z.points])
        fX = np.concatenate([samp.Y.values, samp.Z.values])

        n_quad = p * (p + 1) // 2
        A = np.zeros((samp.mTotal-1, p + n_quad))
        A[:, 0:p] = X

        idx = p
        for i in range(p):
            for j in range(i, p):
                if i == j:
                    A[:, idx] = 0.5 * X[:, i] * X[:, j]
                else:
                    A[:, idx] = X[:, i] * X[:, j]
                idx += 1

        theta, *_ = np.linalg.lstsq(A, fX - samp.fc, rcond=None)

        Hsub = np.zeros((p, p))
        k = p
        for i in range(p):
            for j in range(i, p):
                Hsub[i, j] = Hsub[j, i] = theta[k]
                k += 1

        # lift to full space
        self.g = theta[0:p]
        self.H =  samp.Q @ Hsub @ samp.Q.T

    # -------------------------------------------
    # COMPUTE TRUST REGION STEP
    # -------------------------------------------
    def minimize(self, samp):
        Q = samp.Q
        p = samp.p
        g_sub = Q.T @ self.g

        # linear model if H is degenerate
        Hsub = Q.T @ self.H @ Q
        if self.type['TRsub'] == 'exact':
            v_sub, val = trust_sub_exact(Hsub, g_sub, self.delta)
        else:
            v_sub, val = trust_sub_CG(Hsub, g_sub, self.delta)

        v = Q @ v_sub
        return v, -val
