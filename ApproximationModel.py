import numpy as np
from trust_sub import *
import copy

class ApproximationModel:
    """Quadratic or linear model in full space or subspace."""

    def __init__(self, n, options):
        self.n = n
        self.type = {
            'model': options['alg_model'],
            'TR': options['alg_TR'],
            'TRsub': options['alg_TRsub']
        }

        self.c = 0.0
        self.g = np.zeros(n)
        self.H = np.zeros((n, n))

        # trust region parameters
        self.center = np.zeros(n)
        self.delta = 0.0

    # -------------------------------------------
    # FIT MODEL BASED ON SAMPLE
    # -------------------------------------------
    def fit(self, samp):
        m, p = samp.m, samp.p
        Q = samp.Q                     # n × p
        Yc = samp.Ycentered(self.center)  # m × n
        fY = samp.fY                   # (m,)
        
        Ycopy = copy.deepcopy(samp.Y)
        center_idx = np.where(np.all(np.isclose(Ycopy, self.center), axis=1))[0][0]
        Ycopy[[0, center_idx]] = Ycopy[[center_idx, 0]]
        Yc = Ycopy - self.center
        fY_copy = copy.deepcopy(samp.fY)
        fY_copy[0], fY_copy[center_idx] = fY_copy[center_idx], fY_copy[0]
        fY = fY_copy

        # print("Y COPY")
        # print(Ycopy)

        # project samples onto subspace
        Z = Yc @ Q                      # m × p
        # print("Z-Here")
        # print(Z)
        # print("Q-Here")
        # print(Q)
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
        # print("A-Here")
        # print(A)
        # extract c, gradient, Hessian in subspace
        self.c = theta[0]
        self.g_sub = theta[1:1+p]
        Hsub = np.zeros((p, p))
        k = 1 + p
        for i in range(p):
            for j in range(i, p):
                if i == j:
                    Hsub[i, j] = Hsub[j, i] = theta[k] * 2
                else:
                    Hsub[i, j] = Hsub[j, i] = theta[k] 
                k += 1

        # lift to full space
        self.g =  Q @ self.g_sub
        self.H =  Q @ Hsub @ Q.T 
        # print("MODEL STUFF")
        # print("----------")
        # print("H value")
        # print(self.H)
        # print("g value")
        # print(self.g)


    # -------------------------------------------
    # FIT MODEL BASED ON SAMPLE
    # -------------------------------------------
    def fit_old(self, samp):
        m, p = samp.m, samp.p
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
                if i == j:
                    Hsub[i, j] = Hsub[j, i] = theta[k]
                else:
                    Hsub[i, j] = Hsub[j, i] = theta[k] * 0.5
                k += 1

        # lift to full space
        self.g = Q @ self.g_sub
        self.H = Q @ Hsub @ Q.T
        print("MODEL STUFF")
        print("----------")
        print("H value")
        print(self.H)
        print("g value")
        print(self.g)

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
        return self.center + v, -val

    # -------------------------------------------
    # TRUST REGION RADIUS UPDATE
    # -------------------------------------------
    def update_delta(self, rho, stepSize2delta, options):
        if rho >= options['tr_toexpand'] and np.linalg.norm(self.g) >= options['tr_toexpand2'] * self.delta:
            self.delta *= options['tr_expand']
        elif (0 <= rho < options['tr_toaccept']) or rho < options['tr_toshrink']:
            self.delta *= options['tr_shrink']
