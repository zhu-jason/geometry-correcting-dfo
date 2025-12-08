import numpy as np
from itertools import combinations_with_replacement

class Sample:
    def __init__(self, x0, p, options):
        """
        Initialize a sample set.
        x0      : initial point (n,)
        p       : subspace dimension (<= n)
        options : dict with options, must contain 'tr_delta'
        """
        self.n = len(x0)
        self.p = p

        # initial orthonormal subspace
        Q, _ = np.linalg.qr(np.random.randn(self.n, self.p))
        Y = x0.reshape(self.n, 1) + options['tr_delta'] * Q
        self.Y = np.vstack([Y.T, x0.reshape(1, self.n)])  # (p+1) × n
        self.fY = np.full(len(self.Y), np.nan)
        self.big_lambda = options['big_lambda']
        self.Q = Q

    @property
    def m(self):
        return len(self.Y)

    def Ycentered(self, center):
        return self.Y - center

    def distance(self, center):
        return np.linalg.norm(self.Ycentered(center), axis=1)

    def addpoint(self, point):
        self.Y = np.vstack([self.Y, point])
        self.fY = np.append(self.fY, np.nan)

    def removepoint(self, idx):
        self.Y = np.delete(self.Y, idx, axis = 0)
        self.fY = np.delete(self.fY, idx)

    def auto_delete(self, model, options):
        distance = self.distance(model.center)
        farthest = np.argsort(distance)[-(self.m - options['sample_min']):]
        toofar = np.where(distance > model.delta)[0]
        if toofar.size == 0 and self.m > options['sample_max']:
            toremove = farthest[-(self.m - options['sample_max']):]
        else:
            toremove = np.intersect1d(farthest, toofar)
        self.Y = np.delete(self.Y, toremove, axis=0)
        self.fY = np.delete(self.fY, toremove)

    def _updateQR(self):
        A = np.random.randn(self.n, self.n)
        Qfull, R = np.linalg.qr(A)
        L = np.diag(np.sign(np.diag(R)))
        Qfull = Qfull @ L
        self.Q = Qfull[:, :self.p]

    # -----------------------------
    # GEOMETRY IMPROVEMENT USING LAGRANGE POLYNOMIALS
    # -----------------------------
    def improve_geometry(self, center, Q, delta):
        """
        Return a new point in the trust region that maximizes
        the norm of Lagrange polynomials in the subspace.

        center : current trust-region center
        Q      : n × p orthonormal subspace
        delta  : trust-region radius
        """

        # Project current samples into subspace coordinates
        Z = (self.Y - center) @ Q  # m × p

        # Construct all linear Lagrange polynomials
        m, p = Z.shape
        # For p-dimensional subspace, the basis of linear polynomials is: [1, z1, ..., zp]
        # The Vandermonde matrix for samples
        V = np.hstack([Z])  # m × (p+1)

        # Compute Lagrange coefficients for each sample

        L_coefs = np.linalg.pinv(V)  # (p+1) × m

        # The "poorest" Lagrange polynomial is the one with largest norm
        norms = np.linalg.norm(L_coefs, axis=0)
        idx = np.argmax(norms)

        # Evaluate new point in subspace: pick along direction of max Lagrange polynomial
        direction = L_coefs[:, idx][0:]  # skip constant term

        # We are lambda poised!
        if np.linalg.norm(direction) < self.big_lambda:
            return None, None

        direction = direction / np.linalg.norm(direction)
        new_point = center + delta * Q @ direction

        # Ensure new point is not duplicate
        if any(np.allclose(new_point, y) for y in self.Y):
            return None, None

        return idx, new_point
