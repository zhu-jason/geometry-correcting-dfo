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

        # initial orthonormal subspace (we start with only p + 1 points)
        Q, _ = np.linalg.qr(np.random.randn(self.n, self.p))
        Y = x0.reshape(self.n, 1) + options['tr_delta'] * Q
        self.Y = np.vstack([Y.T, x0.reshape(1, self.n)])  # (p+1) × n
        self.fY = np.full(len(self.Y), np.nan)
        self.big_lambda = options['big_lambda']
        self.Q = Q
        print("Initial Sample:")
        print("Initial interpolation set")
        print(self.Y)
        print("Function values")
        print(self.fY)

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

    def addpoint_secondlast(self, point):
        self.Y = np.insert(self.Y, -1, point, axis=0)
        self.fY = np.insert(self.fY, -1, np.nan)

    def has_point_greater_delta(self, model, options):
        distance = self.distance(model.center)
        furthest_index = np.argsort(distance)[-1]
        print("Distances: ")
        print(distance)
        print(furthest_index)
        return np.linalg.norm(self.fY[furthest_index]) > model.delta

    def delete_point(self, index):
        self.Y = np.delete(self.Y, index, axis=0)
        self.fY = np.delete(self.fY, index)
        
    def delete_furthest(self, model, options):
        distance = self.distance(model.center)
        furthest_index = np.argsort(distance)[-1]
        self.Y = np.delete(self.Y, furthest_index, axis=0)
        self.fY = np.delete(self.fY, furthest_index)

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
        A = np.random.randn(self.n, self.p)
        Qfull, _ = np.linalg.qr(A)
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
        Y_copy = self.Y[:-1].copy()
        Z = (Y_copy - center) @ Q  # m × p

        # Construct all linear Lagrange polynomials
        m, p = Z.shape
        # For p-dimensional subspace, the basis of linear polynomials is: [1, z1, ..., zp]
        # The Vandermonde matrix for samples
        V = np.hstack([np.ones((m, 1)), Z])  # m × (p+1)

        # Compute Lagrange coefficients for each sample
        try:
            L_coefs = np.linalg.inv(V)  # (p+1) × m
        except np.linalg.LinAlgError:
            # If V is singular, fall back to pseudo-inverse
            L_coefs = np.linalg.pinv(V)

        # The "poorest" Lagrange polynomial is the one with largest norm
        norms = np.linalg.norm(L_coefs, axis=0)
        idx = np.argmax(norms)

        # Evaluate new point in subspace: pick along direction of max Lagrange polynomial
        direction = L_coefs[:, idx][1:]  # skip constant term

        if np.linalg.norm(direction) < 1e-12:
            return None, None
        
        # We are lambda poised!
        if np.linalg.norm(direction) < self.big_lambda:
            return None, None

        direction = direction / np.linalg.norm(direction)
        new_point = center + delta * Q @ direction

        # Ensure new point is not duplicate
        if any(np.allclose(new_point, y) for y in self.Y):
            return None, None

        return idx, new_point
