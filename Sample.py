import numpy as np
from itertools import combinations_with_replacement
from trust_sub import trust_sub_exact

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
        self.Q = Q
        self.model_type = options.get('alg_model', 'quadratic') # default to quadratic

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

    def deletepoint(self, idx):
        self.Y = np.delete(self.Y, idx, axis=0)
        self.fY = np.delete(self.fY, idx)

    # Originally from Liyuan's code -- removes several points all at once 
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
        Z = (self.Y - center) @ Q  # m × p
        m, p = Z.shape

        quad_terms = p * (p + 1) // 2
        if self.model_type == 'quadratic' and m >= (1 + p + quad_terms):
            new_point, bad_idx = self._geometry_from_quadratic(Z, center, Q, delta)
            if new_point is not None:
                return new_point, bad_idx

        new_point, bad_idx = self._geometry_from_linear(Z, center, Q, delta)
        return new_point, bad_idx

    # ---------------------------------------------
    # GEOMETRY HELPERS
    # ---------------------------------------------
    def _build_vandermonde(self, Z, use_quadratic):
        m, p = Z.shape
        cols = [np.ones((m, 1)), Z]
        quad_pairs = []
        if use_quadratic:
            quad_pairs = list(combinations_with_replacement(range(p), 2))
            for (i, j) in quad_pairs:
                cols.append((Z[:, i] * Z[:, j]).reshape(m, 1))
        Vmat = np.hstack(cols)
        return Vmat, quad_pairs

    def _solve_lagrange_coeffs(self, V):
        eye_m = np.eye(V.shape[0])
        coeffs, *_ = np.linalg.lstsq(V, eye_m, rcond=None)
        return coeffs

    def _geometry_from_linear(self, Z, center, Q, delta):
        m = Z.shape[0]
        V, _ = self._build_vandermonde(Z, use_quadratic=False)
        L_coefs = self._solve_lagrange_coeffs(V)  # (p+1) × m
        norms = np.linalg.norm(L_coefs, axis=0)
        idx = np.argmax(norms)
        direction = L_coefs[1:, idx]
        if np.linalg.norm(direction) < 1e-12:
            return None, None
        direction /= np.linalg.norm(direction)
        new_point = center + delta * Q @ direction
        if self._is_duplicate(new_point):
            return None, None
        return new_point, idx

    def _geometry_from_quadratic(self, Z, center, Q, delta):
        m, p = Z.shape
        V, quad_pairs = self._build_vandermonde(Z, use_quadratic=True)
        L_coefs = self._solve_lagrange_coeffs(V)  # basis_dim × m

        def coeffs_to_quad(coeffs):
            a = coeffs[0]
            b = coeffs[1:1+p]
            H = np.zeros((p, p))
            idx = 1 + p
            for (i, j) in quad_pairs:
                cij = coeffs[idx]
                if i == j:
                    H[i, j] += 2.0 * cij
                else:
                    H[i, j] += cij
                    H[j, i] += cij
                idx += 1
            return a, b, H

        def eval_poly(a, b, H, z):
            return a + b @ z + 0.5 * z @ (H @ z)

        best_val = 0.0
        best_z = None
        best_idx = None

        for idx in range(m):
            coeffs = L_coefs[:, idx]
            a, b, H = coeffs_to_quad(coeffs)
            for sign in (1, -1):
                step, _ = trust_sub_exact(-sign * H, -sign * b, delta)
                z = step.flatten()
                val = abs(eval_poly(a, b, H, z))
                if val > best_val + 1e-12:
                    best_val = val
                    best_z = z
                    best_idx = idx

        if best_z is None or np.linalg.norm(best_z) < 1e-12:
            return None, None

        new_point = center + Q @ best_z
        if self._is_duplicate(new_point):
            return None, None
        return new_point, best_idx

    def _is_duplicate(self, point):
        return any(np.allclose(point, y) for y in self.Y)
