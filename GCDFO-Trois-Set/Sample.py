import numpy as np
from itertools import combinations_with_replacement
from trust_sub import *
from InterpolationSet import InterpolationSet

class Sample:
    def __init__(self, x0, oracle, p, options):
        """
        Initialize a sample set.
        x0      : initial point (n,)
        p       : subspace dimension (<= n)
        options : dict with options, must contain 'tr_delta'
        """
        
        self.n = len(x0)
        self.p = p
        self.oracle = oracle
        self.Q = np.eye(p)
        self.big_lambda = options['big_lambda']

        # Center
        self.center = x0
        self.fc = oracle(x0)

        # Linear Interpolation Set
        Q, _ = np.linalg.qr(np.random.randn(self.n, self.p))
        print(Q)
        Y_pts = options['tr_delta'] * Q.T  # (p) × n  (note: codebase uses this convention)
        print(Y_pts)
        # Y_pts = options['tr_delta'] * np.eye(self.n)
        Y_vals = oracle(Y_pts + self.center)
        self.Y = InterpolationSet(Y_pts, Y_vals, p)
        
        # Hessian Interpolation Set, (p)*(p+1)/2 points sampled from delta sphere
        num_z_points = (p + 1) * p // 2
        random_vectors = np.random.randn(num_z_points, p)
        norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
        unit_vectors = random_vectors / norms
        Z_pts = options['tr_delta'] * unit_vectors
        Z_vals = oracle(Z_pts + self.center)

        # Z_pts = np.array([])
        # Z_vals = np.array([])
        self.Z = InterpolationSet(Z_pts, Z_vals, p)
        

    @property
    def mY(self):
        return len(self.Y)
    
    @property
    def mZ(self):
        return len(self.Z)
    
    @property
    def mTotal(self):
        return 1 + len(self.Y) + len(self.Z)

    def _updateQR(self):
        A = np.random.randn(self.n, self.p)
        Qfull, _ = np.linalg.qr(A)
        self.Q = Qfull[:, :self.p]

        # FOR DEBUG PURPOSES ONLY 
        self.Q = np.eye(self.n)

    def get_lin_lagrange_coef(self):
        try:
            L_coefs = np.linalg.inv(self.Y.points) 
        except np.linalg.LinAlgError:
            print("WEE WOO Linear Lagrange Polynomial Singular")
            L_coefs = np.linalg.pinv(self.Y.points)
        return L_coefs