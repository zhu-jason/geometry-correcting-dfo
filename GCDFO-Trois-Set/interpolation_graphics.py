"""This file executes the derivative free optimization (DFO) solver to
"""
import os, sys, inspect
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Circle

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"Python3")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

print(os.listdir(os.path.dirname(__file__)))

import numpy as np
from GCDFO import gcdfo
from funcs_defs import arwhead, rosen, sphere
from Oracle import Oracle

for iter in range(0, 10, 1):
    np.random.seed(42)

    # choose function
    # func = arwhead
    # func = rosen
    oracle = Oracle(rosen)

    # starting point
    n = p = 2
    x0 = -1.5 * np.ones(n)
    #x0 = np.repeat(np.array([[-1.2, 1]]), 5, axis=0).flatten()

    # overwrite default settings
    customOptions = {'alg_model': 'quadratic',
                    'alg_TRsub': 'exact',
                    'tr_delta': 0.5,
                    'tr_toaccept': 0.1,
                    'tr_toexpand': 0.5,
                    'tr_expand': 1.3,
                    'tr_shrink':0.65,
                    'stop_iter': iter,
                    'stop_nfeval': 1500,
                    'stop_predict': 0.,
                    'verbosity': 2
                    }

    # optimization with class function
    x, fx, info = gcdfo.optimize(x0, oracle, p, customOptions)
    final_samples = info["sample"]
    hessian_norms = info['hessian_norms'];
    print(hessian_norms)



    # -----------------------------
    # Create 2D grid
    # -----------------------------
    x1 = np.linspace(-3.0, 3.0, 400)
    x2 = np.linspace(-3.0, 3.0, 400)
    X1, X2 = np.meshgrid(x1, x2)

    # -----------------------------
    # Evaluate Rosenbrock on grid
    # -----------------------------
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = rosen(np.array([X1[i, j], X2[i, j]]))

    # -----------------------------
    # Points known to lie on surface
    # -----------------------------
    center1 = final_samples.center[0]
    center2 = final_samples.center[1]

    # Shift points onto world coordinates
    Ypts = final_samples.Y.points + final_samples.center
    Zpts = final_samples.Z.points + final_samples.center

    Y1 = Ypts[:, 0]
    Y2 = Ypts[:, 1]

    Z1 = Zpts[:, 0]
    Z2 = Zpts[:, 1]

    # -----------------------------
    # Plot contour + points
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Contour plot
    contours = ax.contour(X1, X2, Z, levels=50, cmap="viridis")
    ax.clabel(contours, inline=True, fontsize=8)

    # Overlay points
    ax.scatter(center1, center2, color="red", s=80, label="center")
    ax.scatter(Y1, Y2, color="blue", s=80, label="Y set")
    ax.scatter(Z1, Z2, color="green", s=80, label="Z set")

    # Add trust region
    r = info['tr_final']
    print(r)
    circle = Circle((center1, center2), r, color='blue', fill=False, linewidth=2, label='Trust region')
    ax.add_patch(circle)

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Rosenbrock Function Contours, iteration {}".format(iter))
    plt.legend(loc="upper right")
    plt.show()
