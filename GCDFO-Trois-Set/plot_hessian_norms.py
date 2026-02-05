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

np.random.seed(42)

# choose function
# func = arwhead
# func = rosen
oracle = Oracle(rosen)

# starting point
n = p = 3
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
                'stop_iter': 1000,
                'stop_nfeval': 1500,
                'stop_predict': 0.,
                'verbosity': 2
                }

# optimization with class function
x, fx, info = gcdfo.optimize(x0, oracle, p, customOptions)
final_samples = info["sample"]
hessian_norms = info['hessian_norms'];


# -----------------------------
# Plot contour + points
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(hessian_norms)
plt.xlabel('iteration')
plt.ylabel('Hessian norm')
plt.title('Hessian norms for Rosenbrock')
print(sorted(hessian_norms))
plt.show()
