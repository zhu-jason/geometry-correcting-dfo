"""This file executes the derivative free optimization (DFO) solver to
minimize a blackbox function.

User is required to import a blackbox function, and provide a
starting point. User can overwrite the default algorithm and/or the
default parameters used in the solver.
"""
import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"Python3")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

print(os.listdir(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from GCDFO import gcdfo
from funcs_defs import arwhead, rosen, sphere, beale, booth, bukin, ackley


# np.random.seed(42)

# choose function
# func = arwhead
# func = rosen
func = rosen

# starting point
n = 50
subspace_dims = [50]
x0 = np.random.randn(n)
#x0 = np.repeat(np.array([[-1.2, 1]]), 5, axis=0).flatten()

# optimization with class function
customOptions = {'alg_model': 'quadratic',
            'alg_TRsub': 'exact',
            'tr_delta': 0.5,
            'stop_iter': 1000,
            'stop_nfeval': 1000,
            'stop_predict': 0.,
            'verbosity': -1,
            'big_lambda': 1.5
            }
                
gc_info = []
# with geometry correction
for p in subspace_dims:
    # overwrite default settings
    x, fx, info = gcdfo.optimize(func, x0, p, customOptions)
    gc_info.append(info['best_objectives'])


non_gc_info = []
# without geometry correction
for p in subspace_dims:
    x, fx, info = gcdfo.optimize(func, x0, p, customOptions, False)
    non_gc_info.append(info['best_objectives'])

# Plotting
plt.figure()
# Use a consistent color per dimension, changing only line style between GC and non-GC
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, curve in enumerate(gc_info):
    xs = [pt[0] for pt in curve]
    ys = [pt[1] for pt in curve]
    color = colors[i % len(colors)]
    plt.semilogy(xs, ys, linestyle='--', color=color,
             label=f'p={subspace_dims[i]}, GC')

# for i, curve in enumerate(non_gc_info):
#     xs = [pt[0] for pt in curve]
#     ys = [pt[1] for pt in curve]
#     color = colors[i % len(colors)]
#     plt.semilogy(xs, ys, linestyle='-', color=color,
#              label=f'p={subspace_dims[i]}, non-GC')

plt.xlabel('Function evaluations')
plt.ylabel('Objective value')
plt.title('Subspace-GeomCorr-TR')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
