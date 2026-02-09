"""This file executes the derivative free optimization (DFO) solver to
"""
import os, sys, inspect
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Circle

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from DFOTR.Python3.dfo_tr import dfo_tr


cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"Python3")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

print(os.listdir(os.path.dirname(__file__)))

import numpy as np
from GCDFO import gcdfo
from funcs_defs import arwhead, rosen, sphere, ackley, booth, beale, bukin
from Oracle import Oracle


# Choose function in {arwhead, rosen, sphere, ackley, booth, beale, bukin}
func = bukin
oracle = Oracle(func)

# Set starting point
n = p = 10
x0 = -5 * np.ones(n)
#x0 = np.repeat(np.array([[-1.2, 1]]), 5, axis=0).flatten()

#  OPTIMIZE WITH GCDFO
customOptions = {'alg_model': 'quadratic',
                'alg_TRsub': 'exact',
                'tr_delta': 0.5,
                'tr_toaccept': 0.1,
                'tr_toexpand': 0.5,
                'tr_expand': 1.3,
                'tr_shrink':0.65,
                'stop_iter': 1500,
                'stop_nfeval': 1500,
                'stop_predict': 0.,
                'verbosity': 2
                }

# optimization with class function
x, fx, info = gcdfo.optimize(x0, oracle, p, customOptions)
final_samples = info["sample"]
eval_history = oracle.get_evaluation_history()

# x0 = np.repeat(np.array([[-1.2, 1]]), 2, axis=0).flatten()

# overwrite default settings
customOptions = {'alg_model': 'quadratic',
                'alg_TRsub': 'exact',
                'tr_delta': 0.5,
                'tr_shrink': 0.65,
                'stop_iter': 1500,
                'stop_nfeval': 1500,
                'sample_toremove': 30,
                'stop_predict': 0.,
                'verbosity': 2,
                }

# OPTIMIZE WITH DFO_TR
optimizer = dfo_tr(x0, customOptions)
while True:
    x = optimizer.ask()
    fx = [func(x[0])]
    optimizer.tell(x,fx)
    if optimizer._stop():
        break
idx = np.nanargmin(optimizer.samp.fY)
x = optimizer.samp.Y[idx]
fx = optimizer.samp.fY[idx]

dfo_tr_history = optimizer.info['best_obj']
plt.semilogy(dfo_tr_history, label="DFOTR")

fvals = [eval_history[i][1] for i in range(len(eval_history))]
best_fvals = np.minimum.accumulate(fvals)
plt.semilogy(best_fvals, label="GCDFO")
plt.title('bukin function n = 10')
plt.ylabel('log objective value')
plt.xlabel('Function Evaluations')
plt.legend(loc = "upper right")
plt.show()
