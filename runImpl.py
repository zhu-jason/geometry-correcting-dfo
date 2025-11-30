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
from GCDFO import gcdfo
from funcs_defs import arwhead, rosen, sphere


# choose function
#func = arwhead
#func = rosen
func = sphere

# starting point
x0 = np.ones(100) * 0.5
#x0 = np.repeat(np.array([[-1.2, 1]]), 1, axis=0).flatten()

# overwrite default settings
customOptions = {'alg_model': 'quadratic',
                'alg_TRsub': 'exact',
                'tr_delta': 0.5,
                'sample_toremove': 30,
                'stop_iter': 10000,
                'stop_predict': 0.,
                'verbosity': 2,
                }

# optimization with class function
# x, fx, info = dfo_tr.optimize(func, x0, customOptions)

# optimization in ask and tell frameword
p = 30
optimizer = gcdfo(x0, p, customOptions)
while True:
    x = optimizer.ask()
    fx = [func(x[0])]
    optimizer.tell(x,fx)
    if optimizer._stop():
        break
idx = np.nanargmin(optimizer.samp.fY)
x = optimizer.samp.Y[idx]
fx = optimizer.samp.fY[idx]

# print result
print("Printing result for function " + func.__name__ + ":")
print("best point: {}, with obj: {:.6f}".format(
    np.around(x, decimals=5), float(fx)))
