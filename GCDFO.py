"""Subspace DFO Trust-Region (GCDFO)"""
import numpy as np
import time
from Sample import Sample
from ApproximationModel import ApproximationModel

class gcdfo:
    def __init__(self, x0, p=None, options=None, geom_correcting = True):
        self.n = len(x0)
        self.p = p or self.n

        # default options
        self.options = {
            'alg_model': 'quadratic',
            'alg_TR': 'ball',
            'alg_TRsub': 'exact',
            'sample_initial': self.p + 1,
            'sample_min': int((self.p+1) * 1.1),
            'sample_max': min(3000, (self.p+1)*(self.p+2)//2),
            'sample_toremove': 10,
            'tr_delta': 1.0,
            'tr_toaccept': 0.0,
            'tr_toexpand': 0.5,
            'tr_toexpand2': 0.5,
            'tr_expand': 1.3,
            'tr_toshrink': -5e-3,
            'tr_shrink': 0.95,
            'stop_iter': 2000,
            'stop_nfeval': 2000,
            'stop_delta': 1e-6,
            'stop_predict': 1e-8,
            'verbosity': 2,
            'big_lambda': 3
        }

        if options:
            for key in options:
                if key not in self.options:
                    raise ValueError(f"{key!r} is not a valid option name.")
                self.options[key] = options[key]

        self.geom_correcting = geom_correcting
        self.options['p'] = self.p
        self.info = {
            'start_time': time.time(), 
            'iteration': 0, 
            'success': 0, 
            'nfeval': 0,
            'best_objectives': [],  # Track best objective at each evaluation
        }

        # initial sample
        self.samp = Sample(x0, self.p, self.options)
        self.model = ApproximationModel(self.p, self.options)
        
        # Track initial best (will be updated as evaluations come in)
        self._best_obj = np.inf

    # -----------------------------
    # ASK / TELL INTERFACE
    # -----------------------------
    def ask(self, nAsk=1):
        idx = np.isnan(self.samp.fY).argmax()
        return [self.samp.Y[idx]]

    def tell(self, X, fX):
        assert np.all(X == X[0])
        idx = np.all(self.samp.Y == X[0], axis=1).argmax()
        self.samp.fY[idx] = np.mean(fX)
        self.info['nfeval'] += 1
        
        self.info['best_objectives'].append((self.info['nfeval'] + self.p, np.nanmin(self.samp.fY)))

        if np.any(np.isnan(self.samp.fY)):
            return

        self.__suggest()

    # -----------------------------
    # MAIN ITERATION
    # -----------------------------
    def __suggest(self):
        if self.info['iteration'] == 0:
            self.model.center = self.samp.Y[np.nanargmin(self.samp.fY)]
            self.model.delta = self.options['tr_delta']

            if self.options['verbosity'] >= 2:
                print("\n Iteration Report ")
                print('|  iter |suc|  objective  | TR_radius |    rho    | m  |')
                print("| {:5d} |---| {:11.5e} | {:9.6f} | --------- | {} "
                      .format(self.info['iteration'],
                              np.nanmin(self.samp.fY),
                              self.model.delta,
                              self.samp.m))
            self._success = 1
        else:
            # predicted vs actual reduction
            rho = (self.model.c - self.samp.fY[-1]) / self.info['predicted_decrease']
            stepSize = np.linalg.norm(self.samp.Y[-1] - self.model.center)
            stepSize2delta = stepSize / self.model.delta

            if rho >= self.options['tr_toaccept'] and \
               np.linalg.norm(self.model.g) >= self.options['tr_toexpand2'] * self.model.delta:
                self._success = 1
                self.info['success'] += 1
                self.model.center = self.samp.Y[-1]
                self.samp.auto_delete(self.model, self.options)
                self.samp._updateQR()
            else:
                self._success = 0

            self.model.update_delta(rho, stepSize2delta, self.options)

            if self.options['verbosity'] >= 2:
                print("| {:5d} | {} | {:11.5e} | {:9.6f} | {:9.6f} | {} |"
                      .format(self.info['iteration'],
                              self._success,
                              self.samp.fY[-1],
                              self.model.delta,
                              rho,
                              self.samp.m))

        # fit model
        self.model.fit(self.samp)

        # geometry improvement only if step unsuccessful
        if self.geom_correcting:
            if not self._success:
                geom_point = self.samp.improve_geometry(self.model.center, self.samp.Q, self.model.delta)
                if geom_point is not None:
                    self.samp.auto_delete(self.model, self.options)
                    self.samp.addpoint(geom_point)
                else:
                    #self.samp._updateQR()
                    pass

        # solve trust-region subproblem
        x1, self.info['predicted_decrease'] = self.model.minimize(self.samp)
        self.samp.addpoint(x1)
        self.info['iteration'] += 1

    # -----------------------------
    # STOPPING CRITERIA
    # -----------------------------
    def _stop(self):
        STOP = False
        if self.info['iteration'] == 0:
            return STOP

        if self.info['iteration'] >= self.options['stop_iter']:
            STOP = True
            print('Exiting: max iterations reached.')
        elif self.info['nfeval'] >= self.options['stop_nfeval']:
            STOP = True
            print('Exiting: max function evaluations reached.')
        elif self.model.delta <= self.options['stop_delta']:
            STOP = True
            print('Exiting: minimum trust-region radius reached.')
        elif self.info['predicted_decrease'] <= self.options['stop_predict']:
            STOP = True
            print('Exiting: minimum predicted decrease reached.')

        if STOP and self.options['verbosity'] >= 1:
            print('***************** FINAL REPORT ************************')
            self.info['end_time'] = time.time()
            print('total elapsed time: {} seconds\n'.format(self.info['end_time'] - self.info['start_time']))
            print("|#iter|#success|#fevals| best fvalue |final tr_radius|final predicted decrease|")
            print("|{:5d}| {:5d}  | {:5d} | {:11.5e} |   {:9.6f}   |       {:11.5e}      |\n"
                  .format(self.info['iteration'],
                          self.info['success'],
                          self.info['nfeval'],
                          min(self.samp.fY),
                          self.model.delta,
                          self.info['predicted_decrease']                          ))

        return STOP

    # -----------------------------
    # CLASS METHOD: OPTIMIZE
    # -----------------------------
    @classmethod
    def optimize(cls, obj, x0, p=None, options=None, geom_correcting = True):
        optimizer = cls(x0, p, options, geom_correcting)

        while True:
            x = optimizer.ask()
            fx = [obj(x[0])]
            optimizer.tell(x, fx)
            if optimizer._stop():
                break
        idx = np.nanargmin(optimizer.samp.fY)
        return optimizer.samp.Y[idx], optimizer.samp.fY[idx], optimizer.info
