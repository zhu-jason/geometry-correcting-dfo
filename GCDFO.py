"""Subspace DFO Trust-Region (GCDFO)"""
import numpy as np
import time
from Sample import Sample
from ApproximationModel import ApproximationModel

class gcdfo:
    def __init__(self, x0, p=None, options=None):
        self.n = len(x0)
        self.p = min(p, self.n)

        # default options
        self.options = {
            'alg_model': 'quadratic',
            'alg_TR': 'ball',
            'alg_TRsub': 'exact',
            'sample_initial': self.n + 1,
            'sample_min': int((self.n+1) * 1.1),
            'sample_max': min(3000, (self.n+1)*(self.n+2)//2),
            'sample_toremove': 10,
            'tr_delta': 1.0,
            'tr_toaccept': 0.0,
            'tr_toexpand': 0.5,
            'tr_toexpand2': 0.5,
            'tr_expand': 1.3,
            'tr_toshrink': -5e-3,
            'tr_shrink': 0.99,
            'stop_iter': 2000,
            'stop_nfeval': 2000,
            'stop_delta': 1e-6,
            'stop_predict': 1e-8,
            'verbosity': 2
        }

        if options:
            for key in options:
                if key not in self.options:
                    raise ValueError(f"{key!r} is not a valid option name.")
                self.options[key] = options[key]

        self.options['p'] = self.p
        self.info = {'start_time': time.time(), 'iteration': 0, 'success': 0, 'nfeval': 0}

        # required sample size depends on model order
        if self.options['alg_model'] == 'quadratic':
            self.required_sample_size = (self.p + 1) * (self.p + 2) // 2
        else:  # treat everything else as linear
            self.required_sample_size = self.p + 1

        # initial sample
        self.samp = Sample(x0, self.p, self.options)
        self.model = ApproximationModel(self.p, self.options)



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

            self.info['predicted_decrease'] = np.inf

            self.info['iteration'] = 1
        else:
            # fit model
            self.model.fit(self.samp)

            # solve trust-region subproblem
            new_point, self.info['predicted_decrease'] = self.model.minimize(self.samp)
            self.info['iteration'] += 1

            # predicted vs actual reduction
            # Why is this the correct calculation for RHO?
            rho = (self.model.c - self.samp.fY[-1]) / self.info['predicted_decrease']

            # If successful replace point and expand TR
            if rho >= self.options['tr_toaccept'] and \
                np.linalg.norm(self.model.g) >= self.options['tr_toexpand2'] * self.model.delta:
                self._success = 1
                self.info['success'] += 1

                # replace furthest interpolation point
                dist = self.samp.distance(new_point)
                j_star = np.argmax(dist)
                self.samp.Y[j_star] = new_point
                self.samp.fY[j_star] = np.nan
                self.model.center = new_point
                #self.samp._updateQR()
                self.model.delta = self.model.delta  self.options['tr_expand']
            else: # If unsuccessful, try to improve geometry
                self._success = 0
                center = self.model.center
                delta = self.model.delta
                m = self.samp.m
                sample_size = self.required_sample_size

                # Geometry correction steps as in Algorithm 4:
                # 1) add point if sample set is too small
                # 2) replace a far point if any lies outside the TR
                # 3) replace a "bad" point identified by Lagrange polynomials
                # 4) if geometry is good, shrink TR radius and refresh subspace

                if new_point is not None:
                    if m < sample_size:
                        # Geometry correction by adding a point
                        self.samp.addpoint(new_point)
                    else:
                        # Check for far point
                        dist = self.samp.distance(center)
                        j_star = np.argmax(dist)
                        #bad_idx = 0 # placeholder for bad geometry correcting step
                        if dist[j_star] > delta:
                            # Geometry correction by replacing a far point
                            self.samp.Y[j_star] = new_point
                            self.samp.fY[j_star] = np.nan
                        #elif bad_idx is not None:
                            # Geometry correction by replacing a "bad" point
                            #self.samp.Y[bad_idx] = new_point
                            #self.samp.fY[bad_idx] = np.nan
                        #    self.samp.auto_delete(self.model, self.options)  #placeholder
                        else:
                            # Geometry is good: shrink TR and refresh subspace
                            self.model.delta *= self.options['tr_shrink']
                            self.samp._updateQR()

            # Print iteration report
            if self.options['verbosity'] >= 2:
                print("| {:5d} | {} | {:11.5e} | {:9.6f} | {:9.6f} | {} |"
                        .format(self.info['iteration'],
                                self._success,
                                self.samp.fY[-1],
                                self.model.delta,
                                rho,
                                self.samp.m))

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
    def optimize(cls, obj, x0, p=None, options=None):
        optimizer = cls(x0, p, options)

        while True:
            x = optimizer.ask()
            fx = [obj(x[0])]
            optimizer.tell(x, fx)
            if optimizer._stop():
                break
        idx = np.nanargmin(optimizer.samp.fY)
        return optimizer.samp.Y[idx], optimizer.samp.fY[idx], optimizer.info
