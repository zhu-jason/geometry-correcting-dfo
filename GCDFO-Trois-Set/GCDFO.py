"""Subspace DFO Trust-Region (GCDFO)"""
import numpy as np
import time
from Sample import Sample
from ApproximationModel import ApproximationModel

class gcdfo:
    def __init__(self, x0, oracle, p=None, options=None):
        self.n = len(x0)
        self.p = p or self.n
        self.oracle = oracle

        # default options 
        self.options = {
            'alg_model': 'quadratic',
            'alg_TR': 'ball',
            'alg_TRsub': 'exact',
            'tr_delta': 1.0,
            'tr_toaccept': 0.1,
            'tr_toexpand': 0.5,
            'tr_expand': 1.3,
            'tr_shrink': 0.6,
            'stop_iter': 2000,
            'stop_nfeval': 2000,
            'stop_delta': 1e-6,
            'stop_predict': 1e-8,
            'verbosity': 2,
            'big_lambda': 2
        }

        if options:
            for key in options:
                if key not in self.options:
                    raise ValueError(f"{key!r} is not a valid option name.")
                self.options[key] = options[key]

        self.info = {
            'start_time': time.time(), 
            'iteration': 0, 
            'success': 0, 
            'nfeval': 0,
            'lagrange_step': 0,
            'iteration_info': [],  # Track best objective at each evaluation
        }

        # initial sample
        self.samp = Sample(x0, oracle, self.p, self.options)
        self.model = ApproximationModel(self.p, self.options)
        
        # Track initial best (will be updated as evaluations come in)
        self._best_obj = np.inf

    # -----------------------------
    # CLASS METHOD: OPTIMIZE
    # -----------------------------
    @classmethod
    def optimize(cls, x0, oracle, p=None, options=None):
        opt = cls(x0, oracle, p, options)
        opt.model.delta = opt.options['tr_delta']
        opt.print_iteration()
        while True:
            opt.info['iteration'] += 1
            opt.info["nfeval"] = oracle.get_evaluation_count()
            print("Iteration: {}".format(opt.info['iteration']))
            print("---------------")
            print("Center:")
            print(opt.samp.center)
            print("Center Value")
            print(opt.samp.fc)
            print("Linear Interpolation Set:")
            print(opt.samp.Y.points)
            print("Linear Function Values:")
            print(opt.samp.Y.values)
            print("Hessian Interpolation Set:")
            print(opt.samp.Z.points)
            print("Hessian Function Values:")
            print(opt.samp.Z.values)
            print("---------------")
            print(opt.oracle.evaluation_count)
            opt.info["iteration_info"].append((oracle.get_evaluation_count, opt.samp.center, opt.samp.fc))
            
            # Build Model
            opt.model.fit_full_quadratic(opt.samp)
            step, opt.info['predicted_decrease'] = opt.model.minimize(opt.samp)
            print("Predicted Decrease")
            print(opt.info['predicted_decrease'])

            # print("MODEL INFO")
            # print("--------------")
            # print(opt.model.g)
            # print(opt.model.H)
            # print(np.linalg.norm(opt.model.H))
            # Evluate step
            f_new = oracle(opt.samp.center + step)
            rho = (opt.samp.fc - f_new) / opt.info['predicted_decrease']

            # Successful Step
            if rho >= opt.options['tr_toaccept'] and \
                np.linalg.norm(opt.model.g) >= opt.options['tr_toexpand'] * opt.model.delta:
                # Logging
                opt._success = 1
                opt.info['success'] += 1
                
                # Update Sets
                y_max_idx, y_furthest = opt.samp.Y.get_furthest(step)
                z_max_idx, z_furthest = opt.samp.Z.get_furthest(step)
                if z_max_idx and np.linalg.norm(y_furthest) > np.linalg.norm(z_furthest):
                    opt.samp.Y.delete_point(y_max_idx)
                    opt.samp.Y.append_origin(value=opt.samp.fc)
                    print("one")
                elif opt.samp.mZ == (p * (p+1)) // 2:
                    opt.samp.Z.delete_point(z_max_idx)
                    opt.samp.Z.append_origin(value = opt.samp.fc)
                    print("two")
                else:
                    print("three")
                    opt.samp.Z.append_origin(value = opt.samp.fc)

                opt.samp.Y.shift(step)
                opt.samp.Z.shift(step)

                # Update Iterates
                opt.samp.center = opt.samp.center + step
                opt.samp.fc = f_new
                opt.model.delta *= opt.options['tr_expand']


            # Unsuccessful Step
            else:
                opt._success = 0
                improve_flag = 0
                kicked_list = []

                # Replace a far point
                far_idx, far_point = opt.samp.Y.get_furthest()
                if np.linalg.norm(far_point) > opt.model.delta + 1e-4:
                    kicked_list.append(opt.samp.Y[far_idx])
                    opt.samp.Y.delete_point(far_idx)
                    opt.samp.Y.add_point(step, value=f_new)
                    improve_flag = 1
                    print()
                    print("GC: Replace far point")
                    print()
                else:
                    # Lagrange polynomial 
                    L_coefs = opt.samp.get_lin_lagrange_coef()

                    # GC by replacing a point with Lag-poly
                    lagrange_at_s = np.abs( step @ L_coefs )
                    idx = np.argmax(lagrange_at_s)

                    if lagrange_at_s[idx] > opt.options['big_lambda']:
                        kicked_list.append(opt.samp.Y[idx])
                        opt.samp.Y.delete_point(idx)
                        opt.samp.Y.add_point(step, value=f_new)
                        print()
                        print("GC: Replace by Lagrange poly")
                        print()
                    else:
                        kicked_list.append((step, f_new))

                    # GC by replacing a bad point
                    norms = np.linalg.norm(L_coefs, axis=0)
                    idx = np.argmax(norms)
                    direction = L_coefs[:, idx]
                    lag_step = (direction / np.linalg.norm(direction)) * opt.model.delta
                    
                    if direction @ lag_step > opt.options['big_lambda']:
                        kicked_list.append(opt.samp.Y[idx])
                        opt.samp.Y.delete_point(idx)
                        lag_value = oracle(opt.samp.center + lag_step)
                        opt.samp.Y.add_point(lag_step, value = lag_value)
                        improve_flag = 1
                        print()
                        print("GC: Replace bad point by Lagrange poly")
                        print()

                for kicked_point, kicked_val in kicked_list:
                    if opt.samp.mZ < (p * (p+1))// 2:
                        opt.samp.Z.add_point(kicked_point, kicked_val)
                    else:
                        far_idx, far_point = opt.samp.Z.get_furthest()
                        if np.linalg.norm(far_point) > np.linalg.norm(kicked_point):
                            opt.samp.Z.delete_point(far_idx)
                            opt.samp.Z.add_point(kicked_point, kicked_val)
                        
                if improve_flag == 0:    
                    print("SHRINK")
                    print("SHRINK")
                    opt.model.delta *= opt.options["tr_shrink"]


            opt.print_iteration(rho)

            # Check Stopping Criteria
            if opt._stop():
                break
            print()
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print()
        # idx = np.nanargmin(opt.samp.Y.values)
        # return opt.samp.Y.points[idx], opt.samp.Y.values[idx], opt.info

        # idx = np.nanargmin(opt.samp.Y.values)
        return opt.samp.center, opt.samp.fc, opt.info
        
        
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
                          self.samp.fc,
                          self.model.delta,
                          self.info['predicted_decrease']                          ))

        return STOP

    # -----------------------------
    # OUTPUT STREAM
    # -----------------------------

    def print_iteration(self, rho=None):
        if self.options['verbosity'] < 2:
            return
        if self.info['iteration'] == 0:

            print("Initialization: Iteration 0")
            print("---------------")
            print("Initial Center:")
            print(self.samp.center)
            print("Initial Center Value")
            print(self.samp.fc)
            print("Initial Linear Interpolation Set:")
            print(self.samp.Y.points)
            print("Initial Linear Function Values:")
            print(self.samp.Y.values)
            print("Initial Hessian Interpolation Set:")
            print(self.samp.Z.points)
            print("Initial Hessian Function Values:")
            print(self.samp.Z.values)
            print("---------------")
            print(self.oracle.evaluation_count)
            print("")

            print("\n Iteration Report ")
            print('|  iter |suc|  objective  | TR_radius |    rho    | m  |')
            print("| {:5d} |---| {:11.5e} | {:9.6f} | --------- | {} "
                  .format(self.info['iteration'],
                          self.samp.fc,
                          self.model.delta,
                          self.samp.mTotal))
        else:
            print("| {:5d} | {} | {:11.5e} | {:9.6f} | {:9.6f} | {} |"
                  .format(self.info['iteration'],
                          self._success,
                          self.samp.fc,
                          self.model.delta,
                          rho,
                          self.samp.mTotal))
