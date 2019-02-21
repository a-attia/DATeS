
#
# ########################################################################################## #
#                                                                                            #
#   DATeS: Data Assimilation Testing Suite.                                                  #
#                                                                                            #
#   Copyright (C) 2016  A. Sandu, A. Attia, P. Tranquilli, S.R. Glandon,                     #
#   M. Narayanamurthi, A. Sarshar, Computational Science Laboratory (CSL), Virginia Tech.    #
#                                                                                            #
#   Website: http://csl.cs.vt.edu/                                                           #
#   Phone: 540-231-6186                                                                      #
#                                                                                            #
#   This program is subject to the terms of the Virginia Tech Non-Commercial/Commercial      #
#   License. Using the software constitutes an implicit agreement with the terms of the      #
#   license. You should have received a copy of the Virginia Tech Non-Commercial License     #
#   with this program; if not, please contact the computational Science Laboratory to        #
#   obtain it.                                                                               #
#                                                                                            #
# ########################################################################################## #
#


"""
    This module contains implementations of EnKF flavors with adaptive components; e.g. adaptive localization or inflation

    EnKF_OED_Inflation:
    ------------------------
        A class implementing the deterministic ensemble Kalman Filter [Sakov, P. and Bertino, L. 2010], with adaptive inflation.
        The strategy is to use OED-based approach with A-optimality criterion!


    Remarks:
    --------
        - This is research-in-progress with ultimate goal of online adaptive localization and inflation

    References
    ------------
    -

"""

import sys
import scipy.linalg as slinalg

from EnKF import *
import scipy.optimize as optimize

from scipy.optimize import minimize

import re

try:
    import cpickle
except:
    import cPickle as pickle



class EnKF_OED_Inflation(DEnKF):
    """
    A class implementing the deterministic ensemble Kalman Filter [Sakov, P. and Bertino, L. 2010], with adaptive inflation.
    The strategy is to use OED-based approach with A-optimality criterion!

    Args:
        filter_configs:  dict,
            A dictionary containing EnKF filter configurations.
            Supported configuarations:
            --------------------------
                * model (default None):  model object
                * filter_name (default None): string containing name of the filter; used for output.
                * hybrid_background_coeff (default 0.0): used when hybrid background errors are used,
                    this multiplies the modeled Background error covariance matrix.
                    Will be effective only if the covariance localization is done in full space, otherwise it is meaningless.
                * inflation_factor (default 1.09): covariance inflation factor
                * obs_covariance_scaling_factor (default 1): observation covariance scaling factor (rfactor2 in Sakov's Code)
                * obs_adaptive_prescreening_factor (default None): Added adaptive observation prescreening (kfactor in Sakov's Code)
                * localize_covariances (default True): bool,
                    apply covariance localization to ensemble covariances.
                    This is done by default using Shur product, and is requested from the model.
                    This is likely to be updated in future releases to be carried out here with more options.
                * localization_method: method used to carry out filter localization to remove sporious correlations.
                  Three localization methods are supported:
                     - 'covariance_filtering': involves modification of the update equations by replacing
                        the state error covariance by its element-wise product with some distance-dependent
                        correlation matrix. This is done by localizing covariances projected in the observation space.
                     -  'local_analysis': uses a local approximation of the forecast covariance for updating
                        a state vector element, calculated by building a local window around this element.
                * localization_radius (default np.infty): radius of influence of covariance decorrelation
                * localization_function ('gaspari-cohn'): the covariance localization function
                    'gaspari-cohn', 'gauss', etc.
                    These functions has to be supported by the model to be used here.
                * prior_distribution (default 'gaussian'): prior probability distribution;
                    this shoule be either 'gaussian' or 'GMM'.
                    - 'Gaussian': the prior distribution is approximated based on the forecast ensemble,
                    - 'GMM': the prior distribution is approximated by a GMM constructed using EM algorithm
                             using the forecast ensemble.
                * gmm_prior_settings: dict,
                    This is a configurations dictionary of the GMM approximation to the prior.
                    This will be used only if the prior is assumed to be non-Gaussian, and better estimate is needed,
                    i.e. it is used only if 'prior_distribution' is set to 'GMM'.
                    The implementation in this case follows the cluster EnKF described by [cite].
                    The configurations supported are:
                       - clustering_model:
                       - cov_type:
                       - localize_covariances:
                       - localization_radius:
                       - localization_function:
                       - inf_criteria:
                       - number_of_components:
                       - min_number_of_components:
                       - max_number_of_components:
                       - min_number_of_points_per_component:
                       - invert_uncertainty_param:
                * ensemble_size (default None): size of the ensemble; this has to be set e.g. in a driver
                * analysis_ensemble (default None): a placeholder of the analysis ensemble.
                    All ensembles are represented by list of model.state_vector objects
                * analysis_state (default None): model.state_vector object containing the analysis state.

                    All ensembles are represented by list of model.state_vector objects
                * forecast_state (default None): model.state_vector object containing the forecast state.
                * filter_statistics: dict,
                    A dictionary containing updatable filter statistics. This will be updated by the filter.

        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
            --------------------------
                * scr_output (default False): Output results to screen on/off switch
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default 'Assimilation_Results'): relative path (to DATeS root directory)
                    of the directory to output results in

                * filter_statistics_dir (default 'Filter_Statistics'): directory where filter statistics (such as RMSE, ESS,...) are saved
                * model_states_dir (default 'Model_States_Repository'): directory where model-states-like objects (including ensemble) are saved
                * observations_dir (default 'Observations_Rpository'): directory where observations and observations operators are saved
                * file_output_moment_only (default True): output moments of the ensembles (e.g. ensemble mean) or the full ensembles.
                * file_output_moment_name (default 'mean'): Name of the first order moment to save.
                    used only if file_output_moment_only is True
                * file_output_file_name_prefix (default 'EnKF_results'): name/prefix of output file
                * file_output_file_format (default 'mat'): file ouput format.
                    Supported formats:
                        - 'mat': matlab matrix files,
                        - 'pickle': python pickled objects,
                        - 'txt' or 'ascii': text files
                * file_output_separate_files (default True): save all results to a single or multiple files

    Returns:

    """
    #
    _def_local_filter_configs = dict(model=None,
                                     filter_name="EnKF_OED_Inflation",
                                     forecast_inflation_factor=1.05,  # applied to forecast ensemble; taken as initial value for the optimizer
                                     inflation_factor_bounds=(1+1e-9, 2.0),  # ounds on alpha; this +1 is is the bounds put on the inflation factor 1+alpha
                                     reset_inflation_factor=True,
                                     inflation_design_penalty=0.05,  # penalty of the regularization parameter
                                     inflation_factor=1.0,  # make sure analysis ensemble is not inflated
                                     OED_criterion='A',
                                     regularization_norm='l2',  # L1, L2 are supported
                                     moving_average_radius=0,
                                     optimizer_configs=dict(method='SLSQP',  # 'COBYLA', 'SLSQP'
                                                            maxiter=10000,
                                                            maxfun=5000,
                                                            tol=1e-9,
                                                            reltol=1e-7,
                                                            pgtol=1e-07,
                                                            epsilon=1e-10,
                                                            factr=10000000.0,
                                                            disp=1,
                                                            maxls=50,
                                                            iprint=-1
                                                            )
                                     )
    _local_def_output_configs = dict()
    #
    __round_num_digits = 4  # round optimal solution of inflation parameters

    #
    def __init__(self, filter_configs=None, output_configs=None):
        #
        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, EnKF_OED_Inflation._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, EnKF_OED_Inflation._local_def_output_configs)
        #
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(filter_configs=filter_configs, output_configs=output_configs)
        else:
            # old-stype class
            super(EnKF_OED_Inflation, self).__init__(filter_configs=filter_configs, output_configs=output_configs)
        #
        # Additional class-specific configurations, e.g. for adaptive inflation:
        # self.__shifted_proj_cov_inv = self.model.observation_matrix()  # an empty place holder for (R + H B H^T )^{-1}
        # self.__original_forecast_state = None  # this creates a copy of the forecast ensemble

        orig_f = self.filter_configs['forecast_inflation_factor']
        if np.isscalar(orig_f):
            self.__original_inflation_factor = orig_f
        else:
            self.__original_inflation_factor = orig_f.copy()
        #
        default_bounds = EnKF_OED_Inflation._def_local_filter_configs['inflation_factor_bounds']
        inflation_factor_bounds = self.filter_configs['inflation_factor_bounds']
        try:
            lb, ub = inflation_factor_bounds
            if None not in [lb, ub]:
                if lb >= ub:
                    print("The bounds must be ranked increasingly with upper bound < lower bound!")
                    raise ValueError
        except:
            print("Failed to get the optimizer bounds on the inflation factor; using default values % " % str(default_bounds))
            self.filter_configs.update({'inflation_factor_bounds':default_bounds})

        # set numpy random seed, and preserve current state:
        try:
            random_seed = self.filter_configs['random_seed']
        except KeyError:
            random_seed = None

        if random_seed is None:
            self._random_state = None
        else:
            self._random_state = np.random.get_state()
            #
        np.random.seed(random_seed)

            #
        self.__initialized = True
        #

    #
    def analysis(self, all_to_numpy=True):
        """
        Analysis step:

        Args:
            all_to_numpy (default False): bool,
                convert all data structures to Numpy and re-place results into target structures only in the end.

        Returns:
            None. Only self.filter_configs is updated.

        """
        model = self.filter_configs['model']

        # make sure the original forecast ensemble and background state are packedup
        original_forecast_ensemble = [ v.copy() for v in self.filter_configs['forecast_ensemble'] ]  # this creates a copy of the
        self.filter_configs.update({'original_forecast_ensemble': original_forecast_ensemble})
        self.filter_configs.update({'original_forecast_state': utility.ensemble_mean(original_forecast_ensemble)})

        state_size = self.model.state_size()

        #
        # 1- Initilize an inflation factor vector
        x0 = self.filter_configs['forecast_inflation_factor']

        if x0 is None:
            x0 = np.empty(state_size)  # default initial inflation factors vector
            x0[:] = 1.05
        elif np.isscalar(x0):
            if x0 <=1:
                x0 = 1.05
            x0 = np.ones(state_size) * float(x0)
        else:
            x0 = np.array(x0).flatten()
            if (x0 <=1).any():
                print("The inflation is set to 1 in all entries! No Inflation is needed with this solution;")
                x0[np.where(x0<=1)] = 1.05
            pass

        #
        # 2- Create an optimizer
        obj_fun = lambda x: self.obj_inflation_fun_value(x)
        obj_grad = lambda x: self.obj_inflation_fun_gradient(x)
        callback_fun = lambda x: self._inflation_iter_callback(x)
        #
        optimizer_configs = self.filter_configs['optimizer_configs']
        lb, ub = self.filter_configs['inflation_factor_bounds']
        bounds = [(lb, ub)] * state_size  # can be passed in the configurations dictionary; TODO.

        #
        if False:
            try:
                opt_x, f, d = optimize.fmin_l_bfgs_b(obj_fun,
                                                     x0,
                                                     fprime=obj_grad,
                                                     bounds=bounds,
                                                     m=50,
                                                     factr=float(optimizer_configs['factr']),
                                                     pgtol=optimizer_configs['pgtol'],
                                                     epsilon=optimizer_configs['epsilon'],
                                                     iprint=optimizer_configs['iprint'],
                                                     maxfun=optimizer_configs['maxfun'],
                                                     maxiter=optimizer_configs['maxiter'],
                                                    #  maxls=optimizer_configs['maxls'],
                                                     disp=optimizer_configs['disp'],
                                                     callback=callback_fun
                                                     )
            except ValueError:
                opt_x = x0
                f = 0
                d = {'warnflag': -1}
            flag = d['warnflag']
            if flag == 0:
                msg = 'Converged: %s' % d['task']
                success = True
            elif flag in [1, 2]:
                msg = d['task']
                success = False
            opt_res = {'success':success,
                       'status': flag,
                       'message': msg,
                       'nfev': d['funcalls'],
                       'nit': d['nit']
                      }
        else:
            #
            opts = {'maxiter':10000,
                    'ftol':1e-06,
                    'gtol':1e-05,
                    'xtol':1e-05
                    }
            if False:
                const=()
            else:
                const = ({'type': 'ineq',
                          'fun': lambda x: self.obj_inflation_fun_value(x) - 1e-02,
                          # 'fun': obj_fun,
                          'jac': obj_grad})
            method = self.filter_configs['optimizer_configs']['method']
            try:
                res = minimize(obj_fun, x0, method=method, jac=obj_grad, hess=None, hessp=None, bounds=bounds, constraints=const, tol=1e-08, callback=callback_fun, options=opts)
                # print(res)
                opt_x = res.x
                f = res.fun
                d = {'warnflag': int(not res.success)}
                opt_res = {'success':res.success, 'status':res.status, 'message':res.message, 'nfev':res.nfev, 'njev':res.njev, 'nit':res.nit}
            except:
                opt_x = x0
                f = 0
                d = {'warnflag': 1}
                opt_res = {'success':0, 'status':1, 'message':'Optimizer failed', 'nfev':0, 'njev':0, 'nit':0}

        # print(opt_x)
        orig_opt_x = opt_x.copy()

        # apply a moving average to the optimal solution
        model_name = model._model_name
        try:
            if re.match(r'\Alorenz', model_name, re.IGNORECASE):
                periodic = True
            elif re.match(r'\AQ(_|-)*G', model_name, re.IGNORECASE):
                periodic = False
            else:
                print("The model you selected ['%s'] is not supported" % model_name)
                periodic = True
        except:
            periodic = False

        moving_average_radius = self.filter_configs['moving_average_radius']
        r = min(moving_average_radius, state_size/2-1)
        # print("Original Solution: ", orig_opt_x)
        # print("Smoothing with periodic bc set to: %s, and moving average radius = %f" % (periodic, r))
        if r > 0:
            opt_x[:] = utility.moving_average(opt_x, radius=r, periodic=periodic)
            # print("Smoothed Solution: ", opt_x)

        opt_x = np.round(opt_x, self.__round_num_digits)
        if False:
            max_bnd = bounds[0][1]
            min_bnd = bounds[0][0]
            opt_x[np.where(opt_x>max_bnd)[0]] = max_bnd
            opt_x[np.where(opt_x<min_bnd)[0]] = min_bnd
        if self._verbose:
            print("Optimal solution: ", opt_x)
            print("res: ", res)


        #
        if self._verbose:
             # This is to be rewritten appropriately after debugging
            sepp = "\n%s\n" % ("{|}"*50)
            print(sepp + "OED-Inflation RESULTS: %s" % '-'*15)
            print("optimal inflation_fac:", opt_x)
            print("Original optimal inflation_fac: ", orig_opt_x)
            print("Minimum inflation factor entry:", opt_x.min())
            print("Maximum inflation factor entry:", opt_x.max())
            print("Average inflation factor:", np.mean(opt_x))
            print("Standard Deviation of inflation factor entries:", np.std(opt_x))
            print(" >> Minimum Objective (posterior-covariance trace): ", f)
            print("flags: ", d)
            print(sepp)
        #

        # Save the results, and calculate the results' statistics
        failed = d['warnflag']  # 0 flag --> converged
        if failed:
            print(d)
            self.filter_configs['analysis_state'] = None
            sep = "/^\\"*30
            print(sep + "\n\n\tThe Optimizer algorithm Miserably failed!\n\n" + sep)
            # raise ValueError
            pass

        # add regularization term (starting with L1 norm here):
        alpha = self.filter_configs['inflation_design_penalty']
        regularization_norm = self.filter_configs['regularization_norm']  # get regularization norm
        post_trace = f
        if re.match(r"\Al(_|-)*1\Z", regularization_norm, re.IGNORECASE):
            if alpha !=0:
                regularizer = alpha * np.sum(opt_x - 1)
            else:
                regularizer = 0.0
            post_trace += regularizer
        elif re.match(r"\Al(_|-)*2\Z", regularization_norm, re.IGNORECASE):
            if alpha !=0:
                regularizer = alpha * np.linalg.norm(opt_x-1, 2)
            else:
                regularizer = 0.0
            pass
            post_trace += regularizer
        else:
            print("Unrecognized norm %s " % regularization_norm)
            raise ValueError
        self.filter_configs.update({'inflation_opt_results':(orig_opt_x, opt_x, f, d, post_trace, opt_res)})


        # Reset forecast information
        self.filter_configs['forecast_ensemble'] = self.filter_configs['original_forecast_ensemble']
        self.filter_configs['forecast_state'] = self.filter_configs['original_forecast_state']
        #
        # Analysis with optimal inflation factor
        self.filter_configs['forecast_inflation_factor'] = opt_x
        self.filter_configs.update({'optimal_inflation_factor': opt_x})
        #
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            res = super().analysis(all_to_numpy=all_to_numpy)
        else:
            # old-stype class
            res = super(EnKF_OED_Inflation, self).analysis(all_to_numpy=all_to_numpy)

        if self.filter_configs['reset_inflation_factor']:
            self.filter_configs['forecast_inflation_factor'] = self.__original_inflation_factor

        return res
        #

    #
    def obj_inflation_fun_value(self, inflation_fac):
        """
        Evaluate the value of the A-optimal objective given an inflation factor

        Args:
            inflation_fac:

        Returns:
            objective_value: the value of the A-optimality objective function

        """
        model = self.model
        observation_size = model.observation_size()

        # Check the inflation factor
        if np.isscalar(inflation_fac):
            _scalar = True
            _inflation_fac = float(inflation_fac)
        elif utility.isiterable(inflation_fac):
            _inflation_fac = np.asarray(inflation_fac)
            if _inflation_fac.size == 1:  # Extract inflation factor if it's a single value wrappend in an iterable
                for i in xrange(_inflation_fac.ndim):
                    _inflation_fac = _inflation_fac[i]
                _scalar = True
            else:
                _scalar = False
        else:
            print("inflation factor has to be a scalar or an iterable of length equal to model.state_size")
            raise AssertionError

        #
        # Retrieve forecast information:
        forecast_ensemble = list(self.filter_configs['original_forecast_ensemble'])
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        self.filter_configs['forecast_inflation_factor'] = _inflation_fac
        ensemble_size = len(forecast_ensemble)

        #
        OED_criterion = self.filter_configs['OED_criterion']
        if re.match('\AA(-|_)*(opt)*\Z', OED_criterion, re.IGNORECASE):

            # OneD array with ensemble-based forecast variances
            forecast_variances = utility.ensemble_variances(forecast_ensemble, sample_based=True, ddof=1.0, return_state_vector=False)
            first_term = np.sum( forecast_variances * _inflation_fac )
            # DEBUG: First term checks...  (DONE) ...

            # Get forecast anomalies, and compute the scaled+inflated forecast ensemble of forecast anomalies:
            Xf_tilde = utility.ensemble_anomalies(forecast_ensemble, scale=True, ddof=1.0, in_place=False, inflation_factor=np.sqrt(_inflation_fac), model=model)
            Xf_tilde_np = utility.ensemble_to_np_array(Xf_tilde, state_as_col=True, model=model)

            # Xf_tilde muyltipled (from left) by the observation operator jacobian (consider preallocation)
            H_Xf_tilde_np = np.empty((observation_size, ensemble_size))
            for i in xrange(ensemble_size):
                H_Xf_tilde_np[:, i] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=Xf_tilde[i]).get_numpy_array()

            # First approach:
            # --------------------
            # What follows can be optimized by solving linear systems with columns of H_Xf_tilde_np: TODO...
            # HBHt = np.dot(H_Xf_tilde_np, H_Xf_tilde_np.T)
            HBHt = self._obs_proj_covariance(Xf_tilde, anomalies=True, scale=False, return_np=True)  # check

            G = HBHt + model.observation_error_model.R.get_numpy_array()
            # Calculate SVD of HBHt; DONOT INVERT; BAD Condition number!
            # G_inv = slinalg.inv(G)
            U, s, Vh = np.linalg.svd(G)
            G_inv = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))  # check

            ens_proj_Xf = np.dot(Xf_tilde_np.T, Xf_tilde_np)

            S1 = np.dot(H_Xf_tilde_np, ens_proj_Xf)
            S2 = np.dot(S1, H_Xf_tilde_np.T)
            S3 = np.dot(G_inv, S2)
            second_term = np.trace(S3)
            # DEBUG: second term checks...  (DONE) ...

            # trace of posterior error covariance matrix
            objective_value = first_term - second_term
            if self._verbose:
                print("\n Objective Function: first term=%f, second_term=%f, objective=%f \n\n" % (first_term, second_term, objective_value))

            # Second(optimized) approach:
            # ----------------------------
            pass
            #

            #
            # add regularization term (starting with L1 norm here):
            alpha = self.filter_configs['inflation_design_penalty']
            regularization_norm = self.filter_configs['regularization_norm']  # get regularization norm
            if re.match(r"\Al(_|-)*1\Z", regularization_norm, re.IGNORECASE):
                if alpha != 0:
                    if _scalar:
                        regularizer = alpha * state_size * (_inflation_fac - 1)
                    else:
                        regularizer = alpha * np.sum(_inflation_fac - 1)
                else:
                    regularizer = 0.0
                objective_value -= regularizer
            elif re.match(r"\Al(_|-)*2\Z", regularization_norm, re.IGNORECASE):
                regularizer = alpha * np.linalg.norm(_inflation_fac-1, 2)
                objective_value -= regularizer
            else:
                print("Unrecognized norm %s " % regularization_norm)
                raise ValueError

            if self._verbose:
                sep = "\n%s\n" % ("*"*80)
                print(sep)
                print("Inside 'obj_inflation_fun_value'. Passed inflation factor is: ", _inflation_fac)
                print("Regularizer: ", regularizer)
                print("First Term: ", first_term)
                print("Second Term: ", second_term)
                print("objective_value:", objective_value)
                print(sep)

        elif re.match('\AD(-|_)*(opt)*\Z', OED_criterion, re.IGNORECASE):
            #
            raise NotImplementedError("D optimality is not yet implemented...")
            pass

        else:
            print("Unrecognized Adaptive inflation criterion [%s]!" % OED_criterion)
            raise ValueError

        #
        return objective_value
        #


    #
    def obj_inflation_fun_gradient(self, inflation_fac, FD_Validation=False, FD_eps=1e-7, FD_central=True):
        """
        Evaluate the gradient of the A-optimal objective given an inflation factor

        Args:
            inflation_fac:

        Returns:
            grad:

        """
        # Check the inflation factor
        if np.isscalar(inflation_fac):
            _scalar = True
            _inflation_fac = float(inflation_fac)
        elif utility.isiterable(inflation_fac):
            _inflation_fac = np.array(inflation_fac).flatten()
            if inflation_fac.size == 1:  # Extract inflation factor if it's a single value wrappend in an iterable
                for i in xrange(_inflation_fac.ndim):
                    _inflation_fac = _inflation_fac[i]
                _scalar = True
            else:
                _scalar = False
        else:
            print("inflation factor has to be a scalar or an iterable of length equal to model.state_size")
            raise AssertionError

        # Retrieve forecast information:
        # self.filter_configs['forecast_ensemble'] = list(self.__original_forecast_ensemble)
        #
        # forecast_ensemble = self.filter_configs['forecast_ensemble']
        forecast_ensemble = list(self.filter_configs['original_forecast_ensemble'])
        forecast_state = utility.ensemble_mean(forecast_ensemble)

        self.filter_configs['forecast_state'] = forecast_state.copy()
        self.filter_configs['forecast_inflation_factor'] = _inflation_fac

        #
        OED_criterion = self.filter_configs['OED_criterion']
        if re.match('\AA(-|_)*(opt)*\Z', OED_criterion, re.IGNORECASE):
            # Calculate the gradient of the posterior-covariance trace:
            grad = self._infl_post_cov_trace_gradient(inflation_fac=_inflation_fac, Xf=forecast_ensemble)
            #
            # add regularization term (starting with L1 norm here):
            alpha = self.filter_configs['inflation_design_penalty']
            regularization_norm = self.filter_configs['regularization_norm']  # get regularization norm
            if re.match(r"\Al(_|-)*1\Z", regularization_norm, re.IGNORECASE):
                regularizer = alpha
                if alpha != 0:
                    grad -= regularizer
            elif re.match(r"\Al(_|-)*2\Z", regularization_norm, re.IGNORECASE):
                if alpha != 0:
                    if _scalar:
                        regularizer = alpha * np.sqrt(state_size)
                    else:
                        regularizer = (float(alpha)/np.linalg.norm(_inflation_fac-1, 2)) * (_inflation_fac-1)
                    grad -= regularizer
            else:
                print("Unrecognized norm %s " % regularization_norm)
                raise ValueError
            #

            if self._verbose:
                sep = "\n%s\n" % ("*"*80)
                print(sep)
                print("Inside 'obj_inflation_fun_gradient'. Passed inflation factor is: ", _inflation_fac)
                print("\nGradient is: ", grad)
                print(sep)

        elif re.match('\AD(-|_)*(opt)*\Z', OED_criterion, re.IGNORECASE):
            raise NotImplementedError("TODO")
            pass

        else:
            print("Unrecognized Adaptive inflation criterion [%s]!" % OED_criterion)
            raise ValueError

        if FD_Validation:
            self.validate_gradient(_inflation_fac, grad, objective=self.obj_inflation_fun_value, FD_eps=FD_eps, FD_central=FD_central)

        #
        return grad
        #

    #
    def _infl_post_cov_trace_gradient(self, inflation_fac, Xf):
        """
        Given An analysis ensemble of model states, calculate the derivative of the trace with respect to inflation parameters applied to the forecast ensemble

        Args:
            inflation_fac: a vector (any iterable will work) of size equal to state vector with inflation factors;
                it can also be a scalar, if spcae-independent inflation is carried out.
            Xf: Forecast ensemble

        Returns:
            grad: Numpy array; gradient of the posterior covariance trace with respect to the inflation factors applied to the forecast ensemble

        """
        assert isinstance(Xf, list), "Passed FORECAST ensemble Xf must be a list of model states!"

        model = self.model
        state_size = model.state_size()

        # Check ensemble size:
        ens_size = len(Xf)
        if ens_size <= 1 :
            print("Ensemble(s) must contain at least one state vector; ensemble passed is of length %d?!" % ens_size)
            raise ValueError

        # Check the inflation factor:
        if np.isscalar(inflation_fac):
            _scalar = True
            _inflation_fac = float(inflation_fac)
        elif utility.isiterable(inflation_fac):
            _inflation_fac = np.asarray(inflation_fac)
            if inflation_fac.size == 1:  # Extract inflation factor if it's a single value wrappend in an iterable
                for i in xrange(_inflation_fac.ndim):
                    _inflation_fac = _inflation_fac[i]
                _scalar = True
            else:
                _scalar = False
        else:
            print("inflation factor has to be a scalar or an iterable of length equal to model.state_size")
            raise AssertionError

        #
        OED_criterion = self.filter_configs['OED_criterion']
        if re.match('\AA(-|_)*(opt)*\Z', OED_criterion, re.IGNORECASE):
            #
            forecast_state = utility.ensemble_mean(Xf)

            # Initialize the gradient:
            if _scalar:
                state_size = 1
            else:
                state_size = _inflation_fac.size
                #

            # get forecast and analysis ensemble anomalies
            Xf_prime = utility.ensemble_anomalies(Xf, scale=True, ddof=1.0, in_place=False, inflation_factor=np.sqrt(_inflation_fac), model=model)

            # Use forecast information only:
            HBHt = self._obs_proj_covariance(Xf_prime, anomalies=True, scale=False, return_np=True)
            G = HBHt + model.observation_error_model.R.get_numpy_array()
            # Calculate SVD of HBHt; DONOT INVERT; BAD Condition number!
            # G_inv = slinalg.inv(G)
            U, s, Vh = np.linalg.svd(G)
            G_inv = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))

            def _Ht_Ginv_H_vec(state):  # Q * state
                # this is internal to multiply H^T G^{-1} H by a vector x
                #
                Hx = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=state)  # x -> H x
                G_inv_Hx = np.dot(G_inv, Hx.get_numpy_array())  # TODO: remove copying!
                Hx[:] = G_inv_Hx[:]
                out = model.observation_operator_Jacobian_T_prod_vec(in_state=forecast_state, observation=Hx)
                #
                return out
                #


            # Initialize, and fill-in the gradient:
            grad = np.zeros(state_size)
            # grad2 = np.zeros_like(grad)

            e_i = model.state_vector()
            #
            for i in xrange(state_size):  # TODO: Remove non-necessary copying after debugging...
                e_i[:] = 0.0
                e_i[i] = 1.0  # ith cardinal vector in R^n; n = state_size

                z1 = self._cov_prod_cardinal(Xf_prime, cardinal=i)  # z1 = B_tilde * e_i

                z2 = _Ht_Ginv_H_vec(self._cov_prod_vec(Xf_prime, in_state=z1))  # z2 = Q B_tilde z1

                z3 = self._cov_prod_vec(Xf_prime, in_state=_Ht_Ginv_H_vec(z1))  # z3 = B_tilde Q ei

                z4 = _Ht_Ginv_H_vec(self._cov_prod_vec(Xf_prime, in_state=z3))  # z4 = Ht Q B_tilde z3

                if _scalar:
                    scl = 1.0 / _inflation_fac
                else:
                    scl = 1.0 / _inflation_fac[i]

                grad[i] = scl * ( z1[i] - z2[i] - z3[i] + z4[i] )


                if False:
                    # second one
                    zz1 = self._cov_prod_cardinal(Xf_prime, cardinal=i)
                    Qe = _Ht_Ginv_H_vec(e_i)
                    zz2 = self._cov_prod_vec(Xf_prime, in_state=self._cov_prod_vec(Xf_prime, in_state=Qe))
                    zz3 = self._cov_prod_vec(Xf_prime, in_state=_Ht_Ginv_H_vec(zz1))
                    zz4 = self._cov_prod_vec(Xf_prime, in_state=_Ht_Ginv_H_vec(zz2))

                    grad2[i] = scl * (zz1[i] - zz2[i] - zz3[i] + zz4[i])

                    print("grad1, grad2...")
                    print("grad1:", grad)
                    print("grad2: ", grad2)
                    print("grad1 == grad2:", np.isclose(grad, grad2).all())

        elif re.match('\AD(-|_)*(opt)*\Z', OED_criterion, re.IGNORECASE):
            pass

        else:
            print("Unrecognized Adaptive inflation criterion [%s]!" % OED_criterion)
            raise ValueError

        if self._verbose:
            print("Gradient info:")
            print("Inflation factor(s):", _inflation_fac)
            print("Gradient:", grad)

        return grad


    def _obs_proj_covariance(self, ensemble, anomalies=False, scale=True, ddof=1.0, return_np=False):
        """
        Given an ensemble, return the covariance matrix projected into the obsevations space,
        that is, given Xf, return H Xf Xf^T H^T = (H Xf) (H Xf)^T,

        Args:
            ensemble: list of model states
            anomalies: if True, the passed ensemble, is actually an ensemble of anomalies (Xf - mean(Xf)) rather than the original ensemble
            scale: if anomalies is used, this multiplies anomalies (if not already scaled) by 1/sqrt(ensemble size - ddof)
            ddof: used if anomalies is True, and scale is True as indicated above
            return_np: return the result as a Numpy array rather than a model state_matrix object

        Returns:
            proj_cov: The ensemble-based covariance matrix projected into the obsevations space

        """
        assert isinstance(ensemble, list), "Passed ANALYSIS ensemble Xf must be a list of model states!"

        ens_size = len(ensemble)
        if ens_size <= 1 :
            print("Ensemble must contain at least one state vector; ensemble passed is of length %d?!" % ens_size)
            raise ValueError

        model = self.model
        observation_size = model.observation_size()

        if anomalies:
            if scale:
                scl = np.sqrt(float(ens_size - ddof))
                ens_anomalies = [v.scale(scl, in_place=False) for v in ensemble]
            else:
                ens_anomalies = [v.copy() for v in ensemble]
        else:
            ens_anomalies = utility.ensemble_anomalies(ensemble, scale=True, ddof=1.0, in_place=False, inflation_factor=None, model=model)

        #
        proj_anomalies_np = np.empty((observation_size, ens_size))
        forecast_state = self.filter_configs['forecast_state']
        for i in xrange(ens_size):
            obs = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=ens_anomalies[i])
            proj_anomalies_np[:, i] = obs.get_numpy_array()

        if return_np:
            proj_cov = np.dot(proj_anomalies_np, proj_anomalies_np.T)
        else:
            proj_cov = model.observation_matrix()
            proj_cov[:, :] = proj_anomalies_np.dot(proj_anomalies_np.T)

        return proj_cov
        #


    #
    def validate_gradient(self, x, gradient, objective, FD_eps=1e-7, FD_central=False):
        """
        Use Finite Difference to validate the gradient

        Args:
            x: inflation factor (np array)
            gradient:
            objective: objective function
            FD_eps:
            FD_central:

        """
        #
        # Use finite differences to validate the Gradient (if needed):
        #
        # Finite Difference Validation:
        eps = FD_eps

        # Get local copies:
        grad_np = np.asarray(gradient).copy()
        fd_grad = np.zeros_like(grad_np)

        x_np = np.asarray(x).copy()
        _size = x_np.size

        prtrb = np.zeros(_size)

        sep = "\n"+"~"*80+"\n"
        # print some statistics for monitoring:
        print(sep + "FD Validation of the Gradient" + sep)
        print("  + Maximum gradient entry :", grad_np.max())
        print("  + Minimum gradient entry :", grad_np.min())

        #
        #
        if not FD_central:
            f0 = objective(x_np)
        #
        for i in xrange(_size):
            prtrb[:] = 0.0; prtrb[i] = eps
            #
            if FD_central:
                f0 = objective(x_np- prtrb)
            #
            f1 = objective(x_np + prtrb)

            if FD_central:
                print("f1, f0, eps:", f1, f0, eps)
                fd_grad[i] = (f1-f0)/(2.0*eps)
            else:
                print("f1, f0, eps:", f1, f0, eps)
                fd_grad[i] = (f1-f0)/(eps)

            err = (grad_np[i] - fd_grad[i]) / fd_grad[i]
            print(">>>>Gradient/FD>>>> %4d| Grad = %+8.6e\t FD-Grad = %+8.6e\t Rel-Err = %+8.6e <<<<" % (i, grad_np[i], fd_grad[i], err))
            #
        print("sep" + "Gradient Validation is DONE... Returning" +  sep + "\n")
        return fd_grad



    def _inflation_iter_callback(self, inf_fac):
        """
        """
        # to be called after each iteration of the optimizer
        #
        forecast_time = self.filter_configs['forecast_time']
        analysis_time = self.filter_configs['analysis_time']
        observation_time = self.filter_configs['observation_time']
        #
        reference_state = self.filter_configs['reference_state']
        forecast_state = utility.ensemble_mean(self.filter_configs['forecast_ensemble'])
        analysis_state = self.filter_configs['analysis_state']

        forecast_rmse = utility.calculate_rmse(reference_state, forecast_state)
        if analysis_state is None:
            analysis_rmse = -np.inf
        else:
            analysis_rmse = utility.calculate_rmse(reference_state, analysis_state)

        filter_name = self.filter_configs['filter_name']
        output_line = "Reanalysis\n RMSE Results: Filter: '%s' \n %s \t %s \t %s \t %s \t %s \n" % (filter_name,
                                                                                  'Observation-Time'.rjust(20),
                                                                                  'Forecast-Time'.rjust(20),
                                                                                  'Analysis-Time'.rjust(20),
                                                                                  'Forecast-RMSE'.rjust(20),
                                                                                  'Analysis-RMSE'.rjust(20),
                                                                                  )
        output_line += u" {0:20.14e} \t {1:20.14e} \t {2:20.14e} \t {3:20.14e} \t {4:20.14e} \n".format(observation_time,
                                                                                                       forecast_time,
                                                                                                       analysis_time,
                                                                                                       forecast_rmse,
                                                                                                       analysis_rmse
                                                                                                       )
        if self._verbose:
            sepp = "\n"*2 + "*^="*120 + "\n"*2
            print(sepp + "... Iteration Callback:")
            print(" > Inflation Factor passed by LBFGSB:", inf_fac)
            print(output_line)
            print("... Getting out or iteration callback function... " + sepp)


    @staticmethod
    def cov_trace(ensemble):
        """
        Given an ensemble of model states, calculate the trace of the covariance matrix of this ensemble
        """
        return utility.covariance_trace(ensemble)


    @staticmethod
    def _cov_prod_cardinal(anomalies, cardinal):
        """ Given ensemble of scaled anomalies, multiply the covariance matrix by e_i """
        vec = utility.ensemble_T_dot_state(anomalies, in_state=None, cardinal=cardinal, return_np_array=True)
        out_state = utility.ensemble_dot_vec(anomalies, vec)
        return out_state

    @staticmethod
    def _cov_prod_vec(anomalies, in_state):
        """ Given ensemble of scaled anomalies, multiply the covariance matrix by a given state """
        vec = utility.ensemble_T_dot_state(anomalies, in_state=in_state, cardinal=None, return_np_array=True)
        out_state = utility.ensemble_dot_vec(anomalies, vec)
        return out_state


    def save_cycle_results(self, output_dir=None, cleanup_out_dir=False):
        """
        Save filtering results from the current cycle to file(s).
        Check the output directory first. If the directory does not exist, create it.

        Args:
            output_dir: full path of the directory to save results in

        Returns:
            None

        """
        #
        # TODO: We need to save all inflation factor values over time, and when the localization approach is carried out, we need to do the same.
        # TODO: I will either do hdf5, or pickle
        #
        # Retrieve output configurations
        output_configs = self.output_configs
        file_output = output_configs['file_output']
        if not file_output:
            raise ValueError("The output flag is turned of. The method 'save_cycle_results' is called though!")

        # We are good to go! --> Start preparing directories (if necessary) then save results...
        if output_dir is not None:
            file_output_directory = output_dir
        else:
            file_output_directory = output_configs['file_output_dir']
        # clean-up output directory; this is set to true only if the filter is called once, otherwise filtering_process should handle it.
        if cleanup_out_dir:
            parent_path, out_dir = os.path.split(file_output_directory)
            utility.cleanup_directory(directory_name=out_dir, parent_path=parent_path)
        # check the output sub-directories...
        filter_statistics_dir = os.path.join(file_output_directory, output_configs['filter_statistics_dir'])
        model_states_dir = os.path.join(file_output_directory, output_configs['model_states_dir'])
        observations_dir = os.path.join(file_output_directory, output_configs['observations_dir'])
        file_output_variables = output_configs['file_output_variables']  # I think it's better to remove it from the filter base...

        if not os.path.isdir(filter_statistics_dir):
            os.makedirs(filter_statistics_dir)
        if not os.path.isdir(model_states_dir):
            os.makedirs(model_states_dir)
        if not os.path.isdir(observations_dir):
            os.makedirs(observations_dir)

        # check if results are to be saved to separate files or appended on existing files.
        # This may be overridden if not adequate for some output (such as model states), we will see!
        file_output_separate_files = output_configs['file_output_separate_files']
        # This is useful for saving filter statistics but not model states or observations as models should handle both
        file_output_file_format = output_configs['file_output_file_format'].lower()
        file_output_file_name_prefix = output_configs['file_output_file_name_prefix']  # this is useless!

        # SAVING MODEL STATES (Either Moments Only or Full Ensembles)
        # write cycle configurations:
        model_conf = self.model.get_model_configs()
        utility.write_dicts_to_config_file('setup.dat', file_output_directory,
                                           model_conf, 'Model Configs')
        # get a proper name for the folder (cycle_*) under the model_states_dir path
        suffix = 0
        cycle_prefix = 'cycle_'
        while True:
            cycle_dir = cycle_prefix + str(suffix)
            cycle_states_out_dir = os.path.join( model_states_dir, cycle_dir)  # full path where states will be saved for the current cycle
            if not os.path.isdir(cycle_states_out_dir):
                cycle_observations_out_dir = os.path.join( observations_dir, cycle_dir)
                if os.path.isdir(cycle_observations_out_dir):
                    raise IOError("There is inconsistency problem. Naming mismatch in cycles folders for states and observations!")
                os.makedirs(cycle_states_out_dir)
                os.makedirs(cycle_observations_out_dir)
                break
            else:
                suffix += 1

        # Now we have all directories cleaned-up and ready for outputting.
        output_dir_structure_file = os.path.join(file_output_directory, 'output_dir_structure.txt')
        if not os.path.isfile(output_dir_structure_file):
            # First, we need to save the output paths info to a file to be used later by results' reader
            # print('writing output directory structure to config file \n \t%s \n' % output_dir_structure_file)
            out_dir_tree_structure = dict(file_output_separate_files=file_output_separate_files,
                                          file_output_directory=file_output_directory,
                                          model_states_dir=model_states_dir,
                                          observations_dir=observations_dir,
                                          filter_statistics_dir=filter_statistics_dir,
                                          cycle_prefix=cycle_prefix
                                          )
            utility.write_dicts_to_config_file(file_name='output_dir_structure.txt',
                                               out_dir=file_output_directory,
                                               dicts=out_dir_tree_structure,
                                               sections_headers='out_dir_tree_structure'
                                               )

        #  save states
        file_output_moment_only = output_configs['file_output_moment_only']
        if file_output_moment_only:
            file_output_moment_name = output_configs['file_output_moment_name'].lower()
            if file_output_moment_name in ['mean', 'average']:
                # start outputting ensemble means... (both forecast and analysis of course).
                # save forecast mean
                forecast_state = self.filter_configs['forecast_state']
                self.model.write_state(state=forecast_state, directory=cycle_states_out_dir, file_name='forecast_mean')
                # save analysis mean
                analysis_state = self.filter_configs['analysis_state']
                self.model.write_state(state=analysis_state, directory=cycle_states_out_dir, file_name='analysis_mean')
            else:
                raise ValueError("Unsupported ensemble moment: '%s' !" % (file_output_moment_name))
        else:
            # start outputting the whole ensemble members (both forecast and analysis ensembles of course).
            # check if all ensembles are to be saved or just one of the supported ensemble moments
            for ens_ind in xrange(self.sample_size):
                if file_output_separate_files:
                    # print('saving ensemble member to separate files: %d' % ens_ind)
                    forecast_ensemble_member = self.filter_configs['forecast_ensemble'][ens_ind]
                    self.model.write_state(state=forecast_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble_'+str(ens_ind),
                                            append=False
                                            )
                    #
                    analysis_ensemble_member = self.filter_configs['analysis_ensemble'][ens_ind]
                    self.model.write_state(state=analysis_ensemble_member,
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble_'+str(ens_ind),
                                            append=False
                                            )
                else:
                    # print('saving ensemble member to same file with resizing: %d' % ens_ind)
                    # save results to different files. For moments
                    forecast_ensemble_member = self.filter_configs['forecast_ensemble'][ens_ind]
                    self.model.write_state(state=forecast_ensemble_member.copy(),
                                            directory=cycle_states_out_dir,
                                            file_name='forecast_ensemble',
                                            append=True
                                            )
                    #
                    analysis_ensemble_member = self.filter_configs['analysis_ensemble'][ens_ind]
                    self.model.write_state(state=analysis_ensemble_member.copy(),
                                            directory=cycle_states_out_dir,
                                            file_name='analysis_ensemble',
                                            append=True
                                            )
        # save reference state
        reference_state = self.filter_configs['reference_state']
        self.model.write_state(state=reference_state, directory=cycle_states_out_dir, file_name='reference_state')

        #
        # Save observation to file; use model to write observations to file(s)
        # save analysis mean
        observation = self.filter_configs['observation']
        self.model.write_observation(observation=observation, directory=cycle_observations_out_dir, file_name='observation', append=False)

        # Save filter statistics to file
        # 1- Output filter RMSEs: RMSEs are saved to the same file. It's meaningless to create a new file for each cycle
        rmse_file_name = 'rmse'
        if file_output_file_format in ['txt', 'ascii']:
            rmse_file_name += '.dat'
            rmse_file_path = os.path.join(filter_statistics_dir, rmse_file_name)
            if not os.path.isfile(rmse_file_path):
                # rmse file does not exist. create file and add header.
                filter_name = self.filter_configs['filter_name']
                header = "RMSE Results: Filter: '%s' \n %s \t %s \t %s \t %s \t %s \n" % (filter_name,
                                                                                          'Observation-Time'.rjust(20),
                                                                                          'Forecast-Time'.rjust(20),
                                                                                          'Analysis-Time'.rjust(20),
                                                                                          'Forecast-RMSE'.rjust(20),
                                                                                          'Analysis-RMSE'.rjust(20),
                                                                                          )
                # get the initial RMSE and add it if forecast is done first...
                if self.filter_configs['forecast_first']:
                    initial_time = self.filter_configs['timespan'][0]
                    initial_rmse = self.output_configs['filter_statistics']['initial_rmse']
                    header += " %20s \t %20.14e \t %20.14e \t %20.14e \t %20.14e \n" % ('0000000',
                                                                                        initial_time,
                                                                                        initial_time,
                                                                                        initial_rmse,
                                                                                        initial_rmse
                                                                                        )
                # dump the header to the file
                with open(rmse_file_path, mode='w') as file_handler:
                    file_handler.write(header)
            else:
                pass
            # rmse file exists --> Append rmse results to the file.
            forecast_time = self.filter_configs['forecast_time']
            analysis_time = self.filter_configs['analysis_time']
            observation_time = self.filter_configs['observation_time']
            #
            forecast_rmse = self.output_configs['filter_statistics']['forecast_rmse']
            analysis_rmse = self.output_configs['filter_statistics']['analysis_rmse']
            output_line = u" {0:20.14e} \t {1:20.14e} \t {2:20.14e} \t {3:20.14e} \t {4:20.14e} \n".format(observation_time,
                                                                                                           forecast_time,
                                                                                                           analysis_time,
                                                                                                           forecast_rmse,
                                                                                                           analysis_rmse
                                                                                                           )
            #
            with open(rmse_file_path, mode='a') as file_handler:
                file_handler.write(output_line)

            # save filter and model configurations (a copy under observation directory and another under state directory)...
            filter_configs = self.filter_configs
            filter_conf= dict(filter_name=filter_configs['filter_name'],
                              prior_distribution=filter_configs['prior_distribution'],
                              gmm_prior_settings=filter_configs['gmm_prior_settings'],
                              ensemble_size=filter_configs['ensemble_size'],
                              apply_preprocessing=filter_configs['apply_preprocessing'],
                              apply_postprocessing=filter_configs['apply_postprocessing'],
                              timespan=filter_configs['timespan'],
                              analysis_time=filter_configs['analysis_time'],
                              observation_time=filter_configs['observation_time'],
                              forecast_time=filter_configs['forecast_time'],
                              forecast_first=filter_configs['forecast_first']
                              )
            io_conf = output_configs
            #
            if self.prior_distribution in ['gmm', 'gaussian_mixture', 'gaussian-mixture']:
                gmm_conf = self.output_configs['filter_statistics']['gmm_prior_statistics']
                utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                                   [filter_conf, io_conf, gmm_conf], ['Filter Configs', 'Output Configs', 'GMM-Prior Configs'])
                utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                                   [filter_conf, io_conf, gmm_conf], ['Filter Configs', 'Output Configs', 'GMM-Prior Configs'])
            else:
                utility.write_dicts_to_config_file('setup.dat', cycle_observations_out_dir,
                                                   [filter_conf, io_conf], ['Filter Configs', 'Output Configs'])
                utility.write_dicts_to_config_file('setup.dat', cycle_states_out_dir,
                                                   [filter_conf, io_conf], ['Filter Configs', 'Output Configs'])
        else:
            print("Unsupported output format: '%s' !" % file_output_file_format)
            raise ValueError()
            #
        #
        # 2- Save optimization results, and optimal inflation, and localization factors
        if 'inflation_opt_results' in self.filter_configs:
            #
            x = self.model.state_vector()
            if 'optimal_inflation_factor' in self.filter_configs:
                x[:] = self.filter_configs['optimal_inflation_factor']
            else:
                x[:] = self.filter_configs['forecast_inflation_factor']
            self.model.write_state(state=x, directory=filter_statistics_dir, file_name='inflation_factors')

            #
            # pickle optimization results to monitor objective and stuff
            opt_dict = {}
            opt_dict = {'orig_opt_x':self.filter_configs['inflation_opt_results'][0],
                        'opt_x':self.filter_configs['inflation_opt_results'][1],
                        'min_f':self.filter_configs['inflation_opt_results'][2],
                        'opt_info_d':self.filter_configs['inflation_opt_results'][3],
                        'post_trace':self.filter_configs['inflation_opt_results'][4],
                        'alpha':self.filter_configs['inflation_design_penalty'],
                        'full_opt_results':self.filter_configs['inflation_opt_results'][5],
                        }
            target_file_path = os.path.join(cycle_states_out_dir, 'inflation_opt_results.p')
            pickle.dump(opt_dict, open(target_file_path, "wb"))
            #
        if 'localization_opt_results' in self.filter_configs:
            localization_space = self.filter_configs['localization_space']
            if re.match(r'\AB\Z', localization_space, re.IGNORECASE):
                # B-Localization
                x = self.model.state_vector()
            elif re.match(r'\AR(-| |_)*(1|2)\Z', localization_space, re.IGNORECASE):
                # R localization
                x = self.model.observation_vector()
            else:
                print("Unsupported Localization space %s" % repr(localization_space))
                print("Localization space must be B, R1, or R2")
                raise ValueError
            #
            if 'optimal_localization_radius' in self.filter_configs:
                x[:] = self.filter_configs['optimal_localization_radius']
            else:
                x[:] = self.filter_configs['localization_radius']

            if re.match(r'\AB\Z', localization_space, re.IGNORECASE):
                # B-Localization
                self.model.write_state(state=x, directory=filter_statistics_dir, file_name='localization_radii')
            elif re.match(r'\AR(-| |_)*(1|2)\Z', localization_space, re.IGNORECASE):
                # R localization
                self.model.write_observation(observation=x, directory=filter_statistics_dir, file_name='localization_radii')
            else:
                print("Unsupported Localization space %s" % repr(localization_space))
                print("Localization space must be B, R1, or R2")
                raise ValueError

            # pickle optimization results to monitor objective and stuff
            opt_dict = {}
            opt_dict = {'orig_opt_x':self.filter_configs['localization_opt_results'][0],
                        'opt_x':self.filter_configs['localization_opt_results'][1],
                        'min_f':self.filter_configs['localization_opt_results'][2],
                        'opt_info_d':self.filter_configs['localization_opt_results'][3],
                        'post_trace':self.filter_configs['localization_opt_results'][4],
                        'alpha':self.filter_configs['localization_design_penalty'],
                        'full_opt_results':self.filter_configs['localization_opt_results'][5],
                        }
            target_file_path = os.path.join(cycle_states_out_dir, 'localization_opt_results.p')
            pickle.dump(opt_dict, open(target_file_path, "wb"))
            #


class EnKF_OED_Localization(EnKF_OED_Inflation):
    # A class implementing the deterministic vanilla ensemble Kalman Filter, with adaptive localization.
    # The strategy is to use OED-based approach with A-optimality criterion!
    #

    _def_local_filter_configs = dict(model=None,
                                     filter_name="EnKF_OED_Localization",
                                     forecast_inflation_factor=1.09,  # applied to forecast ensemble
                                     reset_localization_radius=True,
                                     localization_radii_bounds=(0, 20),
                                     localization_design_penalty=0.05,  # penalty of the regularization parameter
                                     loc_direct_approach=7,  # 6 (See My Notes on OED_Localization) cases to apply localization with different radii
                                     inflation_factor=1.0,  # make sure analysis ensemble is not inflated
                                     regularization_norm='l1',  # L1, L2 are supported
                                     moving_average_radius=0,
                                     localization_space='B',  # B, R1, R2 --> B-localization, R-localization (BHT only), R-localization (both BHT, and HBHT)
                                     optimizer_configs=dict(method='SLSQP',  # 'COBYLA', 'SLSQP'
                                                            maxiter=1000,
                                                            maxfun=500,
                                                            tol=1e-5,
                                                            reltol=1e-6,
                                                            pgtol=1e-05,
                                                            epsilon=1e-8,
                                                            factr=10000000.0,
                                                            disp=1,
                                                            maxls=50,
                                                            iprint=-1
                                                            ),
                                     localize_covariances=True,
                                     localization_method='covariance_filtering',
                                     localization_radius=np.infty,
                                     localization_function='gaspari-cohn'
                                     )
    _local_def_output_configs = dict()
    #
    __round_num_digits = 4  # round optimal solution of localization parameters

    #
    def __init__(self, filter_configs=None, output_configs=None):
        #
        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, EnKF_OED_Localization._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, EnKF_OED_Localization._local_def_output_configs)
        #
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(filter_configs=filter_configs, output_configs=output_configs)
        else:
            # old-stype class
            super(EnKF_OED_Localization, self).__init__(filter_configs=filter_configs, output_configs=output_configs)
        #
        # Additional class-specific configurations, e.g. for adaptive inflation:
        # self.__shifted_proj_cov_inv = self.model.observation_matrix()  # an empty place holder for (R + H B H^T )^{-1}
        self.__original_forecast_state = None  # this creates a copy of the forecast ensemble

        orig_l = self.filter_configs['localization_radius']
        if np.isscalar(orig_l):
            self.__original_localization_radius = orig_l
        else:
            self.__original_localization_radius = orig_l.copy()
        #
        #
        default_bounds = EnKF_OED_Localization._def_local_filter_configs['localization_radii_bounds']
        localization_radii_bounds = self.filter_configs['localization_radii_bounds']
        try:
            state_size = self.model.state_size()
            # Get and correct the bounds:
            lb, ub = localization_radii_bounds
            if lb is None:
                lb = 0
            lb = max(lb, 0)
            if ub is None:
                ub = state_size/2
            ub = min(ub, state_size/2)
            if lb >= ub:
                print("The bounds must be ranked increasingly with upper bound < lower bound!")
                raise ValueError
            else:
                adjusted_bounds = (lb, ub)
        except:
            print("Failed to get the optimizer bounds on the localization radii; using default values % " % str(default_bounds))
            adjusted_bounds = default_bounds
        self.filter_configs.update({'localization_radii_bounds':adjusted_bounds})
        
        self.__opt_tracker = None

        self.__initialized = True
        #

    #
    def analysis(self, all_to_numpy=True):
        """
        Analysis step:

        Args:
            all_to_numpy (default False): bool,
                convert all data structures to Numpy and re-place results into target structures only in the end.

        Returns:
            None. Only self.filter_configs is updated.

        """
        model = self.model

        # make sure the original forecast ensemble and background state are packedup
        original_forecast_ensemble = [ v.copy() for v in self.filter_configs['forecast_ensemble'] ]  # this creates a copy of the
        self.filter_configs.update({'original_forecast_ensemble': original_forecast_ensemble})
        self.filter_configs.update({'original_forecast_state': utility.ensemble_mean(original_forecast_ensemble)})

        orig_l = self.filter_configs['localization_radius']
        if np.isscalar(orig_l):
            self.__original_localization_radius = orig_l
        else:
            self.__original_localization_radius = orig_l.copy()

        # Model state, and observation sizes
        state_size = model.state_size()
        observation_size = model.observation_size()

        # Get the localization space:
        localization_space = self.filter_configs['localization_space']
        if re.match(r'\AB\Z', localization_space, re.IGNORECASE):
            # B-Localization
            opt_space_size = state_size
        elif re.match(r'\AR(-| |_)*(1|2)\Z', localization_space, re.IGNORECASE):
            # R localization
            opt_space_size = observation_size
        else:
            print("Unsupported Localization space %s" % repr(localization_space))
            print("Localization space must be B, R1, or R2")
            raise ValueError

        x0 = orig_l
        if x0 is None:
            x0 = np.empty(opt_space_size)  # default initial inflation factors vector
            x0[:] = 4.0
        elif np.isscalar(x0):
            x0 = np.ones(opt_space_size) * float(x0)
        else:
            pass
        #
        # 2- Create an optimizer
        self.__opt_tracker = None  # refresh tracker
        obj_fun = lambda x: self.obj_localization_fun_value(x)
        obj_grad = lambda x: self.obj_localization_fun_gradient(x)
        callback_fun = lambda x: self._localization_iter_callback(x)
        #
        optimizer_configs = self.filter_configs['optimizer_configs']
        lb, ub = self.filter_configs['localization_radii_bounds']
        bounds = [(lb, ub)] * opt_space_size  # can be passed in the configurations dictionary; TODO.
        #
        if False:
            try:
               opt_x, f, d = optimize.fmin_l_bfgs_b(obj_fun,
                                                     x0,
                                                     fprime=obj_grad,
                                                     bounds=bounds,
                                                     m=50,
                                                     factr=float(optimizer_configs['factr']),
                                                     pgtol=optimizer_configs['pgtol'],
                                                     epsilon=optimizer_configs['epsilon'],
                                                     iprint=optimizer_configs['iprint'],
                                                     maxfun=optimizer_configs['maxfun'],
                                                     maxiter=optimizer_configs['maxiter'],
                                                    maxls=optimizer_configs['maxls'],
                                                     disp=optimizer_configs['disp'],
                                                     callback=callback_fun
                                                     )
            except NameError:
                obj_vals = np.asarray([x[-1] for x in self.__opt_tracker])
                mask = np.where(obj_vals>0)[0]
                try:
                    loc = mask[np.argsort(obj_vals[mask])[0]]
                    opt_x = self.__opt_tracker[loc][0]
                    f = self.__opt_tracker[loc][0][1]
                    print("optimal solution set to: %s" % repr(opt_x))
                    d = {'warnflag': 0, 'task':'Reset to the last known good solution from optimizer log', 'funcalls':len(self.__opt_tracker), 'nit':len(self.__opt_tracker)}
                except(IndexError):
                    opt_x = x0
                    f = 0
                    d = {'warnflag': -1, 'task':'Failed. No good iterations ', 'funcalls':len(self.__opt_tracker), 'nit':len(self.__opt_tracker)}

            except ValueError:
                opt_x = x0
                f = 0
                success = False
                d = {'warnflag': -1, 'task':'Failed. No good iterations ', 'funcalls':len(self.__opt_tracker), 'nit':len(self.__opt_tracker)}

            flag = d['warnflag']
            if flag == 0:
                msg = 'Converged: %s' % d['task']
                success = True
            elif flag in [1, 2]:
                msg = d['task']
                success = False
            else:
                msg = d['task']
                success = False
            opt_res = {'success':success,
                       'status': flag,
                       'message': msg,
                       'nfev': d['funcalls'],
                       'nit': d['nit']
                      }
            opt_x[opt_x<lb] = lb
            opt_x[opt_x>ub] = ub
            print("OptX: ", opt_x)
        else:
            #
            ftol = 1e-6
            maxiter = optimizer_configs['maxiter']
            opts = {'maxiter':maxiter,
                    'ftol':ftol,
                    'disp':optimizer_configs['disp']
                    }
            if True:
                const=()
            else:
                const = ({'type': 'ineq',
                          'fun': lambda x: self.obj_localization_fun_value(x) - 0.55,
                          # 'fun': obj_fun,
                          'jac': obj_grad})
            method = self.filter_configs['optimizer_configs']['method']
            try:
                res = minimize(obj_fun, x0,
                               method=method,
                               jac=obj_grad,
                               hess=None,
                               hessp=None,
                               bounds=bounds,
                               constraints=const,
                               tol=ftol,
                               callback=callback_fun,
                               options=opts)
                # print(res)
                opt_x = res.x
                f = res.fun
                d = {'warnflag': int(not res.success)}
                opt_res = {'success':res.success, 'status':res.status, 'message':res.message, 'nfev':res.nfev}
                try:
                    opt_res.update({'njev':res.njev})
                except:
                    pass
                try:
                    opt_res.update({'nit':res.nit})
                except:
                    pass

            except NameError:
                raise
                print("NAME ERR RAISED")
                obj_vals = np.asarray([x[-1] for x in self.__opt_tracker])
                if np.max(obj_vals) <= 0:
                    mask = np.where(obj_vals==np.max(obj_vals))
                else:
                    mask = np.where(obj_vals>0)[0]
                try:
                    loc = mask[np.argsort(obj_vals[mask])[0]]
                    opt_x = self.__opt_tracker[loc][0]
                    f = self.__opt_tracker[loc][0][1]
                    print("optimal solution set to: %s" % repr(opt_x))
                    d = {'warnflag': 0}
                    opt_res = {'warnflag': 0, 'task':'Reset to the last known good solution from optimizer log', 'funcalls':len(self.__opt_tracker), 'nit':len(self.__opt_tracker)}
                except(IndexError):
                    print("INDEX ERR RAISED")
                    opt_x = x0
                    f = 0
                    d = {'warnflag': -1}
                    opt_res = {'warnflag': -1, 'task':'Failed. No good iterations ', 'funcalls':len(self.__opt_tracker), 'nit':len(self.__opt_tracker)}

            except(ValueError):
                print("VALUE ERR RAISED")
                opt_x = x0
                f = 0
                d = {'warnflag': 1}
                opt_res = {'success':0, 'status':1, 'message':'Optimizer failed', 'nfev':0, 'njev':0, 'nit':0}
                raise
        orig_opt_x = opt_x.copy()
        # print("Optimal localization radii:", opt_x)
        # apply a moving average to the optimal solution
        moving_average_radius = self.filter_configs['moving_average_radius']
        r = min(moving_average_radius, opt_space_size/2-1)
        if r > 0:
            model_name = model._model_name
            try:
                if re.match(r'\Alorenz', model_name, re.IGNORECASE):
                    periodic = True
                elif re.match(r'\AQ(_|-)*G', model_name, re.IGNORECASE):
                    periodic = False
                else:
                    print("The model you selected ['%s'] is not supported" % model_name)
                    periodic = True
            except:
                periodic = False

            # print("Applying Moving average of r=%d " % r)
            # print("Opt_x PRE: ", opt_x)
            #
            opt_x = utility.moving_average(orig_opt_x, radius=r, periodic=periodic)
            # avg_opt_x = [np.mean(opt_x[i-r:i+r]) for i in xrange(r, opt_space_size-r)]
            # l_avg, u_avg = np.mean(opt_x[: r+1]), np.mean(opt_x[opt_space_size-2-r: ])
            # for i in xrange(r):
            #     # avg_opt_x.insert(0, opt_x[i])
            #     avg_opt_x.insert(0, l_avg)
            #     # avg_opt_x.append(opt_x[opt_space_size-1-r+i])
            #     avg_opt_x.append(u_avg)
            # try:
            #     opt_x[:] = avg_opt_x[:]
            # except:
            #     print("avg_opt_x", avg_opt_x)
            # # print("opt_x POST ", opt_x)

            # print("Utility-based smoothed Opt_x: ", opt_x2)
            # print("Two methods match? ", opt_x[:] == opt_x2[:])
            # print

        opt_x = np.round(opt_x, self.__round_num_digits)


        if self._verbose:
            print("Optimal solution: ", opt_x)
            print("res: ", res)

        #
        if self._verbose:
             # This is to be rewritten appropriately after debugging
            sepp = "\n%s\n" % ("{|}"*50)
            print(sepp + "OED-Localization RESULTS: %s" % '-'*15)
            print("optimal localization radii:", opt_x)
            print("Minimum localization radii entry:", opt_x.min())
            print("Maximum localization radii entry:", opt_x.max())
            print("Average localization radii:", np.mean(opt_x))
            print("Standard Deviation of localization radii entries:", np.std(opt_x))
            print(" >> Minimum Objective (posterior-covariance trace): ", f)
            print("flags: ", d)
            print(sepp)
        #

        # Save the results, and calculate the results' statistics
        failed = d['warnflag']  # 0 flag --> converged
        if failed:
            # self.filter_configs['analysis_state'] = None
            sep = "*"*30
            print(sep + "\n\n\tThe Optimizer algorithm Miserably failed!\n\n" + sep)
            obj_vals = np.asarray([x[-1] for x in self.__opt_tracker])
            mask = np.where(obj_vals>0)[0]
            try:
                loc = mask[np.argsort(obj_vals[mask])[0]]
                opt_x = self.__opt_tracker[loc][0]
                print("optimal solution set to: %s" % repr(opt_x))
            except(IndexError):
                opt_x = x0
            # raise ValueError
            pass

        # add regularization term (starting with L2 norm here):
        gamma = self.filter_configs['localization_design_penalty']
        post_trace = f
        regularization_norm = self.filter_configs['regularization_norm']  # get regularization norm
        if re.match(r"\Al(_|-)*1\Z", regularization_norm, re.IGNORECASE):
            regularizer = gamma * np.sum(opt_x)
            post_trace -= regularizer
        elif re.match(r"\Al(_|-)*2\Z", regularization_norm, re.IGNORECASE):
            regularizer = gamma * np.linalg.norm(opt_x, 2)
            post_trace -= regularizer
        else:
            print("Unrecognized norm %s " % regularization_norm)
            raise ValueError
        #
        self.filter_configs.update({'localization_opt_results':(orig_opt_x, opt_x, f, d, post_trace, opt_res)})

        #
        # Analysis with optimal inflation factor
        self.filter_configs['localization_radius'] = opt_x
        self.filter_configs.update({'optimal_localization_radius': opt_x})
        #
        # Reset forecast information
        self.filter_configs['forecast_ensemble'] = self.filter_configs['original_forecast_ensemble']
        self.filter_configs['forecast_state'] = self.filter_configs['original_forecast_state']

        # print("Re Analysis with localixzation radius: ", self.filter_configs['localization_radius'])
        # print("Original optx is : ", orig_opt_x)
        #
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            res = super().super().analysis(all_to_numpy=all_to_numpy)
        else:
            # old-stype class
            res = super(EnKF_OED_Inflation, self).analysis(all_to_numpy=all_to_numpy)
            # res = DEnKF.analysis(self, all_to_numpy=all_to_numpy)

        #
        if self.filter_configs['reset_localization_radius']:
            # print("\n\n\n ++++++++ Resetting radius to %s  \n\n++++++\n\n" % repr(self.__original_localization_radius))
            self.filter_configs['localization_radius'] = self.__original_localization_radius

        return res
        #

    def obj_localization_fun_value(self, loc_radius):
        """
        Evaluate the value of the A-optimal objective given a localization radius

        Args:
            loc_radius:

        Returns:
            objective_value: the value of the A-optimality objective function

        """
        # print("*** obj_localization_fun_value ***")
        # print("Loc_radius: ", loc_radius)
        
        # Check the inflation factor
        if np.isscalar(loc_radius):
            _scalar = True
            _loc_radius = float(loc_radius)
        elif utility.isiterable(loc_radius):
            _loc_radius = np.asarray(loc_radius).copy()
            if _loc_radius.size == 1:  # Extract inflation factor if it's a single value wrappend in an iterable
                for i in xrange(_loc_radius.ndim):
                    _loc_radius = _loc_radius[i]
                _scalar = True
            else:
                _scalar = False
        else:
            print("localization radius has to be a scalar or an iterable of length equal to model.state_size, or model.observation_size")
            raise AssertionError

        model = self.model
        model_name = model._model_name

        # Model state, and observation sizes
        state_size = model.state_size()
        observation_size = model.observation_size()

        # Get localization information::
        localization_function = loc_func = self.filter_configs['localization_function']
        loc_direct_approach = self.filter_configs['loc_direct_approach']
        localization_space = self.filter_configs['localization_space']

        if re.match(r'\AB\Z', localization_space, re.IGNORECASE):
            # B-Localization
            opt_space_size = state_size
        elif re.match(r'\AR(-| |_)*(1|2)\Z', localization_space, re.IGNORECASE):
            # R localization
            #
            opt_space_size = observation_size
        else:
            print("Unsupported Localization space %s" % repr(localization_space))
            print("Localization space must be B, R1, or R2")
            raise ValueError

        # model_grid, and observation_grid, and dimensions; for calculating distances:
        model_grid = model.get_model_grid()
        observation_grid = model.get_observation_grid()
        if observation_grid.ndim == 1:
            observation_grid = np.reshape(observation_grid.flatten(), (1, observation_size))
        if model_grid.ndim == 1:
            model_grid = np.reshape(model_grid.flatten(), (1, state_size))
        num_dimensions = np.size(model_grid, 1)
        num_obs_dims = np.size(observation_grid, 1)
        #
        if num_obs_dims != num_dimensions:
            print("Observational grid dimension mismatches the model grid dimension!. \n  \
                    Observation dimensions = %d, Model state dimensions = %d" % (num_obs_dims, num_dimensions))
            raise ValueError
        else:
            num_dims = num_obs_dims

        try:
            periodic_bc = model.model_configs['periodic']
        except(KeyError, NameError, AttributeError):
            periodic_bc = False

        if periodic_bc:
            dx = model.model_configs['dx']
        else:
            dx = 1
        dy = observation_grid[1] - observation_grid[0]

        # Retrieve forecast information:
        forecast_ensemble = [v.copy() for v in self.filter_configs['forecast_ensemble']]
        # forecast_ensemble = self.filter_configs['forecast_ensemble']
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        ensemble_size = len(forecast_ensemble)

        # OneD array with ensemble-based forecast variances
        first_term = utility.covariance_trace(forecast_ensemble)
        # print(" > Prior covariance matrix TRACE = ", first_term)

        #
        # REMARK: Both B, and R1 are checked, gradient works, and results are good.
        #         R2 objective and gradient are currently being validate (using Finite Differences)
        #
        # Second term; depends on the type of localization;
        if re.match(r'\AB\Z', localization_space, re.IGNORECASE):
            # Unlocalized ensemble background error covariance matrix
            B = utility.ensemble_covariance_matrix(forecast_ensemble)
            x = model.state_vector()
            # B-Localization
            B_hat = self._cov_mat_loc(B, _loc_radius, loc_func)
            BHt = np.empty((state_size, observation_size))
            for i in xrange(state_size):
                x[:] = B_hat[i, :].copy()
                BHt[i, :] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            HBHt = np.empty((observation_size, observation_size))
            for i in xrange(observation_size):
                x[:] = BHt[:, i]
                HBHt[:, i] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            G = HBHt + model.observation_error_model.R.get_numpy_array()
            U, s, Vh = np.linalg.svd(G)
            G_inv = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))
            # G_inv = slinalg.inv(G)
            #
            # objective_value = first_term - np.trace(np.dot(G_inv, BHt.T.dot(BHt)))
            # return objective_value

            HBBHT = np.dot(BHt.T, BHt)
            second_term = 0.
            for i in xrange(observation_size):
                up = G_inv.dot(HBBHT[:, i])[i]
                second_term += up

            if False:
                GHB = np.dot(G_inv, BHt.T)
                second_term2 = 0.
                for i in xrange(state_size):
                    up = BHt.dot(GHB[:, i])[i]
                    second_term2 += up
                    #

            # trace of posterior error covariance matrix
            objective_value = first_term - second_term
        #
        elif re.match(r'\AR(-| |_)*1\Z', localization_space, re.IGNORECASE):
            x = model.state_vector()  # temp vector
            #
            # Unlocalized version of HB, and HBHT:
            B = utility.ensemble_covariance_matrix(forecast_ensemble)
            BHt = np.empty((state_size, observation_size))
            for i in xrange(state_size):
                x[:] = B[i, :].copy()
                BHt[i, :] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            HBHt = np.empty((observation_size, observation_size))
            for i in xrange(observation_size):
                x[:] = BHt[:, i]
                HBHt[:, i] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            # Localize HB (or equivalently BHT), in both cases:
            for obs_ind in xrange(observation_size):
                ref_obs_coord = observation_grid[obs_ind, :]
                distances = utility.euclidean_distance(model_grid, ref_obs_coord)
                # print(">>++ ref_obs_coord, model_grid, distances: ", ref_obs_coord, model_grid, distances)

                if periodic_bc:
                    # rem_distances = utility.euclidean_distance(cir_grid, ref_obs_coord)
                    for i in xrange(distances.size):
                        distances[i] = min(distances[i], state_size*dx-distances[i])
                # print("++<< distances: ", distances)
                #
                loc_coeffs = utility.calculate_localization_coefficients(_loc_radius[obs_ind], distances, localization_function)
                # print("distances, loc_coeffs: ", distances, loc_coeffs)
                BHt[:, obs_ind] *= loc_coeffs

            # Now calculate second term
            G = HBHt + model.observation_error_model.R.get_numpy_array()
            U, s, Vh = np.linalg.svd(G)
            G_inv = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))

            # objective_value = first_term - np.trace(np.dot(G_inv, BHt.T.dot(BHt)))
            # return objective_value

            # HBBHT = BHt.T.dot(BHt)
            # print("Full second term: ", np.trace(HBBHT.dot(G_inv)))
            second_term = 0.
            for i in xrange(observation_size):
                # up = HBBHT.dot(G_inv[:, i])[i]
                up = BHt.T.dot(BHt).dot(G_inv[:, i])[i]
                second_term += up
    
            # print(" > R1 localization space: second term = ", second_term)

            # trace of posterior error covariance matrix
            objective_value = first_term - second_term
        #
        elif re.match(r'\AR(-| |_)*2\Z', localization_space, re.IGNORECASE):
            #
            x = model.state_vector()  # temp vector
            #
            # Unlocalized version of HB, and HBHT:
            B = utility.ensemble_covariance_matrix(forecast_ensemble)
            BHt = np.empty((state_size, observation_size))
            for i in xrange(state_size):
                x[:] = B[i, :].copy()
                BHt[i, :] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            HBHt = np.empty((observation_size, observation_size))
            for i in xrange(observation_size):
                x[:] = BHt[:, i]
                HBHt[:, i] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            # Localize HB (or equivalently BHT), in both cases:
            for obs_ind in xrange(observation_size):
                ref_obs_coord = observation_grid[obs_ind, :]
                distances = utility.euclidean_distance(model_grid, ref_obs_coord)
                if periodic_bc:
                    for i in xrange(distances.size):
                        distances[i] = min(distances[i], state_size*dx-distances[i])
                #
                loc_coeffs = utility.calculate_localization_coefficients(_loc_radius[obs_ind], distances, localization_function)
                # print("distances, loc_coeffs: ", distances, loc_coeffs)
                BHt[:, obs_ind] *= loc_coeffs
            #
            # Localize HBHT
            # Localization in the observation space:
            for obs_ind in xrange(observation_size):
                ref_obs_coord = observation_grid[obs_ind, :]
                distances = utility.euclidean_distance(observation_grid, ref_obs_coord)
                # update distances for periodic models/boundary-conditions (e.g. Lorenz 96)
                if periodic_bc:
                    for i in xrange(distances.size):
                        distances[i] = min(distances[i], state_size*dx-distances[i])
                if loc_direct_approach == 7:
                    obs_loc_coeffs = utility.calculate_mixed_localization_coefficients( (_loc_radius, _loc_radius[obs_ind]), distances, localization_function)
                else:
                    obs_loc_coeffs = utility.calculate_localization_coefficients(_loc_radius[obs_ind], distances, localization_function)
                    #
                if loc_direct_approach == 1:
                    # radius is fixed over rows of the covariance matrix:
                    HBHt[obs_ind, : ] *= obs_loc_coeffs

                elif loc_direct_approach == 2:
                    # radius is fixed over rows of the covariance matrix:
                    HBHt[:, obs_ind] *= obs_loc_coeffs

                elif loc_direct_approach == 3 or loc_direct_approach == 6:
                    # radius is fixed over rows of the covariance matrix:
                    #
                    vert_coeffs = utility.calculate_localization_coefficients(_loc_radius, distances, localization_function)
                    fac = 0.5 * (obs_loc_coeffs[obs_ind: ] + vert_coeffs[obs_ind: ])
                    HBHt[obs_ind:, obs_ind] *= fac
                    HBHt[obs_ind, obs_ind: ] *= fac

                elif loc_direct_approach == 4:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    HBHt[obs_ind, obs_ind: ] *= obs_loc_coeffs[obs_ind: ]
                    HBHt[obs_ind: , obs_ind] *= obs_loc_coeffs[obs_ind: ]

                elif loc_direct_approach == 5:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied upwards:
                    HBHt[obs_ind, : obs_ind+1 ] *= obs_loc_coeffs[obs_ind: ]
                    HBHt[ : obs_ind+1, obs_ind] *= obs_loc_coeffs[obs_ind: ]

                elif loc_direct_approach == 7:
                    HBHt[obs_ind:, obs_ind ] *= obs_loc_coeffs[obs_ind: ]
                    HBHt[obs_ind, obs_ind: ] *= obs_loc_coeffs[obs_ind: ]

                else:
                    print("loc_direct_approach MUST be an integer from 1-6, see Attia's Notes on OED_Localization!")
                    raise ValueError

            # Now calculate second term
            G = HBHt + model.observation_error_model.R.get_numpy_array()
            U, s, Vh = np.linalg.svd(G)
            G_inv = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))
            #
            # objective_value = first_term - np.trace(np.dot(G_inv, BHt.T.dot(BHt)))
            # return objective_value

            # HBBHT = BHt.T.dot(BHt)
            # print("Full second term: ", np.trace(HBBHT.dot(G_inv)))
            second_term = 0.
            for i in xrange(observation_size):
                # up = HBBHT.dot(G_inv[:, i])[i]
                up = BHt.T.dot(BHt).dot(G_inv[:, i])[i]
                second_term += up
            # print("Aggregated Second term: ", second_term)
            
            # print(" > R1 localization space: second term = ", second_term)

            # trace of posterior error covariance matrix
            objective_value = first_term - second_term

        #
        else:
            print("Unsupported Localization space %s" % repr(localization_space))
            print("Localization space must be B, R1, or R2")
            raise ValueError
            
        # print(" > Posterior Trace = ", objective_value)
        if objective_value <= 0 and self._verbose:
            msg = "*** WARGNING: NEGATIVE POSTERIOR TRACE ***"
            print("*"*len(msg))
            print(msg)
            print("localization_space: ", localization_space)
            print("The posterior covariance trace is calcualted to be negative!")
            print("The localization radius: ", loc_radius)
            print(" > Prior covariance matrix TRACE = ", first_term)
            print(" > second term = ", second_term)
            print("The posterior trace at this radius is: ", objective_value)
            # HBBHT = BHt.T.dot(BHt)
            # print("Full second term: ", np.trace(HBBHT.dot(G_inv)))
            # print("\t the other approach gives: %f" % (first_term - np.trace(np.dot(G_inv, BHt.T.dot(BHt))) ) )
            print("*"*len(msg))
            # raise ValueError
        #
        # add regularization term (starting with L2 norm here):
        gamma = self.filter_configs['localization_design_penalty']
        regularization_norm = self.filter_configs['regularization_norm']  # get regularization norm
        if re.match(r"\Al(_|-)*1\Z", regularization_norm, re.IGNORECASE):
            if gamma!=0:
                if _scalar:
                    reg_update = gamma * opt_space_size * (_loc_radius)
                else:
                    reg_update = gamma * np.sum(_loc_radius)
                    #
            else:
                reg_update = 0.0
            objective_value += reg_update
        elif re.match(r"\Al(_|-)*2\Z", regularization_norm, re.IGNORECASE):
            if gamma != 0:
                if _scalar:
                    reg_update = gamma * np.sqrt(opt_space_size) * _loc_radius
                else:
                    reg_update = gamma * np.linalg.norm(_loc_radius, 2)
            else:
                reg_update = 0.0
            objective_value += reg_update
        else:
            print("Unrecognized norm %s " % regularization_norm)
            raise ValueError
        
        # print(" > Objective function := posterior trace + regularization = ", objective_value)

        if self._verbose:
            sep = "\n%s\n" % ("*"*80)
            print(sep)
            regularizer = reg_update
            print("Inside 'obj_inflation_fun_value'. Passed localization radius is: ", _loc_radius)
            print("Regularizer: ", regularizer)
            print("First Term: ", first_term)
            print("Second Term: ", second_term)
            print("objective_value:", objective_value)
            print(sep)

        #
        # Second(optimized) approach:
        # ----------------------------
        pass

        return objective_value
        #

    @staticmethod
    def l_term(i, j, l, loc_func, model_name, domain_size, grid=None, dxi=None, dxj=None, verbose=False):
        """
        Vector of derivatives of the localization kernel by the localization radii
        Note; for zero radius, and zer distances the localization kernel is a delta function, and hence the localization coefficient is 1, and the derivative is zero
        """
        if l == 0:
            if verbose:
                print("The localization is set to zero; We assume the coefficient is one in this case, and the derivative is zero")
            # raise ValueError
            d = 0
            return d

        if dxi is None:
            dxi = 1
        if dxj is None:
            dxj = 1
        if not np.isscalar(l):
            try:
                li, lj = l
                l_isscalar = False
            except:
                print("l is expected to be iterable with two entries!\nCouldn't broadcast the localization radii %s of Type %s" % (str(l) , type(l)) )
                raise
        else:
            l_isscalar = True

        # l is scalar (cases 1-6)
        if re.match(r'\Alorenz', model_name, re.IGNORECASE):
            dist = min(abs(i*dxi-j*dxj), domain_size-abs(i*dxi-j*dxj))

        elif re.match(r'\AQ(_|-)*G', model_name, re.IGNORECASE):
            raise NotImplementedError("Need to reconsider cases where model/obs locliatization is carried out...")  # something already exist in the QG model module
            if grid is None:
                print("For 2D models, a model/observation grid MUST be provided")
                raise ValueError
            dist = np.sqrt( (grid[i, 0]-grid[j, 0])**2 + (grid[i, 1]-grid[j, 1])**2 )
        else:
            print("The model you selected ['%s'] is not supported" % model_name)
            raise ValueError()

        if re.match(r'\Agauss\Z', loc_func, re.IGNORECASE):
            if l_isscalar:
                d = dist**2 / (l**3) * np.exp( - 0.5 * dist**2 / (l**2) )
            else:
                d = dist**2 / (2* (li**2) *lj ) * np.exp( - 0.5 * dist**2 / (li * lj) )

        elif re.match(r'\Agaspari(_|-)*cohn\Z', loc_func, re.IGNORECASE):
            if l_isscalar:
                thresh = l * 1.7386  # This is C in the utility module,a nd in papers; Change of variables is straigt forward here
                # dist *= 1.7386
                if dist <= thresh:
                    red = dist/thresh
                    r2 = dist**2 / thresh**3
                    r3 = dist**4 / thresh**5
                    d = (5.0/4.0)*(red*r3) - 2.0*r3 - (15.0/8.0)*(red*r2) + (10.0/3.0)*r2
                elif dist <= thresh*2:
                    red = dist/thresh
                    r2 = dist**2 / thresh**3
                    r3 = dist**4 / thresh**5
                    d = -(5.0/12.0)*(red*r3) + 2.0*r3 - (15.0/8.0)*(red*r2) - (10.0/3.0)*r2 + 5.0*red/thresh -(2.0/3.0/dist)
                else:
                    d = 0.0
            else:
                thresh = li * 1.7386  # This is C in the utility module,a nd in papers; Change of variables is straigt forward here
                dist /= lj
                if dist <= thresh:
                    red = dist/thresh
                    r2 = dist**2 / thresh**3
                    r3 = dist**4 / thresh**5
                    d = (5.0/4.0)*(red*r3) - 2.0*r3 - (15.0/8.0)*(red*r2) + (10.0/3.0)*r2
                elif dist <= thresh*2:
                    red = dist/thresh
                    r2 = dist**2 / thresh**3
                    r3 = dist**4 / thresh**5
                    d = -(5.0/12.0)*(red*r3) + 2.0*r3 - (15.0/8.0)*(red*r2) - (10.0/3.0)*r2 + 5.0*red/thresh -(2.0/3.0/dist)
                else:
                    d = 0.0
            #
            d *= 1.7386  # coz I am optimizing for c directly

        else:
            print("Unknown Localization function %s" %repr(loc_func))
            raise ValueError
        return d


    def obj_localization_fun_gradient(self, loc_radius, FD_Validation=False, FD_eps=1e-5, FD_central=False):
        """
        """
        # print(" >> GOING INTO obj_localization_fun_gradient\n")
        # Check the inflation factor
        if np.isscalar(loc_radius):
            _scalar = True
            _loc_radius = float(loc_radius)
        elif utility.isiterable(loc_radius):
            _loc_radius = np.asarray(loc_radius).copy()
            if _loc_radius.size == 1:  # Extract inflation factor if it's a single value wrappend in an iterable
                for i in xrange(_loc_radius.ndim):
                    _loc_radius = _loc_radius[i]
                _scalar = True
            else:
                _scalar = False
        else:
            print("localization radius has to be a scalar or an iterable of length equal to model.state_size, or model.observation_size")
            raise AssertionError

        model = self.model
        model_name = model._model_name
        try:
            if re.match(r'\Alorenz', model_name, re.IGNORECASE):
                periodic = True
            elif re.match(r'\AQ(_|-)*G', model_name, re.IGNORECASE):
                periodic = False
            else:
                print("The model you selected ['%s'] is not supported" % model_name)
                raise ValueError()
        except:
            periodic = False


        # Model state, and observation sizes
        state_size = model.state_size()
        observation_size = model.observation_size()

        # Get localization information::
        localization_function = loc_func = self.filter_configs['localization_function']
        loc_direct_approach = self.filter_configs['loc_direct_approach']
        localization_space = self.filter_configs['localization_space']

        if re.match(r'\AB\Z', localization_space, re.IGNORECASE):
            # B-Localization
            opt_space_size = state_size
        elif re.match(r'\AR(-| |_)*(1|2)\Z', localization_space, re.IGNORECASE):
            # R localization
            #
            opt_space_size = observation_size
        else:
            print("Unsupported Localization space %s" % repr(localization_space))
            print("Localization space must be B, R1, or R2")
            raise ValueError

        # model_grid, and observation_grid, and dimensions; for calculating distances:
        model_grid = model.get_model_grid()
        observation_grid = model.get_observation_grid()
        if observation_grid.ndim == 1:
            observation_grid = np.reshape(observation_grid.flatten(), (1, observation_size))
        if model_grid.ndim == 1:
            model_grid = np.reshape(model_grid.flatten(), (1, state_size))
        num_dimensions = np.size(model_grid, 1)
        num_obs_dims = np.size(observation_grid, 1)
        #
        if num_obs_dims != num_dimensions:
            print("Observational grid dimension mismatches the model grid dimension!. \n  \
                    Observation dimensions = %d, Model state dimensions = %d" % (num_obs_dims, num_dimensions))
            raise ValueError
        else:
            num_dims = num_obs_dims

        try:
            periodic_bc = model.model_configs['periodic']
        except(KeyError, NameError, AttributeError):
            periodic_bc = False

        if periodic_bc:
            dx = model.model_configs['dx']
        else:
            dx = 1
        dy = observation_grid[1] - observation_grid[0]

        # Retrieve forecast information:
        forecast_ensemble = [v.copy() for v in self.filter_configs['forecast_ensemble']]
        # forecast_ensemble = self.filter_configs['forecast_ensemble']
        forecast_state = utility.ensemble_mean(forecast_ensemble)
        ensemble_size = len(forecast_ensemble)

        #
        # Retrieve forecast information:
        # self.filter_configs['localization_radius'] = _loc_radius

        # Initialize, and fill-in the gradient:
        grad = np.zeros(opt_space_size)
        if re.match(r'\AB\Z', localization_space, re.IGNORECASE):
            # B-Localization
            B = utility.ensemble_covariance_matrix(forecast_ensemble)

            B_hat = B.copy()
            B_hat = self._cov_mat_loc(B_hat, _loc_radius, loc_func)
            #

            BHt = np.empty((state_size, observation_size))
            x = model.state_vector()
            for i in xrange(state_size):
                x[:] = B_hat[i, :].copy()
                BHt[i, :] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            HBHt = np.empty((observation_size, observation_size))
            for i in xrange(observation_size):
                x[:] = BHt[:, i]
                HBHt[:, i] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            G = HBHt + model.observation_error_model.R.get_numpy_array()
            U, s, Vh = np.linalg.svd(G)
            G_inv = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))
            # G_inv = slinalg.inv(G)
            #

            #
            I_Bi = model.state_vector()
            e_i = model.state_vector()
            l_vec = np.empty(state_size)
            #
            for i in xrange(state_size):  # TODO: Remove non-necessary copying after debugging...
                e_i[:] = 0.0; e_i[i] = 1.0  # ith cardinal vector in R^n; n = state_size
                eta = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=e_i).get_numpy_array()  # H e_i
                eta = G_inv.dot(eta)
                eta = BHt.dot(eta)
                # eta = e_i[:] - eta
                eta[i] -= 1.0  # e_i[i]
                eta *= -1

                if loc_direct_approach == 7:
                    l_vec[:] = [self.l_term(i, j, (_loc_radius[i],_loc_radius[j]), loc_func, model_name, state_size*dx, grid=model_grid, dxi=dx, dxj=dx) for j in xrange(state_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                else:
                    l_vec[:] = [self.l_term(i, j, _loc_radius[i], loc_func, model_name,  state_size*dx, grid=model_grid, dxi=dx, dxj=dx) for j in xrange(state_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                #

                # Notice that in all cases, except 1, 3,  B_hat is no longer symmetric, and the transpose must be considered!

                if loc_direct_approach == 1:
                    # radius is fixed over rows/columns/both of the covariance matrix:
                    raise NotImplementedError

                elif loc_direct_approach == 2:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    raise NotImplementedError

                elif loc_direct_approach == 3 or loc_direct_approach == 6:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    lb = B[i, :].copy()  # copying is not necessary here!
                    lb *= l_vec.copy()

                    I_Bi[:] = lb.copy()  # I_{B,i}
                    zeta = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=I_Bi).get_numpy_array()  # H I_Bi
                    zeta = G_inv.dot(zeta)
                    zeta = BHt.dot(zeta)
                    zeta = lb - zeta

                    grad_entry = eta.dot(zeta)
                    # print("grad_entry: ", grad_entry)
                    grad[i] = grad_entry

                elif loc_direct_approach == 4:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    raise NotImplementedError

                    lb = B[i, :].copy()  # copying is not necessary here!
                    lb[:i ] = 0
                    lb *= l_vec

                    I_Bi[:] = lb[:].copy()  # I_{B,i}
                    zeta = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=I_Bi).get_numpy_array()
                    zeta = G_inv.dot(zeta)
                    zeta = BHt.dot(zeta)
                    zeta = lb - zeta

                    grad[i] = eta[i: ].dot(zeta[i: ])

                elif loc_direct_approach == 5:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied upwards:
                    raise NotImplementedError

                    lb = B[i, :].copy()  # copying is not necessary here!
                    if i < state_size:
                        lb[i+1 : ] = 0
                    lb *= l_vec

                    I_Bi[:] = lb[:].copy()  # I_{B,i}
                    zeta = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=I_Bi).get_numpy_array()
                    zeta = G_inv.dot(zeta)
                    zeta = BHt.dot(zeta)
                    zeta = lb - zeta

                    grad[i] = eta[: i+1 ].dot(zeta[: i+1 ])

                elif loc_direct_approach == 7:
                    #
                    lb = B[i, :].copy()  # copying is not necessary here!
                    lb *= l_vec.copy()

                    I_Bi[:] = lb.copy()  # I_{B,i}
                    zeta = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=I_Bi).get_numpy_array()  # H I_Bi
                    zeta = G_inv.dot(zeta)
                    zeta = BHt.dot(zeta)
                    zeta = lb - zeta

                    grad_entry = 2 * eta.dot(zeta)
                    # print("grad_entry: ", grad_entry)
                    grad[i] = grad_entry

                else:
                    print("loc_direct_approach MUST be an integer from 1-7, see Attia's Notes on OED_Localization!")
                    raise ValueError

        elif re.match(r'\AR(-| |_)*1\Z', localization_space, re.IGNORECASE):
            # R localization
            B = utility.ensemble_covariance_matrix(forecast_ensemble)
            x = model.state_vector()
            unloc_BHt = np.empty((state_size, observation_size))
            for i in xrange(state_size):
                x[:] = B[i, :].copy()
                unloc_BHt[i, :] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            HBHt = np.empty((observation_size, observation_size))
            for i in xrange(observation_size):
                x[:] = unloc_BHt[:, i]
                HBHt[:, i] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            # Localize HB (or equivalently BHT), in both cases:
            BHt = unloc_BHt.copy()
            #
            for obs_ind in xrange(observation_size):
                ref_obs_coord = observation_grid[obs_ind, :]
                distances = utility.euclidean_distance(model_grid, ref_obs_coord)
                if periodic_bc:
                    for i in xrange(distances.size):
                        distances[i] = min(distances[i], state_size*dx-distances[i])
                loc_coeffs = utility.calculate_localization_coefficients(_loc_radius[obs_ind], distances, localization_function)
                BHt[:, obs_ind] *= loc_coeffs
            #
            G = HBHt + model.observation_error_model.R.get_numpy_array()
            U, s, Vh = np.linalg.svd(G)
            G_inv = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))

            e_i = model.observation_vector()
            l_vec = np.empty(state_size)
            #
            for i in xrange(observation_size):  # TODO: Remove non-necessary copying after debugging...
                e_i[:] = 0.0; e_i[i] = 1.0  # ith cardinal vector in R^n; n = state_size
                psi = G_inv.dot(e_i)
                psi = BHt.dot(psi)

                l_vec[:] = [self.l_term(i, j, _loc_radius[i], loc_func, model_name, state_size*dx, dxi=dx, dxj=dy) for j in xrange(state_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                #
                # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                l_vec *= unloc_BHt[:, i]
                grad[i] = -2 * psi.dot(l_vec)

        elif re.match(r'\AR(-| |_)*2\Z', localization_space, re.IGNORECASE):
            # R localization
            B = utility.ensemble_covariance_matrix(forecast_ensemble)
            x = model.state_vector()
            unloc_BHt = np.empty((state_size, observation_size))
            for i in xrange(state_size):
                x[:] = B[i, :].copy()
                unloc_BHt[i, :] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            unloc_HBHt = np.empty((observation_size, observation_size))
            for i in xrange(observation_size):
                x[:] = unloc_BHt[:, i]
                unloc_HBHt[:, i] = model.observation_operator_Jacobian_prod_vec(in_state=forecast_state, state=x).get_numpy_array()

            # Localize HB (or equivalently BHT), in both cases:
            BHt = unloc_BHt.copy()
            #
            for obs_ind in xrange(observation_size):
                ref_obs_coord = observation_grid[obs_ind, :]
                distances = utility.euclidean_distance(model_grid, ref_obs_coord)
                if periodic_bc:
                    for i in xrange(distances.size):
                        distances[i] = min(distances[i], state_size*dx-distances[i])
                loc_coeffs = utility.calculate_localization_coefficients(_loc_radius[obs_ind], distances, localization_function)
                # print("GRAD: distances, loc_coeffs: ", distances, loc_coeffs)
                BHt[:, obs_ind] *= loc_coeffs
            #
            # Localize HBHT
            HBHt = unloc_HBHt.copy()
            # Localization in the observation space:
            for obs_ind in xrange(observation_size):
                ref_obs_coord = observation_grid[obs_ind, :]
                distances = utility.euclidean_distance(observation_grid, ref_obs_coord)
                # update distances for periodic models/boundary-conditions (e.g. Lorenz 96)
                if periodic_bc:
                    for i in xrange(distances.size):
                        distances[i] = min(distances[i], state_size*dx-distances[i])
                if loc_direct_approach == 7:
                    obs_loc_coeffs = utility.calculate_mixed_localization_coefficients( (_loc_radius, _loc_radius[obs_ind]), distances, localization_function)
                else:
                    obs_loc_coeffs = utility.calculate_localization_coefficients(_loc_radius[obs_ind], distances, localization_function)
                    #
                if loc_direct_approach == 1:
                    # radius is fixed over rows of the covariance matrix:
                    HBHt[obs_ind, : ] *= obs_loc_coeffs

                elif loc_direct_approach == 2:
                    # radius is fixed over rows of the covariance matrix:
                    HBHt[:, obs_ind] *= obs_loc_coeffs

                elif loc_direct_approach == 3 or loc_direct_approach == 6:
                    # radius is fixed over rows of the covariance matrix:
                    #
                    vert_coeffs = utility.calculate_localization_coefficients(_loc_radius, distances, localization_function)
                    fac = 0.5 * (obs_loc_coeffs[obs_ind: ] + vert_coeffs[obs_ind: ])
                    HBHt[obs_ind:, obs_ind] *= fac
                    HBHt[obs_ind, obs_ind: ] *= fac

                elif loc_direct_approach == 4:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    HBHt[obs_ind, obs_ind: ] *= obs_loc_coeffs[obs_ind: ]
                    HBHt[obs_ind: , obs_ind] *= obs_loc_coeffs[obs_ind: ]

                elif loc_direct_approach == 5:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied upwards:
                    HBHt[obs_ind, : obs_ind+1 ] *= obs_loc_coeffs[obs_ind: ]
                    HBHt[ : obs_ind+1, obs_ind] *= obs_loc_coeffs[obs_ind: ]

                elif loc_direct_approach == 7:
                    HBHt[obs_ind:, obs_ind ] *= obs_loc_coeffs[obs_ind: ]
                    HBHt[obs_ind, obs_ind: ] *= obs_loc_coeffs[obs_ind: ]

                else:
                    print("loc_direct_approach MUST be an integer from 1-6, see Attia's Notes on OED_Localization!")
                    raise ValueError
            #
            G = HBHt + model.observation_error_model.R.get_numpy_array()
            U, s, Vh = np.linalg.svd(G)
            G_inv = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))

            e_i = model.observation_vector()
            l_vec_s = np.empty(state_size)
            l_vec_o = np.empty(observation_size)
            #

            #
            I_Bi = np.empty(observation_size)
            # second term
            for i in xrange(observation_size):  # TODO: Remove non-necessary copying after debugging...
                # Notice that in all cases, except 1, 3,  HBH_hat is no longer symmetric, and the transpose must be considered!

                # first term:
                e_i[:] = 0.0; e_i[i] = 1.0  # ith cardinal vector in R^n; n = state_size

                l_vec_s[:] = [self.l_term(i, j, _loc_radius[i], loc_func, model_name, state_size*dx, dxi=dx, dxj=dy) for j in xrange(state_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                l_vec_s *= unloc_BHt[:, i]

                psi = G_inv.dot(e_i)
                psi = BHt.dot(psi)

                omega = BHt.T.dot(l_vec_s)
                omega = G_inv.dot(omega)

                grad[i] = - l_vec_s.dot(psi) - omega[i]

                if loc_direct_approach in [1, 3, 6]:
                    # radius is fixed over rows/columns/both of the covariance matrix:
                    l_vec_o[:] = [self.l_term(i, j, _loc_radius[i], loc_func, model_name, state_size*dx, dxi=dy, dxj=dy) for j in xrange(observation_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                    I_Bi[:] = unloc_HBHt[i, :].copy()  # I_{B,i}
                    I_Bi *= l_vec_o

                    eta_i = BHt.dot(G_inv.T.dot(I_Bi))
                    grad[i] += eta_i.dot(psi)

                elif loc_direct_approach ==2:

                    zeta_i = BHt.dot(G_inv.T.dot(e_i))

                    l_vec_o[:] = [self.l_term(i, j, _loc_radius[i], loc_func, model_name, state_size*dx, dxi=dy, dxj=dy) for j in xrange(observation_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                    I_Bi[:] = unloc_HBHt[i, :].copy()  # I_{B,i}
                    I_Bi *= l_vec_o

                    xi_i = BHt.dot(G_inv.dot(I_Bi))
                    grad[i] += xi_i.dot(zeta_i)

                elif loc_direct_approach == 4:
                    l_vec_o[:] = [self.l_term(i, j, _loc_radius[i], loc_func, model_name, state_size*dx, dxi=dy, dxj=dy) for j in xrange(observation_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                    I_Bi[:] = unloc_HBHt[i, :].copy()  # I_{B,i}
                    I_Bi *= l_vec_o
                    I_Bi[: i] = 0.0

                    eta_i = BHt.dot(G_inv.T.dot(I_Bi))
                    grad[i] += 2 * eta_i.dot(psi)

                elif loc_direct_approach == 5:
                    l_vec_o[:] = [self.l_term(i, j, _loc_radius[i], loc_func, model_name, state_size*dx, dxi=dy, dxj=dy) for j in xrange(observation_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                    I_Bi[:] = unloc_HBHt[i, :].copy()  # I_{B,i}
                    I_Bi *= l_vec_o
                    I_Bi[i: ] = 0.0

                    eta_i = BHt.dot(G_inv.T.dot(I_Bi))
                    grad[i] += 2 * eta_i.dot(psi)

                elif loc_direct_approach == 7:
                    #
                    l_vec_o[:] = [self.l_term(i, j, (_loc_radius[i],_loc_radius[j]), loc_func, model_name, state_size*dx, grid=model_grid, dxi=dx, dxj=dx) for j in xrange(observation_size)]  # I_Bi = B[i,:] odot l_vec^T ei
                    I_Bi[:] = unloc_HBHt[i, :].copy()  # I_{B,i}
                    I_Bi *= l_vec_o

                    eta_i = BHt.dot(G_inv.T.dot(I_Bi))
                    grad[i] += 2 * eta_i.dot(psi)

                else:
                    print("loc_direct_approach MUST be an integer from 1-7, see Attia's Notes on OED_Localization!")
                    raise ValueError

        else:
            print("Unsupported Localization space %s" % repr(localization_space))
            print("Localization space must be B, R1, or R2")
            raise ValueError


        # add regularization term:
        gamma = self.filter_configs['localization_design_penalty']
        regularization_norm = self.filter_configs['regularization_norm']  # get regularization norm
        if re.match(r"\Al(_|-)*1\Z", regularization_norm, re.IGNORECASE):
            if gamma != 0:
                grad += gamma
            else:
                pass
        #
        elif re.match(r"\Al(_|-)*2\Z", regularization_norm, re.IGNORECASE):
            if gamma != 0:
                if _scalar:
                    grad += gamma * np.sqrt(state_size)
                else:
                    grad += (float(gamma)/np.linalg.norm(_loc_radius, 2)) * _loc_radius
            else:
                pass
        else:
            print("Unrecognized norm %s " % regularization_norm)
            raise ValueError
        #

        if self._verbose:
            sep = "\n%s\n" % ("*"*80)
            print(sep)
            print("gamma: ", gamma)
            print("Inside 'obj_localization_fun_gradient'. Passed Localization radius is: ", _loc_radius)
            print("\nGradient is: ", grad)
            print(sep)

        if FD_Validation:
            self.validate_gradient( _loc_radius, grad, objective=self.obj_localization_fun_value, FD_eps=FD_eps, FD_central=FD_central)


        if self._verbose:
            print("Gradient info:")
            print("Localization radius factor(s):", _loc_radius)
            print("Gradient:", grad)

        return grad

    def _localization_iter_callback(self, loc_radius, raise_for_negative_vals=False):
        """
        """
        obj_val = self.obj_localization_fun_value(loc_radius)
        if self.__opt_tracker is None:
            self.__opt_tracker = [(loc_radius, obj_val)]
        else:
            self.__opt_tracker.append((loc_radius, obj_val))
        #
        # to be called after each iteration of the optimizer
        if self._verbose:
            sepp = "\n"*2 + "*^="*120 + "\n"*2
            print(sepp + "... Iteration Callback:")
            print(" > Localization Radii passed by LBFGSB:", loc_radius)
            print("... Getting out or iteration callback function... " + sepp)
        
        if obj_val < 0 and raise_for_negative_vals:
            raise NameError
        else:
            pass

    def _cov_mat_loc(self, Pb, loc_radius, loc_func):
        """
        Localize the covariance matrix via pointwise product
        """
        # print("Input Pb", Pb)
        model = self.model
        state_size = model.state_size()

        Pb_copy = Pb.copy()


        # Naive Localization:

        loc_direct_approach = self.filter_configs['loc_direct_approach']
        if loc_direct_approach<1 or loc_direct_approach>7:
            print("loc_direct_approach MUST be an integer from 1-7, see Attia's Notes on OED_Localization!")
            raise ValueError

        if loc_func is None:
            localization_function = self.filter_configs['localization_function']
        else:
            localization_function = loc_func
        localization_radius = loc_radius

        # number of model-grid dimensions
        try:
            num_dimensions = self.model.model_configs['num_dimensions']
        except(KeyError):
            num_dimensions = None
        # get the observational grid
        observations_positions = self.model.get_observations_positions()
        num_obs_dims = np.size(observations_positions, 1)
        #
        if num_dimensions is None:
            num_dimensions = num_obs_dims
        elif (max(num_dimensions, num_obs_dims) == 1):
            num_dimensions = num_obs_dims = 1
        else:
            pass
        #
        try:
            periodic_bc = self.model.model_configs['periodic']
        except(KeyError, NameError, AttributeError):
            periodic_bc = False

        if periodic_bc:
            dx = model.model_configs['dx']
        else:
            dx = 1

        if num_dimensions <= 1:
            try:
                model_grid = self.model.get_model_grid()
                model_grid = np.asarray(model_grid).squeeze()
            except:
                model_grid = np.arange(state_size)

            # Localization in the observation space:
            for st_ind in xrange(state_size):
                ref_st_coord = model_grid[st_ind]
                # distances = np.abs(model_grid[st_ind: ] - ref_st_coord).squeeze()
                distances = np.abs(model_grid[:] - ref_st_coord).squeeze()
                if distances.ndim == 0:
                    distances = np.array([distances.item()])

                #
                # update distances for periodic models/boundary-conditions (e.g. Lorenz 96)
                if periodic_bc:
                    rem_distances = (model_grid[-1]-model_grid[0]) +(model_grid[1]-model_grid[0])  - distances
                    up_distances = distances > rem_distances
                    distances[up_distances] = rem_distances[up_distances]
                #
                # print("distances", distances)
                if np.isscalar(localization_radius):
                    loc_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                else:
                    if loc_direct_approach == 7:
                        loc_coeffs = utility.calculate_mixed_localization_coefficients( (localization_radius, localization_radius[st_ind]), distances, localization_function)
                    else:
                        # loc_coeffs = utility.calculate_localization_coefficients(localization_radius[st_ind: ], distances, localization_function)
                        loc_coeffs = utility.calculate_localization_coefficients(localization_radius[st_ind], distances, localization_function)
                #

                if loc_direct_approach == 1:
                    # radius is fixed over rows of the covariance matrix:
                    Pb[st_ind, : ] *= loc_coeffs

                elif loc_direct_approach == 2:
                    # radius is fixed over rows of the covariance matrix:
                    Pb[:, st_ind ] *= loc_coeffs

                elif loc_direct_approach == 3 or loc_direct_approach == 6:
                    # radius is fixed over rows of the covariance matrix:
                    #
                    if np.isscalar(localization_radius):
                        Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                        Pb[st_ind:, st_ind ] *= loc_coeffs[st_ind: ]

                    else:
                        vert_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                        Pb[st_ind:, st_ind ] *= (0.5 * (loc_coeffs[st_ind: ] + vert_coeffs[st_ind: ]))
                        Pb[st_ind, st_ind: ] *= (0.5 * (loc_coeffs[st_ind: ] + vert_coeffs[st_ind: ]))


                elif loc_direct_approach == 4:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                    Pb[st_ind: , st_ind] *= loc_coeffs[st_ind: ]

                elif loc_direct_approach == 5:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied upwards:
                    Pb[st_ind, : st_ind+1 ] *= loc_coeffs[st_ind: ]
                    Pb[ : st_ind+1, st_ind] *= loc_coeffs[st_ind: ]

                elif loc_direct_approach == 7 :
                    #
                    if np.isscalar(localization_radius):
                        Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                        Pb[st_ind:, st_ind ] *= loc_coeffs[st_ind: ]

                    else:
                        Pb[st_ind:, st_ind ] *= loc_coeffs[st_ind: ]
                        Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]

                else:
                    print("loc_direct_approach MUST be an integer from 1-7loc_direct_approach, see Attia's Notes on OED_Localization!")
                    raise ValueError
                # Pb[st_ind, st_ind] = 0.9 * Pb[st_ind, st_ind] + 0.1


        elif num_dimensions==2:
            #
            # WARNING: This should not be called online for big models. Consider calculating offline, and writing to file!
            try:
                model_grid = self.model.get_model_grid()
            except:
                try:
                    nx = self.model.model_configs['nx']
                    ny = self.model.model_configs['ny']
                    dx = self.model_configs['dx']
                    dy = self.model_configs['dy']
                except:
                    nx = np.int(np.floor(np.sqrt(state_size)))
                    ny = state_size / nx
                    dx = 1.0
                    dy = 1.0
                    model_grid = np.empty((state_size, 2))

                    x_indexes = np.arange(nx) * dx
                    y_indexes = np.arange(ny) * dy
                    model_grid[:, 0] = list(x_indexes) * ny  # test if reshaping is faster!
                    model_grid[:, 1] = np.repeat(y_indexes, nx)
                    #

            for st_ind, ref_coord in enumerate(model_grid):
                # coords = model_grid[st_ind: , :]
                coords = model_grid
                distance = np.sqrt( (coords[:, 0]-ref_coord[0])**2 + (coords[:, 1]-ref_coord[1])**2 )
                # print("distance", distance, localization_radius)

                if False:  # TODO: remove after debugging
                    if np.isscalar(localization_radius):
                        loc_coeffs = utility.calculate_localization_coefficients(radius=localization_radius,
                                                                                 distances=distance,
                                                                                 method=localization_function)
                    else:
                        loc_coeffs = utility.calculate_localization_coefficients(radius=localization_radius[st_ind],
                                                                                 distances=distance,
                                                                                 method=localization_function)
                else:
                    if np.isscalar(localization_radius):
                        loc_coeffs = utility.calculate_localization_coefficients(localization_radius, distances, localization_function)
                    else:
                        if loc_direct_approach == 7:
                            # loc_coeffs = utility.calculate_localization_coefficients(np.sqrt(np.asarray(localization_radius[st_ind:]))*localization_radius[st_ind], distances, localization_function)
                            loc_coeffs = utility.calculate_mixed_localization_coefficients( (localization_radius, localization_radius[st_ind]), distances, localization_function)
                        else:
                            loc_coeffs = utility.calculate_localization_coefficients(localization_radius[st_ind], distances, localization_function)

                # print("loc_coeffs", loc_coeffs)

                # Pb[st_ind, st_ind:] *= loc_coeffs
                # Pb[st_ind:, st_ind] *= loc_coeffs
                if loc_direct_approach == 1:
                    # radius is fixed over rows of the covariance matrix:
                    Pb[st_ind, : ] *= loc_coeffs

                elif loc_direct_approach == 2:
                    # radius is fixed over rows of the covariance matrix:
                    Pb[:, st_ind ] *= loc_coeffs

                elif loc_direct_approach == 3 or loc_direct_approach == 6:
                    # radius is fixed over rows of the covariance matrix:
                    #
                    if np.isscalar(localization_radius):
                        Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                        Pb[st_ind:, st_ind ] *= loc_coeffs[st_ind: ]

                    else:
                        vert_coeffs = utility.calculate_localization_coefficients(localization_radius, distance, localization_function)
                        Pb[st_ind:, st_ind ] *= (0.5 * (loc_coeffs[st_ind: ] + vert_coeffs[st_ind: ]))
                        Pb[st_ind, st_ind: ] *= (0.5 * (loc_coeffs[st_ind: ] + vert_coeffs[st_ind: ]))
                    #

                elif loc_direct_approach == 4:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied downwards:
                    Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                    Pb[st_ind: , st_ind] *= loc_coeffs[st_ind: ]

                elif loc_direct_approach == 5:
                    # radius is fixed over rows, and columnsn of the covariance matrix, and are varied upwards:
                    Pb[st_ind, : st_ind+1 ] *= loc_coeffs[st_ind: ]
                    Pb[ : st_ind+1, st_ind] *= loc_coeffs[st_ind: ]

                elif loc_direct_approach == 7 :
                    #
                    if np.isscalar(localization_radius):
                        Pb[st_ind, st_ind: ] *= loc_coeffs[st_ind: ]
                        Pb[st_ind:, st_ind ] *= loc_coeffs[st_ind: ]

                    else:
                        Pb[st_ind:, st_ind ] *= loc_coeffs
                        Pb[st_ind, st_ind: ] *= loc_coeffs

                else:
                    print("loc_direct_approach MUST be an integer from 1-6, see Attia's Notes on OED_Localization!")
                    raise ValueError
                pass
                #
        #
        else:
            print("Only up to 2D models are handled Here!")
            raise NotImplementedError

        # print("Pb: ", Pb)

        #
        return Pb


# This class needs a double  check! It should perform identially to the other classes when only on
# of them is required; this is not the case though!
class EnKF_OED_Adaptive(EnKF_OED_Localization, EnKF_OED_Inflation):

    __round_num_digits = 4  # round optimal solution of inflation parameters
    #
    def __init__(self, filter_configs=None, output_configs=None):
        #
        # ADAPYIVE INFLATION SETTINGS:
        # -----------------------------
        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, EnKF_OED_Inflation._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, EnKF_OED_Inflation._local_def_output_configs)
        EnKF_OED_Inflation.__init__(self, filter_configs=filter_configs, output_configs=output_configs)
        #
        orig_f = self.filter_configs['forecast_inflation_factor']
        if np.isscalar(orig_f):
            self.__original_inflation_factor = orig_f
        else:
            self.__original_inflation_factor = orig_f.copy()
        #
        default_bounds = EnKF_OED_Inflation._def_local_filter_configs['inflation_factor_bounds']
        inflation_factor_bounds = self.filter_configs['inflation_factor_bounds']
        try:
            lb, ub = inflation_factor_bounds
            if None not in [lb, ub]:
                if lb >= ub:
                    print("The bounds must be ranked increasingly with upper bound < lower bound!")
                    raise ValueError
        except:
            print("Failed to get the optimizer bounds on the inflation factor; using default values % " % str(default_bounds))
            self.filter_configs.update({'inflation_factor_bounds':default_bounds})

        # set numpy random seed, and preserve current state:
        try:
            random_seed = self.filter_configs['random_seed']
        except KeyError:
            random_seed = None

        if random_seed is None:
            self._random_state = None
        else:
            self._random_state = np.random.get_state()
            #
        np.random.seed(random_seed)

            #
        # ADAPYIVE LOCALIZATION SETTINGS:
        # --------------------------------
        # aggregate configurations, and attach filter_configs, output_configs to the filter object.
        filter_configs = utility.aggregate_configurations(filter_configs, EnKF_OED_Localization._def_local_filter_configs)
        output_configs = utility.aggregate_configurations(output_configs, EnKF_OED_Localization._local_def_output_configs)
        EnKF_OED_Localization.__init__(self, filter_configs=filter_configs, output_configs=output_configs)
        #
        self.__original_forecast_state = None  # this creates a copy of the forecast ensemble
        orig_l = self.filter_configs['localization_radius']
        if np.isscalar(orig_l):
            self.__original_localization_radius = orig_l
        else:
            self.__original_localization_radius = orig_l.copy()
        #
        default_bounds = EnKF_OED_Localization._def_local_filter_configs['localization_radii_bounds']
        localization_radii_bounds = self.filter_configs['localization_radii_bounds']
        try:
            state_size = self.model.state_size()
            # Get and correct the bounds:
            lb, ub = localization_radii_bounds
            if lb is None:
                lb = 0
            lb = max(lb, 0)
            if ub is None:
                ub = state_size/2
            ub = min(ub, state_size/2)
            if lb >= ub:
                print("The bounds must be ranked increasingly with upper bound < lower bound!")
                raise ValueError
            else:
                adjusted_bounds = (lb, ub)
        except:
            print("Failed to get the optimizer bounds on the localization radii; using default values % " % str(default_bounds))
            adjusted_bounds = default_bounds
        self.filter_configs.update({'localization_radii_bounds':adjusted_bounds})
        
        self.__opt_tracker = None

        self.__initialized = True
        #

    #
    def analysis(self, all_to_numpy=True):
        """
        Analysis step:

        Args:
            all_to_numpy (default False): bool,
                convert all data structures to Numpy and re-place results into target structures only in the end.

        Returns:
            None. Only self.filter_configs is updated.

        """
        # Model information
        model = self.filter_configs['model']
        state_size = model.state_size()
        observation_size = model.observation_size()
        model_name = model._model_name
        try:
            if re.match(r'\Alorenz', model_name, re.IGNORECASE):
                periodic = True
            elif re.match(r'\AQ(_|-)*G', model_name, re.IGNORECASE):
                periodic = False
            else:
                print("The model you selected ['%s'] is not supported" % model_name)
                periodic = True
        except:
            periodic = False

        # make sure the original forecast ensemble and background state are packedup
        original_forecast_ensemble = [ v.copy() for v in self.filter_configs['forecast_ensemble'] ]  # this creates a copy of the
        self.filter_configs.update({'original_forecast_ensemble': original_forecast_ensemble})
        self.filter_configs.update({'original_forecast_state': utility.ensemble_mean(original_forecast_ensemble)})

        #
        print("--> Adaptive Inflation <--")
        # Optimize for an inflation factor:
        # =========================================
        # 1- Initilize an inflation factor vector
        x0 = self.filter_configs['forecast_inflation_factor']
        if x0 is None:
            x0 = np.empty(state_size)  # default initial inflation factors vector
            x0[:] = 1.05
        elif np.isscalar(x0):
            if x0 <=1:
                x0 = 1.05
            x0 = np.ones(state_size) * float(x0)
        else:
            x0 = np.array(x0).flatten()
            if (x0 <=1).any():
                print("The inflation is set to 1 in all entries! No Inflation is needed with this solution;")
                x0[np.where(x0<=1)] = 1.05

        #
        # 2- Create an optimizer
        obj_fun = lambda x: self.obj_inflation_fun_value(x)
        obj_grad = lambda x: self.obj_inflation_fun_gradient(x)
        callback_fun = lambda x: self._inflation_iter_callback(x)
        #
        optimizer_configs = self.filter_configs['optimizer_configs']
        lb, ub = self.filter_configs['inflation_factor_bounds']
        bounds = [(lb, ub)] * state_size  # can be passed in the configurations dictionary; TODO.

        #
        opts = {'maxiter':10000,
                'ftol':1e-06,
                'gtol':1e-05,
                'xtol':1e-05
                }
        const = ({'type': 'ineq',
                  'fun': lambda x: self.obj_inflation_fun_value(x) - 1e-02,
                  # 'fun': obj_fun,
                  'jac': obj_grad})
        method = self.filter_configs['optimizer_configs']['method']
        try:
            res = minimize(obj_fun, x0, method=method, jac=obj_grad, hess=None, hessp=None, bounds=bounds, constraints=const, tol=1e-08, callback=callback_fun, options=opts)
            # print(res)
            opt_x = res.x
            f = res.fun
            d = {'warnflag': int(not res.success)}
            opt_res = {'success':res.success, 'status':res.status, 'message':res.message, 'nfev':res.nfev, 'njev':res.njev, 'nit':res.nit}
        except:
            opt_x = x0
            f = 0
            d = {'warnflag': 1}
            opt_res = {'success':0, 'status':1, 'message':'Optimizer failed', 'nfev':0, 'njev':0, 'nit':0}
        orig_opt_x = opt_x.copy()

        # apply a moving average to the optimal solution
        moving_average_radius = self.filter_configs['moving_average_radius']
        r = min(moving_average_radius, state_size/2-1)
        if r > 0:
            opt_x[:] = utility.moving_average(opt_x, radius=r, periodic=periodic)

        opt_x = np.round(opt_x, self.__round_num_digits)
        if self._verbose:
            sepp = "\n%s\n" % ("{|}"*50)
            print(sepp + "OED-Inflation RESULTS: %s" % '-'*15)
            print("res: ", res)
             # This is to be rewritten appropriately after debugging
            print("optimal inflation_fac:", opt_x)
            print("Original optimal inflation_fac: ", orig_opt_x)
            print("Minimum inflation factor entry:", opt_x.min())
            print("Maximum inflation factor entry:", opt_x.max())
            print("Average inflation factor:", np.mean(opt_x))
            print("Standard Deviation of inflation factor entries:", np.std(opt_x))
            print(" >> Minimum Objective (posterior-covariance trace): ", f)
            print("flags: ", d)
            print(sepp)
        #

        # Save the results, and calculate the results' statistics
        failed = d['warnflag']  # 0 flag --> converged
        if failed:
            print(d)
            self.filter_configs['analysis_state'] = None
            sep = "/^\\"*30
            print(sep + "\n\n\tThe Optimizer algorithm Miserably failed!\n\n" + sep)
            # raise ValueError
            pass

        # add regularization term (starting with L1 norm here):
        alpha = self.filter_configs['inflation_design_penalty']
        regularization_norm = self.filter_configs['regularization_norm']  # get regularization norm
        post_trace = f
        if re.match(r"\Al(_|-)*1\Z", regularization_norm, re.IGNORECASE):
            if alpha !=0:
                regularizer = alpha * np.sum(opt_x - 1)
            else:
                regularizer = 0.0
            post_trace += regularizer
        elif re.match(r"\Al(_|-)*2\Z", regularization_norm, re.IGNORECASE):
            if alpha !=0:
                regularizer = alpha * np.linalg.norm(opt_x-1, 2)
            else:
                regularizer = 0.0
            pass
            post_trace += regularizer
        else:
            print("Unrecognized norm %s " % regularization_norm)
            raise ValueError
        self.filter_configs.update({'inflation_opt_results':(orig_opt_x, opt_x, f, d, post_trace, opt_res)})

        #
        # Reset forecast information
        self.filter_configs['forecast_ensemble'] = [x.copy() for x in self.filter_configs['original_forecast_ensemble']]
        self.filter_configs['forecast_state'] = self.filter_configs['original_forecast_state'].copy()

        # Analysis with optimal inflation factor
        self.filter_configs['forecast_inflation_factor'] = opt_x
        self.filter_configs.update({'optimal_inflation_factor': opt_x})

        #
        print("--> Adaptive Localization <--")
        # Optimize for an inflation factor:
        # =========================================
        orig_l = self.filter_configs['localization_radius']
        if np.isscalar(orig_l):
            self.__original_localization_radius = orig_l
        else:
            self.__original_localization_radius = orig_l.copy()

        # Model state, and observation sizes
        state_size = model.state_size()
        observation_size = model.observation_size()

        # Get the localization space:
        localization_space = self.filter_configs['localization_space']
        if re.match(r'\AB\Z', localization_space, re.IGNORECASE):
            # B-Localization
            opt_space_size = state_size
        elif re.match(r'\AR(-| |_)*(1|2)\Z', localization_space, re.IGNORECASE):
            # R localization
            opt_space_size = observation_size
        else:
            print("Unsupported Localization space %s" % repr(localization_space))
            print("Localization space must be B, R1, or R2")
            raise ValueError

        x0 = orig_l
        if x0 is None:
            x0 = np.empty(opt_space_size)  # default initial inflation factors vector
            x0[:] = 4.0
        elif np.isscalar(x0):
            x0 = np.ones(opt_space_size) * float(x0)
        else:
            pass
        #
        # 2- Create an optimizer
        self.__opt_tracker = None  # refresh tracker
        obj_fun = lambda x: self.obj_localization_fun_value(x)
        obj_grad = lambda x: self.obj_localization_fun_gradient(x)
        callback_fun = lambda x: self._localization_iter_callback(x)
        #
        optimizer_configs = self.filter_configs['optimizer_configs']
        lb, ub = self.filter_configs['localization_radii_bounds']
        bounds = [(lb, ub)] * opt_space_size  # can be passed in the configurations dictionary; TODO.
        #
        #
        ftol = 1e-6
        maxiter = optimizer_configs['maxiter']
        opts = {'maxiter':maxiter,
                'ftol':ftol,
                'disp':optimizer_configs['disp']
                }
        if False:
            const=()
        else:
            const = ({'type': 'ineq',
                      'fun': lambda x: self.obj_localization_fun_value(x) - 0.50,
                      # 'fun': obj_fun,
                      'jac': obj_grad})
        method = self.filter_configs['optimizer_configs']['method']
        try:
            res = minimize(obj_fun, x0,
                           method=method,
                           jac=obj_grad,
                           hess=None,
                           hessp=None,
                           bounds=bounds,
                           constraints=const,
                           tol=ftol,
                           callback=callback_fun,
                           options=opts)
            # print(res)
            opt_x = res.x
            f = res.fun
            d = {'warnflag': int(not res.success)}
            opt_res = {'success':res.success, 'status':res.status, 'message':res.message, 'nfev':res.nfev}
            try:
                opt_res.update({'njev':res.njev})
            except:
                pass
            try:
                opt_res.update({'nit':res.nit})
            except:
                pass

        except NameError:
            obj_vals = np.asarray([x[-1] for x in self.__opt_tracker])
            mask = np.where(obj_vals>0)[0]
            try:
                loc = mask[np.argsort(obj_vals[mask])[0]]
                opt_x = self.__opt_tracker[loc][0]
                f = self.__opt_tracker[loc][0][1]
                print("optimal solution set to: %s" % repr(opt_x))
                d = {'warnflag': 0}
                opt_res = {'warnflag': 0, 'task':'Reset to the last known good solution from optimizer log', 'funcalls':len(self.__opt_tracker), 'nit':len(self.__opt_tracker)}
            except(IndexError):
                opt_x = x0
                f = 0
                d = {'warnflag': -1}
                opt_res = {'warnflag': -1, 'task':'Failed. No good iterations ', 'funcalls':len(self.__opt_tracker), 'nit':len(self.__opt_tracker)}

        except(ValueError):
            opt_x = x0
            f = 0
            d = {'warnflag': 1}
            opt_res = {'success':0, 'status':1, 'message':'Optimizer failed', 'nfev':0, 'njev':0, 'nit':0}
            raise
        orig_opt_x = opt_x.copy()
        # print("Optimal localization radii:", opt_x)
        # apply a moving average to the optimal solution
        moving_average_radius = self.filter_configs['moving_average_radius']
        r = min(moving_average_radius, opt_space_size/2-1)
        if r > 0:
            model_name = model._model_name
            try:
                if re.match(r'\Alorenz', model_name, re.IGNORECASE):
                    periodic = True
                elif re.match(r'\AQ(_|-)*G', model_name, re.IGNORECASE):
                    periodic = False
                else:
                    print("The model you selected ['%s'] is not supported" % model_name)
                    periodic = True
            except:
                periodic = False

            # print("Applying Moving average of r=%d " % r)
            # print("Opt_x PRE: ", opt_x)
            #
            opt_x = utility.moving_average(orig_opt_x, radius=r, periodic=periodic)
            # avg_opt_x = [np.mean(opt_x[i-r:i+r]) for i in xrange(r, opt_space_size-r)]
            # l_avg, u_avg = np.mean(opt_x[: r+1]), np.mean(opt_x[opt_space_size-2-r: ])
            # for i in xrange(r):
            #     # avg_opt_x.insert(0, opt_x[i])
            #     avg_opt_x.insert(0, l_avg)
            #     # avg_opt_x.append(opt_x[opt_space_size-1-r+i])
            #     avg_opt_x.append(u_avg)
            # try:
            #     opt_x[:] = avg_opt_x[:]
            # except:
            #     print("avg_opt_x", avg_opt_x)
            # # print("opt_x POST ", opt_x)

            # print("Utility-based smoothed Opt_x: ", opt_x2)
            # print("Two methods match? ", opt_x[:] == opt_x2[:])
            # print

        opt_x = np.round(opt_x, self.__round_num_digits)


        if self._verbose:
            print("Optimal solution: ", opt_x)
            print("res: ", res)

        #
        if self._verbose:
             # This is to be rewritten appropriately after debugging
            sepp = "\n%s\n" % ("{|}"*50)
            print(sepp + "OED-Localization RESULTS: %s" % '-'*15)
            print("optimal localization radii:", opt_x)
            print("Minimum localization radii entry:", opt_x.min())
            print("Maximum localization radii entry:", opt_x.max())
            print("Average localization radii:", np.mean(opt_x))
            print("Standard Deviation of localization radii entries:", np.std(opt_x))
            print(" >> Minimum Objective (posterior-covariance trace): ", f)
            print("flags: ", d)
            print(sepp)
        #

        # Save the results, and calculate the results' statistics
        failed = d['warnflag']  # 0 flag --> converged
        if failed:
            # self.filter_configs['analysis_state'] = None
            sep = "*"*30
            print(sep + "\n\n\tThe Optimizer algorithm Miserably failed!\n\n" + sep)
            obj_vals = np.asarray([x[-1] for x in self.__opt_tracker])
            mask = np.where(obj_vals>0)[0]
            try:
                loc = mask[np.argsort(obj_vals[mask])[0]]
                opt_x = self.__opt_tracker[loc][0]
                print("optimal solution set to: %s" % repr(opt_x))
            except(IndexError):
                opt_x = x0
            # raise ValueError
            pass

        # add regularization term (starting with L2 norm here):
        gamma = self.filter_configs['localization_design_penalty']
        post_trace = f
        regularization_norm = self.filter_configs['regularization_norm']  # get regularization norm
        if re.match(r"\Al(_|-)*1\Z", regularization_norm, re.IGNORECASE):
            regularizer = gamma * np.sum(opt_x)
            post_trace -= regularizer
        elif re.match(r"\Al(_|-)*2\Z", regularization_norm, re.IGNORECASE):
            regularizer = gamma * np.linalg.norm(opt_x, 2)
            post_trace -= regularizer
        else:
            print("Unrecognized norm %s " % regularization_norm)
            raise ValueError
        #
        self.filter_configs.update({'localization_opt_results':(orig_opt_x, opt_x, f, d, post_trace, opt_res)})

        #
        # Analysis with optimal inflation factor
        self.filter_configs['localization_radius'] = opt_x
        self.filter_configs.update({'optimal_localization_radius': opt_x})
        #
        # Reset forecast information
        self.filter_configs['forecast_ensemble'] = self.filter_configs['original_forecast_ensemble']
        self.filter_configs['forecast_state'] = self.filter_configs['original_forecast_state']

        # print("Re Analysis with localixzation radius: ", self.filter_configs['localization_radius'])
        # print("Original optx is : ", orig_opt_x)
        #
        # Now, carry out The analysis step of EnKF with optimal inflation factor and localization radii
        #
        print("--> EnKF Analysis <--")
        # print("Re Analysis with localixzation radius: ", self.filter_configs['localization_radius'])
        # print("Original optx is : ", orig_opt_x)
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            res = super().super().analysis(all_to_numpy=all_to_numpy)
        else:
            # old-stype class
            res = DEnKF.analysis(self, all_to_numpy=all_to_numpy)

        # Reset inflation and localization parameters:
        if self.filter_configs['reset_inflation_factor']:
            self.filter_configs['forecast_inflation_factor'] = self.__original_inflation_factor
        if self.filter_configs['reset_localization_radius']:
            self.filter_configs['localization_radius'] = self.__original_localization_radius

        return res
        #


