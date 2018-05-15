
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
    TDVAR:
    ------
    A class implementing the standard Three Dimansional Variational DA scheme.
    This is a Vanilla implementation, and is intended for illustration purposes.
    Will be tuned as we go!
    
    Given an observation vector, forecast state, and forecast error covariance matrix 
    (or a representative forecast ensemble), the analysis step updates the 'analysis state'.
    
    **Remark**
        I am thinking of 3D-Var a variational scheme for filtering. This is why 'TDVAR' inherits 'FiltersBase'.
    
"""


import numpy as np
import os

try:
    import cPickle as pickle
except:
    import pickle
    
from scipy.sparse.linalg import LinearOperator
import scipy.optimize as optimize  # may be replaced with dolfin_adjoint or PyIPOpt later!

import dates_utility as utility
from filters_base import FiltersBase
from state_vector_base import StateVectorBase as StateVector
from state_matrix_base import StateMatrixBase as StateMatrix
from observation_vector_base import ObservationVectorBase as ObservationVector


class TDVAR(FiltersBase):
    """
    A class implementing the vanilla 3D-Var DA scheme.
    I am considering 3D-Var a filtering scheme...

    Given an observation vector, forecast state, and forecast error covariance matrix 
        (encapsualted in model.background_error_model) the analysis step updates the 'analysis state';
        Note that there is no time dimension involved here; everything is assumed to be given at the same time instance.
    
    Args:
        filter_configs:  dict, a dictionary containing filter configurations.
            Supported configuarations:
            ---------------------------
                * model (default None):  model object
                * filter_name (default None): string containing name of the filter; used for output.
                * optimizer: the optimization algorithm to be used to minimize the objective
                * optimizer_configs: settings of the optimizer
                    - Supported configurations:
                    ----------------------------
                        + maxiter:
                        + maxfun:
                        + tol:
                        + reltol:
                        + pgtol:
                        + epsilon:
                        + factr:
                        + disp:
                        + maxls:
                        + iprint:
                * linear_system_solver (default 'lu'): String containing the name of the system solver 
                    used to solver for the inverse of internal matrices. e.g. $(HBH^T+R)^{-1}(y-Hx)$
                * analysis_state (default None): model.state_vector object containing the analysis state.
                    This is where the filter output (analysis state) will be saved and returned.
                * forecast_state (default None): model.state_vector object containing the forecast state.
                * observation (default None): model.observation_vector object
                * reference_state(default None): model.state_vector object containing the reference/true state
                * apply_preprocessing (default False): call the pre-processing function before filtering
                * apply_postprocessing (default False): call the post-processing function after filtering
                * screen_output (default False): Output results to screen on/off switch
                
        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default 'Results'): relative path (to DATeS root directory) 
                    of the directory to output results in
                * file_output_separate_files (default True): save all results to a single or multiple files
                * file_output_file_name_prefix (default 'TDVAR_results'): name/prefix of output file
                * file_output_file_format (default 'mat'): file ouput format.
                    Supported formats:
                        - 'mat': matlab matrix files,
                        - 'pickle': python pickled objects,
                        - 'txt' or 'ascii': text files
                        
                * file_output_variables (default ['filter_statistics']): a list of variables to ouput. 
            
    """
    _filter_name = "3D-VAR"
    _local_def_3DVAR_filter_configs = dict(filter_name=_filter_name,
                                             optimizer='lbfgs',  # TODO: Revisit this...
                                             optimizer_configs=dict(maxiter=10000,
                                                                    maxfun=1000,
                                                                    tol=1e-12,
                                                                    reltol=1e-5,
                                                                    pgtol=1e-05, 
                                                                    epsilon=1e-08,
                                                                    factr=10000000.0,
                                                                    disp=1, 
                                                                    maxls=50,
                                                                    iprint=-1
                                                                    )
                                             )
    _local_def_3DVAR_output_configs = dict(scr_output=False,
                                           file_output=True,
                                           file_output_separate_files=True,
                                           file_output_file_name_prefix='TDVAR_results',
                                           file_output_file_format='mat',
                                           filter_statistics_dir='Filter_Statistics',
                                           model_states_dir='Model_States_Repository',
                                           observations_dir='Observations_Rpository',
                                           )
    
    #
    local__time_eps = 1e-7  # this can be useful to compare time instances
    try:
        #
        __time_eps = os.getenv('DATES_TIME_EPS')
        if __time_eps is not None:
            __time_eps = eval(__time_eps)
        else:
            pass
        #
    except :
        __time_eps = None
    finally:
        if __time_eps is None:
            __time_eps = local__time_eps
        elif np.isscalar(__time_eps):
            pass
        else:
            print("\n\n** Failed to get/set the value of '__time_eps'**\n setting it to %f\n " % local__time_eps)
            __time_eps = local__time_eps
            # raise ValueError
    
        
    #
    def __init__(self, filter_configs=None, output_configs=None):
        
        self.filter_configs = utility.aggregate_configurations(filter_configs, TDVAR._local_def_3DVAR_filter_configs)
        self.output_configs = utility.aggregate_configurations(output_configs, TDVAR._local_def_3DVAR_output_configs)
        
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(filter_configs=filter_configs, output_configs=self.output_configs)
        else:
            # old-type class
            super(TDVAR, self).__init__(filter_configs=filter_configs, output_configs=self.output_configs)
        #
        self.model = self.filter_configs['model']
        #
        try:
            self._model_step_size = self.model._default_step_size
        except:
            self._model_step_size = TDVAR.__time_eps
        self._time_eps = TDVAR.__time_eps
        
        
        #
        self.__initialized = True
        #
            
    #
    def filtering_cycle(self, update_reference=False):
        """
        Carry out a single assimilation cycle. This is just the analysis in this case
        All required variables are obtained from 'filter_configs' dictionary.
        
        Args:
            update_reference (default True): bool,
                A flag to decide whether to update the reference state in the filter or not.
                This should be removed!
                  
           
        """
        #
        model = self.model
        
        try:
            observation_time = self.filter_configs['observation_time']
        except(NameError, KeyError, AttributeError):
            observation_time = None
            
        try:
            forecast_time = self.filter_configs['forecast_time']
        except(NameError, KeyError, AttributeError):
            forecast_time = None
            
        try:
            analysis_time = self.filter_configs['analysis_time']
        except(NameError, KeyError, AttributeError):
            analysis_time = None
        
        try:
            reference_time = self.filter_configs['reference_time']
        except(NameError, KeyError, AttributeError):
            reference_time = None
            
        #
        try:
            # Retrieve the reference state and evaluate initial root-mean-squared error
            reference_state = self.filter_configs['reference_state'].copy()
        except(NameError, KeyError, AttributeError):
            reference_state = None
            #
            if self._verbose:
                print("Couldn't retrieve the reference state/time! ")
                print("True observations must be present in this case, and RMSE can't be evaluated!")
            
        
        if reference_state is not None and reference_time is not None:
            # If the reference time is below window final limit, march it forward:
            if analysis_time is not None:
                new_time = analysis_time
            elif forecast_time is not None:
                new_time = forecast_time
            elif reference_time is not None:
                new_time = reference_time
                
            if (new_time - reference_time) > self._time_eps:
                # propagate forward the reference state, and update it
                local_trajectory = model.integrate_state(reference_state, [reference_time, new_time])
                reference_time = new_time
                if isinstance(local_trajectory, list):
                    reference_state = local_trajectory[-1]
                else:
                    reference_state = local_trajectory
                
            elif (reference_time - new_time) > self._time_eps:
                print("Can't have the reference time after the forecast/analysis time!")
                raise ValueError
            
            
            if update_reference:
                self.filter_configs.update({'reference_state': reference_state})
                self.filter_configs.update({'reference_time': reference_time})
                
            
        
        # Check time instances:
        times_array = np.array([observation_time, analysis_time, forecast_time, reference_time])
        val_inds = np.where(times_array)[0]
        if len(val_inds)>0:
            if not np.allclose(times_array[val_inds], times_array[val_inds[0]], atol=self._time_eps, rtol=self._time_eps):
                print("Given time instances, e.g. forecast time, observation time, etc. MUST match! ")
                print("[observation_time, analysis_time, forecast_time, reference_time]", [observation_time, analysis_time, forecast_time, reference_time])
                raise AssertionError
            else:
                # Good to go!
                pass
        else:
            # No times are given; just proceed
            pass
        
        
        #
        try:
            forecast_state = self.filter_configs['forecast_state'].copy()
        except(NameError, KeyError, AttributeError):
            print("Couldn't retrieve the forecast state! This can't work; Aborting!")
            raise AssertionError
        #
        try:
            observation = self.filter_configs['observation']
        except(NameError, KeyError, AttributeError):
            observation = None
        finally:
            if observation is None:
                print("The observation is not found!")
                raise AssertionError
                
        #
        # Forecast state/ensemble should be given:
        if forecast_state is None:
            try:
                forecast_state = utility.ensemble_mean(self.filter_configs['forecast_ensemble'])
            except:
                print("Couldn't find either forecast state or forecast ensemble while analysis should be done first!")
                raise AssertionError
        
        #
        # calculate the initial/forecast RMSE: this is the RMSE before assimilation at the first entry of the time-span
        
        state_size = model.state_size()
        
        #
        if forecast_state is not None and reference_state is not None:
            f_rmse = initial_rmse = utility.calculate_rmse(forecast_state, reference_state, state_size)
        else:
            f_rmse = initial_rmse = 0
            
        #
        # Start the filtering process: preprocessing -> filtering(forecast->+<-anslsysis) -> postprocessing
        if self.filter_configs['apply_preprocessing']:
            self.cycle_preprocessing()
        
        #
        # Analysis step (This calls the filter's analysis step.)
        # > --------------------------------------------||
        sep = "\n" + "*"*80 + "\n"
        if self._verbose:
            print("%s ANALYSIS STEP %s" % (sep, sep) )
        self.analysis()
        # > --------------------------------------------||
        #
        
        analysis_state = self.filter_configs['analysis_state'].copy()        
        
        
        # Analysis RMSE:
        if reference_state is not None:
            a_rmse = utility.calculate_rmse(reference_state, analysis_state, state_size)
        else:
            a_rmse = 0
        
        #
        if True:
            print("Initial (f_rmse, a_rmse) :", (f_rmse, a_rmse))
            print("analysis_state", analysis_state)
            print("forecast_state", forecast_state)
            print("reference_state", reference_state)
        
        
        #
        # Apply post-processing if required
        if self.filter_configs['apply_postprocessing']:
            self.cycle_postprocessing()
        
        #
        # Update filter statistics (including RMSE)
        if 'filter_statistics' not in self.output_configs:
            self.output_configs.update(dict(filter_statistics=dict(initial_rmse=None,forecast_rmse=None, analysis_rmse=None)))
        else:
            if 'analysis_rmse' not in self.output_configs['filter_statistics']:
                self.output_configs['filter_statistics'].update(dict(analysis_rmse=None))
            if 'forecast_rmse' not in self.output_configs['filter_statistics']:
                self.output_configs['filter_statistics'].update(dict(forecast_rmse=None))
            if 'initial_rmse' not in self.output_configs['filter_statistics']:
                self.output_configs['filter_statistics'].update(dict(initial_rmse=None))

        # now update the RMSE's
        self.output_configs['filter_statistics'].update({'initial_rmse': initial_rmse})
        self.output_configs['filter_statistics'].update({'forecast_rmse': f_rmse})
        #
        self.output_configs['filter_statistics'].update({'analysis_rmse': a_rmse})

        # output and save results if requested
        if self.output_configs['scr_output']:
            self.print_cycle_results()
        if self.output_configs['file_output']:
            self.save_cycle_results()
        #
     
        
    #
    def analysis(self):
        """
        Analysis step of the (Vanilla 3DVar) filter. 
            
        """
        # state and observation vector dimensions
        observation = self.filter_configs['observation']
        if observation is None:
            print("Nothing to assimilate...")
            print("Setting the analysis state to the forecast state...")
            
            self.filter_configs['analysis_state'] = self.filter_configs['forecast_state'].copy()
            return
            #
        else:
            # Good to go!
            pass
        
        #
        # > --------------------------------------------||
        # START the 3D-VAR process:
        # > --------------------------------------------||
        #
        # set the optimizer, and it's configurations (e.g. tol, reltol, maxiter, etc.)
        # start the optimization process:
        optimizer = self.filter_configs['optimizer'].lower()
        #
        if optimizer == 'lbfgs':  # TODO: Consider replacing with the general interface optimize.minimize...
            #
            # Retrieve/Set the objective function, and the gradient of the objective function
            tdvar_value = lambda x: self.objective_function_value(x)
            tdvar_gradient = lambda x: self.objective_function_gradient(x)
            #
            optimizer_configs = self.filter_configs['optimizer_configs']
            x0 = self.filter_configs['forecast_state'].get_numpy_array()
            #
            x, f, d = optimize.fmin_l_bfgs_b(tdvar_value, 
                                             x0, 
                                             fprime=tdvar_gradient, 
                                             m=10, 
                                             factr=float(optimizer_configs['factr']), 
                                             pgtol=optimizer_configs['pgtol'], 
                                             epsilon=optimizer_configs['epsilon'], 
                                             iprint=optimizer_configs['iprint'], 
                                             maxfun=optimizer_configs['maxfun'], 
                                             maxiter=optimizer_configs['maxiter'],
                                             maxls=optimizer_configs['maxls'],
                                             disp=optimizer_configs['disp']
                                             )
            # print "3D-VAR RESULTS:", "optimal state:", x, "Minimum:", f, "flags:", d  # This is to be rewritten appropriately after debugging 
            
            # Save the results, and calculate the results' statistics
            failed = d['warnflag']  # 0 flag --> converged
            if failed:
                print d
                self.filter_configs['analysis_state'] = None
                print("The 3D-Var algorithm Miserably failed!")
                raise ValueError
                
            analysis_state_numpy = x
            analysis_state = self.model.state_vector()
            analysis_state[:] = analysis_state_numpy
            self.filter_configs['analysis_state'] = analysis_state.copy()
            #        
        else:
            print("The optimizer '%s' is not recognized or not yet supported!" % optimizer)
            raise ValueError
            #        
            
        #
        # > --------------------------------------------||
        # END the 3D-VAR process:
        # > --------------------------------------------||
        #
    
    #
    #
    def tdvar_objective(self, state):
        """
        Return the value and the gradient of the objective function evaluated at the given state
        
        Args:
            state:
        
        Returns:
            value:
            gradient:
            
        """
        value = self.objective_function_value(state)
        gradient = self.objective_function_gradient(state)
        return value, gradient
        #
        
    #
    def objective_function_value(self, state):
        """
        Evaluate the value of the 3D-Var objective function
        
        Args:
            state:
        
        Returns:
            objective_value
            
        """  
        #
        model = self.model
        #
        if isinstance(state, np.ndarray):
            local_state = self.model.state_vector()
            local_state[:] = state.copy()
        else:
            local_state = state.copy()
        
        #
        forecast_state = self.filter_configs['forecast_state'].copy()
        
        #
        # Evaluate the Background term:  # I am assuming forecast vs. analysis times match correctly here.        
        state_dev = forecast_state.copy().scale(-1.0)  # <- state_dev = - forecast_state
        state_dev = state_dev.add(local_state, in_place=False)  # state_dev = x - forecast_state
        scaled_state_dev = state_dev.copy()
        scaled_state_dev = model.background_error_model.invB.vector_product(scaled_state_dev)
        background_term = scaled_state_dev.dot(state_dev)
        
        #
        # Evaluate the observation terms:
        observation = self.filter_configs['observation']
        
        #
        if self._verbose:
            print("In objective_function_value:")
            print("in-state", state)
            print("forecast_state", forecast_state)
            print("observation", observation)
            print("background_term", background_term)
            raw_input("\npress Enter to continue")
        
            #
        # an observation exists at the analysis time
        Hx = model.evaluate_theoretical_observation(local_state)
        obs_innov = observation.copy().scale(-1.0)
        obs_innov = obs_innov.add(Hx)
        scaled_obs_innov = obs_innov.copy()
        scaled_obs_innov = model.observation_error_model.invR.vector_product(scaled_obs_innov)
        observation_term = scaled_obs_innov.dot(obs_innov)
        
        #
        # Add observation and background terms and return:
        objective_value = 0.5 * (background_term + observation_term)
        
        #
        if self._verbose:
            print("In objective_function_value:")
            print("bacground state_dev: ", state_dev)
            print("bacground scaled_state_dev: ", scaled_state_dev)
            print("background_term", background_term)
            print()
            print("observation_term", observation_term)
            print("objective_value", objective_value)
            raw_input("\npress Enter to continue")
            
            #
        #
        return objective_value
        #
        
    #
    def objective_function_gradient(self, state, FD_Validation=False, FD_eps=1e-7, FD_central=True):
        """
        Evaluate the gradient of the 3D-Var objective function
        
        Args:
            state: 
            FD_Validation: 
            FD_eps: 
            FD_central:
            
        Returns:
            objective_gradient:
            
        """
        # get a pointer to the model object
        model = self.model
        #
        if isinstance(state, np.ndarray):
            local_state = model.state_vector()
            local_state[:] = state.copy()
        else:
            local_state = state.copy()
        
        #
        # Start Evaluating the gradient:
        
        #
        forecast_state = self.filter_configs['forecast_state'].copy()
        
        #
        # Evaluate the Background term:  # I am assuming forecast vs. analysis times match correctly here.        
        # state_dev = local_state.axpy(-1.0, forecast_state, in_place=False)
        state_dev = forecast_state.copy().scale(-1.0)
        state_dev = state_dev.add(local_state)
        
        background_term = model.background_error_model.invB.vector_product(state_dev, in_place=False)
        
        #
        # Evaluate the observation terms:
        # get observations list:
        obs = observation = self.filter_configs['observation']
        
        #
        if self._verbose:
            print("In objective_function_gradient:")
            print("in-state", state)
            print("forecast_state", forecast_state)
            print("observation", observation)
            print("background_term", background_term)
            raw_input("\npress Enter to continue")
        
        
        # The observation term:
        Hx = model.evaluate_theoretical_observation(local_state)
        obs_innov = Hx.axpy(-1.0, obs, in_place=False)  # innov = H(x) - y
        scaled_obs_innov = model.observation_error_model.invR.vector_product(obs_innov, in_place=False)
        observation_term = model.observation_operator_Jacobian_T_prod_vec(local_state, scaled_obs_innov)
        
        #
        # Add observation and background terms and return:
        objective_gradient = background_term.add(observation_term, in_place=False)
        
        #
        if self._verbose:
            print("In objective_function_gradient:")
            print("observation", observation)
            print("observation_term", observation_term)
            print("objective_gradient", objective_gradient)
            raw_input("\npress Enter to continue")
        
        
        #
        if FD_Validation:
            self.__validate_gradient(state, objective_gradient, FD_eps=FD_eps, FD_central=FD_central)
        #
        if isinstance(state, np.ndarray):
            objective_gradient = objective_gradient.get_numpy_array()
        
        #
        return objective_gradient
        #
        
    #
    #
    def __validate_gradient(self, state, gradient, FD_eps=1e-5, FD_central=False):
        """
        Use Finite Difference to validate the gradient
        
        Args:
            state: 
            gradient: 
            FD_eps: 
            FD_central:
            
        """
        #
        # Use finite differences to validate the Gradient (if needed):
        state_size = self.model.state_size()
        # Finite Difference Validation:
        eps = FD_eps
        grad_array = gradient.get_numpy_array()
        if isinstance(state, np.ndarray):
            state_array = state.copy()
        elif isinstance(state, (StateVector, StateMatrix)):
            state_array = state.get_numpy_array()
        else:
            print("Passed state is of unrecognized Type: %s" % repr(type(state)) )
            raise TypeError
        
        sep = "\n"+"~"*80+"\n"
        # print some statistics for monitoring:
        print(sep + "FD Validation of the Gradient" + sep)
        print("  + Maximum gradient entry :", grad_array.max())
        print("  + Minimum gradient entry :", grad_array.min())
        
        #
        perturbed_state = self.model.state_vector()
        state_perturb = np.zeros_like(grad_array)
        fd_grad = np.zeros_like(grad_array)
        #
        if not FD_central:
            f0 = self.objective_function_value(state)
        #
        
        for i in xrange(state_size):
            state_perturb[:] = 0.0
            state_perturb[i] = eps
            #
            if FD_central:
                perturbed_state[:] = state_array - state_perturb
                f0 = self.objective_function_value(perturbed_state)
            #
            perturbed_state[:] = state_array + state_perturb
            f1 = self.objective_function_value(perturbed_state)
            
            if FD_central:
                fd_grad[i] = (f1-f0)/(2.0*eps)
            else:
                fd_grad[i] = (f1-f0)/(eps)
            
            err = (grad_array[i] - fd_grad[i]) / fd_grad[i]
            print(">>>>Gradient/FD>>>> %4d| Grad = %+8.6e\t FD-Grad = %+8.6e\t Rel-Err = %+8.6e <<<<" % (i, grad_array[i], fd_grad[i], err))
    
        
    #
    def print_cycle_results(self):
        """
        Print results from the current cycle to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        """
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().print_cycle_results()
        else:
            # old-stype class
            super(TDVAR, self).print_cycle_results()
        pass  # Add more...
                  
        #
        
    
    #
    def save_cycle_results(self, output_dir=None, cleanup_out_dir=False, save_err_covars=False):
        """
        Save filtering results from the current cycle to file(s).
        A check on the correspondidng options in the configurations dictionary is made to make sure
        saving is requested.
        
        Args:
            out_dir (default None): directory to put results in. 
                The output_dir is created (with all necessary parent paths) if it is not on disc.
                The directory is relative to DATeS root directory.
            
            cleanup_out_dir (default None): bool,
                Takes effect if the output directory is not empty. True: remove directory contents.
            
        
        """
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
        
        # 3- save reference, forecast, and analysis states; use model to write states to file(s)
        # i- save reference state
        reference_state = self.filter_configs['reference_state'].copy()
        self.model.write_state(state=reference_state, directory=cycle_states_out_dir, file_name='reference_state')
        # ii- save forecast state
        forecast_state = self.filter_configs['forecast_state']
        self.model.write_state(state=forecast_state, directory=cycle_states_out_dir, file_name='forecast_state')
        # iii- save analysis state
        analysis_state = self.filter_configs['analysis_state'].copy()
        self.model.write_state(state=analysis_state, directory=cycle_states_out_dir, file_name='analysis_state')
        
        # 4- Save observations to files; use model to write observations to file(s)
        observation = self.filter_configs['observation']
        file_name = utility.try_file_name(directory=cycle_observations_out_dir, file_prefix='observation')
        self.model.write_observation(observation=observation, directory=cycle_observations_out_dir, file_name=file_name, append=False)

        # 4- Save filter configurations and statistics to file,
        # i- Output the configurations dictionaries:
        assim_cycle_configs_file_name = 'assim_cycle_configs'
        if file_output_file_format in ['txt', 'ascii', 'mat']:
            # Output filter and model configurations; this goes under state directory
            assim_cycle_configs_file_name += '.txt'
            utility.write_dicts_to_config_file(assim_cycle_configs_file_name, cycle_states_out_dir,
                                                   [self.filter_configs, self.output_configs], ['Filter Configs', 'Output Configs'])
            
        elif file_output_file_format in ['pickle']:
            #
            # Output filter and model configurations; this goes under state directory
            assim_cycle_configs_file_name += '.pickle'
            assim_cycle_configs = dict(filter_configs=self.filter_configs   , output_configs=self.output_configs)
            pickle.dump(assim_cycle_configs, open(os.path.join(cycle_states_out_dir, assim_cycle_configs_file_name)))
        
        else:
            raise ValueError("Unsupported output format for configurations dictionaries: '%s' !" % file_output_file_format)
            #
        
        # ii Output the RMSE results; it's meaningless to create a new file for each cycle:
        rmse_file_name = 'rmse.txt'  # RMSE are always saved in text files
        rmse_file_path = os.path.join(filter_statistics_dir, rmse_file_name)
        # Create a header for the file if it is newely created
        if not os.path.isfile(rmse_file_path):
            # rmse file does not exist. create file and add header.
            header = "RMSE Results: DA: '%s' \n %s \t %s \n" % (self._filter_name,
                                                                      'Forecast-RMSE'.rjust(20),
                                                                      'Analysis-RMSE'.rjust(20),
                                                                      )
            # dump the header to the file
            with open(rmse_file_path, mode='w') as file_handler:
                file_handler.write(header)
        else:
            # rmse file does exist. Header should be already there!
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
        # now write the rmse results to file
        with open(rmse_file_path, mode='a') as file_handler:
            file_handler.write(output_line)
        #
    
        # save error covariance matrices if requested; these will go in the state output directory
        if save_err_covars:
            Pf = self.filter_configs['forecast_error_covariance']
            Pa = self.filter_configs['forecast_error_covariance']
            print("Saving covariance matrices is not supported yet. CDF will be considered soon!")
            raise NotImplementedError()        
        else:
            pass
    
    
    #
    def read_cycle_results(self, output_dir, read_err_covars=False):
        """
        Read filtering results from file(s).
        Check the output directory first. If the directory does not exist, raise an IO error.
        If the directory, and files exist, Start retrieving the results properly
        
        Args:
            output_dir: directory where TDVAR results are saved. 
                We assume the structure is the same as the structure created by the TDVAR implemented here
                in 'save_cycle_results'.
            read_err_covars:
            
        Returns:
            reference_state:
            forecast_state:
            analysis_state:
            observation:
            forecast_err_covar:
            analysis_err_covar:
            
        """
        # TODO: To be written!
        raise NotImplementedError
        #
        
        

