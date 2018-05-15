
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
    AlphabeticalOED:
    ------
    A class implementing the standard alphabetical OED criteria for PDE-based applications.
    The standard approach, is the alphabetical OED approach, that includes:
        - A-Optimality,
        - C-Optimality,
        - D-Optimality,
        - G-Optimality,
        
"""


import numpy as np
import re
import os

try:
    import cPickle as pickle
except:
    import pickle
    
from scipy.sparse.linalg import LinearOperator
import scipy.optimize as optimize  # may be replaced with dolfin_adjoint or PyIPOpt later!

import dates_utility as utility
from oed_base import OEDBase, PriorBase
from state_vector_base import StateVectorBase as StateVector
from state_matrix_base import StateMatrixBase as StateMatrix
from observation_vector_base import ObservationVectorBase as ObservationVector



class Prior(PriorBase):
    """
    A class to construct a prior-distribution object. This should inherit the model-based background error model!
    It is used in the Bayesian approach for regularization
    
    """
    def __init__():
        raise NotImplementedError
        


class AlphabeticalOED(OEDBase):
    """
    A class implementing the standard (alphabetical) approach for optimal design of experiments.

    Args:
        oed_configs: dict, A dictionary containing configurations of the OED object.
            Supported configuarations:
                * oed_name (default None): string containing name of the OED object; used for output.
                * model (default None):  model object
                * window_bounds(default None): bounds of the experimental design window (should be iterable of lenght 2)                
                * observations_list (default None): a list of model.observation_vector objects;
                * obs_checkpoints (default None): time instance at which observations are taken/collected.
                    These are necessaryt for adjoint calculation
                * reference_time (default None): time instance at which the reference state is provided (if available);
                * reference_state(default None): model.state_vector object containing the reference/true state;
                    this is provided only if observations_list is not available, and is used to create synthetic observations!
                * initial_design: the initial design vector,
                * optimal_design: the optimal experimental design,
                    Assumptions about the design:
                        a) the design is a vector of weights corresponding to observation vector entries;
                            i.e. the design size is equal to the size of model-based observation vector,
                        b) design is assumed, at least for now, to be time-independent.
           
        output_configs: dict,
            A dictionary containing screen/file output configurations.
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * verbose (default False): This is used for extensive outputting e.g. while debugging
                * file_output (default True): Save results to file on/off switch
                * file_output_dir (default True): full path of the directory to output results in
                * file_output_separate_files (default True): save all results to a single or multiple files
                * file_output_file_name_prefix (default 'OED_results'): name/prefix of output file
                * file_output_file_format (default 'txt'): file ouput format.
                    Supported formats:
                        - 'txt' or 'ascii': text files
                        - 'pickle': python pickled objects,                        
                * file_output_variables (default ['oed_statistics']): a list of variables to ouput. 
            
    """
    _oed_name = "Alphabetical-OED"
    _local_def_oed_configs = dict(oed_name=_oed_name,
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
                                                         ),
                                  )
    _local_def_4DVAR_output_configs = dict(scr_output=False,
                                           file_output=True,
                                           file_output_separate_files=True,
                                           file_output_file_name_prefix='OED_results',
                                           file_output_file_format='txt',
                                           oed_statistics_dir='OED_Statistics'
                                           )
       
    _supported_oed_criteria = ['A', 'D']
    _supported_reg_norms = ['L1',]
    
    
    #
    def __init__(self, oed_configs=None, output_configs=None):
        
        self.oed_configs = utility.aggregate_configurations(oed_configs, AlphabeticalOED._local_def_oed_configs)
        self.output_configs = utility.aggregate_configurations(output_configs, AlphabeticalOED._local_def_4DVAR_output_configs)
        
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().__init__(oed_configs=oed_configs, output_configs=self.output_configs)
        else:
            # old-type class
            super(AlphabeticalOED, self).__init__(oed_configs=oed_configs, output_configs=self.output_configs)
        #
        self.model = self.oed_configs['model']
        #
        try:
            self._model_step_size = self.model._default_step_size
        except:
            self._model_step_size = AlphabeticalOED.__time_eps
        self._time_eps = AlphabeticalOED.__time_eps
        
        self._supported_oed_criteria = [c.lower() for c in AlphabeticalOED._supported_oed_criteria]
        self._supported_reg_norms = [c.lower() for c in AlphabeticalOED._supported_reg_norms]
        
        self.__initialized = True
        #
            
    
    def regularization_term_value(self, design, reg_norm):
        """
        evaluate the value of the regularization term, evaluated at the given design
        
        Args:
            design:
            reg_param:
            reg_norm:
            
        Returns:
            reg_term:
            
        """
        raise NotImplementedError
    
    #
    def regularization_term_gradient(self, design, reg_norm):
        """
        evaluate the gradient of the regularization term, evaluated at the given design
        
        Args:
            design:
            reg_param:
            reg_norm:
            
        Returns:
            reg_term:
            
        """
        raise NotImplementedError
    
    
    #
    #
    def Hessian_solve(self, state, design=None):
        """
        Evaluate the product of the Hessian-inverse by the given state.
        The passed design udpates the observation error covariance matrix diagonal if not None.
        
        Args:
            state:
            design:
            
        Returns:
            out_vec:
            
        """
        raise NotImplementedError
    
    
    #
    #
    def oed_objective(self, design, criterion, reg_param, reg_norm):
        """
        Evaluate the value and the gradient of the objective function evaluated at the given design
        
        Args:
            design:
            criterion: optimality criterion;
                supported criteria (so far):
                    'A':
                    'D':
                    
            reg_param: regularization parameter (weight of the regularization term)
            reg_norm: the norm used in the regularization term
            
        
        Returns:
            value:
            gradient:
            
        """
        value = self.objective_function_value(design, criterion, reg_param, reg_norm)
        gradient = self.objective_function_gradient(design, reg_param, reg_norm)
        return value, gradient
        #
        
    #
    def objective_function_value(self, design, criterion, reg_param, reg_norm):
        """
        Evaluate the value of the objective function evaluated at the given design
        
        Args:
            design:
            criterion: optimality criterion;
                supported criteria (so far):
                    'A':
                    'D':
                    
            reg_param: regularization parameter (weight of the regularization term)
            reg_norm: the norm used in the regularization term
        
        Returns:
            objective_value
            
        """
        if not isinstance(criterion, str):
            msg = "criterion must be a string representation of the OED optimality criterion!"
            for i, c in zip(len(self._supported_oed_criteria), self._supported_oed_criteria):
                msg += "\n\t%d: %s" % (i, str(c))
            print(msg)
            raise AssertionError
        else:
            criterion = criterion.lower()
        # assert isinstance(criterion, str), msg
        if criterion not in self._supported_oed_criteria:
            msg = "criterion must be a string representation of the OED optimality criterion!"
            for i, c in zip(len(self._supported_oed_criteria), self._supported_oed_criteria):
                msg += "\n\t%d: %s" % (i, str(c))
            print(msg)
            raise AssertionError
        else:
            pass
        
        # TODO: Proceed here:
        
        #
        model = self.model
        #
        # TODO: check if copying is actually necessary after finishing the implementation...
        if isinstance(design, np.ndarray):
            local_design = self.model.observation_vector()
            local_design[:] = design.copy()
        else:
            local_design = design.copy()
        
        # Evaluate the observation term:
        if criterion == 'a':
            observation_term = self.a_opt_objective(local_design)
            
        elif criterion == 'd':
            observation_term = 
            
        else:
            msg = "Unsupported OED optimality criterion [%s]!" % criterion
            print(msg)
            raise ValueError
            
        # Evaluate the regularization term:
        regularization_term = self.regularization_term_gradient(local_design, reg_norm):
        
        # Add the two terms and return:
        objective_value = observation_term + reg_param * regularization_term
        
        raise NotImplementedError
        #
        return objective_value
        #
        
    def a_opt_objective(self, design, randomized_approach=False):
        """
        The A-Optimality objective function at the given design; this does NOT include any regularization term.
        
        Args:
            desgin: 
            randomized_approach: flag to use randomized algorithm(s) to calculate/approximate the trace of the posterior covariance matrix
            
        Returns:
            a_opt_val: value of the A-Optimality objective
            
        """
        # Retrieve a pointer to the model object:
        model = self.model
        state_size = model.state_size()
        #
        
        if verbose:
            print "A-Optimality"
        #
        e_vec = model.state_vector()
        e_vec[:] = 1.0
        
        if randomized_approach:
            # TODO: to be implemented
            print("Randomized approach for trace estimation is To-Be-Implemented")
            raise NotImplementedError
        
        else:
            trace = 0.0            
            for e_ind in xrange(state_size):
                e_vec[:] = 0.0
                e_vec[e_ind] = 1.0
                #
                H_inv_a = Hessian_solve(e_vec)
                #
                if verbose:
                    print("Evaluating diagonal of P H^{-1} ; entry [%d] out of [%d]. \
                           Variance[%d]=%8.6e" % (e_ind, state_size, e_ind, H_inv_a[0]))
                trace += float(H_inv_a[e_ind])
            a_opt_val = trace
        #
        
        if False:
            # this can be used if an analysis ensemble is available
            a_opt_val = utility.covariance_trace(analysis_ensemble, model=filter_obj.model)
        #
        if self._verbose:
            print "Objective Value:", a_opt_val
        #
        return a_opt_val
        #
    
    
    
        
    def d_opt_objective(self, design, randomized_approach=False):
        """
        The D-Optimality objective function at the given design; this does NOT include any regularization term.
        
        Args:
            desgin: 
            randomized_approach: flag to use randomized algorithm(s) to calculate/approximate the trace of the posterior covariance matrix
            
        Returns:
            a_opt_val: value of the D-Optimality objective
            
        """
        # Retrieve a pointer to the model object:
        model = self.model
        state_size = model.state_size()
        #
        
        if verbose:
            print "D-Optimality"
        #
        e_vec = model.state_vector()
        e_vec[:] = 1.0
        
        if randomized_approach:
            # TODO: to be implemented
            print("Randomized approach for trace estimation is To-Be-Implemented")
            raise NotImplementedError
        
        else:
            #
            
            #
            posterior_cov = np.empty((state_size, state_size))
            e_vec = model.state_vector()
            for e_ind in xrange(state_size):
                e_vec[:] = 0
                e_vec[e_ind] = 1.0
                posterior_cov[:, e_ind] = Hessian_solve(e_vec)
                    
            #
            full_det = False
            if not full_det:  
                try:
                    eigvals = sc_linalg.eigvalsh(posterior_cov)
                    if verbose:
                        print "*** scipy.linalg.eigvalsh succeeded ***"
                        sys.stdout.flush()
                except(sc_linalg.LinAlgError):
                    try:    
                        if verbose:
                            print("*** scipy.linalg.eigvalsh Failed *** \n*** \
                                   Trying eigvals instead: ... "),
                        eigvals = sc_linalg.eigvals(posterior_cov)
                        if verbose:
                            print "***\n*** scipy.linalg.eigvals sucedded ***"
                            sys.stdout.flush()
                    except(sc_linalg.LinAlgError):
                        print "\nCouldn't use Eigenvalues to evaluate log-det"
                        sys.stdout.flush()
                        raise
                #
                cost = np.sum(np.log(eigvals))
                #
                if verbose:
                    print "\n\nfull det:", np.log(sc_linalg.det(posterior_cov))
                    # print "sum log det", cost,"\n\n"
            #  
            else:
                print "Evaluating Full determinant ..."
                # sys.stdout.flush()
                # print "posterior_cov", posterior_cov
                cost = np.log(sc_linalg.det(posterior_cov))
            
            d_opt_val = trace
        #
        
        if False:
            # this can be used if an analysis ensemble is available
            
        #
        if self._verbose:
            print "Objective Value:", D_opt_val
        #
        return d_opt_val
        #
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
    #
    def objective_function_gradient(self, design, criterion, reg_param, reg_norm, FD_Validation=False, FD_eps=1e-7, FD_central=True):
        """
        Evaluate the gradient of the objective function evaluated at the given design
        
        Args:
            state: 
            FD_Validation: 
            FD_eps: 
            FD_central:
            
        Returns:
            objective_gradient:
            
        """
        if not isinstance(criterion, str):
            msg = "criterion must be a string representation of the OED optimality criterion!"
            for i, c in zip(len(self._supported_oed_criteria), self._supported_oed_criteria):
                msg += "\n\t%d: %s" % (i, str(c))
            print(msg)
            raise AssertionError
        else:
            criterion = criterion.lower()
        # assert isinstance(criterion, str), msg
        if criterion not in self._supported_oed_criteria:
            msg = "criterion must be a string representation of the OED optimality criterion!"
            for i, c in zip(len(self._supported_oed_criteria), self._supported_oed_criteria):
                msg += "\n\t%d: %s" % (i, str(c))
            print(msg)
            raise AssertionError
        else:
            pass
        
        # get a pointer to the model object
        model = self.model
        #
        
        # TODO: Proceed here:
        
        if isinstance(state, np.ndarray):
            local_state = model.state_vector()
            local_state[:] = state.copy()
        else:
            local_state = state.copy()
        
        #
        # Start Evaluating the gradient:
        
        
        raise NotImplementedError
        #
        # return objective_gradient
        #
    
    #
    #
    def __validate_gradient(self, gradient, design, criterion, reg_param, reg_norm, FD_eps=1e-5, FD_central=False):
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
    def print_results(self):
        """
        Print OED results to the main terminal
        A check on the corresponding options in the configurations dictionary is made to make sure
        saving is requested.
        """
        class OldStyle: pass
        if issubclass(OldStyle().__class__, object):
            # object-inherited class
            super().print_results()
        else:
            # old-type class
            super(AlphabeticalOED, self).print_results()
        #
        pass  # Add more...
        #
    
    #
    def save_results(self, output_dir=None, cleanup_out_dir=False, save_err_covars=False):
        """
        Save OED results to file(s).
        A check on the correspondidng options in the configurations dictionary is made to make sure
        saving is requested.
        
        Args:
            out_dir (default None): directory to put results in. 
                The output_dir is created (with all necessary parent paths) if it is not on disc.
                The directory is relative to DATeS root directory.
            
            cleanup_out_dir (default None): bool,
                Takes effect if the output directory is not empty. True: remove directory contents.
            
        
        """
        raise NotImplementedError
        #
        #
        # The first code block that prepares the output directory can be moved to parent class later...
        # Retrieve output configurations
        output_configs = self.output_configs
        file_output = output_configs['file_output']
        if not file_output:
            raise ValueError("The output flag is turned of. The method 'save_results' is called though!")

        # We are good to go! --> Start preparing directories (if necessary) then save results...
        if output_dir is not None:
            file_output_directory = output_dir
        else:
            file_output_directory = output_configs['file_output_dir']
        # clean-up output directory;
        if cleanup_out_dir:
            parent_path, out_dir = os.path.split(file_output_directory)
            utility.cleanup_directory(directory_name=out_dir, parent_path=parent_path)
        # check the output sub-directories...
        oed_statistics_dir = os.path.join(file_output_directory, output_configs['oed_statistics_dir'])
        model_states_dir = os.path.join(file_output_directory, output_configs['model_states_dir'])
        observations_dir = os.path.join(file_output_directory, output_configs['observations_dir'])
        file_output_variables = output_configs['file_output_variables']  # I think it's better to remove it from the oed base...

        if not os.path.isdir(oed_statistics_dir):
            os.makedirs(oed_statistics_dir)
        if not os.path.isdir(model_states_dir):
            os.makedirs(model_states_dir)
        if not os.path.isdir(observations_dir):
            os.makedirs(observations_dir)

        # check if results are to be saved to separate files or appended on existing files.
        # This may be overridden if not adequate for some output (such as model states), we will see!
        file_output_separate_files = output_configs['file_output_separate_files']
        
        # This is useful for saving oed statistics but not model states or observations as models should handle both
        file_output_file_name_prefix = output_configs['file_output_file_name_prefix']  # this is useless!
        
        # Format of the ouput files
        file_output_file_format = output_configs['file_output_file_format'].lower()
        if file_output_file_format not in ['mat', 'pickle', 'txt', 'ascii']:
            print("The file format ['%s'] is not supported!" % file_output_file_format )
            raise ValueError()
        
        # Retrieve oed and ouput configurations needed to be saved
        oed_configs = self.oed_configs  # we don't need to save all configs
        oed_conf= dict(oed_name=oed_configs['oed_name'],
                       oed_criterion=oed_configs['oed_criterion'],
                       initial_time=oed_configs['initial_time'],
                       obs_checkpoints=oed_configs['obs_checkpoints'],
                       observations_list=oed_configs['observations_list'],
                       )
        io_conf = output_configs
        
        # Start writing settings, and reuslts:
        # 1- write model configurations configurations:
        model_conf = self.model.get_model_configs()
        if file_output_file_format == 'pickle':
            pickle.dump(model_conf, open(os.path.join(file_output_directory, 'model_configs.pickle')))
        elif file_output_file_format in ['txt', 'ascii', 'mat']:  # 'mat' here has no effect.
            utility.write_dicts_to_config_file('model_configs.txt', file_output_directory,
                                                model_conf, 'Model Configs'
                                                )
          
        # 2- get a proper name for the folder (cycle_*) under the model_states_dir path;
        # I will keep it as it is, so that we can extend it to the case where the design is updated sequentially (e.g. for averaging!)
        suffix = 0
        while True:
            cycle_dir = 'cycle_' + str(suffix)
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
        
        # 3- save reference, forecast, and analysis states; use model to write states to file(s)
        # i- save reference state
        reference_state = self.oed_configs['reference_state'].copy()
        self.model.write_state(state=reference_state, directory=cycle_states_out_dir, file_name='reference_state')
        # ii- save forecast state
        forecast_state = self.oed_configs['forecast_state']
        self.model.write_state(state=forecast_state, directory=cycle_states_out_dir, file_name='forecast_state')
        # iii- save analysis state
        analysis_state = self.oed_configs['analysis_state'].copy()
        self.model.write_state(state=analysis_state, directory=cycle_states_out_dir, file_name='analysis_state')
        
        # 4- Save observations to files; use model to write observations to file(s)
        for observation in self.oed_configs['observations_list']:
            file_name = utility.try_file_name(directory=cycle_observations_out_dir, file_prefix='observation')
            self.model.write_observation(observation=observation, directory=cycle_observations_out_dir, file_name=file_name, append=False)

        # 4- Save oed configurations and statistics to file,
        # i- Output the configurations dictionaries:
        assim_cycle_configs_file_name = 'assim_cycle_configs'
        if file_output_file_format in ['txt', 'ascii', 'mat']:
            # Output oed and model configurations; this goes under state directory
            assim_cycle_configs_file_name += '.txt'
            utility.write_dicts_to_config_file(assim_cycle_configs_file_name, cycle_states_out_dir,
                                                   [oed_conf, io_conf], ['Smoother Configs', 'Output Configs'])
            
        elif file_output_file_format in ['pickle']:
            #
            # Output oed and model configurations; this goes under state directory
            assim_cycle_configs_file_name += '.pickle'
            assim_cycle_configs = dict(oed_configs=oed_conf, output_configs=io_conf)
            pickle.dump(assim_cycle_configs, open(os.path.join(cycle_states_out_dir, assim_cycle_configs_file_name)))
        
        else:
            raise ValueError("Unsupported output format for configurations dictionaries: '%s' !" % file_output_file_format)
            #
        
        # ii Output the RMSE results; it's meaningless to create a new file for each cycle:
        rmse_file_name = 'rmse.txt'  # RMSE are always saved in text files
        rmse_file_path = os.path.join(oed_statistics_dir, rmse_file_name)
        # Create a header for the file if it is newely created
        if not os.path.isfile(rmse_file_path):
            # rmse file does not exist. create file and add header.
            header = "RMSE Results: Smoother: '%s' \n %s \t %s \t %s \t %s \n" % (self._oed_name,
                                                                                  'Forecast-Time'.rjust(20),
                                                                                  'Analysis-Time'.rjust(20),
                                                                                  'Forecast-RMSE'.rjust(20),
                                                                                  'Analysis-RMSE'.rjust(20),
                                                                                  )
            if False:
                # get the initial RMSE and add it if forecast is done first...
                initial_time = self.oed_configs['forecast_time']
                initial_rmse = self.output_configs['oed_statistics']['initial_rmse']
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
            # rmse file does exist. Header should be already there!
            pass
            
        # Now rmse results file exists --> Append rmse results to the file:
        forecast_time = self.oed_configs['forecast_time']
        analysis_time = self.oed_configs['analysis_time']
        #
        forecast_rmse = self.output_configs['oed_statistics']['forecast_rmse'][0]
        analysis_rmse = self.output_configs['oed_statistics']['analysis_rmse'][0]
        if self._verbose:
            print("forecast_time, forecast_rmse, analysis_time, analysis_rmse")
            print(forecast_time, forecast_rmse, analysis_time, analysis_rmse)
        output_line = u" {0:20.14e} \t {1:20.14e} \t {2:20.14e} \t {3:20.14e} \n".format(forecast_time,
                                                                                         analysis_time,
                                                                                         forecast_rmse,
                                                                                         analysis_rmse
                                                                                         )
        # now write the rmse results to file
        with open(rmse_file_path, mode='a') as file_handler:
            file_handler.write(output_line)
        #
    
        # save error covariance matrices if requested; these will go in the state output directory
        if save_err_covars:
            Pf = self.oed_configs['forecast_error_covariance']
            Pa = self.oed_configs['forecast_error_covariance']
            print("Saving covariance matrices is not supported yet. CDF will be considered soon!")
            raise NotImplementedError()        
        else:
            pass
    
    
    #
    def read_cycle_results(self, output_dir, read_err_covars=False):
        """
        Read OED results from file(s).
        Check the output directory first. If the directory does not exist, raise an IO error.
        If the directory, and files exist, Start retrieving the results properly
        
        Args:
            output_dir: directory where AlphabeticalOED results are saved. 
                We assume the structure is the same as the structure created by the AlphabeticalOED implemented here
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
        
        

