
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



class ModelsBase(object):
    """
    Abstract class for DATeS dynamical models' (wrappers') implementation.
    
    A base class for dynamical models implementations/wrappers.
    The implementation in classes inheriting this base class should carry out all essential tasks should be generally provided by dynamical model (wrappers).           

    **Remarks**                                                                                       
    The methods defined here are all the essential methods needed by all the assimilation implemented by default in DATeS releases.                  
    
    
    Args:
        model_configs: dict,
            A configurations dictionary for model settings
            This is expected to vary greately from one model to another. 
            The configs provided here are just guidelines.
            Supported Configurations:
            * model_name: string representation of the model name
            * time_integration_scheme: decides what time integration used to propagate model state and/or 
                perturbations forward/backward in time.
            * default_step_size: optimized (stable) step size for model propagation. 
                This can be usefule when long timesteps in timespans passed to the forward propagation method
            * num_prognostic_variables: Number of prognostic/physical variables at each gridpoint.
            * model_grid_type: 'cartesian', 'spherical', etc.
            * num_spatial_dimensions: number of dimensions in the physical domain.
            * model_errors_distribution: probability distribution of model errors (e.g. for imperfect model 
                with additive noise)
            * model_error_variances: an iterable that contains the model errors' variances 
                (e.g. for models with Gaussian model errors)
            * model_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
            * model_errors_covariance_localization_radius: radius of influence for model error covariance 
                matrix localization
            * model_errors_covariance_method': an indicator of the method used to construct model error 
                covariances
            * model_errors_steps_per_model_steps: Non-negative integer; number of model steps after which 
                (additive) model noise are added to the model state when the time integrator is called to 
                propagate the model.state_vector forward in time.
                If the value is zero, no model errors are incorporated (Perfect Model).
            * observation_operator_type (default 'linear'): type of the relation between model state and 
                observations, e.g. 'linear', etc.
            * observation_vector_size: size of the observation vector. This can be used e.g. to control 
                the observation operator construction process
            * observation_errors_distribution (default 'gaussian'): probability distribution of observational
                errors (e.g. defining the likelihood fuction in the data assimialtion settings)
            * observation_errors_variances: an iterable that contains the observational errors' variances 
            * observation_errors_covariance_method: an indicator of the method used to construct observational
                error covariances
            * observation_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
            * observation_errors_covariance_localization_radius: radius of influence for observation error 
                covariance matrix localization
            * background_error_variances: an iterable that contains the background errors' variances 
            * background_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
            * background_errors_covariance_localization_radius: radius of influence for background error 
                covariance matrix localization
            * background_errors_covariance_method': an indicator of the method used to construct background
                error covariances
           
        output_configs: dict,
            A dictionary containing screen/file output configurations for the model
            Supported configuarations:
                * scr_output (default False): Output results to screen on/off switch
                * scr_output_iter (default 1): printing to screen after this number of model steps 
                * verbose (default False): This is used for extensive screen outputting e.g. while debugging.
                    This overrides scr_output.
                * file_output (default False): Output results to file on/off switch
                * file_output_iter (default 1): saving model results to file after this number of model steps 
                * file_output_dir: Location to save model output to (if file_output is set to True).
                    Some default directory should be provided by model or an exception will be thrown.            
        
    """
    # Default model configurations
    # These will definitely vary much for each model. Additional configurations (passed to the class 
    # constructor) will be aggregated with these. Options in the configurations dictionary passed to the
    # class constructor overrides the default configurations of course.
    # Here we provide what we believe to be the common features. Will be updated periodically.
    _default_model_configs = {'model_name': None,
                              'time_integration_scheme':None,
                              'default_step_size':None,
                              'num_prognostic_variables':1,
                              'model_grid_type': 'cartesian',
                              'num_spatial_dimensions':1,
                              'model_errors_distribution': 'gaussian',
                              'model_error_variances':None,
                              'model_errors_covariance_localization_function':None,
                              'model_errors_covariance_localization_radius':None,                          
                              'model_errors_covariance_method':'empirical',
                              'model_errors_steps_per_model_steps': 0,
                              'observation_operator_type':'linear',
                              'observation_vector_size':None,
                              'observation_errors_distribution':'gaussian',
                              'observation_errors_variances':None,
                              'observation_errors_covariance_localization_function':None,
                              'observation_errors_covariance_localization_radius':None,   
                              'observation_errors_covariance_method':'empirical',                          
                              'background_errors_distribution':'gaussian',
                              'background_error_variances':None,
                              'background_errors_covariance_localization_function':None,
                              'background_errors_covariance_localization_radius':None,
                              'background_errors_covariance_method':'empirical'                        
                              }

    # Default I/O for the model.
    _default_output_configs = {'scr_output': False,
                               'scr_output_iter': 1,
                               'verbose':False,  # this can be used for extensive output, e.g. while debugging
                               'file_output': False,
                               'file_output_iter': 1,
                               'file_output_dir':None
                               }

    def __init__(self, model_configs=None, output_configs=None, *argv, **kwargs):
        
        self._default_step_size = None
        self._reference_initial_condition = self.create_initial_condition()
        self._initialized = False
        raise NotImplementedError
        #

    def state_vector(self, state_vector_ref=None, *argv, **kwargs):
        """
        Create a wrapper for a state vector data structure, and return a reference to it.
        state_vector_ref could be a numpy array, pointer to C/Fortran array, etc.
        
        Args:
            state_vector_ref:
            *argv:
            **kwargs:
            
        Returns:
            state_vector:
            
        """
        raise NotImplementedError
        #

    def _initialize_model_state_vector(self, *argv, **kwargs):
        """
        Create an empty vector data structure and return its reference.
        This could be a numpy array, pointer to C/Fortran array, etc.
        It should be called by self.state_vector() to create a wrapper that can be manipulated by the 
            associated linear algebra module. 
        
        Args:
            *argv:
            **kwargs:
            
        Returns:
            state_vector:
            
        """
        raise NotImplementedError
        #

    def state_vector_size(self):
        """
        Return the size of the state vector
        
        Returns:
            state_size:
            
        """
        # return self._state_size
        raise NotImplementedError
        #
    # add a useful alias to remove confusion
    state_size = state_dimension = state_vector_size
        #

    def observation_vector(self, observation_vector_ref=None, *argv, **kwargs):
        """
        Create a wrapper for a observation vector data structure, and return a reference to it.
        observation_vector_ref could be a numpy array, pointer to C/Fortran array, etc.
        
        Args:
            observation_vector_ref:
            *argv:
            **kwargs:
            
        Returns:
            observation_vector:
            
        """
        raise NotImplementedError
        #

    def _initialize_model_observation_vector(self, *argv, **kwargs):
        """
        Create an empty observation data strucgture and return its reference.
        This could be a numpy array, pointer to C/Fortran array, etc.
        It should be called by self.observation_vector() to create a wrapper that can be manipulated by the 
            associated linear algebra module. 
        
        Args:
            *argv:
            **kwargs:
            
        Returns:
            observation_vector:
        
        """
        raise NotImplementedError
        #

    def observation_vector_size(self):
        """
        return the size of an observation vector given the currently defined observation operator
        
        Returns:
            state_size:
            
        """
        raise NotImplementedError
        #
    # add a useful alias to remove confusion
    observation_size = observation_dimension = state_vector_size
        #

    def state_matrix(self, state_matrix_ref=None, *argv, **kwargs):
        """
        Create a wrapper for a model matrix data structure, and return a reference to it.
        This can hold things such as model or background error covariance matrices.
        state_matrix_ref could be a numpy array, pointer to C/Fortran array, etc.
        
        Args:
            state_matrix_ref:
            *argv:
            **kwargs:
            
        Returns:
            state_matrix:
            
        """
        raise NotImplementedError
        #

    def _initialize_model_state_matrix(self, *argv, **kwargs):
        """
        Create an empty state-based matrix data strucgture and return its reference.
        This could be a numpy array, pointer to C/Fortran array, etc.
        It should be called by self.state_matrix() to create a wrapper that can be manipulated by the 
            associated linear algebra module. 
        
        Args:
            *argv:
            **kwargs:
            
        Returns:
            state_matrix:
        
        """
        raise NotImplementedError
        #

    def observation_matrix(self, observation_matrix_ref=None, *argv, **kwargs):
        """
        Create a wrapper for a model matrix data structure, and return a reference to it.
        This can hold things such as observation error covariance matrix.
        observation_matrix_ref could be a numpy array, pointer to C/Fortran array, etc.
        
        Args:
            observation_matrix_ref:
            *argv:
            **kwargs:
            
        Returns:
            observation_matrix:
            
        """
        raise NotImplementedError
        #

    def _initialize_model_observation_matrix(self, *argv, **kwargs):
        """
        Create an empty observation-based matrix data strucgture and return its reference.
        This could be a numpy array, pointer to C/Fortran array, etc.
        It should be called by self.observation_matrix() to create a wrapper that can be manipulated by the 
            associated linear algebra module. 
        
        Args:
            *argv:
            **kwargs:
            
        Returns:
            observation:
            
        """
        raise NotImplementedError
        #

    def construct_model_grids(self, *argv, **kwargs):
        """
        Either construct the spatial grids of the model (if the model is fully implemented in Python),
        or obtain model grids from the given implementation (probably by a simple run and read output 
            files or so.)
        
        Args:
            *argv: 
            **kwargs:
        Returns:
            model_grid:
            
        """
        raise NotImplementedError
        #

    def step_forward_function(self, time_point, in_state, *argv, **kwargs):
        """
        In the simplest case, this implements the right-hand side of the model and evaluates it at 
            the given model state_vector
        
        Args:
            time_point: 
            in_state: 
            *argv: 
            **kwargs:
            
        Returns:
            right-hand-side function value
        """
        raise NotImplementedError
        #

    def step_forward_function_Jacobian(self, time_point, in_state, *argv, **kwargs):
        """
        The Jacobian of the right-hand side of the model and evaluate it at the given model state.
            rhs: the derivative/Jacobian of the right-hand side function evaluated at 'in_state'
        """
        raise NotImplementedError
        #

    def integrate_state(self, initial_state, checkpoints, *argv, **kwargs):
        """
        March the model state forward in time (backward for negative step or decreasing list).
        
        Args:
            initial_state: model.state_vector to be propagated froward according to checkpoints
            checkpoints: a timespan that should be a float scalar (taken as one step) or an iterable 
                including beginning, end, or a full timespan to build model trajectory at.
            
        Returns:
            trajectory: model trajectory. this can be either a list of model state_vector objects 
                (e.g. if checkpoints is an iterable) or simply a single state_vector if checkpoints is a scalar.
            
        """
        raise NotImplementedError
        #

    def integrate_state_perturbations(self, initial_perturbations, checkpoints, *argv, **kwargs):
        """
        March the model state perturbations forward in time (backward for negative step or decreasing list).
        
        Args:
            initial_perturbations: model.state_vector to be propagated froward according to checkpoints
            checkpoints: a timespan that should be a float scalar (taken as one step) or an iterable 
                including beginning, end, or a full timespan to build model trajectory at.
            
        Returns:
            trajectory: model trajectory. this can be either a list of model state_vector objects 
                (e.g. if checkpoints is an iterable) or simply a single state_vector if checkpoints is a scalar.
            
        """
        raise NotImplementedError
        #

    def update_observation_operator(self, time, *argv, **kwargs):
        """
        This should be called for each assimilation cycle if the observation operator is time-varying.
        
        Args:
            time: 
            *argv: 
            **kwargs:
        
        """
        raise NotImplementedError
        #

    def construct_observation_operator(self, *argv, **kwargs):
        """
        Construct the observation operator (H) in full. This should be avoided in practice.
        We need it's (or it's TLM) effect on a state vector always.
        
        Args:
            *argv: 
            **kwargs:
            
        """
        raise NotImplementedError
        #

    def evaluate_theoretical_observation(self, in_state, *argv, **kwargs):
        """
        Evaluate the theoretical observation corresponding to a model state vector,
        i.e. evaluate H(state), where H is the observation operator.
        
        Args:
            in_state: model.state_vector to be projected onto the observation space
            *argv: 
            **kwargs:
        
        Returns:
            observation: equivalent observation vector of the passed in_state. observation = H(in_state)
        
        """
        raise NotImplementedError
        #

    def construct_observation_operator_Jacobian(self, *argv, **kwargs):
        """
        This creates the Jacobian of the observation operator (forward operator).
        This could be a constant matrix if the observation operator is linear, or state-dependent matrix
        This might be called by evaluate_observation_operator_Jacobian if needed. 
        Most of the models won't need it.
        
        Args:
            *argv: 
            **kwargs:
            
        """
        raise NotImplementedError
        #

    def evaluate_observation_operator_Jacobian(self, in_state, *argv, **kwargs):
        """
        Evaluate the Jacobian of the observation operator at specific model state.
            i.e. evaluate $\mathbf{H}$ evaluated at in_state.
        
        Args:
            in_state: model.state_vector to evaluate the Jacobian of the observation operator at
            *argv: 
            **kwargs:
        
        Returns:
            Jacobian: the Jacobian of the observation operator evaluated at the passed in_state
        
        """
        raise NotImplementedError
        #

    def observation_operator_Jacobian_prod_vec(self, in_state, state, *argv, **kwargs):
        """
        Multiply the Jacobian of the observation operator (evaluated at in_state) by state
            i.e. evaluate $\mathbf{H} \times \text{state}$.
        The result is an observation vector
        
        Args:
            in_state: model.state_vector to evaluate the Jacobian of the observation operator at
            *argv: 
            **kwargs:
            
        """
        raise NotImplementedError
        #

    def observation_operator_Jacobian_T_prod_vec(self, in_state, observation, *argv, **kwargs):
        """
        Multiply the transpose of the Jacobian of the observation operator evaluated at in_state by observation
            i.e. evaluate $\mathbf{H}^T \times \text{observation}$, where $\mathbf{H}$ is evaluated at 
            in_state . Should generally be time dependent...
        The result is a state vector of course
        
        Args:
            in_state: model.state_vector to evaluate the Jacobian of the observation operator at
            observation: 
            *argv: 
            **kwargs:
            
        """
        raise NotImplementedError
        #

    def apply_state_covariance_localization(self, covariance_array, localization_function=None, 
                                            localization_radius=None, *argv, **kwargs):
        """
        Apply localization/decorrelation to a given state-based square array.
        This generally a point-wise multiplication of the decorrelation array/function and the passed 
            covariance_array
        
        Args:
            covariance_array: 
            localization_function:
            localization_radius: 
            *argv: 
            **kwargs:
        
        Returns:
            localized_covariance
            
        """
        raise NotImplementedError
        #

    def apply_observation_covariance_localization(self, covariance_array, localization_function=None, 
                                                  localization_radius=None, *argv, **kwargs):
        """
        Apply localization/decorrelation to a given observation square array.
        This generally a point-wise multiplication of the decorrelation array/function and the passed 
            covariance_array
        
        Args:
            covariance_array: 
            localization_function:
            localization_radius: 
            *argv: 
            **kwargs:
        
        Returns:
            localized_covariance
            
        """
        raise NotImplementedError
        #

    def create_observation_error_model(self, configs=None, *argv, **kwargs):
        """
        Create an observation error model
        
        Args:
            configs: 
            *argv: 
            **kwargs:
        
        Returns:
            observation_error_model
            
        """
        raise NotImplementedError
        #

    def create_background_error_model(self, configs=None, *argv, **kwargs):
        """
        Create an observation error model
        
        Args:
            configs: 
            *argv: 
            **kwargs:
        
        Returns:
            background_error_model
            
        """
        raise NotImplementedError
        #

    def create_model_error_model(self, configs=None, *argv, **kwargs):
        """
        Create a model error model
        
        Args:
            configs: 
            *argv: 
            **kwargs:
        
        Returns:
            model_error_model
            
        """
        raise NotImplementedError
        #
        
    def create_initial_condition(self, *argv, **kwargs):
        """
        Create initial condition state for the model. This state can be used as reference initial condition 
            in the experimental data assimilation settings.
        
        Args:
            *argv: 
            **kwargs:
        
        Returns:
            model_state
            
        """
        raise NotImplementedError
        #
        
    def create_initial_ensemble(self, ensemble_size, *argv, **kwargs):
        """
        create an ensemble of states. This can be used for example as an initial ensemble in the 
            ensemble-based data assimilation.
        All ensembles in DATeS are lists of state_vector objects.
        
        Args:
            ensemble_size:
            *argv: 
            **kwargs:
        
        Returns:
            initial_ensemble: list of model states
            
        """
        raise NotImplementedError
        #

    def get_neighbors_indexes(self, index, radius, *argv, **kwargs):
        """
        A function that returns the indexes within a given radius of influence w.r.t. a given index in the state vector.
        Of course this will be different for non-zero dimensional models.
        For zero dimensional models this will be straight forward, but for non-zeros dimensional models, this has to
        be handled carefully.
        
        Args:
            index: state vector index (0:state_size-1)
            radius: radius of influence
            *argv: 
            **kwargs:
        
        Returns:
            indexes_list: a list containing spatial indexes with the radius of influence from the passed index
            
        """
        raise NotImplementedError
        #

    def ensemble_covariance_matrix(self, ensemble, localize=False, *argv, **kwargs):
        """
        Construct an ensemble-based covariance matrix
        
        Args:
            ensemble: a list (ensemble) of model states
            localize (default False): flag upon which spatial decorrelation is applied or not.
            *argv: 
            **kwargs:
        
        Returns:
            ensemble_covariances: state_matrix containing ensemble-based covariances.
        
        """
        raise NotImplementedError
        #

    def write_state(self, state, directory, file_name, *argv, **kwargs):
        """
        Save a state vector to a file.
        
        Args:
            state: state_vector to be written to file
            directory: location where state vector will be saved
            file_name: name of the target file
            *argv: 
            **kwargs:
            
        Returns:
            None
        
        """
        raise NotImplementedError
        #

    def write_observation(self, observation, directory, file_name, *argv, **kwargs):
        """
        Save an observation vector to a file.
        
        Args:
            observation: observation_vector to be written to file
            directory: location where observation vector will be saved
            file_name: name of the target file
            *argv: 
            **kwargs:
            
        Returns:
            None
        
        """
        raise NotImplementedError
        #

    def read_state(self, file_name, directory, *argv, **kwargs):
        """
        Save a state vector to a file.
        
        Args:
            file_name: name of the target file
            directory: location where state vector will be read from
            *argv: 
            **kwargs:
            
        Returns:
            state: state_vector read from file
        
        """
        raise NotImplementedError
        #

    def read_observation(self, file_name, directory, *argv, **kwargs):
        """
        Read an observation vector from a file.
        
        Args:
            file_name: name of the target file
            directory: location where observation vector will be read from
            *argv: 
            **kwargs:
        
        Returns:
            observation: observation_vector read from file
        
        """
        raise NotImplementedError
        #

    def get_model_configs(self, *argv, **kwargs):
        """
        Return a dictionary containing model configurations.
        
        Args:
            *argv: 
            **kwargs:
        
        Returns:
            configs: a dictionary containing model configurations. can be used e.g. for file output.
        """
        raise NotImplementedError
        #



