
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
    A module providing implementations of several Lorenz models.
    This initially includes Lorenz-3, and Lorenz-96 (A.K.A Lorenz-40).
"""


import numpy as np
import scipy.sparse as sparse
import scipy.io as sio
import os
import re

import dates_utility as utility
from models_base import ModelsBase

# FATODE integrator (Experimental):
import fatode_erk_adjoint
import fatode_erk_forward
from fatode_erk_forward import FatODE_ERK_FWD as ForwardIntegrator  # This should be made more flexible...
from fatode_erk_adjoint import FatODE_ERK_ADJ as AdjointIntegrator  # This should be made more flexible...
#
from explicit_runge_kutta import ExplicitRungeKutta as ExplicitTimeIntegrator
from linear_implicit_runge_kutta import LIRK as ImplicitTimeIntegrator
#

from state_vector_numpy import StateVectorNumpy as StateVector
from state_matrix_numpy import StateMatrixNumpy as StateMatrix
from state_matrix_sp_scipy import StateMatrixSpSciPy as SparseStateMatrix
from observation_vector_numpy import ObservationVectorNumpy as ObservationVector
from observation_matrix_numpy import ObservationMatrixNumpy as ObservationMatrix
from observation_matrix_sp_scipy import ObservationMatrixSpSciPy as SparseObservationMatrix
from error_models_numpy import BackgroundErrorModelNumpy as BackgroundErrorModel
from error_models_numpy import ObservationErrorModelNumpy as ObservationErrorModel


class Lorenz96(ModelsBase):
    """
    An implementation of the Lorenz96 model (aka Lorenz-40).

    Lorenz96 Model class constructor

    Args:
        model_configs: dict; a configurations dictionary for model settings.
            Supported Configurations for Lorenz-96 Model:
            * model_name: string representation of the model name
            * initial_state (default None): an iterable (e.g. one dimensional numpy array) containing
                a reference inintial state. length of initial_state must be the same as
                num_prognostic_variables here.
            * num_prognostic_variables (default 40): Number of prognostic/physical variables at each
                gridpoint.
            * model_grid_type: 'cartesian', 'spherical', etc. For Lorenz-96, this is not useful
            * force (default 8.0): the force (F) constant in Lorenz96 model. F=8 creates chaotic system.
            * forward_integration_scheme (default 'LIRK'): decides what time integration used to propagate model
                state and/or perturbations forward/backward in time.
                Supported schemes:
                    - 'ERK': Explicit Runge-Kutta
                    - 'LIRK': Lightly Implicit Runge-Kutta
            * adjoint_integration_scheme (default None): decides what adjoint integration used to
                calculate sensitivity matrix lambda
                Supported schemes:
                    - 'ADJ-ERK': Adjoint Explicit Runge-Kutta
            * default_step_size (default 0.05): optimized (stable) step size for model propagation.
                This can be usefule when long timesteps in timespans passed to the forward propagation method.
            * model_errors_distribution: probability distribution of model errors (e.g. for imperfect model
                with additive noise).
            * model_noise_level: used for creating model error variances from reference state/trajectory if
                the model_errors_covariance_method is 'empirical'.
            * model_errors_variances: an iterable that contains the model errors' variances
                (e.g. for models with Gaussian model errors)
            * create_model_errors_correlations (default False): If True; create correlation structure making
                the model error covariance matrix (Q) dense. If False, Q will be diagonal.
                This is of course if the model errors are Gaussian.
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
                the observation operator construction process.
            * observation_indexes: an iterable that contains the indices of state vector coordinates/entries
                to observe. This, if not None, overrides observation_vector_size.
            * observation_errors_distribution (default 'gaussian'): probability distribution of observational
                errors (e.g. defining the likelihood fuction in the data assimialtion settings).
            * observation_noise_level: used for creating observation error variances from observations
                equivalent to reference trajectory if the observation_errors_covariance_method is 'empirical'.
            * observation_errors_variances: an iterable that contains the observational errors' variances.
            * create_observation_errors_correlations (default False): If True; create correlation structure
                making the observation error covariance matrix (R) dense. If False, R will be diagonal.
                This is of course if the observation errors are Gaussian.
            * observation_errors_covariance_method: an indicator of the method used to construct observational
                error covariances
            * observation_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
            * observation_errors_covariance_localization_radius: radius of influence for observation error
                covariance matrix localization.
            * background_errors_distribution: probability distribution of the prior errors.
                This can be used to create prior ensemble or initial forecast state.
            * background_errors_variances: an iterable that contains the background errors' variances.
            * create_background_errors_correlations (default True): If True; create correlation structure
                making the background/prior error covariance matrix (B) dense. If False, B will be diagonal.
                This is of course if the background errors are Gaussian.
            * background_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
            * background_errors_covariance_localization_radius: radius of influence for background error
                covariance matrix localization
            * background_errors_covariance_method': an indicator of the method used to construct background
                error covariances

        output_configs: dict,
            A dictionary containing screen/file output configurations for the model
            Supported configuarations:
            --------------------------
                * scr_output (default False): Output results to screen on/off switch
                * scr_output_iter (default 1): printing to screen after this number of model steps
                * verbose (default False): This is used for extensive screen outputting e.g. while debugging.
                    This overrides scr_output.
                * file_output (default False): Output results to file on/off switch
                * file_output_iter (default 1): saving model results to file after this number of model steps
                * file_output_dir: Location to save model output to (if file_output is set to True).
                    Some default directory should be provided by model or an exception will be thrown.

    Returns:
        None

    """
    # Default model configurations
    _model_name = "Lorenz96"
    _default_model_configs = dict(model_name=_model_name,
                                  initial_state=None,
                                  num_prognostic_variables=40,
                                  model_grid_type=None,
                                  force=8.0,
                                  forward_integration_scheme='ERK',
                                  adjoint_integration_scheme='ERK',
                                  default_step_size=0.005,
                                  model_errors_distribution='gaussian',
                                  model_noise_level=0.0,
                                  model_errors_variances=None,
                                  create_model_errors_correlations=False,
                                  model_errors_covariance_localization_function=None,
                                  model_errors_covariance_localization_radius=None,
                                  model_errors_covariance_method='empirical',
                                  model_errors_steps_per_model_steps=0,
                                  observation_operator_type='linear',
                                  observation_vector_size=40,
                                  observation_indexes=None,
                                  observation_errors_distribution='gaussian',
                                  observation_noise_level=0.05,
                                  observation_errors_variances=None,
                                  create_observation_errors_correlations=False,
                                  observation_errors_covariance_localization_function=None,
                                  observation_errors_covariance_localization_radius=None,
                                  observation_errors_covariance_method='empirical',
                                  background_errors_distribution='gaussian',
                                  background_noise_level=0.08,
                                  create_background_errors_correlations=True,
                                  background_errors_variances=None,
                                  background_errors_covariance_localization_function='gauss',
                                  background_errors_covariance_localization_radius=4,
                                  background_errors_covariance_method='empirical'
                                  )
    __supported_observation_operators = ['linear', 'quadratic', 'cubic']

    __def_verbose = True

    def __init__(self, model_configs=None, output_configs=None):

        # Aggregate passed model configurations with default configurations
        model_configs = utility.aggregate_configurations(model_configs, Lorenz96._default_model_configs)
        self.model_configs = utility.aggregate_configurations(model_configs, ModelsBase._default_model_configs)

        # Aggregate passed output configurations with default configurations
        self._output_configs = utility.aggregate_configurations(output_configs, ModelsBase._default_output_configs)

        # model verbosity (output level)
        try:
            self._verbose = self._output_configs['verbose']
        except KeyError:
            self._verbose = Lorenz96.__def_verbose
        finally:
            if self._verbose is None:
                self._verbose = Lorenz96.__def_verbose

        #
        # Model-related settings:
        # ---------------------------------
        self._model_name = self.model_configs['model_name']
        self._num_prognostic_variables = self.model_configs['num_prognostic_variables']
        self._num_dimensions = 0
        # Set/update model configurations:
        self.model_configs.update(dict(periodic=True, nx=1, dx=1,
                                       num_prognostic_variables=self._num_prognostic_variables,
                                       num_dimensions=self._num_dimensions,
                                       )
                                  )
        self._model_constants = dict(F=self.model_configs['force'])
        self._state_size = self.model_configs['num_prognostic_variables']

        #
        # Model time integration settings (Forward and Adjoint)
        # 1- Forward:
        fwd_integration_scheme = self.model_configs['forward_integration_scheme']
        if self._verbose:
            print("Initiating model Forward Integrator [%s]..." % fwd_integration_scheme)
        #
        self._default_step_size = self.model_configs['default_step_size']
        #
        if fwd_integration_scheme is not None:
            if re.match(r'\Afwd(_|-)*erk\Z', fwd_integration_scheme, re.IGNORECASE) or \
                re.match(r'\Aerk(_|-)*fwd\Z', fwd_integration_scheme, re.IGNORECASE)or \
                re.match(r'\Aerk\Z', fwd_integration_scheme, re.IGNORECASE):
                #
                # Get options from fwdtester.py here...
                fwd_configs = fatode_erk_forward.initialize_forward_configs(self, fwd_integration_scheme)
                fwd_configs.update({'atol':np.ones(self._state_size)*1e-8,
                                    'rtol':np.ones(self._state_size)*1e-8
                                    })
                #
                self._forward_integrator = ForwardIntegrator(fwd_configs)
            else:
                print("The forward integration/solver scheme %s is not supported by this Model!" \
                       % fwd_integration_scheme)
                raise ValueError
        else:
            print("There is no Time Integration scheme attached to this model!")
            print("fwd_integration_scheme is None!")
            raise ValueError

        if self._verbose:
            print(" %s Model: Forward Integrator [%s] Is Successfully Initialized." % (self._model_name, fwd_integration_scheme))

        #
        # 2- Adjoint:
        adj_integration_scheme = self.model_configs['adjoint_integration_scheme']
        if self._verbose:
            print(" %s Model: Initiating Adjoint Integrator [%s]..." % (self._model_name, adj_integration_scheme))
        #
        if adj_integration_scheme is not None:
            if re.match(r'\Aadj(_|-)*erk\Z', adj_integration_scheme, re.IGNORECASE) or \
                re.match(r'\Aerk(_|-)*adj\Z', adj_integration_scheme, re.IGNORECASE)or \
                re.match(r'\Aerk\Z', adj_integration_scheme, re.IGNORECASE):
                #
                # Get options from adjtester.py here...
                adj_configs = fatode_erk_adjoint.initialize_adjoint_configs(self, adj_integration_scheme)
                adj_configs.update({'model':self,
                                    'fun':self.step_forward_function,
                                    'jac':self.step_forward_function_Jacobian,
                                    'atol':np.ones(self._state_size)*1e-8,
                                    'rtol':np.ones(self._state_size)*1e-8
                                    })
                #
                self._adjoint_integrator = AdjointIntegrator(adj_configs)
            else:
                print("The adjoint integration/solver scheme %s is not supported by this Model!" \
                       % adj_integration_scheme)
                raise ValueError
        else:
            self._adjoint_integrator = None

        if self._verbose:
            print(" %s Model: Adjoint Integrator [%s] Is Successfully Initialized." % (self._model_name, adj_integration_scheme))

        #
        # Create reference initial condition
        initial_state = self.model_configs['initial_state']
        if initial_state is None:
            self._reference_initial_condition = self.create_initial_condition()
        else:
            self._reference_initial_condition = self.state_vector(np.squeeze(np.asarray(initial_state[:])))
        #
        # Model Error model settings and initialization
        model_errors_steps_per_model_steps = self.model_configs['model_errors_steps_per_model_steps']
        if model_errors_steps_per_model_steps > 0:
            self._perfect_model = False
            model_err_configs = dict(errors_distribution=self.model_configs['model_errors_distribution'],
                                     model_noise_level=self.model_configs['model_noise_level'],
                                     model_errors_variances=self.model_configs['model_errors_variances'],
                                     create_errors_correlations=self.model_configs['create_model_errors_correlations'],
                                     errors_covariance_method=self.model_configs['model_errors_covariance_method']
                                     )
            self.create_model_errors_model(configs=model_err_configs)
            self._model_errors_steps_per_model_steps = model_errors_steps_per_model_steps
        else:
            self._perfect_model = True
            self.model_error_model = None
            self._model_errors_steps_per_model_steps = 0  # Can be removed...

        #
        # Observation-related settings:
        # ---------------------------------
        self._observation_operator_type = self.model_configs['observation_operator_type'].lower()
        if self._observation_operator_type not in Lorenz96.__supported_observation_operators:
            print("The observation operator '%s' is not supported by the Lorenz96 implementation!")
            raise NotImplementedError
        #
        observation_indexes = self.model_configs['observation_indexes']
        if observation_indexes is not None:
            observation_indexes = np.squeeze(np.asarray(observation_indexes))
            if min(observation_indexes) < 0  or max(observation_indexes) >= self._state_size:
                print("Indexes to observe are out of range. Indexes must be such that: %d < Indexes < %d !" % (0, self._state_size))
                raise IndexError
            else:
                self._observation_indexes = observation_indexes
                self._observation_vector_size = observation_indexes.size
                #
        else:
            # Observations will be linearly spaced if they are less than the
            observation_vector_size = self.model_configs['observation_vector_size']

            if observation_vector_size is not None and not isinstance(observation_vector_size, int):
                print("The option 'observation_vector_size' in the configurations dictionary has to be either None or a positive integer")
                raise ValueError
            elif observation_vector_size is None:
                observation_vector_size = self._state_size
            elif isinstance(observation_vector_size, int):
                if observation_vector_size <=0:
                    print("The option 'observation_vector_size' in the configurations dictionary has to be either None or a positive integer")
                    raise ValueError
                elif observation_vector_size > self._state_size:
                    observation_vector_size = self._state_size
                else:
                    # The observation vector size is legit
                    pass
            #
            # Generate evenly spaced indexes in the state vector:
            observation_indexes = np.empty(observation_vector_size, dtype=int)
            observation_indexes[:]= np.rint(np.linspace(0, self._state_size, observation_vector_size, endpoint=False))
            # observation_indexes = np.rint(np.linspace(0, self._state_size, observation_vector_size, endpoint=False), out=observation_indexes)
            #
            self._observation_indexes = observation_indexes
            self._observation_vector_size = observation_vector_size

        # Observation Error model settings and initialization
        obs_err_configs = dict(errors_distribution=self.model_configs['observation_errors_distribution'],
                               observation_noise_level=self.model_configs['observation_noise_level'],
                               observation_errors_variances=self.model_configs['observation_errors_variances'],
                               create_errors_correlations=self.model_configs['create_observation_errors_correlations'],
                               errors_covariance_method=self.model_configs['observation_errors_covariance_method']
                               )
        self.create_observation_error_model(configs=obs_err_configs)

        #
        # Additional settings
        # ---------------------------------
        background_err_configs = dict(errors_distribution=self.model_configs['background_errors_distribution'],
                                      background_noise_level=self.model_configs['background_noise_level'],
                                      background_errors_variances=self.model_configs['background_errors_variances'],
                                      create_errors_correlations=self.model_configs['create_background_errors_correlations'],
                                      errors_covariance_method=self.model_configs['background_errors_covariance_method']
                                      )
        #
        localize_covariances = self.model_configs['create_background_errors_correlations'] and \
                               self.model_configs['background_errors_covariance_localization_function'] is not None and \
                               self.model_configs['background_errors_covariance_localization_radius'] is not None
        background_err_configs.update({'localize_errors_covariances':localize_covariances})

        self.create_background_error_model(background_err_configs)

        # a placeholder for the sparse Jacobian; this prevents reconstruction
        self.__Jacobian = None

        #
        self._initialized = True
        #

    def state_vector(self, state_vector_ref=None):
        """
        Create a wrapper for a state vector data structure, and return a reference to it.

        Args:
            state_vector_ref: a 1D-numpy array that will be wrapped, to be handled by the linear algebra module.

        Returns:
            initial_state: a wrapped 1D-numpy to be handled by the linear algebra module.

        """
        if state_vector_ref is None:
            initial_state = self._initialize_model_state_vector()
        else:
            if not isinstance(state_vector_ref, np.ndarray):
                print(" data structure passed to (instnce of) Lorenze96.state_vector() has to be np.ndarray!")
                raise AssertionError
            elif state_vector_ref.ndim != 1:
                print("A numpy.ndarray passed to (instnce of) Lorenze96.state_vector() has to one dimensional!")
                raise AssertionError
            else:
                initial_state = StateVector(state_vector_ref)
        return initial_state
        #

    def _initialize_model_state_vector(self):
        """
        Create an empty 1D numpy.ndarray, wrap it in StateVectorNumpy, and return its reference.

        Args:
            None

        Returns:
            initial_vec: 1D-numpy array wrapped by StateVectorNumpy to be handled by the linear algebra module.

        """
        initial_vec_ref = np.zeros(self.state_size())
        initial_vec = StateVector(initial_vec_ref)
        return initial_vec
        #

    def state_vector_size(self):
        """
        Return the size of the state vector (dimensions of the model space).

        Args:
            None

        Returns:
            Model state space size/dimension

        """
        return self._state_size
        #
    # add a useful alias to remove confusion
    state_size = state_dimension = state_vector_size
        #

    def observation_vector(self, observation_vector_ref=None):
        """
        Create a wrapper for a observation vector data structure, and return a reference to it.

        Args:
            observation_vector_ref: a 1D-numpy array that will be wrapped, to be handled by the linear algebra module.

        Returns:
            observation_vector: a wrapped 1D-numpy to be handled by the linear algebra module.

        """
        if observation_vector_ref is None:
            observation_vector = self._initialize_model_observation_vector()
        else:
            if not isinstance(observation_vector_ref, np.ndarray):
                print(" data structure passed to (instnce of) Lorenze96.observation_vector() has to be np.ndarray!")
                raise AssertionError
            elif observation_vector_ref.ndim != 1:
                print("A numpy.ndarray passed to (instnce of) Lorenze96.observation_vector() has to one dimensional!")
                raise AssertionError
            else:
                observation_vector = ObservationVector(observation_vector_ref)
        return observation_vector
        #

    def _initialize_model_observation_vector(self):
        """
        Create an empty 1D numpy.ndarray, wrap it in ObservationVectorNumpy, and return its reference.

        Args:
            None

        Returns:
            observation_vec: 1D-numpy array wrapped by ObservationVectorNumpy to be handled by the linear algebra module.

        """
        observation_vec_ref = np.zeros(self.observation_vector_size())
        observation_vec = ObservationVector(observation_vec_ref)
        return observation_vec
        #

    def observation_vector_size(self):
        """
        Return the size of the observation vector (dimension of the observation space).

        Args:
            None

        Returns:
            Observation state space size/dimension

        """
        return self._observation_vector_size
        #
    # add a useful alias to remove confusion
    observation_size = observation_dimension = observation_vector_size
        #

    def state_matrix(self, state_matrix_ref=None, create_sparse=False):
        """
        Create a wrapper for a model matrix data structure, and return a reference to it.
        This will hold a dense (2D numpy.ndarray) or a sparse (2D scipy.sparse.*_matrix).

        Args:
            state_matrix_ref: a 2D-numpy array that will be wrapped, to be handled by the linear algebra module.
            create_sparse: If create_sparse is True, the matrix created is empty sparse.lil_matrix, otherwise it is
                dense/full numpy.ndarray.
                If state_matrix_ref is a sparse matrix, create_sparse is ignored even if it is set to False

        Returns:
            state_matrix: 2D-numpy array wrapped by StateMatrixNumpy, or 2D sparse matrix wrapped by
                StateMatrixSpScipy to be handled by the linear algebra module.
                The size of the matrix is (state_size x state_size).

        """
        if state_matrix_ref is None:
            state_matrix = self._initialize_model_state_matrix(create_sparse=create_sparse)
        else:
            if not isinstance(state_matrix_ref, np.ndarray) and not sparse.issparse(state_matrix_ref):
                print(" data structure passed to (instnce of) Lorenze96.state_matrix() has to be np.ndarray or a sparse (scipy.*matrix) data structure!")
                raise AssertionError
            elif state_matrix_ref.ndim != 2:
                print("A numpy array or sparse matrix passed to (instnce of) Lorenze96.state_matrix() has to two dimensional!")
                raise AssertionError
            else:
                if sparse.issparse(state_matrix_ref):
                    state_matrix = SparseStateMatrix(state_matrix_ref)
                else:
                    state_matrix = StateMatrix(state_matrix_ref)
        return state_matrix
        #

    def _initialize_model_state_matrix(self, create_sparse=False):
        """
        Create an empty 2D numpy.ndarray, wrap it in StateMatrixNumpy, and return its reference.
        This returns a dense (2D numpy.ndarray), or a sparse (2D scipy.sparse.csr_matrix) based on
        sparse value. The returned data structure is wrapped by StateMatrixNumpy or StateMatrixSpScipy.

        Args:
            create_sparse: If create_sparse is True, the matrix created is empty sparse.lil_matrix, otherwise it is
                dense/full numpy.ndarray

        Returns:
            state_matrix: 2D-numpy array wrapped by StateMatrixNumpy, or 2D sparse matrix wrapped by
                StateMatrixSpScipy to be handled by the linear algebra module.
                The size of the matrix is (state_size x state_size).

        """
        state_size = self.state_size()
        if create_sparse:
            state_matrix_ref = sparse.lil_matrix((state_size, state_size), dtype=np.float32)
            state_matrix = SparseStateMatrix(state_matrix_ref)
        else:
            state_matrix_ref = np.zeros((state_size, state_size))
            state_matrix = StateMatrix(state_matrix_ref)
        return state_matrix
        #

    def observation_matrix(self, observation_matrix_ref=None, create_sparse=False):
        """
        Create a wrapper for a model matrix data structure, and return a reference to it.
        This will hold a dense (2D numpy.ndarray) or a sparse (2D scipy.sparse.*_matrix).

        Args:
            observation_matrix_ref: a 2D-numpy array that will be wrapped, to be handled by the linear algebra module.
            create_sparse: If create_sparse is True, the matrix created is empty sparse.lil_matrix, otherwise it is
                dense/full numpy.ndarray.
                If observation_matrix_ref is a sparse matrix, create_sparse is ignored even if it is set to False

        Returns:
            observation_matrix: 2D-numpy array wrapped by ObservationMatrixNumpy, or 2D sparse matrix wrapped by
                ObservationMatrixSpScipy to be handled by the linear algebra module.
                The size of the matrix is (observation_size x observation_size).

        """
        if observation_matrix_ref is None:
            observation_matrix = self._initialize_model_observation_matrix(create_sparse=create_sparse)
        else:
            if not isinstance(observation_matrix_ref, np.ndarray) and not sparse.issparse(observation_matrix_ref):
                print(" data structure passed to (instnce of) Lorenze96.observation_matrix() has to be np.ndarray or a sparse (scipy.*matrix) data structure!")
                raise AssertionError
            elif observation_matrix_ref.ndim != 2:
                print("A numpy array or sparse matrix passed to (instnce of) Lorenze96.observation_matrix() has to two dimensional!")
                raise AssertionError
            else:
                if sparse.issparse(observation_matrix_ref):
                    observation_matrix = SparseObservationMatrix(observation_matrix_ref)
                else:
                    observation_matrix = ObservationMatrix(observation_matrix_ref)
        return observation_matrix
        #

    def _initialize_model_observation_matrix(self, create_sparse=False):
        """
        Create an empty 2D numpy.ndarray, wrap it in ObservationMatrixNumpy, and return its reference.
        This returns a dense (2D numpy.ndarray), or a sparse (2D scipy.sparse.csr_matrix) based on
        sparse value. The returned data structure is wrapped by ObservationMatrixNumpy or ObservationMatrixSpScipy.

        Args:
            create_sparse: If create_sparse is True, the matrix created is empty sparse.lil_matrix, otherwise it is
                dense/full numpy.ndarray

        Returns:
            observation_matrix: 2D-numpy array wrapped by ObservationMatrixNumpy, or 2D sparse matrix wrapped by
                ObservationMatrixSpScipy to be handled by the linear algebra module.
                The size of the matrix is (observation_size x observation_size).

        """
        observation_size = self.observation_vector_size()
        if create_sparse:
            observation_matrix_ref = sparse.lil_matrix((observation_size, observation_size), dtype=np.float32)
            observation_matrix = SparseObservationMatrix(observation_matrix_ref)
        else:
            observation_matrix_ref = np.zeros((observation_size, observation_size))
            observation_matrix = ObservationMatrix(observation_matrix_ref)
        return observation_matrix
        #

    def step_forward_function(self, time_point, in_state):
        """
        Evaluate the right-hand side of the Lorenz96 model at the given model state (in_state) and time_piont.

        Args:
            time_point: scalar time instance to evaluate right-hand-side at
            in_state: current model state to evaluate right-hand-side at

        Returns:
            rhs: the right-hand side function evaluated at state=in_state, and time=time_point

        """
        assert isinstance(in_state, (StateVector, np.ndarray)), "in_state passed to (instance of) \
        #                                 Lorenz96.step_forward_function() has to be a valid StateVector object"
        #
        x = in_state
        state_size = self.state_size()
        F = self._model_constants['F']

        #
        # Evaluate the right-hand-side:
        rhs = self.state_vector()

        for i in xrange(state_size):
            j = i+1 if i<state_size-1 else i+1-state_size  # this is i+1 in round-robin fashion
            #
            rhs[i] = x[i-1] * (x[j] - x[i-2]) - x[i] + F
        #
        if isinstance(in_state, np.ndarray):
            rhs = rhs.get_numpy_array()
        return rhs
        #

    def step_forward_function_Jacobian(self, time_point, in_state, create_sparse=True):
        """
        The Jacobian of the right-hand side of the model and evaluate it at the given model state.

        Args:
            time_point: scalar time instance to evaluate Jacobian of the right-hand-side at
            in_state: current model state to evaluate Jacobian of the right-hand-side at
            create_sparse (default True): return the Jacobian as a sparse csr_matrix

        Returns:
            Jacobian: the derivative/Jacobian of the right-hand side function evaluated at
                state=in_state, and time=time_point

        """
        x = in_state
        state_size = self.state_size()

        if self.__Jacobian is None:
            #
            # initialize with components' self derivatives
            if create_sparse:
                init_Jacobian = sparse.diags(diagonals=-np.ones(state_size), offsets=0, format='lil')
            else:
                init_Jacobian = np.diag(-np.ones(state_size))

            # Update the jacobian; loop over all rows:
            for i in xrange(state_size):
                j = i+1 if i<state_size-1 else i+1-state_size  # this is i+1 in round-robin fashion
                #
                init_Jacobian[i, i-2] = - in_state[i-1]
                init_Jacobian[i, i-1] = in_state[j] - in_state[i-2]
                init_Jacobian[i, j] = in_state[i-1]
            #
            if create_sparse:
                Jacobian = SparseStateMatrix(init_Jacobian.tocsr())
            else:
                Jacobian = StateMatrix(init_Jacobian)

            # Update self.__Jacobian:
            self.__Jacobian = Jacobian

        else:
            # Update the jacobian; loop over all rows:
            Jacobian = self.__Jacobian

            if create_sparse:
                if not isinstance(Jacobian, SparseStateMatrix):
                    print("\n\n  * Attempted to ReUse sparse Lorenz.__Jacobian but ist's not instance of SparseStateMatrix! *\n\n ")
                    raise AssertionError
            else:
                if not isinstance(Jacobian, StateMatrix):
                    print("\n\n  * Attempted to ReUse dense Lorenz.__Jacobian but ist's not instance of StateMatrix! *\n\n ")
                    raise AssertionError


            for i in xrange(state_size):
                j = i+1 if i<state_size-1 else i+1-state_size  # this is i+1 in round-robin fashion
                #
                Jacobian[i, i-2] = - in_state[i-1]
                Jacobian[i, i-1] = in_state[j] - in_state[i-2]
                Jacobian[i, j] = in_state[i-1]

        #
        return Jacobian
        #

    def integrate_state(self, initial_state=None, checkpoints=None, step_size=None, rel_tol=1.e-9, abs_tol=1.e-9):
        """
        March the model state forward in time (backward for negative step or decreasing list).
        checkpoints should be a float scalar (taken as one step) or a list including beginning,end or a full timespan.
        The output is a list containing StateVector(s) generated by propagation over the given checkpoints.
        If checkpoints is not None, it has to be an iterable of length greater than one.
        If checkpoints is None, a single step of size (step_size) is taken and a list containing the
            initial_state, and the resulting StateVector instance is returned.
            i.e. checkpoints are replaced by [0, step_size].

        Args:
            initial_state: model.state_vector to be propagated froward according to checkpoints
            checkpoints: a timespan that should be a float scalar (taken as one step) or an iterable
                including beginning, end, or a full timespan to build model trajectory at.
                If None, a single step of size (step_size) is taken and a list containing the initial_state, and
                the resulting StateVector instance is returned. i.e. checkpoints are replaced by [0, step_size].
                If both checkpoints, and step_size are None, an assertion error is raised
            step_size: time integration step size. If None, the model default step size is used.
            rel_tol: relative tolerance for adaptive integration
            abs_tol: absolute tolerance for adaptive integration

        Returns:
            trajectory: model trajectory.
                This is a list of model state_vector objects;
                This list of model states corresponds to the times in checkpoints.

        """
        # Validate parameters:
        assert isinstance(initial_state, StateVector), "initial_state has to be a StateVector object!"
        local_state = initial_state.copy()  # a local copy of the initial state

        #
        if checkpoints is not None:
            timespan = np.asarray(checkpoints).squeeze()
            assert timespan.size > 1, "A timespan provided in 'checkpoints' has to be an iterable of length greater than one!"
            if step_size is None:
                model_step_size = self._default_step_size
            else:
                assert np.isscalar(step_size), "step_size has to be None or a scalar!"
                model_step_size = float(step_size)
            #
        else:
            if step_size is None:
                model_step_size = self._default_step_size
            else:
                assert np.isscalar(step_size), "step_size has to be None or a scalar!"
                model_step_size = float(step_size)
            timespan = np.asarray([0.0, model_step_size])

        #
        # TODO: Being ReValidated.
        # Now, the output should be a list of states propagated over the timespan with intermediate
        # step sizes as in model_step_size.
        #

        # Start propagation, and adding model noise if necessary
        try:
            self._perfect_model
        except (ValueError, AttributeError):
            self._perfect_model = False

        #
        noise_add_cnt = model_errors_steps_per_model_steps = self.model_configs['model_errors_steps_per_model_steps']
        steps_cntr = 0
        #
        if self._perfect_model:
            # Forwad Model propagation WITHOUT model noise:

            trajectory = self._forward_integrator.integrate(local_state, timespan)

        else:
            #
            # Forwad Model propagation WITH additive model noise:
            trajectory = []
            trajectory.append(initial_state)

            # TODO: This is currently incorrect; to be udpated after validating FATODE forward integrator...
            model_errors_steps_per_model_steps = self.model_configs['model_errors_steps_per_model_steps']
            #
            # loop over the local_checkpoints and add model errors when necessary:
            num_iterations = int(len(local_checkpoints)/model_errors_steps_per_model_steps)
            for iter_ind in xrange(num_iterations):
                init_ind = iter_ind * model_errors_steps_per_model_steps
                sub_timespan = local_checkpoints[init_ind: init_ind+model_errors_steps_per_model_steps+1]
                sub_initial_state = trajectory[-1]
                sub_trajectory = self._time_integrator.integrate(initial_state=sub_initial_state,
                                                                 checkpoints=sub_timespan,
                                                                 step_size=model_step_size,
                                                                 rel_tol=rel_tol,
                                                                 abs_tol=abs_tol
                                                                 )
                model_noise = self.model_error_model.generate_noise_vec()
                sub_trajectory[-1] = sub_trajectory[-1].add(model_noise)  # add model noise only to the last state in the sub_trajectory
                trajectory.append(sub_trajectory[1:])
                #

        return trajectory
        #

    def get_model_grid(self):
        """
        Return a copy of the model grid points.
        This is a numpy array of observation size x number of spatial dimensions.

        Args:
            None

        Returns:
            observational_grid: a copy of ``self.observational_grid".

        """
        model_grid = np.empty((self.state_size(), 1))
        model_grid[:, 0] = xrange(self.state_size())
        return model_grid
        #
    # add an alias

    def get_observational_grid(self):
        """
        Return a copy of the observational grid points.
        This is a numpy array of observation size x number of spatial dimensions.

        Args:
            None

        Returns:
            observational_grid: a copy of ``self.observational_grid".

        """
        observational_grid = np.empty((self.observation_size(), 1))
        observational_grid[:, 0] = self._observation_indexes[:]
        return observational_grid
        #
    # add an alias
    get_observations_positions = get_observation_grid = get_observational_grid

    def construct_observation_operator(self, time_point=None, construct_Jacobian=False):
        """
        Construct the (linear version of) observation operator (H) in full. This should generally be avoided!
        We need it's (or it's TLM) effect on a state vector always.
        If called, the observation operator is attached to the model object as ".observation_operator"
        This may also work as a placeholder for the nonlinear observation case if needed.

        Args:
            time_point: the time at which the observation operator should be created
            construct_Jacobian (default False): construct a data structure (e.g. Numpy.ndarray) holding the
                Jacobian of the observation operator. This can be usefule if it is a constant sparse matrix,
                e.g. if the observation operator is linear.

        Returns:
            None

        """
        # construct the linear version of H incrementally, then convert it to CSR-format
        # This may also work as a placeholder for the nonlinear observation case if needed.
        m = self.observation_vector_size()
        n = self.state_size()
        H = sparse.lil_matrix((m, n), dtype=int)
        H[np.arange(m), self._observation_indexes] = 1
        # Return a compressed sparse row format (for efficient matrix vector product).
        self.observation_operator = H.tocsr()
        if construct_Jacobian:
            self.construct_observation_operator_Jacobian()
        #

    def update_observation_operator(self, time_point=None):
        """
        This should be called for each assimilation cycle if the observation operator is time-varying.
        For this model, the observation operator is fixed. This function simply does nothing.

        Args:
            time_point: the time at which the observation operator should be created/refreshed

        Returns:
            None

        """
        # For now the observation operator for Lorenz model is fixed and time-independent
        # Do nothing...
        pass
        #

    def evaluate_theoretical_observation(self, in_state, time_point=None):
        """
        Evaluate the theoretical observation corresponding to a model state vector,
        i.e. evaluate H(in_state), where H is the observation operator.
        If the observatin operator, time_point is used, however here it is not.

        Args:
            in_state: StatVector at which the observation operator is evaluated
            time_point: time instance at which the observation operator is evaluated

        Returns:
            observation_vec: ObservationVector instance equivalent to in_state transformed by the observation
                operator at time_point time instance.

        """
        #
        observation_vector_numpy = in_state[self._observation_indexes]
        oper_type = self._observation_operator_type
        #
        if oper_type == 'linear':
            # Just apply the observation operator on the state vector and return an observation vector.
            pass

        elif oper_type == 'quadratic':
            # raise all observed components of the state vector to power 2
            observation_vector_numpy = np.power(observation_vector_numpy, 2)

        elif oper_type == 'cubic':
            # raise all observed components of the state vector to power 2
            observation_vector_numpy = np.power(observation_vector_numpy, 3)

        else:
            raise ValueError("Unsupported observation operator type %s" % oper_type)
        #
        # Wrap and return the observation vector
        observation_vec = self.observation_vector(observation_vector_numpy)
        return observation_vec
        #

    def construct_observation_operator_Jacobian(self, time_point=None):
        """
        Create the Jacobian of the observation operator (TLM of forward operator).
        We need the TLM of the forward operator on a state vector. Can be easily extended to effect on a state matrix
        If called, the Jacobian of the observation operator is attached to the model object as ".observation_operator_Jacobian"
        This may also work as a placeholder for the nonlinear observation case if needed.

        Args:
            time_point: the time at which the Jacobaian of the observation operator should be created

        Returns:
            None

        """
        #
        oper_type = self._observation_operator_type
        #
        if oper_type in ['linear', 'quadratic', 'cubic']:
            #
            try:
                observation_operator = self.observation_operator
            except (ValueError, NameError, AttributeError):
                self.construct_observation_operator()
                observation_operator = self.observation_operator
            self.observation_operator_Jacobian = observation_operator.copy()
        else:
            print("Unsupported observation operator type '%s' !" % oper_type)
            raise ValueError
            #

    def evaluate_observation_operator_Jacobian(self, in_state, time_point=None):
        """
        Evaluate the Jacobian of the observation operator (TLM of forward operator) at the given in_state,
            and time_point.

        Args:
            in_state: StatVector at which the Jacobian of the observation operator is evaluated.
            time_point: the time at which the Jacobaian of the observation operator should be evaluated if it
                is time-dependent. In this implementation time_point is ignored.

        Returns:
            observation_operator_Jacobian: a Numpy/Sparse representation of the Jacobian of the observation
                operator (TLM of forward operator) at the given in_state, and time_point.

        """
        oper_type = self._observation_operator_type
        if oper_type not in Lorenz96.__supported_observation_operators:
            print("The observation operator '%s' is not supported by the Lorenz96 implementation!")
            raise NotImplementedError
        #
        try:
            observation_operator_Jacobian = self.observation_operator_Jacobian
        except (ValueError, NameError, AttributeError):
            self.construct_observation_operator_Jacobian()
            observation_operator_Jacobian = self.observation_operator_Jacobian
        #
        obs_coord = np.arange(self.observation_vector_size())
        obs_indexes = self._observation_indexes
        observed_vars = in_state[obs_indexes]  # numpy representation of the observed entries of in_state
        #
        if oper_type == 'linear':
            observation_operator_Jacobian[obs_coord, obs_indexes] = 1.0
        elif oper_type == 'quadratic':
            observation_operator_Jacobian[obs_coord, obs_indexes] = 2.0 * observed_vars
        elif oper_type == 'cubic':
            observation_operator_Jacobian[obs_coord, obs_indexes] = 3.0 * np.power(observed_vars, 2)
        else:
            print("The observation operator '%s' is not supported by the Lorenz96 implementation!")
            raise NotImplementedError
        #
        return observation_operator_Jacobian
        #

    def observation_operator_Jacobian_T_prod_vec(self, in_state, observation, time_point=None):
        """
        Evaluate the transpose of the Jacobian of the observation operator (evaluated at specific model state,
            and time instance) multiplied by an observation vector.
            i.e. evaluate $\\mathbf{H(in_state)}^T \\times observation\_vector$.
            The result is a state vector of course

        Args:
            in_state: StatVector at which the Jacobian of the observation operator is evaluated
            time_point: the time at which the Jacobaian of the observation operator should be evaluated if it
                is time-dependent. In this implementation time_point is ignored.
            observation: ObservationVector to be multiplied by observation operator Jacobian transposed.

        Returns:
            result_state: StatVector containing the result of observation multiplied by the
                observation operator Jacobian transposed.

        """
        oper_type = self._observation_operator_type
        if oper_type not in Lorenz96.__supported_observation_operators:
            print("The observation operator '%s' is not supported by the Lorenz96 implementation!")
            raise NotImplementedError
        #
        # Inintialize the resulting state vector:
        result_state = self.state_vector()  # this already Zeros all entries
        #
        obs_indexes = self._observation_indexes
        #
        if oper_type == 'linear':
            result_state[obs_indexes] = observation[:]
        elif oper_type == 'quadratic':
            observed_vars = in_state[obs_indexes]  # numpy representation of the observed entries of in_state
            result_state[obs_indexes] = 2.0 * np.squeeze(observed_vars) * np.squeeze(observation[:])
        elif oper_type == 'cubic':
            observed_vars = in_state[obs_indexes]  # numpy representation of the observed entries of in_state
            result_state[obs_indexes] = 3.0 * np.power(np.squeeze(observed_vars), 2) * np.squeeze(observation[:])
        else:
            print("The observation operator '%s' is not supported by the Lorenz96 implementation!")
            raise NotImplementedError
            #
        return result_state
        #

    def observation_operator_Jacobian_prod_vec(self, in_state, state, time_point=None):
        """
        Evaluate the Jacobian of the observation operator (evaluated at specific model state, and time
            instance) multiplied by a state vector (state).
            i.e. evaluate $\\mathbf{H(in_state)} \\times state$.
            The result is an observation vector

        Args:
            in_state: StatVector at which the Jacobian of the observation operator is evaluated
            state: state by which the Jacobian of the observation operator is multiplied
            time_point: the time at which the Jacobaian of the observation operator should be evaluated if it
                is time-dependent. In this implementation time_point is ignored.

        Returns:
            result_observation: ObservationVector; the result of the observation operator multiplied by state.

        """
        oper_type = self._observation_operator_type
        if oper_type not in Lorenz96.__supported_observation_operators:
            print("The observation operator '%s' is not supported by the Lorenz96 implementation!")
            raise NotImplementedError
        #
        obs_indexes = self._observation_indexes
        #
        # numpy representation of the result:
        if oper_type == 'linear':
            result_observation = np.squeeze(state[obs_indexes])
            #
        elif oper_type == 'quadratic':
            observed_vars = np.squeeze(in_state[obs_indexes])
            result_observation = 2.0 * observed_vars * np.squeeze(state[obs_indexes])
            #
        elif oper_type == 'cubic':
            observed_vars = np.squeeze(in_state[obs_indexes])
            result_observation = 3.0 * np.power(observed_vars, 2) * np.squeeze(state[obs_indexes])
            #
        else:
            print("The observation operator '%s' is not supported by the Lorenz96 implementation!")
            raise NotImplementedError
            #
        # wrap the observation vector and return
        result_observation = self.observation_vector(result_observation)
        #
        return result_observation
        #

    def _construct_covariance_localization_operator(self, localization_radius=None, localization_function=None):
        """
        Construct the localization/decorrelation operator (Decorr) in full.
        This should be avoided in practice, especially for bigger models. E.g. Observation-localization
            should be considered instead.
        We need it's effect on a square covariance matrix. This should be sparse or evaluated off-line
            and saved to file.
        Here I construct only the upper triangular part. the diagonal is ones, and the matrix is
            symmetric of course (the lower traiangle is then copied).
        When constructed, it is attached to the model instance as "._covariance_localization_operator".
        If the input parameters are None, the parameters are retrieved from the configurations dictionary
            of the model instance .model_configs

        Args:
            localization_radius: covariance radius of influence (decorrelation radius)
            localization_function: 'Gaspari-Cohn', 'Gauss', 'Cosine', etc.

        Returns:
            None

        """
        # Retrieve the covariance localization settings if (any of) the passed arguments are None
        if localization_radius is None:
            localization_radius = self.model_configs['background_errors_covariance_localization_radius']
        if localization_function is None:
            localization_function = self.model_configs['background_errors_covariance_localization_function']
        #
        state_size = self.state_size()
        # evaluate distances in the first row (will be rotated for next rows
        distances_stride = [min(i, state_size-i) for i in xrange(state_size)]
        localization_coeffs = utility.calculate_localization_coefficients(radius=localization_radius,
                                                                          distances=distances_stride,
                                                                          method=localization_function
                                                                          )
        localization_coeffs = [coeff for coeff in localization_coeffs]  # a list is easier for rotation
        #
        loc_matrix = sparse.lil_matrix((state_size, state_size), dtype=np.float)
        for i in xrange(state_size):  # loop over state_vector entries (one dimension of the covariance matrix
            loc_matrix[i, :] = localization_coeffs
            # shift localization coefficients to the right for the next row
            if i < state_size-1:
                localization_coeffs.insert(0, localization_coeffs.pop())
        self._covariance_localization_operator = loc_matrix.tocsr()
        #

    def apply_state_covariance_localization(self, covariance_array, in_place=True):
        """
        Apply localization/decorrelation to a given square array.
        This generally a point-wise multiplication of the decorrelation array and the passed covariance_array

        Args:
            covariance_array: a StateMatrixNumpy or StateMatrixSpScipy containing covariances to be localized.
            in_place (default True): apply localization to covariance_array (in-place) without creating a new
                object. If False, a localized copy of covariance_array is returned.

        Returns:
            localized_covariances: a decorrelated version of (covariance_array), and of the same type.

        """
        assert isinstance(covariance_array, (StateMatrix, SparseStateMatrix)), "Input covariance array has to be of supported type!"
        #
        try:
            localization_operator = self._covariance_localization_operator
        except (NameError, ValueError, AttributeError):
            localization_radius = self.model_configs['background_errors_covariance_localization_radius']
            localization_function = self.model_configs['background_errors_covariance_localization_function']
            self._construct_covariance_localization_operator(localization_radius=localization_radius,
                                                             localization_function=localization_function
                                                             )
            localization_operator = self._covariance_localization_operator
        # localization_operator is a vanilla sparse matrix (e.g. scipy.sparse.csr_matrix)
        #
        if in_place:
            localized_covariances = covariance_array
        else:
            localized_covariances = covariance_array.copy()

        if isinstance(covariance_array, SparseStateMatrix):
            localized_covariances.set_raw_matrix_ref(localization_operator.multiply(localized_covariances.get_raw_matrix_ref()))
        elif isinstance(covariance_array, StateMatrix):
            try:
                localized_covariances.set_raw_matrix_ref(np.asarray(localization_operator.multiply(localized_covariances.get_raw_matrix_ref())))
            except(AssertionError):
                localized_covariances.set_raw_matrix_ref(localization_operator.multiply(localized_covariances.get_raw_matrix_ref()).toarray())
            #
        else:
            print("The 'covariance_array' is of Unsupported Type %s" %type(covariance_array))
            raise TypeError

        #
        return localized_covariances
        #

    def create_observation_error_model(self, configs=None):
        """
        Create and attach an observation error model

        Args:
            configs: dict,
                A configurations dictionary for the observation error model
                Supported configurations:
                ---------------------------
                * errors_distribution (default 'gaussian'): probability distribution of observational
                    errors (e.g. defining the likelihood fuction in the data assimialtion settings).
                * observation_noise_level: used for creating observation error variances from observations
                    equivalent to reference trajectory if the observation_errors_covariance_method is 'empirical'.
                * observation_errors_variances: an iterable that contains the observational errors' variances.
                * errors_covariance_method: an indicator of the method used to construct observational
                    error covariances
                * create_errors_correlations (default False): Whether to create correlations between different
                      components of the observation vector or not. If False, diagonal covariance matrix is construct
                      otherwise it is dense (and probably localized becoming sparse if
                      localize_errors_covariances is set to True),
                * localize_errors_covariances (default False): Use the model to localize the error-covariance
                * observation_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
                * observation_errors_covariance_localization_radius: radius of influence for observation error
                    covariance matrix localization.


                * observation_noise_level: This is used to create variances of the observation errors, such that:
                      error variance = noise_level * signal magnitude (of initial condition).

        Returns:
            None

        """
        if configs is None:
            configs = dict(errors_distribution=self.model_configs['observation_errors_distribution'],
                           observation_noise_level=self.model_configs['observation_noise_level'],
                           observation_errors_variances=self.model_configs['observation_errors_variances'],
                           create_errors_correlations=self.model_configs['create_observation_errors_correlations'],
                           errors_covariance_method=self.model_configs['observation_errors_covariance_method'],
                           variance_adjusting_factor=0.01
                           )
        if not configs.has_key('localize_errors_covariances'):
            try:
                localize_covariances = configs['create_observation_errors_correlations'] and \
                    self.model_configs['observation_errors_covariance_localization_function'] is not None and \
                    self.model_configs['observation_errors_covariance_localization_radius'] is not None
                configs.update({'localize_errors_covariances':localize_covariances})
            except KeyError:
                configs.update({'localize_errors_covariances':False})
        else:
            pass

        self.observation_error_model = ObservationErrorModel(self, configs=configs)
        self.observation_error_model.construct_error_covariances(construct_inverse=True,
                                                                 construct_sqrtm=True,
                                                                 sqrtm_method='cholesky',
                                                                 observation_checkpoints=np.arange(0, 10, 0.05)
                                                                 )
                                                                 #

    def create_background_error_model(self, configs=None):
        """
        create and attach an background error model

        Args:
            configs: dict,
                A configurations dictionary for the background error model
                Supported configurations:
                ---------------------------
                * errors_distribution: probability distribution of the prior errors.
                    This can be used to create prior ensemble or initial forecast state.
                * background_noise_level: This is used to create variances of the background errors, such that:
                      error variance = noise_level * signal magnitude (of initial condition).
                * background_errors_variances: an iterable that contains the background errors' variances.
                * create_errors_correlations (default True): If True; create correlation structure
                    making the background/prior error covariance matrix (B) dense. If False, B will be diagonal.
                    This is of course if the background errors are Gaussian.
                * localize_errors_covariances (default True): Use the model to localize the error-covariance
                      matrix with model's default settings for localization.
                * errors_covariance_method': an indicator of the method used to construct background
                    error covariances

        Returns:
            None

        """
        if configs is None:
            configs = dict(errors_distribution=self.model_configs['background_errors_distribution'],
                           background_noise_level=self.model_configs['background_noise_level'],
                           background_errors_variances=self.model_configs['background_errors_variances'],
                           create_errors_correlations=self.model_configs['create_background_errors_correlations'],
                           errors_covariance_method=self.model_configs['background_errors_covariance_method'],
                           variance_adjusting_factor=0.1
                           )
        if not configs.has_key('localize_errors_covariances'):
            try:
                localize_covariances = configs['create_errors_correlations'] and \
                    self.model_configs['background_errors_covariance_localization_function'] is not None and \
                    self.model_configs['background_errors_covariance_localization_radius'] is not None
                configs.update({'localize_errors_covariances':localize_covariances})
            except KeyError:
                configs.update({'localize_errors_covariances':False})
                raise
        else:
            pass

        #
        self.background_error_model = BackgroundErrorModel(self, configs=configs)
        self.background_error_model.construct_error_covariances(construct_inverse=True,
                                                                construct_sqrtm=True,
                                                                sqrtm_method='cholesky'
                                                                )
                                                                #

    def create_model_error_model(self, configs=None):
        """
        create and attach an model error model

        Args:
            configs: dict,
                A configurations dictionary for the model error model
                Supported configurations:
                ---------------------------
                * errors_distribution: probability distribution of the prior errors.
                    This can be used to create prior ensemble or initial forecast state.
                * model_noise_level: This is used to create variances of the model errors, such that:
                      error variance = noise_level * signal magnitude (of initial condition).
                * model_errors_variances: an iterable that contains the model errors' variances.
                * create_errors_correlations (default True): If True; create correlation structure
                    making the model error covariance matrix (Q) dense. If False, B will be diagonal.
                    This is of course if the model errors are Gaussian.
                * localize_errors_covariances (default True): Use the model to localize the error-covariance
                      matrix with model's default settings for localization.
                * errors_covariance_method': an indicator of the method used to construct model
                    error covariances

        Returns:
            None

        """
        if configs is None:
            configs = dict(errors_distribution=self.model_configs['model_errors_distribution'],
                           model_noise_level=self.model_configs['model_noise_level'],
                           model_errors_variances=self.model_configs['model_errors_variances'],
                           create_errors_correlations=self.model_configs['create_model_errors_correlations'],
                           errors_covariance_method=self.model_configs['model_errors_covariance_method'],
                           variance_adjusting_factor=0.01
                           )
        if not configs.has_key('localize_errors_covariances'):
            try:
                localize_covariances = configs['create_model_errors_correlations'] and \
                    self.model_configs['model_errors_covariance_localization_function'] is not None and \
                    self.model_configs['model_errors_covariance_localization_radius'] is not None
                configs.update({'localize_errors_covariances':localize_covariances})
            except KeyError:
                configs.update({'localize_errors_covariances':False})
        else:
            pass

        self.model_error_model = ModelErrorModel(self, configs=configs)
        self.model_error_model.construct_error_covariances(construct_inverse=True,
                                                                construct_sqrtm=True,
                                                                sqrtm_method='cholesky'
                                                                )
                                                                #

    def create_initial_condition(self, checkpoints=None, initial_vec=None):
        """
        Create and return a (reference) initial condition state for Lorenz96.
        This is carried out by propagating initial_vec over the timespan in checkpoints, and returning
            the final state

        Args:
            checkpoints: an iterable containing burn-in timespan
            initial_vec: iterable (e.g. array-like or StateVector) containing initial

        Returns:
            initial_condition: StateVector containing a valid initial state for Lorenz96 model

        """
        initial_condition = self.state_vector()
        if initial_vec is None:
            #
            initial_condition[:] = np.linspace(-2, 2.001, self._num_prognostic_variables)
        else:
            initial_condition[:] = initial_vec[:]

        try:
            perfect_model = self._perfect_model
            self._perfect_model = True
        except (AttributeError, ValueError, NameError):
            perfect_model = True
            self._perfect_model = True

        if checkpoints is None:
            checkpoints = np.arange(0, 5, 0.005)

        tmp_trajectory = self.integrate_state(initial_state=initial_condition, checkpoints=checkpoints)
        initial_condition = tmp_trajectory[-1]

        # reset the model settings (perfect/non-perfect)
        if not perfect_model:
            self._perfect_model = False

        #
        return initial_condition
        #

    def create_initial_ensemble(self, ensemble_size, ensemble_mean=None):
        """
        Create initial ensemble for Lorenz96 model given the ensemble mean.
        The ensemble is created by adding random noise from the background errors model.

        Args:
            ensemble_size: sample size
            ensemble_mean: StateVector used as the mean of the generated ensemble to which noise is added to
                create the ensemble

        Returns:
            ensemble: list of StateVector objects serving as an initial background ensemble for Lorenze96 model

        """
        try:
            self.background_error_model
        except(ValueError, AttributeError, NameError):
            print("Initial ensemble cannot be created before creating background error model!")
            raise ValueError
        finally:
            if self.background_error_model is None:
                print("Initial ensemble cannot be created before creating background error model!")
                raise ValueError

        if ensemble_mean is None:
            forecast_state = self.create_initial_condition()
            forecast_state = forecast_state.add(self.background_error_model.generate_noise_vec())
        else:
            assert isinstance(ensemble_mean, StateVector), "the passed ensemble mean has to be an instance of StateVector!"
            forecast_state = ensemble_mean

        ensemble = []
        for ens_ind in xrange(ensemble_size):
            state = forecast_state.copy()
            ensemble.append(state.add(self.background_error_model.generate_noise_vec()))
            #
        return ensemble
        #

    def ensemble_covariance_matrix(self, ensemble, localize=True):
        """
        Construct and ensemble-based covariance matrix given an ensemble of states

        Args:
            ensemble: list of StateVector objects
            localize (default True): Apply covariance localization

        Returns:
            covariance_mat: StateMatrix or SparseStateMatrix object containing the (localized) covariances of the
                passed ensemble

        """
        # Here I will just construct the covariance matrix to enable full decorrelation.
        # This can be used for tiny/small models but must be avoided for large models of course
        ensemble_size = len(ensemble)
        state_size = self.state_size()
        perturbations = np.zeros((state_size, ensemble_size))
        ensemble_mean = utility.ensemble_mean(ensemble)
        ensemble_mean = ensemble_mean.scale(-1.0)
        for ens_ind in xrange(ensemble_size):
            member = ensemble[ens_ind].copy()
            member = member.add(ensemble_mean)
            perturbations[:, ens_ind] = member[:]
        covariance_mat = perturbations.dot(perturbations.T)
        # diag_inds = np.arange(state_size)
        # covariance_mat[(diag_inds, diag_inds)] = 0.9* np.diagonal(covariance_mat) + 0.1

        covariance_mat = self.state_matrix(covariance_mat)
        covariance_mat = covariance_mat.scale(1.0/(ensemble_size-1))
        #
        if localize:
            covariance_mat = self.apply_state_covariance_localization(covariance_mat)
        #
        return covariance_mat
        #

    def get_neighbors_indexes(self, index, radius, source_index='state_vector', target_index=None):
        """
        Return the indexes within a given radius of influence w.r.t. a given index;
        The source_index specifies whether the index is coming from a state vector or observation vector,
        target_index specifies whether to look for grid points in the state or the observation space.
        If target_index is None, the same space as the source_index is used.
            state or observation vector.

        Args:
            index: state vector index (0:state_size-1)
            radius: radius of influence
            source_index:
                i)  'state': the index argument is coming from the state vector.
                ii) 'observation': the index argument is coming from the observation vector.
            target_index:
                i)   None; use same target as source_index
                ii)  'state': look for nearby grid points in the state pace
                iii) 'observation': look for nearby grid points in the observation pace

        Returns:
            indexes_list: a list containing indexes within the given radius of influence

        """
        source_index = source_index.lower()
        if target_index is None:
            target_index = source_index

        # dimensionality:
        state_size = self.state_size()
        observation_size = self.observation_size()

        # initilize indexes list
        indexes_list = []

        #
        if source_index.lower().startswith('state'):
            #
            # source index to look around is in the state space

            if radius >= state_size:
                radius = state_size - 1

            #
            if target_index == source_index:
                #  better approach is implemented in the other branches!
                init = index - radius
                if init < 0:
                    indexes_list.append(range(0, index + radius))
                    indexes_list.append(range(state_size+init,state_size))  # because Lorenz-96 is periodic
                else:
                    fin = index + radius
                    if fin >= state_size:
                        indexes_list.append(range(init, state_size))
                        indexes_list.append(range(0, fin-state_size))
                    else:
                        indexes_list.append(range(init, fin+1))
                 #
            elif target_index.lower().startswith('obs'):
                #
                for obs_ind in self._observation_indexes:
                    if abs(obs_ind - index) <= radius:
                        indexes_list.append(obs_ind)
                #
            else:
                print("Unknown target_index: '%s' !" % repr(target_index))
                raise ValueError

            #
        elif source_index.lower().startswith('obs'):
            # source index to look around is in the observation space

            if target_index == source_index:
                for obs_ind in self._observation_indexes:
                    if abs(obs_ind - index) <= radius:
                        indexes_list.append(obs_ind)
                 #
            elif target_index.lower().startswith('state'):
                #
                for st_ind in xrange(state_size):
                    if abs(st_ind - index) <= radius:
                        indexes_list.append(st_ind)
                #
            else:
                print("Unknown target_index: '%s' !" % repr(target_index))
                raise ValueError
            #
        else:
            print("Unknown source_index: '%s' !" % repr(source_index))
            raise ValueError

        #
        return indexes_list
        #

    def write_state(self, state, directory, file_name, append=False, file_format='mat-file'):
        """
        Save a state vector to a file.
        For mat-file format, I am following the same file format and contents and Pavel Sakov of QG-1.5 for
        ease of portability later.
        In this case each column is a state vector.

        Args:
            directory: location where observation vector will be saved
            file_name: name of the target file
            append: If set to True and the file_name exists, the observation will be appended, otherwise files will be overwritten.
            file_format: the type/format of the target file. This should be controlled by the model.

        Returns:
            None

        """
        assert isinstance(state, StateVector)
        supported_file_formats = ['mat-file']
        file_format = file_format.lower()
        if file_format not in supported_file_formats:
            raise AssertionError("File format is not supported."
                                 "Only %s are provided so far" % str(supported_file_formats))
        if not os.path.isdir(directory):
            raise IOError("Output directory [%s] is not found!" % directory)
        if file_format in ['mat-file']:
            #
            if not file_name.endswith('.mat'):
                file_name += '.mat'
            file_path = os.path.join(directory, file_name)
            if append:
                if os.path.exists(file_path):
                    # file exists, open and append.
                    old_dict = sio.loadmat(file_path)
                    try:
                        n = old_dict['n']
                        n_sample = old_dict['n_sample']
                        if isinstance(n, np.ndarray):
                            for i in xrange(n.ndim):
                                n = n[0]
                        if isinstance(n_sample, np.ndarray):
                            for i in xrange(n_sample.ndim):
                                n_sample = n_sample[0]
                        #
                        S = old_dict['S']
                    except(KeyError, NameError, AttributeError):
                        raise KeyError("The file contents do not match the default format!")
                        # raise
                    #
                    if not (np.size(S, 0)==n and np.size(S, 1)==n_sample):
                        raise ValueError("Dimension mismatch. S shape is %s" % str(np.shape(S)))
                    else:
                        # either resize S, or create a larger version.
                        n_sample += 1
                        tmp_S = np.empty((n, n_sample))
                        tmp_S[:, :-1] = S[:, :]
                        S = tmp_S
                        # Now add the new state to the last column
                        S[:, -1] = state.get_numpy_array().squeeze()
                        # Write the new values. This can be optimized by opening the file with r+ permission (will do later)
                        save_dict = dict(n=n, n_sample=n_sample, S=S)
                        sio.savemat(file_path, save_dict, oned_as='column')
                        #
                else:
                    # no file with same name in the same directory exists, it is save to write the observation a new.
                    n = self.state_size()
                    n_sample = 1
                    S = np.empty((n, 1))
                    S[:, 0] = state.get_numpy_array().squeeze()
                    save_dict = dict(n=n, n_sample=n_sample, S=S)
                    sio.savemat(file_path, save_dict, oned_as='column')
            else:
                # Go ahead and write the file, overwrite it if exists
                # no file with same name in the same directory exists, it is save to write the observation a new.
                n = self.state_size()
                n_sample = 1
                # S = np.empty((n, 1))
                S = state.get_numpy_array().squeeze()
                save_dict = dict(n=n, n_sample=n_sample, S=S)
                sio.savemat(file_path, save_dict, oned_as='column')
        #
        else:
            raise AssertionError("File format is not supported.")

    #
    def write_observation(self, observation, directory, file_name, append=False,
                          file_format='mat-file', save_observation_operator=False):
        """
        Save an observation vector to a file. Optionally save the observation operator.
        The default format is a MATLAB mat-file. the names of the variables are as follows:
            - Obs: observation(s) vector(s).
            - m:   observation vector size.
            - n_obs: number of observation vectors. Each column of Obs is an observation vector.
            - H: observation operator(s). a three dimensional array (m x n x n_obs)

        Args:
            directory: location where observation vector will be saved
            file_name: name of the target file
            append: If set to True and the file_name exists, the observation will be appended, otherwise files will be overwritten.
            file_format: the type/format of the target file. This should be controlled by the model.
            save_observation_operator: whether to write the observation operator along with the observation vector or not.

        Returns:
            None

        """
        # TODO: Update, and optimize...
        assert isinstance(observation, ObservationVector)
        supported_file_formats = ['mat-file']
        file_format = file_format.lower()
        if file_format not in supported_file_formats:
            raise AssertionError("File format is not supported."
                                 "Only %s are provided so far" % str(supported_file_formats))
        if not os.path.isdir(directory):
            raise IOError("Output directory [%s] is not found!" % directory)
        #
        if file_format in ['mat-file']:
            #
            if not file_name.endswith('.mat'):
                file_name += '.mat'
            file_path = os.path.join(directory, file_name)
            if append:
                if os.path.isfile(file_path):
                    # file exists, open and append.
                    old_dict = sio.loadmat(file_path)
                    try:
                        m = old_dict['m']
                        n_obs = old_dict['n_obs']
                        Obs = old_dict['Obs']
                        if save_observation_operator:
                            H = old_dict['H']
                    except (KeyError, NameError, AttributeError):
                        raise KeyError("The file contents do not match the default format or some variables are missing!")
                    #
                    if not (np.size(Obs,0)==m and np.size(Obs,1)==n_obs):
                        raise ValueError("Dimension mismatch. S shape is %s" % str(np.shape(S)))
                    else:
                        # either resize S, or create a larger version.
                        n_obs += 1
                        try:
                            # resize the matrix S (in-place) and add the observation after the last column.
                            Obs = np.resize(Obs, (m, n_obs))
                        except:
                            tmp_Obs = np.empty(m, n_obs)
                            tmp_Obs[:, -1] = Obs; Obs = tmp_Obs
                        # Now add the new observation to the last column
                        Obs[:, -1] = observation.get_numpy_array().squeeze()
                        #
                        # Now add the new observation operator if requested. It will be converted to a dense version of course.
                        if save_observation_operator:
                            # get the observation operator and convert it to numpy.ndarray
                            try:
                                observation_operator = self.observation_operator
                            except (ValueError, NameError, AttributeError):
                                self.construct_observation_operator()
                                observation_operator = self.observation_operator
                            observation_operator = observation_operator.toarray()  # convert to numpy (full) array to be saved.
                            # append the observational grid to the existing file
                            n = self.state_size()
                            try:
                                # resize the matrix H (in-place) and add the observation operator after the third index.
                                H = np.resize(H, (m, n, n_obs))
                            except:
                                tmp_H = np.empty(m, n, n_obs)
                                tmp_H[:, :, -1] = observation_operator; H = tmp_H
                            # Now add the new observation operator to the last index
                            H[:, :, -1] = observation_operator
                        #
                        # Write the new values. This can be optimized by opening the file with r+ permission (will do later)
                        if save_observation_operator:
                            save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H)
                        else:
                            save_dict = dict(m=m, n_obs=n_obs, Obs=Obs)
                        sio.savemat(file_path, save_dict)
                else:
                    # no file with same name in the same directory exists, it is save to write the observation a new.
                    m = self.observation_vector_size()
                    n_obs = 1
                    Obs = np.empty((m, 1))
                    Obs[:, 0] = observation.get_numpy_array().squeeze()
                    if save_observation_operator:
                        # get the observation operator and convert it to numpy.ndarray
                        try:
                            observation_operator = self.observation_operator
                        except (ValueError, NameError, AttributeError):
                            self.construct_observation_operator()
                            observation_operator = self.observation_operator
                        H = observation_operator.toarray()  # convert to numpy (full) array to be saved.
                    #
                    # Write the new values. This can be optimized by opening the file with r+ permission (will do later)
                    if save_observation_operator:
                        save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H)
                    else:
                        save_dict = dict(m=m, n_obs=n_obs, Obs=Obs)
                    sio.savemat(file_path, save_dict)
            else:
                # Go ahead and write the file, overwrite it if exists
                m = self.observation_vector_size()
                n_obs = 1
                Obs = np.empty((m, 1))
                Obs[:, 0] = observation.get_numpy_array().squeeze()
                #
                if save_observation_operator:
                    # get the observation operator and convert it to numpy.ndarray
                    try:
                        observation_operator = self.observation_operator
                    except (ValueError, NameError, AttributeError):
                        self.construct_observation_operator()
                        observation_operator = self.observation_operator
                    H = observation_operator.toarray()  # convert to numpy (full) array to be saved.
                # Write the new values. This can be optimized by opening the file with r+ permission (will do later)
                if save_observation_operator:
                    save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H)
                else:
                    save_dict = dict(m=m, n_obs=n_obs, Obs=Obs)
                sio.savemat(file_path, save_dict)
        #
        else:
            raise AssertionError("File format is not supported.")
            #

    def get_model_configs(self):
        """
        Return a dictionary containing model configurations.

        Args:
            None

        Returns:
            model_configs: a dictionary containing a copy of the model configurations

        """
        model_configs = self.model_configs.copy()
        return model_configs
        #


class Lorenz3(ModelsBase):
    """
    A very simple implementation of the Lorenz-3 model.

    Lorenz3 Model constructor:

    Args:
        model_configs: dict; a configurations dictionary for model settings.
            Supported Configurations for Lorenz-3 Model:
            * model_name: string representation of the model name
            * sigma, beta, rho: the Lorenz- model parameters.
                - Default values (sigma=10.0, beta=8.0/3, rho=28) make the model chaotic.
            * initial_state (default None): an iterable (e.g. one dimensional numpy array) containing
                a reference inintial state. length of initial_state must be the same as
                num_prognostic_variables here.
            * forward_integration_scheme (default 'ERK'): decides what time integration used to propagate model
                state and/or perturbations forward/backward in time.
                Supported schemes:
                    - 'ERK': Explicit Runge-Kutta
                    - 'LIRK': Lightly Implicit Runge-Kutta
            * default_step_size (default 0.005): optimized (stable) step size for model propagation.
                This can be usefule when long timesteps in timespans passed to the forward propagation method.
            * model_errors_distribution: probability distribution of model errors (e.g. for imperfect model
                with additive noise).
            * model_noise_level: used for creating model error variances from reference state/trajectory if
                the model_errors_covariance_method is 'empirical'.
            * model_errors_variances: an iterable that contains the model errors' variances
                (e.g. for models with Gaussian model errors)
            * create_model_errors_correlations (default False): If True; create correlation structure making
                the model error covariance matrix (Q) dense. If False, Q will be diagonal.
                This is of course if the model errors are Gaussian.
            * model_errors_covariance_method': an indicator of the method used to construct model error
                covariances
            * model_errors_steps_per_model_steps: Non-negative integer; number of model steps after which
                (additive) model noise are added to the model state when the time integrator is called to
                propagate the model.state_vector forward in time.
                If the value is zero, no model errors are incorporated (Perfect Model).
            * observation_operator_type (default 'linear'): type of the relation between model state and
                observations, e.g. 'linear', etc.
            * observation_vector_size: size of the observation vector. This can be used e.g. to control
                the observation operator construction process.
            * observation_indexes: an iterable that contains the indices of state vector coordinates/entries
                to observe. This, if not None, overrides observation_vector_size.
            * observation_errors_distribution (default 'gaussian'): probability distribution of observational
                errors (e.g. defining the likelihood fuction in the data assimialtion settings).
            * observation_noise_level: used for creating observation error variances from observations
                equivalent to reference trajectory if the observation_errors_covariance_method is 'empirical'.
            * observation_errors_variances: an iterable that contains the observational errors' variances.
            * observation_errors_covariance_method: an indicator of the method used to construct observational
                error covariances
            * background_errors_distribution: probability distribution of the prior errors.
                This can be used to create prior ensemble or initial forecast state.
            * background_errors_variances: an iterable that contains the background errors' variances.
            * create_background_errors_correlations (default True): If True; create correlation structure
                making the background/prior error covariance matrix (B) dense. If False, B will be diagonal.
                This is of course if the background errors are Gaussian.
            * background_errors_covariance_method': an indicator of the method used to construct background
                error covariances


        output_configs: dict,
            A dictionary containing screen/file output configurations for the model
            Supported configuarations:
            --------------------------
                * scr_output (default False): Output results to screen on/off switch
                * scr_output_iter (default 1): printing to screen after this number of model steps
                * verbose (default False): This is used for extensive screen outputting e.g. while debugging.
                    This overrides scr_output.
                * file_output (default False): Output results to file on/off switch
                * file_output_iter (default 1): saving model results to file after this number of model steps
                * file_output_dir: Location to save model output to (if file_output is set to True).
                    Some default directory should be provided by model or an exception will be thrown.

    Returns:
        None

    """
    # Default model configurations
    _model_name = "Lorenz3"
    _default_model_configs = dict(model_name=_model_name,
                                  sigma=10.0,
                                  beta=8.0/3,
                                  rho=28.0,
                                  initial_state=None,
                                  forward_integration_scheme='ERK',
                                  default_step_size=0.005,
                                  model_errors_distribution='gaussian',
                                  model_noise_level=0.03,
                                  model_errors_variances=None,
                                  create_model_errors_correlations=False,
                                  model_errors_covariance_method='empirical',
                                  model_errors_steps_per_model_steps=0,
                                  observation_operator_type='linear',
                                  observation_vector_size=3,
                                  observation_indexes=None,
                                  observation_errors_distribution='gaussian',
                                  observation_noise_level=0.05,
                                  observation_errors_variances=None,
                                  observation_errors_covariance_method='empirical',
                                  background_errors_distribution='gaussian',
                                  background_noise_level=0.08,
                                  create_background_errors_correlations=True,
                                  background_errors_variances=None,
                                  background_errors_covariance_method='empirical'
                                  )
    __supported_observation_operators = ['linear', 'quadratic', 'cubic']

    def __init__(self, model_configs=None, output_configs=None):

        # Aggregate passed model configurations with default configurations
        self.model_configs = utility.aggregate_configurations(model_configs, Lorenz3._default_model_configs)

        # Aggregate passed output configurations with default configurations
        self._output_configs = utility.aggregate_configurations(output_configs, ModelsBase._default_output_configs)


        # Set model constants
        self._model_name = self.model_configs['model_name']
        self._state_size = 3

        # for error models we will observe everything.
        self._observation_vector_size = self._state_size
        self._default_step_size = 0.001

        self._model_constants = {'sigma':self.model_configs['sigma'],
                                 'rho':self.model_configs['rho'],
                                 'beta': self.model_configs['beta']
                                 }

        initial_state = self.model_configs['initial_state']
        if initial_state is None:
            self._reference_initial_condition = self.create_initial_condition()
        else:
            self._reference_initial_condition = self.state_vector(np.squeeze(np.asarray(initial_state)))

        #
        # Model Error model settings and initialization
        model_errors_steps_per_model_steps = self.model_configs['model_errors_steps_per_model_steps']
        if model_errors_steps_per_model_steps > 0:
            self._perfect_model = False
            model_err_configs = dict(model_errors_distribution=self.model_configs['model_errors_distribution'],
                                     model_noise_level=self.model_configs['model_noise_level'],
                                     model_errors_variances=self.model_configs['model_errors_variances'],
                                     create_model_errors_correlations=self.model_configs['create_model_errors_correlations'],
                                     model_errors_covariance_method=self.model_configs['model_errors_covariance_method']
                                     )
            self.model_error_model = self.create_model_errors_model(configs=model_err_configs)
            self._model_errors_steps_per_model_steps = model_errors_steps_per_model_steps
        else:
            self._perfect_model = True
            self.model_error_model = None
            self._model_errors_steps_per_model_steps = 0  # Can be removed...


        # Model time integration settings
        self._default_step_size = self.model_configs['default_step_size']
        def_time_integrator_options = dict(model=self, step_size=self._default_step_size)
        forward_integration_scheme = self.model_configs['forward_integration_scheme'].lower()
        if forward_integration_scheme == 'erk':
            self._time_integrator = ExplicitTimeIntegrator(def_time_integrator_options)
        elif forward_integration_scheme == 'lirk':
            self._time_integrator = ImplicitTimeIntegrator(def_time_integrator_options)
        else:
            print("The time integration scheme %s is not defined for this Model!" % forward_integration_scheme)
            raise ValueError

        #
        # Observation-related settings:
        # ---------------------------------
        self._observation_operator_type = self.model_configs['observation_operator_type'].lower()
        if self._observation_operator_type not in Lorenz3.__supported_observation_operators:
            print("The observation operator '%s' is not supported by the Lorenz3 implementation!")
            raise NotImplementedError
        #
        observation_indexes = self.model_configs['observation_indexes']
        if observation_indexes is not None:
            observation_indexes = np.squeeze(np.asarray(observation_indexes))
            if min(observation_indexes) < 0  or max(observation_indexes) >= self._state_size:
                print("Indexes to observe are out of range. Indexes must be such that: %d < Indexes < %d !" % (0, self._state_size))
                raise IndexError
            else:
                self._observation_indexes = observation_indexes
                self._observation_vector_size = observation_indexes.size
                #
        else:
            # Observations will be linearly spaced if they are less than the
            observation_vector_size = self.model_configs['observation_vector_size']

            if observation_vector_size is not None and not isinstance(observation_vector_size, int):
                print("The option 'observation_vector_size' in the configurations dictionary has to be either None or a positive integer")
                raise ValueError
            elif observation_vector_size is None:
                observation_vector_size = self._state_size
            elif isinstance(observation_vector_size, int):
                if observation_vector_size <=0:
                    print("The option 'observation_vector_size' in the configurations dictionary has to be either None or a positive integer")
                    raise ValueError
                elif observation_vector_size > self._state_size:
                    observation_vector_size = self._state_size
                else:
                    # The observation vector size is legit
                    pass
            #
            # Generate evenly spaced indexes in the state vector:
            observation_indexes = np.empty(observation_vector_size, dtype=int)
            # observation_indexes[:]= np.rint(np.linspace(0, self._state_size, observation_vector_size, endpoint=False))
            observation_indexes = np.rint(np.linspace(0, self._state_size, observation_vector_size, endpoint=False), out=observation_indexes)
            #
            self._observation_indexes = observation_indexes
            self._observation_vector_size = observation_vector_size

        # Observation Error model settings and initialization
        obs_err_configs = dict(observation_errors_distribution=self.model_configs['observation_errors_distribution'],
                               observation_noise_level=self.model_configs['observation_noise_level'],
                               observation_errors_variances=self.model_configs['observation_errors_variances'],
                               observation_errors_covariance_method=self.model_configs['observation_errors_covariance_method']
                               )
        self.observation_error_model = self.create_observation_error_model(configs=obs_err_configs)

        #
        # Additional settings
        # ---------------------------------
        background_err_configs = dict(background_errors_distribution=self.model_configs['background_errors_distribution'],
                                      background_noise_level=self.model_configs['background_noise_level'],
                                      background_errors_variances=self.model_configs['background_errors_variances'],
                                      create_background_errors_correlations=self.model_configs['create_background_errors_correlations'],
                                      background_errors_covariance_method=self.model_configs['background_errors_covariance_method']
                                      )
        self.background_error_model = self.create_background_error_model(configs=background_err_configs)

        #
        self._initialized = True
        #

    def state_vector(self, state_vector_ref=None):
        """
        Create a wrapper for a state vector data structure, and return a reference to it.

        Args:
            state_vector_ref: a 1D-numpy array that will be wrapped, to be handled by the linear algebra module.

        Returns:
            initial_state: a wrapped 1D-numpy to be handled by the linear algebra module.

        """
        if state_vector_ref is None:
            initial_state = self._initialize_model_state_vector()
        else:
            if not isinstance(state_vector_ref, np.ndarray):
                print(" data structure passed to (instnce of) Lorenze3.state_vector() has to be np.ndarray!")
                raise AssertionError
            elif state_vector_ref.ndim != 1:
                print("A numpy.ndarray passed to (instnce of) Lorenze3.state_vector() has to one dimensional!")
                raise AssertionError
            else:
                initial_state = StateVector(state_vector_ref)
        return initial_state
        #

    def _initialize_model_state_vector(self):
        """
        Create an empty 1D numpy.ndarray, wrap it in StateVectorNumpy, and return its reference.

        Args:
            None

        Returns:
            initial_vec: 1D-numpy array wrapped by StateVectorNumpy to be handled by the linear algebra module.

        """
        initial_vec_ref = np.zeros(self.state_size())
        initial_vec = StateVector(initial_vec_ref)
        return initial_vec
        #

    def state_vector_size(self):
        """
        Return the size of the state vector (dimensions of the model space).

        Args:
            None

        Returns:
            Model state space size/dimension

        """
        return self._state_size
        #
    # add a useful alias to remove confusion
    state_size = state_dimension = state_vector_size
        #

    def observation_vector(self, observation_vector_ref=None):
        """
        Create a wrapper for a observation vector data structure, and return a reference to it.

        Args:
            observation_vector_ref: a 1D-numpy array that will be wrapped, to be handled by the linear algebra module.

        Returns:
            observation_vector: a wrapped 1D-numpy to be handled by the linear algebra module.

        """
        if observation_vector_ref is None:
            observation_vector = self._initialize_model_observation_vector()
        else:
            if not isinstance(observation_vector_ref, np.ndarray):
                print(" data structure passed to (instnce of) Lorenze3.observation_vector() has to be np.ndarray!")
                raise AssertionError
            elif observation_vector_ref.ndim != 1:
                print("A numpy.ndarray passed to (instnce of) Lorenze3.observation_vector() has to one dimensional!")
                raise AssertionError
            else:
                observation_vector = ObservationVector(observation_vector_ref)
        return observation_vector
        #

    def _initialize_model_observation_vector(self):
        """
        Create an empty 1D numpy.ndarray, wrap it in ObservationVectorNumpy, and return its reference.

        Args:
            None

        Returns:
            observation_vec: 1D-numpy array wrapped by ObservationVectorNumpy to be handled by the linear algebra module.

        """
        observation_vec_ref = np.zeros(self.observation_vector_size())
        observation_vec = ObservationVector(observation_vec_ref)
        return observation_vec
        #

    def observation_vector_size(self):
        """
        Return the size of the observation vector (dimension of the observation space).

        Args:
            None

        Returns:
            Observation state space size/dimension

        """
        return self._observation_vector_size
        #
    # add a useful alias to remove confusion
    observation_size = observation_dimension = observation_vector_size
        #

    def state_matrix(self, state_matrix_ref=None, create_sparse=False):
        """
        Create a wrapper for a model matrix data structure, and return a reference to it.
        This will hold a dense (2D numpy.ndarray) or a sparse (2D scipy.sparse.*_matrix).

        Args:
            state_matrix_ref: a 2D-numpy array that will be wrapped, to be handled by the linear algebra module.
            create_sparse: If create_sparse is True, the matrix created is empty sparse.lil_matrix, otherwise it is
                dense/full numpy.ndarray.
                If state_matrix_ref is a sparse matrix, create_sparse is ignored even if it is set to False

        Returns:
            state_matrix: 2D-numpy array wrapped by StateMatrixNumpy, or 2D sparse matrix wrapped by
                StateMatrixSpScipy to be handled by the linear algebra module.
                The size of the matrix is (state_size x state_size).

        """
        if state_matrix_ref is None:
            state_matrix = self._initialize_model_state_matrix(create_sparse=create_sparse)
        else:
            if not isinstance(state_matrix_ref, np.ndarray) and not sparse.issparse(state_matrix_ref):
                print(" data structure passed to (instnce of) Lorenze3.state_matrix() has to be np.ndarray or a sparse (scipy.*matrix) data structure!")
                raise AssertionError
            elif state_matrix_ref.ndim != 2:
                print("A numpy array or sparse matrix passed to (instnce of) Lorenze3.state_matrix() has to two dimensional!")
                raise AssertionError
            else:
                if sparse.issparse(state_matrix_ref):
                    state_matrix = SparseStateMatrix(state_matrix_ref)
                else:
                    state_matrix = StateMatrix(state_matrix_ref)
        return state_matrix
        #

    def _initialize_model_state_matrix(self, create_sparse=False):
        """
        Create an empty 2D numpy.ndarray, wrap it in StateMatrixNumpy, and return its reference.
        This returns a dense (2D numpy.ndarray), or a sparse (2D scipy.sparse.csr_matrix) based on
        sparse value. The returned data structure is wrapped by StateMatrixNumpy or StateMatrixSpScipy.

        Args:
            create_sparse: If create_sparse is True, the matrix created is empty sparse.lil_matrix, otherwise it is
                dense/full numpy.ndarray

        Returns:
            state_matrix: 2D-numpy array wrapped by StateMatrixNumpy, or 2D sparse matrix wrapped by
                StateMatrixSpScipy to be handled by the linear algebra module.
                The size of the matrix is (state_size x state_size).

        """
        state_size = self.state_size()
        if create_sparse:
            state_matrix_ref = sparse.lil_matrix((state_size, state_size), dtype=np.float32)
            state_matrix = SparseStateMatrix(state_matrix_ref)
        else:
            state_matrix_ref = np.zeros((state_size, state_size))
            state_matrix = StateMatrix(state_matrix_ref)
        return state_matrix
        #

    def observation_matrix(self, observation_matrix_ref=None, create_sparse=False):
        """
        Create a wrapper for a model matrix data structure, and return a reference to it.
        This will hold a dense (2D numpy.ndarray) or a sparse (2D scipy.sparse.*_matrix).

        Args:
            observation_matrix_ref: a 2D-numpy array that will be wrapped, to be handled by the linear algebra module.
            create_sparse: If create_sparse is True, the matrix created is empty sparse.lil_matrix, otherwise it is
                dense/full numpy.ndarray.
                If observation_matrix_ref is a sparse matrix, create_sparse is ignored even if it is set to False

        Returns:
            observation_matrix: 2D-numpy array wrapped by ObservationMatrixNumpy, or 2D sparse matrix wrapped by
                ObservationMatrixSpScipy to be handled by the linear algebra module.
                The size of the matrix is (observation_size x observation_size).

        """
        if observation_matrix_ref is None:
            observation_matrix = self._initialize_model_observation_matrix(create_sparse=create_sparse)
        else:
            if not isinstance(observation_matrix_ref, np.ndarray) and not sparse.issparse(observation_matrix_ref):
                print(" data structure passed to (instnce of) Lorenze3.observation_matrix() has to be np.ndarray or a sparse (scipy.*matrix) data structure!")
                raise AssertionError
            elif observation_matrix_ref.ndim != 2:
                print("A numpy array or sparse matrix passed to (instnce of) Lorenze3.observation_matrix() has to two dimensional!")
                raise AssertionError
            else:
                if sparse.issparse(observation_matrix_ref):
                    observation_matrix = SparseObservationMatrix(observation_matrix_ref)
                else:
                    observation_matrix = ObservationMatrix(observation_matrix_ref)
        return observation_matrix
        #

    def _initialize_model_observation_matrix(self, create_sparse=False):
        """
        Create an empty 2D numpy.ndarray, wrap it in ObservationMatrixNumpy, and return its reference.
        This returns a dense (2D numpy.ndarray), or a sparse (2D scipy.sparse.csr_matrix) based on
        sparse value. The returned data structure is wrapped by ObservationMatrixNumpy or ObservationMatrixSpScipy.

        Args:
            create_sparse: If create_sparse is True, the matrix created is empty sparse.lil_matrix, otherwise it is
                dense/full numpy.ndarray

        Returns:
            observation_matrix: 2D-numpy array wrapped by ObservationMatrixNumpy, or 2D sparse matrix wrapped by
                ObservationMatrixSpScipy to be handled by the linear algebra module.
                The size of the matrix is (observation_size x observation_size).

        """
        observation_size = self.observation_vector_size()
        if create_sparse:
            observation_matrix_ref = sparse.lil_matrix((observation_size, observation_size), dtype=np.float32)
            observation_matrix = SparseObservationMatrix(observation_matrix_ref)
        else:
            observation_matrix_ref = np.zeros((observation_size, observation_size))
            observation_matrix = ObservationMatrix(observation_matrix_ref)
        return observation_matrix
        #

    def step_forward_function(self, time_point, in_state):
        """
        Evaluate the right-hand side of the Lorenz3 model at the given model state (in_state) and time_piont.

        Args:
            time_point: scalar time instance to evaluate right-hand-side at
            in_state: current model state to evaluate right-hand-side at

        Returns:
            rhs: the right-hand side function evaluated at state=in_state, and time=time_point

        """
        assert isinstance(in_state, StateVector), "in_state passed to (instance of) \
                                        Lorenz3.step_forward_function() has to be a valid StateVector object"
        sigma = self._model_constants['sigma']
        rho = self._model_constants['rho']
        beta = self._model_constants['beta']
        #
        rhs = self.state_vector()
        rhs[0] = sigma * (in_state[1] - in_state[0])
        rhs[1] = in_state[0] * (rho - in_state[2]) - in_state[1]
        rhs[2] = in_state[0] * in_state[1] - (beta * in_state[2])
        #
        return rhs
        #

    def step_forward_function_Jacobian(self, time_point, in_state):
        """
        The Jacobian of the right-hand side of the model and evaluate it at the given model state.

        Args:
            time_point: scalar time instance to evaluate Jacobian of the right-hand-side at
            in_state: current model state to evaluate Jacobian of the right-hand-side at

        Returns:
            Jacobian: the derivative/Jacobian of the right-hand side function evaluated at
                state=in_state, and time=time_point

        """
        sigma = self._model_constants['sigma']
        rho = self._model_constants['rho']
        beta = self._model_constants['beta']
        #
        Jacobian = self.state_matrix()
        Jacobian[0, :] = np.asarray([-sigma, sigma, 0])
        Jacobian[1, :] = np.asarray([rho - in_state[2], -1, -in_state[0]])
        Jacobian[2, :] = np.asarray([in_state[1], in_state[2], -beta])
        #
        return Jacobian
        #

    def integrate_state(self, initial_state=None, checkpoints=None, step_size=None, rel_tol=1.e-9, abs_tol=1.e-9):
        """
        March the model state forward in time (backward for negative step or decreasing list).
        checkpoints should be a float scalar (taken as one step) or a list including beginning,end or a full timespan.
        The output is a list containing StateVector(s) generated by propagation over the given checkpoints.
        If checkpoints is not None, it has to be an iterable of length greater than one.
        If checkpoints is None, a single step of size (step_size) is taken and a list containing the
            initial_state, and the resulting StateVector instance is returned.
            i.e. checkpoints are replaced by [0, step_size].

        Args:
            initial_state: model.state_vector to be propagated froward according to checkpoints
            checkpoints: a timespan that should be a float scalar (taken as one step) or an iterable
                including beginning, end, or a full timespan to build model trajectory at.
                If None, a single step of size (step_size) is taken and a list containing the initial_state, and
                the resulting StateVector instance is returned. i.e. checkpoints are replaced by [0, step_size].
                If both checkpoints, and step_size are None, an assertion error is raised
            step_size: time integration step size. If None, the model default step size is used.
            rel_tol: relative tolerance for adaptive integration
            abs_tol: absolute tolerance for adaptive integration

        Returns:
            trajectory: model trajectory.
                This is a list of model state_vector objects;
                This list of model states corresponds to the times in checkpoints.

        """
        # Validate parameters:
        assert isinstance(initial_state, StateVector), "initial_state has to be a StateVector object!"
        #
        if checkpoints is not None:
            timespan = np.squeeze(np.asarray(checkpoints))
            assert timespan.size > 1, "A timespan provided in 'checkpoints' has to be an iterable of length greater than one!"
            if step_size is None:
                model_step_size = self._default_step_size
            else:
                assert np.isscalar(step_size), "step_size has to be None or a scalar!"
                model_step_size = float(step_size)
            #
        else:
            if step_size is None:
                print("No checkpoints either step_size! You gotta be kidding!")
                raise ValueError
            else:
                assert np.isscalar(step_size), "step_size has to be None or a scalar!"
                model_step_size = float(step_size)
            timespan = np.asarray([0, model_step_size])

        # Validated.
        # Now, the output should be a list of states propagated over the timespan with intermediate
        # step sizes as in model_step_size.

        # Start propagation, and adding model noise if necessary
        try:
            self._perfect_model
        except (ValueError, AttributeError):
            self._perfect_model = False
        #
        if self._perfect_model:
            # Forwad Model propagation WITHOUT model noise:
            trajectory = self._time_integrator.integrate(initial_state=initial_state,
                                                         checkpoints=timespan,
                                                         step_size=model_step_size,
                                                         rel_tol=rel_tol,
                                                         abs_tol=abs_tol
                                                         )

        else:
            # Forwad Model propagation WITH additive model noise:
            trajectory = [initial_state]
            #
            model_errors_steps_per_model_steps = self.model_configs['model_errors_steps_per_model_steps']
            #
            # loop over the timespan and add model errors when necessary:
            num_iterations = int(len(timespan)/model_errors_steps_per_model_steps)
            for iter_ind in xrange(num_iterations):
                init_ind = iter_ind * model_errors_steps_per_model_steps
                sub_timespan = timespan[init_ind: init_ind+model_errors_steps_per_model_steps+1]
                sub_initial_state = trajectory[-1]
                sub_trajectory = self._time_integrator.integrate(initial_state=sub_initial_state,
                                                                 checkpoints=sub_timespan,
                                                                 step_size=model_step_size,
                                                                 rel_tol=rel_tol,
                                                                 abs_tol=abs_tol
                                                                 )
                model_noise = self.model_error_model.generate_noise_vec()
                sub_trajectory[-1] = sub_trajectory[-1].add(model_noise)  # add model noise only to the last state in the sub_trajectory
                trajectory.append(sub_trajectory[1:])
                #
        return trajectory
        #

    def create_initial_condition(self, checkpoints=None, initial_vec=None):
        """
        Create and return a (reference) initial condition state for Lorenz3.
        This is carried out by propagating initial_vec over the timespan in checkpoints, and returning
            the final state

        Args:
            checkpoints: an iterable containing burn-in timespan
            initial_vec: iterable (e.g. array-like or StateVector) containing initial

        Returns:
            initial_condition: StateVector containing a valid initial state for Lorenz3 model

        """
        try:
            self._time_integrator
        except:
            time_integrator_defined = False
            def_time_integrator_options = dict(model=self, step_size=self._default_step_size)
            forward_integration_scheme = self.model_configs['forward_integration_scheme'].lower()
            if forward_integration_scheme == 'erk':
                self._time_integrator = ExplicitTimeIntegrator(def_time_integrator_options)
            elif forward_integration_scheme == 'lirk':
                self._time_integrator = ImplicitTimeIntegrator(def_time_integrator_options)
            else:
                print("The time integration scheme %s is not defined for this Model!" % forward_integration_scheme)
                raise ValueError
        else:
            time_integrator_defined = True

        initial_condition = self.state_vector()
        if initial_vec is None:
            initial_condition[:] = [1.5, 1.5, 1]
        else:
            initial_condition[:] = initial_vec[:]

        try:
            perfect_model = self._perfect_model
            self._perfect_model = True
        except (AttributeError, ValueError, NameError):
            perfect_model = True
            self._perfect_model = True

        if checkpoints is None:
            checkpoints = np.arange(0, 10, 0.001)

        tmp_trajectory = self.integrate_state(initial_state=initial_condition, checkpoints=checkpoints)
        initial_condition = tmp_trajectory[-1]

        # reset the model settings (perfect/non-perfect)
        if not perfect_model:
            self._perfect_model = False

        if not time_integrator_defined:
            self._time_integrator = None
        #
        return initial_condition
        #

    def construct_observation_operator(self, time_point=None, construct_Jacobian=False):
        """
        Construct the (linear version of) observation operator (H) in full. This should generally be avoided!
        We need it's (or it's TLM) effect on a state vector always.
        If called, the observation operator is attached to the model object as ".observation_operator"
        This may also work as a placeholder for the nonlinear observation case if needed.

        Args:
            time_point: the time at which the observation operator should be created
            construct_Jacobian (default False): construct a data structure (e.g. Numpy.ndarray) holding the
                Jacobian of the observation operator. This can be usefule if it is a constant sparse matrix,
                e.g. if the observation operator is linear.

        Returns:
            None

        """
        # construct the linear version of H incrementally, then convert it to CSR-format
        # This may also work as a placeholder for the nonlinear observation case if needed.
        m = self.observation_vector_size()
        n = self.state_size()
        H = sparse.lil_matrix((m, n), dtype=int)
        H[np.arange(m), self._observation_indexes] = 1
        # Return a compressed sparse row format (for efficient matrix vector product).
        self.observation_operator = H.tocsr()
        if construct_Jacobian:
            self.construct_observation_operator_Jacobian()
        #

    def update_observation_operator(self, time_point=None):
        """
        This should be called for each assimilation cycle if the observation operator is time-varying.
        For this model, the observation operator is fixed. This function simply does nothing.

        Args:
            time_point: the time at which the observation operator should be created/refreshed

        Returns:
            None

        """
        # For now the observation operator for Lorenz model is fixed and time-independent
        # Do nothing...
        pass
        #

    def evaluate_theoretical_observation(self, in_state, time_point=None):
        """
        Evaluate the theoretical observation corresponding to a model state vector,
        i.e. evaluate H(in_state), where H is the observation operator.
        If the observatin operator, time_point is used, however here it is not.

        Args:
            in_state: StatVector at which the observation operator is evaluated
            time_point: time instance at which the observation operator is evaluated

        Returns:
            observation_vec: ObservationVector instance equivalent to in_state transformed by the observation
                operator at time_point time instance.

        """
        #
        observation_vector_numpy = in_state[self._observation_indexes]
        oper_type = self._observation_operator_type
        #
        if oper_type == 'linear':
            # Just apply the observation operator on the state vector and return an observation vector.
            pass

        elif oper_type == 'quadratic':
            # raise all observed components of the state vector to power 2
            observation_vector_numpy = np.power(observation_vector_numpy, 2)

        elif oper_type == 'cubic':
            # raise all observed components of the state vector to power 2
            observation_vector_numpy = np.power(observation_vector_numpy, 3)

        else:
            raise ValueError("Unsupported observation operator type %s" % oper_type)
        #
        # Wrap and return the observation vector
        observation_vec = self.observation_vector(observation_vector_numpy)
        return observation_vec
        #

    def construct_observation_operator_Jacobian(self, time_point=None):
        """
        Create the Jacobian of the observation operator (TLM of forward operator).
        We need the TLM of the forward operator on a state vector. Can be easily extended to effect on a state matrix
        If called, the Jacobian of the observation operator is attached to the model object as ".observation_operator_Jacobian"
        This may also work as a placeholder for the nonlinear observation case if needed.

        Args:
            time_point: the time at which the Jacobaian of the observation operator should be created

        Returns:
            None

        """
        #
        oper_type = self._observation_operator_type
        #
        if oper_type in ['linear', 'quadratic', 'cubic']:
            #
            try:
                observation_operator = self.observation_operator
            except (ValueError, NameError, AttributeError):
                self.construct_observation_operator()
                observation_operator = self.observation_operator
            self.observation_operator_Jacobian = observation_operator.copy()
        else:
            print("Unsupported observation operator type '%s' !" % oper_type)
            raise ValueError
            #

    def evaluate_observation_operator_Jacobian(self, in_state, time_point=None):
        """
        Evaluate the Jacobian of the observation operator (TLM of forward operator) at the given in_state,
            and time_point.

        Args:
            in_state: StatVector at which the Jacobian of the observation operator is evaluated.
            time_point: the time at which the Jacobaian of the observation operator should be evaluated if it
                is time-dependent. In this implementation time_point is ignored.

        Returns:
            observation_operator_Jacobian: a Numpy/Sparse representation of the Jacobian of the observation
                operator (TLM of forward operator) at the given in_state, and time_point.

        """
        oper_type = self._observation_operator_type
        if oper_type not in Lorenz3.__supported_observation_operators:
            print("The observation operator '%s' is not supported by the Lorenz3 implementation!")
            raise NotImplementedError
        #
        try:
            observation_operator_Jacobian = self.observation_operator_Jacobian
        except (ValueError, NameError, AttributeError):
            self.construct_observation_operator_Jacobian()
            observation_operator_Jacobian = self.observation_operator_Jacobian
        #
        obs_coord = np.arange(self.observation_vector_size())
        obs_indexes = self._observation_indexes
        observed_vars = in_state[obs_indexes]  # numpy representation of the observed entries of in_state
        #
        if oper_type == 'linear':
            observation_operator_Jacobian[obs_coord, obs_indexes] = 1.0
        elif oper_type == 'quadratic':
            observation_operator_Jacobian[obs_coord, obs_indexes] = 2.0 * observed_vars
        elif oper_type == 'cubic':
            observation_operator_Jacobian[obs_coord, obs_indexes] = 3.0 * np.power(observed_vars, 2)
        else:
            print("The observation operator '%s' is not supported by the Lorenz3 implementation!")
            raise NotImplementedError
        #
        return observation_operator_Jacobian
        #

    def observation_operator_Jacobian_T_prod_vec(self, in_state, observation, time_point=None):
        """
        Evaluate the transpose of the Jacobian of the observation operator (evaluated at specific model state,
            and time instance) multiplied by an observation vector.
            i.e. evaluate $\\mathbf{H(in_state)}^T \\times observation\_vector$.
            The result is a state vector of course

        Args:
            in_state: StatVector at which the Jacobian of the observation operator is evaluated
            time_point: the time at which the Jacobaian of the observation operator should be evaluated if it
                is time-dependent. In this implementation time_point is ignored.
            observation: ObservationVector to be multiplied by observation operator Jacobian transposed.

        Returns:
            result_state: StatVector containing the result of observation multiplied by the
                observation operator Jacobian transposed.

        """
        oper_type = self._observation_operator_type
        if oper_type not in Lorenz3.__supported_observation_operators:
            print("The observation operator '%s' is not supported by the Lorenz3 implementation!")
            raise NotImplementedError
        #
        # Inintialize the resulting state vector:
        result_state = self.state_vector()  # this already Zeros all entries
        #
        obs_indexes = self._observation_indexes
        observed_vars = in_state[obs_indexes]  # numpy representation of the observed entries of in_state
        #
        if oper_type == 'linear':
            result_state[obs_indexes] = observation[:]
        elif oper_type == 'quadratic':
            result_state[obs_indexes] = 2.0 * np.squeeze(observed_vars) * np.squeeze(observation[:])
        elif oper_type == 'cubic':
            result_state[obs_indexes] = 3.0 * np.power(np.squeeze(observed_vars), 2) * np.squeeze(observation[:])
        else:
            print("The observation operator '%s' is not supported by the Lorenz3 implementation!")
            raise NotImplementedError
            #
        return result_state
        #

    def observation_operator_Jacobian_prod_vec(self, in_state, state, time_point=None):
        """
        Evaluate the Jacobian of the observation operator (evaluated at specific model state, and time
            instance) multiplied by a state vector (state).
            i.e. evaluate $\\mathbf{H(in_state)} \\times state$.
            The result is an observation vector

        Args:
            in_state: StatVector at which the Jacobian of the observation operator is evaluated
            state: state by which the Jacobian of the observation operator is multiplied
            time_point: the time at which the Jacobaian of the observation operator should be evaluated if it
                is time-dependent. In this implementation time_point is ignored.

        Returns:
            result_observation: ObservationVector; the result of the observation operator multiplied by state.

        """
        oper_type = self._observation_operator_type
        if oper_type not in Lorenz3.__supported_observation_operators:
            print("The observation operator '%s' is not supported by the Lorenz3 implementation!")
            raise NotImplementedError
        #
        obs_indexes = self._observation_indexes
        #
        # numpy representation of the result:
        if oper_type == 'linear':
            result_observation = np.squeeze(state[obs_indexes])
            #
        elif oper_type == 'quadratic':
            observed_vars = np.squeeze(in_state[obs_indexes])
            result_observation = 2.0 * observed_vars * np.squeeze(state[obs_indexes])
            #
        elif oper_type == 'cubic':
            observed_vars = np.squeeze(in_state[obs_indexes])
            result_observation = 3.0 * np.power(observed_vars, 2) * np.squeeze(state[obs_indexes])
            #
        else:
            print("The observation operator '%s' is not supported by the Lorenz3 implementation!")
            raise NotImplementedError
            #
        # wrap the observation vector and return
        result_observation = self.observation_vector(result_observation)
        #
        return result_observation
        #

    def create_background_error_model(self, configs=None):
        """
        create and attach an background error model

        Args:
            configs: dict,
                A configurations dictionary for the background error model
                Supported configurations:
                ---------------------------
                * errors_distribution: probability distribution of the prior errors.
                    This can be used to create prior ensemble or initial forecast state.
                * background_noise_level: This is used to create variances of the background errors, such that:
                      error variance = noise_level * signal magnitude (of initial condition).
                * background_errors_variances: an iterable that contains the background errors' variances.
                * create_errors_correlations (default True): If True; create correlation structure
                    making the background/prior error covariance matrix (B) dense. If False, B will be diagonal.
                    This is of course if the background errors are Gaussian.
                * errors_covariance_method': an indicator of the method used to construct background
                    error covariances

        Returns:
            None

        """
        if configs is None:
            configs = dict(errors_distribution=self.model_configs['background_errors_distribution'],
                           background_noise_level=self.model_configs['background_noise_level'],
                           background_errors_variances=self.model_configs['background_errors_variances'],
                           create_errors_correlations=self.model_configs['create_background_errors_correlations'],
                           errors_covariance_method=self.model_configs['background_errors_covariance_method'],
                           create_background_errors_correlations=self.model_configs['create_background_errors_correlations'],
                           )
        configs.update({'localize_errors_covariances':False})

        self.background_error_model = BackgroundErrorModel(self, configs=configs)
        self.background_error_model.construct_error_covariances(construct_inverse=True,
                                                                construct_sqrtm=True,
                                                                sqrtm_method='cholesky'
                                                                )
                                                                #

    def create_model_error_model(self, configs=None):
        """
        create and attach an model error model

        Args:
            configs: dict,
                A configurations dictionary for the model error model
                Supported configurations:
                ---------------------------
                * errors_distribution: probability distribution of the prior errors.
                    This can be used to create prior ensemble or initial forecast state.
                * model_noise_level: This is used to create variances of the model errors, such that:
                      error variance = noise_level * signal magnitude (of initial condition).
                * model_errors_variances: an iterable that contains the model errors' variances.
                * create_errors_correlations (default True): If True; create correlation structure
                    making the model error covariance matrix (Q) dense. If False, B will be diagonal.
                    This is of course if the model errors are Gaussian.
                * errors_covariance_method': an indicator of the method used to construct model
                    error covariances

        Returns:
            None

        """
        if configs is None:
            configs = dict(errors_distribution=self.model_configs['model_errors_distribution'],
                           model_noise_level=self.model_configs['model_noise_level'],
                           model_errors_variances=self.model_configs['model_errors_variances'],
                           create_errors_correlations=self.model_configs['create_model_errors_correlations'],
                           errors_covariance_method=self.model_configs['model_errors_covariance_method'],
                           variance_adjusting_factor=0.01
                           )
        configs.update({'localize_errors_covariances':False})

        self.model_error_model = ModelErrorModel(self, configs=configs)
        self.model_error_model.construct_error_covariances(construct_inverse=True,
                                                           construct_sqrtm=True,
                                                           sqrtm_method='cholesky'
                                                           )
                                                           #

    def create_observation_error_model(self, configs=None):
        """
        Create and attach an observation error model

        Args:
            configs: dict,
                A configurations dictionary for the observation error model
                Supported configurations:
                ---------------------------
                * errors_distribution (default 'gaussian'): probability distribution of observational
                    errors (e.g. defining the likelihood fuction in the data assimialtion settings).
                * observation_noise_level: used for creating observation error variances from observations
                    equivalent to reference trajectory if the observation_errors_covariance_method is 'empirical'.
                * observation_errors_variances: an iterable that contains the observational errors' variances.
                * errors_covariance_method: an indicator of the method used to construct observational
                    error covariances
                * create_errors_correlations (default False): Whether to create correlations between different
                      components of the observation vector or not. If False, diagonal covariance matrix is construct
                      otherwise it is dense (and probably localized becoming sparse if
                      localize_errors_covariances is set to True),

        Returns:
            None

        """
        if configs is None:
            configs = dict(errors_distribution=self.model_configs['observation_errors_distribution'],
                           observation_noise_level=self.model_configs['observation_noise_level'],
                           observation_errors_variances=self.model_configs['observation_errors_variances'],
                           create_errors_correlations=self.model_configs['create_observation_errors_correlations'],
                           errors_covariance_method=self.model_configs['observation_errors_covariance_method'],
                           )
        configs.update({'localize_errors_covariances':False})

        self.observation_error_model = ObservationErrorModel(self, configs=configs)
        self.observation_error_model.construct_error_covariances(construct_inverse=True,
                                                                 construct_sqrtm=True,
                                                                 sqrtm_method='cholesky',
                                                                 observation_checkpoints=np.arange(0, 1, 0.1)
                                                                 )
                                                                 #


#
#
if __name__ == '__main__':
    """
    This is a test procedure
    """
    # Test Lorenz3
    lorenz3 = Lorenz3()
    checkpoints = np.arange(0, 20, 0.1)
    trajectory = lorenz3.integrate_state(initial_state=lorenz3._reference_initial_condition,
                                         checkpoints=checkpoints
                                         )
    for time_ind in xrange(len(trajectory)):
        print(' t= %5.3f : State: %s ' %(checkpoints[time_ind], trajectory[time_ind]) )

    # Test Lorenz96
    lorenz96 = Lorenz96()
    checkpoints = np.arange(0, 10, 0.005)
    trajectory = lorenz96.integrate_state(initial_state=lorenz96._reference_initial_condition,
                                          checkpoints=checkpoints
                                          )
    for time_ind in xrange(len(trajectory)):
        print(' t= %5.3f : State: %s ' %(checkpoints[time_ind], trajectory[time_ind]) )
