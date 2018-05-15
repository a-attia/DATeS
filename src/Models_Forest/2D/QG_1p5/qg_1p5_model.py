
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
    QG1p5:
    ------
    This is a wrapper for the Frotran code implementing the Quasi-Geostrophic model 1.5.
    QG-1.5 source file is developed by Pavel Sakov [cite]
    Matlab code in the "enkf-matlab" package is replaced with this python implementation with some additions
    such as a nonlinear observation operator measuring the magnitude of the wind velocity at
    observation grid-points given the streamfunction at the model grid-points.
    Linear algebra and error models are extended to suit the structure of DATeS.
"""


# TODO : Ahmed Recode the parts responsible for the observational grid, and it's updating code...
import numpy as np
import scipy.sparse as sparse
import scipy.io as sio
import os
import sys
import re
import time
try:
    import cpickle
except:
    import cPickle as pickle

# Import wrapper modules that handles communication between the Python part of the module
# (mainly this driver) and the Fortran implementation of the model.
import qg1p5_wrapper_setup
import dates_utility as utility
from models_base import ModelsBase

from state_vector_numpy import StateVectorNumpy as StateVector
from state_matrix_numpy import StateMatrixNumpy as StateMatrix
from state_matrix_sp_scipy import StateMatrixSpSciPy as SparseStateMatrix
from observation_vector_numpy import ObservationVectorNumpy as ObservationVector
from observation_matrix_numpy import ObservationMatrixNumpy as ObservationMatrix
from qg_1p5_error_models import QGBackgroundErrorModel
from qg_1p5_error_models import QGObservationErrorModel
from qg_1p5_error_models import QGModelErrorModel


class QG1p5(ModelsBase):
    """
    A wrapper for the Quasi-Geostrophic (QG-1.5) model.

    QG-1.5 Model object constructor:

    Args:
        model_configs: dict; a configurations dictionary for model settings
            This is expected to vary greately from one model to another.
            The configs provided here are just guidelines.
            Supported Configurations for QG-1.5 Model:----------------
            * model_name: string representation of the model name
            * MREFIN: this is the variable controling the model grid size in both directions:
                .........................................................................
                ... I kept the naming as in the parameters module in the Fortran code ...
                ... Nx = 2 * 2 ^ (MREFIN-1) +1                                        ...
                ... Ny = Nx                                                           ...
                ...  ----------------------------                                     ...
                ...  > MREFIN=7: Full QG   (QG)                                       ...
                ...  > MREFIN=6: Small QG  (QGs)                                      ...
                ...  > MREFIN=5: Tiny QG   (QGt)                                      ...
                ...  ----------------------------                                     ...
                .........................................................................
            * initial_state (default None): an iterable (e.g. one dimensional numpy array) containing
                a reference inintial state. length of initial_state must be the same as
                num_prognostic_variables here.
            * num_prognostic_variables (default 40): Number of prognostic/physical variables at each
                gridpoint.
            * model_grid_type: 'cartesian', 'spherical', etc. For QG-1.5, the grid is cartesian.
            * model_errors_distribution: probability distribution of model errors (e.g. for imperfect model
                with additive noise).
            * model_errors_variances (default 0.0): an iterable/scalar that contains the model errors'
                variances (e.g. for models with Gaussian model errors).
                Check the implementation of "qg_1p5_error_models" for model errors specifications.
            * create_model_errors_correlations (default False): If True; create correlation structure making
                the model error covariance matrix (Q) dense. If False, Q will be diagonal.
                This is of course if the model errors are Gaussian.
            * model_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
            * model_errors_covariance_localization_radius: radius of influence for model error covariance
                matrix localization
            * model_errors_covariance_method (default 'diagonal'): an indicator of the method used to
                construct model error covariances.
            * model_errors_steps_per_model_steps: Non-negative integer; number of model steps after which
                (additive) model noise are added to the model state when the time integrator is called to
                propagate the model.state_vector forward in time.
                If the value is zero, no model errors are incorporated (Perfect Model).

            * observation_grid_spacing_type (default 'regular'): this controls how the observational
              grid is constructed
                  - 'random': completely random locations
                  - 'regular': regular locations, the same at successive calls
                  - 'urandom': regularly distrubuted locations, with a random offset
            * observation_operator_type (default 'linear'): type of the relation between model state and
                observations, supported observation operators are:
                    - 'linear'
                    - 'wind-magnitue'
            * observation_vector_size: size of the observation vector. This can be used e.g. to control
                the observation operator construction process.
            * observation_errors_distribution (default 'gaussian'): probability distribution of observational
                errors (e.g. defining the likelihood fuction in the data assimialtion settings).
            * observation_errors_variances: scalar, contains the observational errors' variance.
                Observation error covariance matrix is diagonal (no correlations) with constant variances
                along the diagonal.
            * observation_errors_covariance_method: an indicator of the method used to construct observational
                error covariances
            * create_observation_errors_correlations (default False): If True; create correlation structure
                making the observation error covariance matrix (R) dense. If False, R will be diagonal.
                This is of course if the observation errors are Gaussian.
                Note that, we provide observational correlations for experiemental purposes here.
            * observation_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
            * observation_errors_covariance_localization_radius: radius of influence for observation error
                covariance matrix localization.
            * background_errors_variances (default 5.0): scalar, contains the background errors' variance.
                Background error covariance matrix (B) is (initially) diagonal (no correlations) with
                constant variances along the diagonal.
            * background_errors_covariance_localization_function: e.g. 'gaspari-cohn', 'gauss', etc.
            * background_errors_covariance_localization_radius: radius of influence for background error
                covariance matrix localization
            * background_errors_covariance_method': an indicator of the method used to construct background
                error covariances.

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

        model_parameters: dict,
            A dictionary containing parameters of the QG-1.5 model. These parameters are fed to
            the Fortran source code.
            For supported parameters, check documentation of get_default_QG1p5_parameters()

    Returns:
        None

    """
    #
    # Default model, assimilation, and input/output configurations.
    _model_name = "QG-1.5"
    _def_model_configs = dict(model_name=_model_name,
                              MREFIN=5,
                              model_grid_type='cartesian',
                              model_errors_covariance_method='diagonal',
                              model_errors_distribution='gaussian',
                              create_model_errors_correlations=False,
                              model_errors_covariance_localization_function=None,
                              model_errors_covariance_localization_radius=None,
                              model_errors_variances=0.00,
                              model_errors_steps_per_model_steps=0,
                              observation_grid_spacing_type='urandom',
                              observation_operator_type='linear',
                              observation_vector_size=50,
                              observation_errors_distribution='gaussian',
                              observation_errors_variances=4.0,
                              observation_errors_covariance_method='diagonal',
                              background_errors_distribution='gaussian',
                              background_errors_variances=5.0,
                              background_errors_covariance_method='diagonal',
                              background_errors_covariance_localization_function='gaspari_cohn',
                              background_errors_covariance_localization_radius=8,
                              read_covariance_localization_from_file=True,
                              covariance_localization_file_name=None
                              )
    #
    # TODO: Ahmed; the next parameters will be automatically pulled after polishing the utility module
    __model_path = os.path.join(os.environ.get('DATES_ROOT_PATH'), 'src/Models_Forest/2D/QG_1p5/')
    __def_parameters_path = os.path.join(__model_path, 'f90/prm/')
    __def_saved_mats_path = os.path.join(__model_path, 'samples')  # change to the full path of mat files if needed
    #
    __url_enkf_path = "http://enkf.nersc.no/Code/EnKF-Matlab/"
    __url_enkf_samples_path = __url_enkf_path + "QG_samples/"

    __max_psi_allowed = 1e8  # maximum magnitude allowed for the stream function. default 1e3

    def __init__(self, model_configs=None, output_configs=None, model_parameters=None):

        # Aggregate passed model and output configurations with default configurations
        self.model_configs = utility.aggregate_configurations(model_configs, QG1p5._def_model_configs)
        # Aggregate passed output configurations with default configurations
        self.output_configs = utility.aggregate_configurations(output_configs, ModelsBase._default_output_configs)

        # TODO: Update once Utility module is updated
        self._saved_mats_path = QG1p5.__def_saved_mats_path

        # Find the model's state space dimension (size of the model grid x num. of prognostic variables).
        self._num_prognostic_vars = 1
        self._num_dimensions = 2
        MREFIN = self.model_configs['MREFIN']
        self._state_size = self._num_prognostic_vars * pow((2 * pow(2, (MREFIN-1)) + 1), 2)

        # Get default model parameters.
        # These will be written for the fortran code to use while propagating states forward in time...
        self._def_model_parameters, parameters_file_path = self.get_default_QG1p5_parameters(write_to_file=True)
        self._model_parameters_file_path = parameters_file_path
        self._model_parameters = utility.aggregate_configurations(model_parameters, self._def_model_parameters)

        #
        # Create a wrapper for the Fortran code with the desired settings:
        # ==================================================================
        # -----------------<Online WRAPPER GENERATION>----------------------
        # ==================================================================
        qg1p5_wrapper_setup.create_wrapper(in_MREFIN=MREFIN)
        from QG_wrapper import qgstepfunction  # CAN-NOT be moved up.
        self.__qgstepfunction = qgstepfunction
        # ==================================================================
        # ----------------------------<Done!>-------------------------------
        # ==================================================================

        # Create and attach a reference initial condition for the mode.
        # This is read from the enkf-matlab samples DB.
        # If the data set is missing, it will be downloaded from the enk-matlab website.
        self._reference_initial_condition = self.create_initial_condition()

        # Add grid info to the model configurations. Just used for proper outputting
        nx = np.int(np.floor(np.sqrt(self._state_size)))
        dx = 1.0  # 1.0 / (nx - 1); this is assumed 1 only in the Python code to match obs-grid, and Matlab original code
        ny = self.state_size() / nx
        dy = 1.0  # 1.0 / (ny - 1); this is assumed 1 only in the Python code to match obs-grid, and Matlab original code
        boundary_indexes = self._get_boundary_indexes(nx, ny)
        self.model_configs.update(dict(state_size=self._state_size,
                                       nx=nx, ny=ny, dx=dx, dy=dy,
                                       num_prognostic_variables=self._num_prognostic_vars,
                                       num_dimensions=self._num_dimensions, periodic=False,
                                       boundary_indexes=boundary_indexes
                                       )
                                  )
        self_model_name = self.model_configs['model_name']

        # The options of the following grids (and even whether to construct them now) should be revised!
        self.observation_operator = None
        self.observation_operator_Jacobian = None
        #
        self._observation_operator_type = self.model_configs['observation_operator_type'].lower()
        self._observation_vector_size = self.model_configs['observation_vector_size']
        self.construct_model_grids(construct_full_grid=False)
        self._observation_grid_spacing_type = self.model_configs['observation_grid_spacing_type'].lower()
        self.construct_observational_grid()

        # Create Observation error model:
        observation_error_confgis =dict(variance=self.model_configs['observation_errors_variances'],
                                        observation_errors_covariance_method=
                                        self.model_configs['observation_errors_covariance_method']
                                        )
        self.create_observation_error_model(configs=observation_error_confgis)
        #
        # Create Background error model:
        background_error_confgis =dict(variance=self.model_configs['background_errors_variances'],
                                       background_errors_covariance_method=
                                       self.model_configs['background_errors_covariance_method']
                                       )
        self.create_background_error_model(configs=background_error_confgis)

        # Create Model error model:
        # TODO: Model errors for QG-1.5 are very much simplified so far. This will be updated along with
        # my next line of research (MCMC/ClHMC-based sampling with imperfect models)
        model_error_variance = self.model_configs['model_errors_variances']
        if model_error_variance > 0.0:
            model_error_confgis = dict(variance=model_error_variance,
                                       model_errors_covariance_method=self.model_configs['background_errors_covariance_method'])
            self.create_model_error_model(configs=model_error_confgis)
            self._perfect_model = False
        else:
            self._perfect_model = True
        # Override if no number of steps is provided; the model errors are kept though
        model_errors_steps_per_model_steps = self.model_configs['model_errors_steps_per_model_steps']
        if model_errors_steps_per_model_steps <= 0:
            self._perfect_model = True

        self.verbose = self.output_configs['verbose']
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
        initial_vec_ref = np.zeros(self.state_size(), dtype=np.float64)
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
        observation_vec_ref = np.zeros(self.observation_vector_size(), dtype=np.float64)
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

    def _get_boundary_indexes(self, nx, ny):
        """
        The boundary indexes of this specific model.
        This will be used while generating a state-based covariance matrix, e.g. for model or background errors.
        """
        top_bounds = np.arange(nx)
        right_bounds = np.arange(2*nx-1, nx**2-nx+1, nx)
        left_bounds = np.arange(nx, nx**2-nx, nx )
        down_bounds = np.arange(nx**2-nx, nx**2)
        side_bounds = np.reshape(np.vstack((left_bounds, right_bounds)), (left_bounds.size+right_bounds.size), order='F')
        grid_boundary_indexes = np.concatenate((top_bounds, side_bounds, down_bounds))
        return grid_boundary_indexes
        #
    def construct_model_grids(self, construct_full_grid=False):
        """
        The model grid is a numpy array of size (nx x ny x ...)
        construct_indexes: if True the indexes array will be

        Args:
            construct_full_grid (default False): Has to be turned-on for the model grid to be constructed in full,
                and attached to the model instance.
                The model grid is not used explicitly in this implementation, but can be used for
                printing/plotting, etc...

        Returns:
            None

        """
        # self._grid_dict = dict()  # consider removing from all models...
        try:
            nx = self.model_configs['nx']
            dx = self.model_configs['dx']
            ny = self.model_configs['ny']
            dy = self.model_configs['dy']
        except:
            nx = np.int(np.floor(np.sqrt(self.state_size())))
            dx = 1.0 / (nx - 1)
            ny = self.state_size() / nx
            dy = 1.0 / (ny - 1)

        self._grid_sizes = np.array([nx, ny])  # an array containing size of spatial grid in each dimension.
        self._grid_spacings = np.array([dx, dy])  # an array containing step size (dx,dy,...) of spatial grid in each
        #
        # Total number of spatial grid points. Equal to state_size/ number of prognostic variables.
        self._total_grid_size = np.product(self._grid_sizes)

        if construct_full_grid:  # TODO: Validate
            self._model_grid = np.empty((self._state_size, self._num_dimensions))
            x_indexes = np.arange(nx) * dx
            y_indexes = np.arange(ny) * dy
            self._model_grid[:, 0] = list(x_indexes) * ny  # test if reshaping is faster!
            self._model_grid[:, 1] = np.repeat(y_indexes, nx)
        else:
            self._model_grid = None
        try:
            if self.verbose:
                print('model grid:', self._model_grid)
        except AttributeError:
            pass
            #

    def get_model_grid(self):
        """
        Return a copy of the model grid as a numpy array with number of rows equal to state_size,
        and each row cotains the coordinates of each grid point
        """
        try:
            grid = self._model_grid
        except(NameError, ValueError, AttributeError):
            self.construct_model_grids(construct_full_grid=True)
            grid = self._model_grid
        finally:
            if grid is None:
                self.construct_model_grids(construct_full_grid=True)
                grid = self._model_grid
        return grid.copy()
        #

    def integrate_state(self, initial_state, checkpoints):
        """
        March the model state forward in time.
        checkpoints should be a float scalar (taken as one step) or a list including beginning,end or a full timespan.
        The output is a list containing StateVector(s) generated by propagation over the given checkpoints.
        If checkpoints is scalar, a single step of size (self._default_model_step_size) is taken and
            the resulting StateVector instance is returned.
        If checkpoints is an iterable containing a timespan, a list of model states (checked out) at the
            points in the checkpoints.

        Args:
            initial_state: model.state_vector to be propagated froward according to checkpoints
            checkpoints: a timespan that should be a float scalar (taken as one step) or an iterable
                including beginning, end, or a full timespan to build model trajectory at.

        Returns:
            trajectory: model StateVector or a list describing the model trajectory at the given checkpoints.

        """
        # Validate parameters:
        assert isinstance(initial_state, StateVector), "initial_state has to be a StateVector object!"
        #
        checkpoints = np.squeeze(np.asarray(checkpoints))

        # Now, the output should be a list of states propagated over the timespan.
        # The step size is set according to self._model_parameters

        # retrieve grid settings
        nx = self.model_configs['nx']
        ny = self.model_configs['ny']

        # Start propagation, and adding model noise if necessary
        try:
            self._perfect_model
        except (ValueError, AttributeError):
            self._perfect_model = False
        #
        state = initial_state.copy()
        #
        if self._perfect_model:
            # Forwad Model propagation WITHOUT model noise:
            if np.size(checkpoints) == 1:
                t = float(checkpoints)  # to get base type use checkpoints.item()
                state_ref = np.reshape(state.get_raw_vector_ref(), (nx, ny), order='F')
                # state_ref = np.reshape(state.get_raw_vector_ref() ,
                t, state_ref = self.__qgstepfunction(t, state_ref, self._model_parameters_file_path)
                state.set_raw_vector_ref(np.reshape(state_ref, self.state_size(), order='F'))
                # state.set_raw_vector_ref(state_ref)
                out_states = state
            else:
                out_states = []
                out_states.append(state)
                for time_ind in xrange(1, len(checkpoints)):
                    # Start controlling the model step size and tend based on the checkpoints
                    t = checkpoints[time_ind-1]
                    next_t = checkpoints[time_ind]
                    # self._model_parameters['tend'] = 0
                    dt = self._model_parameters['dt']
                    dtout = self._model_parameters['dtout']
                    # self.write_parameters(self._model_parameters)
                    #
                    next_state = state.copy()
                    state_ref = np.reshape(next_state.get_raw_vector_ref(), (nx, ny), order='F')
                    #
                    while t < next_t:
                        if next_t>t and (next_t-t)<dt:
                            restore_dt = True
                            def_dt = self._model_parameters['dt']
                            #
                            self._model_parameters['dt'] = next_t-t
                            self._model_parameters['dtout'] = next_t-t
                            self.write_parameters(self._model_parameters)
                        else:
                            restore_dt = False
                        #
                        t, state_ref = self.__qgstepfunction(t, state_ref, self._model_parameters_file_path)

                    # update state to the state at time=next_t:
                    next_state.set_raw_vector_ref(np.reshape(state_ref, self.state_size(), order='F'))

                    if restore_dt:
                        self._model_parameters['dt'] = def_dt
                        self._model_parameters['dtout'] = def_dt
                        self.write_parameters(self._model_parameters)
                    else:
                        pass
                    # --------------------------------------------------------------------------------------
                    # Check the entries of the state vector:
                    # --------------------------------------------------------------------------------------
                    max_psi, min_psi = next_state.max(), next_state.min()
                    max_psi_val = max(abs(max_psi), abs(min_psi))
                    if np.isnan(max_psi_val) or np.isinf(max_psi_val) or max_psi_val>self.__max_psi_allowed:
                        sys.exit("Model State physics violation: invalid magnitude [%s] at time = %f\n" % (repr(max_psi_val), t))
                    # --------------------------------------------------------------------------------------

                    # Update model trajectory:
                    out_states.append(next_state)
                    state = next_state
                    #
        else:
            # Forwad Model propagation WITH additive model noise:
            model_errors_steps_per_model_steps = self.model_configs['model_errors_steps_per_model_steps']
            #
            if np.size(checkpoints) == 1:
                # A single model step is taken
                t = float(checkpoints)  # to get base type use checkpoints.item()
                state_ref = np.reshape(state.get_raw_vector_ref(), (nx, ny), order='F')
                t, state_ref = self.__qgstepfunction(t, state_ref, self._model_parameters_file_path)
                state.set_raw_vector_ref(np.reshape(state_ref, self.state_size(), order='F'))
                if model_errors_steps_per_model_steps == 1:
                    state = state.add(self.model_error_model.generate_noise_vec())
                    #
            else:
                out_states = []
                out_states.append(state)
                steps_taken = 0  # number of steps the model propagated so far
                for time_ind in xrange(1, len(checkpoints)):
                    # start controlling the model step size and tend based on the checkpoints
                    t = checkpoints[time_ind-1]
                    next_t = checkpoints[time_ind]
                    self._model_parameters['tend'] = 0
                    # self._model_parameters['dtout'] = next_t - t
                    # self.write_parameters(self._model_parameters)
                    dt = self._model_parameters['dt']
                    dtout = self._model_parameters['dtout']
                    #
                    next_state = state.copy()
                    state_ref = np.reshape(next_state.get_raw_vector_ref(), (nx, ny), order='F')
                    #
                    while t < next_t:
                        if next_t>t and (next_t-t)<dt:
                            restore_dt = True
                            def_dt = self._model_parameters['dt']
                            #
                            self._model_parameters['dt'] = next_t-t
                            self._model_parameters['dtout'] = next_t-t
                            self.write_parameters(self._model_parameters)
                        else:
                            restore_dt = False
                        # Advance the model state one time step:
                        t, state_ref = self.__qgstepfunction(t, state_ref, self._model_parameters_file_path)
                        #
                        # Check the steps taken so far, and add noise to the model state when appropriate:
                        steps_taken += 1
                        if (steps_taken % model_errors_steps_per_model_steps) == 0:
                            # add model noise:
                            noise_vec = np.reshape(self.model_error_model.generate_noise_vec().get_raw_vector_ref(), (nx, ny), order='F')
                            state_ref += noise_vec
                            #
                    # update state to the state at time=next_t:
                    next_state.set_raw_vector_ref(np.reshape(state_ref, self.state_size(), order='F'))

                    if restore_dt:
                        self._model_parameters['dt'] = def_dt
                        self._model_parameters['dtout'] = def_dt
                        self.write_parameters(self._model_parameters)
                    else:
                        pass
                    # --------------------------------------------------------------------------------------
                    # Check the entries of the state vector:
                    # --------------------------------------------------------------------------------------
                    #
                    max_psi, min_psi = next_state.max(), next_state.min()
                    max_psi_val = max(abs(max_psi), abs(min_psi))
                    if np.isnan(max_psi_val) or np.isinf(max_psi_val) or max_psi_val>1e3:
                        sys.exit("Model State physics violation: invalid magnitude at time = %f\n" % t)
                    # --------------------------------------------------------------------------------------

                    # Update model trajectory:
                    # out_states.append(next_state)
                    out_states.append(next_state.copy())  # remove copying after debugging
                    state = next_state
                    #
        return out_states
        #

    def construct_observational_grid(self):
        """
        Create observational grid points, and attach observations positions to model object.
        Depending on the value of 'observation_grid_spacing_type' it generates:
            random  - completely random locations
            regular - regular locations, the same at successive calls
            urandom - regularly distrubuted locations, with a random offset
        This should be called by self.construct_observational_grid to get the actual coordinates of the observational
        grid and once set, can be called by the observation operator to evaluate model observation H(x)...
        Note: Here we assume we have a single prognostic variable. For more variables, the grid should be replicated
        based on the number of observed prognostic variable(s)!

        Args:

        Returns:
            None

        """
        try:
            spacing_type = self._observation_grid_spacing_type
        except(AttributeError, NameError):
            try:
                spacing_type = self.model_configs['observation_grid_spacing_type']
            except KeyError:
                print("the observation_grid_spacing_type is missing from the configurations dictionary!")
                raise
        finally:
            if spacing_type is None:
                print("Failed to retreive the observation_grid_spacing_type from model attributes!")
                raise ValueError
            #
        try:
            nx = self.model_configs['nx']
            dx = self.model_configs['dx']
            ny = self.model_configs['ny']
            dy = self.model_configs['dy']
        except:
            nx = np.int(np.floor(np.sqrt(self.state_size())))
            dx = 1.0 / (nx - 1)
            ny = self.state_size() / nx
            dy = 1.0 / (ny - 1)

        observation_size = self.observation_vector_size()
        observations_positions = np.empty((observation_size, self._num_dimensions))
        if re.match(r'\Aregular\Z', spacing_type, re.IGNORECASE):
            nn = (nx - 1) * (ny - 1)
            dn = float(nn) / observation_size
            for obs_ind in xrange(observation_size):
                pp = dn * (obs_ind + 0.5)
                observations_positions[obs_ind, 0] = pp % (nx - 1.0)
                observations_positions[obs_ind, 1] = pp / (nx - 1.0)
                #
        elif re.match(r'\Arandom\Z', spacing_type, re.IGNORECASE):
            for obs_ind in xrange(1, observation_size+1):
                observations_positions[obs_ind, 0] = np.random.rand() * (nx - 2)  # 1 <= px <= nx
                observations_positions[obs_ind, 1] = np.random.rand() * (ny - 2)  # 1 <= py <= ny
                #
        elif re.match(r'\Aurandom\Z', spacing_type, re.IGNORECASE):
            nn = (nx - 1) * (ny - 1)
            offset = np.random.rand() * nn
            dn = float(nn) / observation_size
            for obs_ind in xrange(1, observation_size+1):
                pp = (dn * obs_ind + offset) % nn
                observations_positions[obs_ind-1, 0] = pp % (nx - 1.0)
                observations_positions[obs_ind-1, 1] = pp / (nx - 1.0)
            #
        else:
            print("Observational grid spacing type [' %s '] is not supported!" % spacing_type)
            raise ValueError

        # attach the updated grid to the model
        self._observations_positions = observations_positions
        #


    def get_observations_positions(self, attach_to_model=True):
        """
        Return (a copy) an array containing fractional indexes of observations locations w.r.t model grid indexes.

        Args:

        Returns:
            None

        """
        try:
            observations_positions = self._observations_positions
        except(AttributeError, NameError):
            self.construct_observational_grid()
            observations_positions = self._observations_positions
        finally:
            if observations_positions is None:
                self.construct_observational_grid()
                observations_positions = self._observations_positions
        #
        return observations_positions.copy()
        #
    # add alias
    get_observational_grid = get_observations_positions

    def construct_observation_operator(self, time_point=None, construct_Jacobian=False):
        """
        Construct the observation operator (H) in full. This should be avoided in practice.
        We need it's (or it's TLM) effect on a state vector always.
        For the current model (QG) a sparse structure is used.

        Args:
            time_point: the time at which the observation operator should be created
            construct_Jacobian (default False): construct a data structure (e.g. Numpy.ndarray) holding the
                Jacobian of the observation operator. This can be usefule if it is a constant sparse matrix,
                e.g. if the observation operator is linear.

        Returns:
            None

        """
        if self.verbose:
            print("Warning: The observation operator SHOULD NOT be built in full.\n  \
                            We have replaced the construction of H with it's effect of a state vector. \
                            \nSee 'evaluate_theoretical_observation' for evaluating H(x)\
                            \nSee 'observation_operator_Jacobian_T_prod_vec' for evaluating H^T x ")
        # TODO: This implementation is no longer needed, will keep it as a reference for now.
        # This is now replaced with evaluating it's effect on a state.
        operator_type = self._observation_operator_type.lower()
        if re.match(r'\Alinear\Z', operator_type, re.IGNORECASE):
            #
            n = self._state_size
            npv = self._num_prognostic_vars
            mult = 1
            nv = (n - npv) / mult

            try:
                nx = self.model_configs['nx']
                dx = self.model_configs['dx']
                ny = self.model_configs['ny']
                dy = self.model_configs['dy']
            except:
                nx = np.int(np.floor(np.sqrt(self.state_size())))
                dx = 1.0 / (nx - 1)
                ny = self._state_size / nx
                dy = 1.0 / (ny - 1)

            try:
                observations_positions = self._observations_positions
            except AttributeError:
                self.get_observations_positions(attach_to_model=True)
                observations_positions = self._observations_positions
            pos = observations_positions
            p = np.size(pos, 0)

            # construct H incrementally, then convert it to CSR-format
            H = sparse.lil_matrix((p, n), dtype=np.float)

            for p_ind in xrange(p):
                pp = pos[p_ind, :]
                px = np.floor(pp[0])
                py = np.floor(pp[1])
                fx = pp[0] - px
                fy = pp[1] - py
                # py -= 1
                if fx != 0 and fy != 0:  # Bilinear interpolation:
                    H[p_ind, (px + py * nx)] = (1 - fx) * (1 - fy)  # left-up
                    H[p_ind, (px + 1 + py * nx)] = fx * (1 - fy)    # left-down
                    H[p_ind, (px + (py + 1) * nx)] = (1 - fx) * fy  # right-up
                    H[p_ind, (px + 1 + (py + 1) * nx)] = fx * fy    # right-down
                    #
                elif fx == 0 and fy == 0:
                    H[p_ind, (px + py * nx)] = 1
                    #
                elif fx == 0:
                    H[p_ind, (px + py * nx)] = (1 - fy)
                    H[p_ind, (px + (py + 1) * nx)] = fy
                    #
                elif fy == 0:
                    H[p_ind, (px + py * nx)] = 1 - fx
                    H[p_ind, (px + 1 + py * nx)] = fx
                    #
                else:
                    print("It doesn't make sense to reach this!")
                    raise ValueError

                if self.verbose:
                    print("Assigned on row %d:\n --------------------- " % p_ind)
                    print(H.nonzero())
                    print("px=%d, py=%d, indexes [(px + py * nx), (px + 1 + py * nx), (px + (py + 1) * nx), (px + 1 + (py + 1) * nx)]:" %(px, py))
                    print((px + py * nx), (px + 1 + py * nx), (px + (py + 1) * nx), (px + 1 + (py + 1) * nx))
                    print("Values", H[p_ind, :])

            # Return a compressed sparse row format (for efficient matrix vector product).
            self.observation_operator = H.tocsr()
            #
        elif re.match(r'\Awind(-|_)*magnitude\Z', operator_type, re.IGNORECASE):
            # We need to observation_operator components each to extract one of the velocity components from the stream-function.
            # Since we use linear interpolation, we need the slope of the line at observation locations.
            n = self._state_size
            npv = self._num_prognostic_vars
            mult = 1
            nv = (n - npv) / mult

            try:
                nx = self.model_configs['nx']
                dx = self.model_configs['dx']
                ny = self.model_configs['ny']
                dy = self.model_configs['dy']
            except:
                nx = np.int(np.floor(np.sqrt(self.state_size())))
                dx = 1.0 / (nx - 1)
                ny = np.int(np.floor(self._state_size / nx))
                dy = 1.0 / (ny - 1)

            try:
                observations_positions = self._observations_positions
            except AttributeError:
                observations_positions = self.get_observations_positions(attach_to_model=True)

            pos = observations_positions
            p = self.observation_vector_size()

            # construct H incrementally, then convert it to CSR-format
            Hu = sparse.lil_matrix((p, n), dtype=np.float)
            Hv = sparse.lil_matrix((p, n), dtype=np.float)

            # two lists containing the nonzero indexes of the observation operator for U, and V components
            Hu_y_inds = []
            Hv_y_inds = []

            for p_ind in xrange(p):
                pp = pos[p_ind, :]
                px = np.floor(pp[0])
                py = np.floor(pp[1])
                fx = pp[0] - px
                fy = pp[1] - py
                # py -= 1
                if fx != 0 and fy != 0:
                    #
                    # (d PSI)/dx --> -v component
                    hv_u, hv_d = px + py * nx, px + 1 + py * nx
                    Hv[p_ind, hv_u] = 1.0/dx  # left-up
                    Hv[p_ind, hv_d] = -1.0/dx  # left-down
                    #
                    # (d PSI)/dy --> u component
                    hu_l, hu_r = px + py * nx, px + (py + 1) * nx
                    Hu[p_ind, hu_l] = -1.0/dy  # left-up
                    Hu[p_ind, hu_r] = 1.0/dy  # right-up
                    #
                elif fx == 0 and fy == 0:
                    # (d PSI)/dx --> -v component
                    hv_u, hv_d = px - 1 + py * nx, px + 1 + py * nx
                    Hv[p_ind, hv_u] = 0.5/dx  # up
                    Hv[p_ind, hv_d] = -0.5/dx  # down
                    #
                    # (d PSI)/dy --> u component
                    hu_l, hu_r = px + (py - 1) * nx, px + (py + 1) * nx
                    Hu[p_ind, hu_l] = -0.5/dy  # left
                    Hu[p_ind, hu_r] = 0.5/dy  # right
                    #
                elif fx == 0:
                    # (d PSI)/dx --> -v component
                    hv_u, hv_d = px - 1 + py * nx, px + 1 + py * nx
                    Hv[p_ind, hv_u] = 0.5/dx  # up
                    Hv[p_ind, hv_d] = -0.5/dx  # down
                    #
                    # (d PSI)/dy --> u component
                    hu_l, hu_r = px + py * nx, px + (py + 1) * nx
                    Hu[p_ind, hu_l] = -1.0/dy  # left-up
                    Hu[p_ind, hu_r] = 1.0/dy  # right-up
                    #
                elif fy == 0:
                    # (d PSI)/dx --> -v component
                    hv_u, hv_d = px + py * nx, px + 1 + py * nx
                    Hv[p_ind, hv_u] = 1.0/dx  # left-up
                    Hv[p_ind, hv_d] = -1.0/dx  # left-down
                    #
                    # (d PSI)/dy --> u component
                    hu_l, hu_r = px + (py - 1) * nx, px + (py + 1) * nx
                    Hu[p_ind, hu_l] = -0.5/dy  # left
                    Hu[p_ind, hu_r] = 0.5/dy  # right
                    #
                else:
                    raise ValueError("It doesn't make sense to reach this!")

                # update the lists of nonzero indexes:
                Hu_y_inds.append([hu_l, hu_r])
                Hv_y_inds.append([hv_u, hv_d])

            # Return a tuple of compressed sparse row formats (for efficient matrix vector product).
            # Evaluate and save the indexes of nonzero entries of the observation operator:
            self.observation_operator = (Hu.tocsr(), Hv.tocsr(), Hu_y_inds, Hv_y_inds)
            #
        #
        else:
            print("Observation operator '%s' is not supported!" % operator_type)
            raise ValueError
            #
        if construct_Jacobian:
            self.construct_observation_operator_Jacobian(time_point=time_point)
            #

    def update_observation_operator(self, time_point=None):
        """
        This should be called for each assimilation cycle if the observation operator is time-varying.
        :param time: the time at which the observation operator should be created/refreshed

        Args:
            time_point: the time at which the observation operator should be created/refreshed

        Returns:
            None

        """
        # In this impmlementation, the observaiton operator will not be constructed in full,
        # alternatively, all is required is to change the observational grid
        # The observation operator here is updated randomly in some sense each time it is called.
        try:
            spacing_type = self._observation_grid_spacing_type
        except(AttributeError, NameError):
            try:
                spacing_type = self.model_configs['observation_grid_spacing_type']
            except KeyError:
                print("the observation_grid_spacing_type is missing from the configurations dictionary!")
                raise
        finally:
            if spacing_type is None:
                print("Failed to retreive the observation_grid_spacing_type from model attributes!")
                raise ValueError
            #
        if re.match(r'\Aregular\Z', spacing_type, re.IGNORECASE):
            # Check if the observational grid is already built. Construct only if it is not built yet
            try:
                observations_positions = self._observations_positions
            except (AttributeError, NameError):
                observations_positions = self.get_observations_positions(attach_to_model=True)
            finally:
                if observations_positions is None:
                    observations_positions = self.get_observations_positions(attach_to_model=True)

        elif re.match(r'\Au{0,1}random\Z', spacing_type, re.IGNORECASE):
            # Update the observational grid
            observations_positions = self.get_observations_positions(attach_to_model=True)
        else:
            print("Observational grid spacing type [' %s '] is not supported!" % spacing_type)
            raise ValueError
        #
        return observations_positions.copy()
        #

    def evaluate_theoretical_observation(self, in_state, time_point=None):
        """
        Evaluate the theoretical observation corresponding to a model state vector,
        i.e. evaluate H(state), where H is the observation operator.
        We need to be careful when H here is applied to an observation vector. Specifically make sure you apply the
        sparse operator on the observation_vector.ref_vector while the output is of the same type as that
        reference vector inside the observation vector.

        Args:
            in_state: StatVector at which the observation operator is evaluated
            time_point: time instance at which the observation operator is evaluated

        Returns:
            observation_vec: ObservationVector instance equivalent to in_state transformed by the observation
                operator at time_point time instance.

        """
        if self.verbose:
            elapsed_time = time.time()

        # Just apply the observation operator on the state vector and return an observation vector.
        operator_type = self._observation_operator_type
        if re.match(r'\Alinear\Z', operator_type, re.IGNORECASE):
            # Naive implementation:
            # TODO: This is no longer required. Remove with the code for full construction of the observation operator
            # get the reference vector from the state vector and reshape it
            # Multiply H by the reshaped state vector
            # Attach resulting vector to the observation vector
            # numpy_reshaped_state_vec = np.reshape(in_state.get_numpy_array(), self._state_size, order='F')
            # numpy_state_vec = in_state.get_numpy_array()
            # observation_vector = self.observation_vector(observation_operator.dot(numpy_state_vec))
            #
            # ---------------------------------
            #       Optimization Attempt
            # ---------------------------------
            observation_vector_np = np.empty(self.observation_vector_size())  # this will be the ref_vector later
            #
            n = self._state_size
            npv = self._num_prognostic_vars
            mult = 1
            nv = (n - npv) / mult

            try:
                nx = self.model_configs['nx']
                dx = self.model_configs['dx']
                ny = self.model_configs['ny']
                dy = self.model_configs['dy']
            except:
                nx = np.int(np.floor(np.sqrt(self.state_size())))
                dx = 1.0 / (nx - 1)
                ny = self.state_size() / nx
                dy = 1.0 / (ny - 1)

            try:
                pos = self._observations_positions
            except AttributeError:
                pos = self.get_observations_positions(attach_to_model=True)

            p = np.size(pos, 0)

            # H is not constructed in full, it is just used implicitly to update the observation vector
            for p_ind in xrange(p):
                # set indexes
                pp = pos[p_ind, :]
                pp_0 = pos[p_ind, 0]
                pp_1 = pos[p_ind, 1]
                px = np.floor(pp_0)
                py = np.floor(pp_1)
                fx = pp_0 - px
                fy = pp_1 - py
                # py -= 1
                # set indexes
                lu_ind = px + py * nx  # left-up
                ld_ind = px + 1 + py * nx  # left-down
                ru_ind = px + (py + 1) * nx  # right-up
                rd_ind = px + 1 + (py + 1) * nx  # right-down
                #
                # Evaluate observation vector
                if fx != 0 and fy != 0:  # Bilinear interpolation:
                    observation_vector_np[p_ind] = (((1-fx)*(1-fy)) * in_state[lu_ind]) + \
                                                ((fx*(1-fy)) *in_state[ld_ind]) + \
                                                (((1-fx)*fy) * in_state[ru_ind]) + \
                                                ((fx*fy) * in_state[rd_ind])
                    #
                elif fx == 0 and fy == 0:
                    observation_vector_np[p_ind] = in_state[lu_ind]
                    #
                elif fx == 0:
                    observation_vector_np[p_ind] = (1-fy) * in_state[lu_ind] + fy * in_state[ru_ind]
                    #
                elif fy == 0:
                    observation_vector_np[p_ind] = (1-fx) * in_state[lu_ind] + fx * in_state[ld_ind]
                    #
                else:
                    print("It doesn't make sense to reach this!")
                    raise ValueError
            #
            # wrap into observation vector:
            observation_vector = self.observation_vector()
            observation_vector[:] = observation_vector_np

        #
        elif re.match(r'\Awind(-|_)*magnitude\Z', operator_type, re.IGNORECASE):
            # ---------------------------------
            #      Optimization Attempt
            # ---------------------------------

            state_size = self._state_size
            npv = self._num_prognostic_vars
            # mult = 1
            nv = (state_size - npv)  # / mult

            nx = np.int(np.floor(np.sqrt(state_size)))
            dx = 1.0 / (nx - 1)
            ny = np.int(np.floor(state_size / nx))
            dy = 1.0 / (ny - 1)
            try:
                pos = self._observations_positions
            except AttributeError:
                pos = self.get_observations_positions(attach_to_model=True)

            p = self.observation_vector_size()
            #
            wind_magnitude = np.empty(p)

            # Hu and Hv, are not consrtucted in full; just their effect is required
            for p_ind in xrange(p):

                # set indexes
                pp = pos[p_ind, :]
                pp_0 = pos[p_ind, 0]
                pp_1 = pos[p_ind, 1]
                px = np.floor(pp_0)
                py = np.floor(pp_1)
                fx = pp_0 - px
                fy = pp_1 - py
                # py -= 1
                # set indexes
                lu_ind = px + py * nx  # left-up
                ld_ind = px + 1 + py * nx  # left-down
                ru_ind = px + (py + 1) * nx  # right-up
                rd_ind = px + 1 + (py + 1) * nx  # right-down
                cd_ind = px - 1 + py * nx  # center-down
                cu_ind = px + (py - 1) * nx  # center-up

                if fx != 0 and fy != 0:
                    #
                    # read state entries
                    xlu = in_state[lu_ind]
                    xld = in_state[ld_ind]
                    xru = in_state[ru_ind]
                    # (d PSI)/dx --> -v component
                    # (d PSI)/dy --> u component
                    hv = (xlu-xld) / dx
                    hu = (xru-xlu) / dx
                    #
                elif fx == 0 and fy == 0:
                    #
                    # read state entries
                    xcd = in_state[cd_ind]
                    xld = in_state[ld_ind]
                    xcu = in_state[cu_ind]
                    xru = in_state[ru_ind]
                    # (d PSI)/dx --> -v component
                    # (d PSI)/dy --> u component
                    hv = 0.5 * (xcd-xld) / dx
                    hu = 0.5 * (xru-xcu) / dx
                    #
                elif fx == 0:
                    #
                    # read state entries
                    xcd = in_state[cd_ind]
                    xld = in_state[ld_ind]
                    xlu = in_state[lu_ind]
                    xru = in_state[ru_ind]
                    # (d PSI)/dx --> -v component
                    # (d PSI)/dy --> u component
                    hv = 0.5 * (xcd-xld) / dx
                    hu = (xru-xlu) / dx
                    #
                elif fy == 0:
                    #
                    # read state entries
                    xld = in_state[ld_ind]
                    xlu = in_state[lu_ind]
                    xcu = in_state[cu_ind]
                    xru = in_state[ru_ind]
                    # (d PSI)/dx --> -v component
                    # (d PSI)/dy --> u component
                    hv = (xlu-xld) / dx
                    hu = 0.5 * (xru-xcu) / dx
                    #
                else:
                    print("It doesn't make sense to reach this!")
                    raise ValueError
                #
                # Evaluate the magnitude of the wind given it's velocity components
                wind_magnitude[p_ind] = np.sqrt(hu**2 + hv**2)

            # Wrap the observation vector to return
            observation_vector = self.observation_vector()
            observation_vector[:] = wind_magnitude
            #
        #
        else:
            print("Observation operator '%s' is not supported!" % operator_type)
            raise ValueError
        #
        if self.verbose:
            elapsed_time = time.time() - elapsed_time
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            print("H(x) evaluated in [h:m:s] %ds:%d:%6.4f" %(h, m, s))
        #
        return observation_vector
        #

    def construct_observation_operator_Jacobian(self, time_point=None):
        """
        This creates the Jacobian of the observation operator (forward operator).

        Args:
            time_point: the time at which the Jacobaian of the observation operator should be created

        Returns:
            None

        """
        # TODO: This implementation is no longer required, keep as a reference for now
        #
        operator_type = self._observation_operator_type
        if re.match(r'\Alinear\Z', operator_type, re.IGNORECASE):
            try:
                observation_operator = self.observation_operator
            except:
                self.construct_observation_operator(time_point=time_point)
            finally:
                if self.observation_operator is None:
                    self.construct_observation_operator(time_point=time_point)

            observation_operator = self.observation_operator
            #
            self.observation_operator_Jacobian = observation_operator  # it's a linear observation operator.

        #
        elif re.match(r'\Awind(_|-)*magnitude\Z', operator_type, re.IGNORECASE):
            #
            # No need to construct it here because it has t be constructed incrementally given a model state
            self.observation_operator_Jacobian = None
        else:
                raise ValueError("Observation operator '%s' is not supported!" % operator_type)
                #

    def observation_operator_Jacobian_T_prod_vec(self, in_state, observation,time_point=None):
        """
        Multiply the transpose of the Jacobian of the observation operator evaluated at in_state
            by observation
            i.e. evaluate $\mathbf{H}^T \times \text{observation}$, where $\mathbf{H}$ is evaluated at
                in_state . Should generally be time dependent...
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
        # observation_operator_Jacobian = self.evaluate_observation_operator_Jacobian(in_state)
        # raw_observation = observation.get_numpy_array()
        # result_state = self.state_vector(observation_operator_Jacobian.T.dot(raw_observation))
        #
        elapsed_time = time.time()
        #----------------------------------
        #     Optimization Attempt:
        #----------------------------------
        operator_type = self._observation_operator_type
        #
        # 1- Evaluate the Jacobian-transpose at the input state:
        if re.match(r'\Alinear\Z', operator_type, re.IGNORECASE):
            # All we need is to loop over nonzeros of the observaiton operator and apply them to the passed vector
            result_state_np = np.zeros(self.state_size(), dtype=np.float64)  # this will be the ref_vector later

            #
            n = self._state_size
            npv = self._num_prognostic_vars
            mult = 1
            nv = (n - npv) / mult

            try:
                nx = self.model_configs['nx']
                dx = self.model_configs['dx']
                ny = self.model_configs['ny']
                dy = self.model_configs['dy']
            except:
                nx = np.int(np.floor(np.sqrt(self.state_size())))
                dx = 1.0 / (nx - 1)
                ny = self.state_size() / nx
                dy = 1.0 / (ny - 1)

            try:
                pos = self._observations_positions
            except AttributeError:
                pos = self.get_observations_positions(attach_to_model=True)

            p = np.size(pos, 0)

            # H is not constructed in full, It's (it's transpose) effect only is evaluated
            for p_ind in xrange(p):
                # set indexes
                pp = pos[p_ind, :]
                pp_0 = pos[p_ind, 0]
                pp_1 = pos[p_ind, 1]
                px = np.floor(pp_0)
                py = np.floor(pp_1)
                fx = pp_0 - px
                fy = pp_1 - py
                # py -= 1
                # set indexes of observation operator
                lu_ind = int(px + py * nx)  # left-up
                ld_ind = int(px + 1 + py * nx)  # left-down
                ru_ind = int(px + (py + 1) * nx)  # right-up
                rd_ind = int(px + 1 + (py + 1) * nx)  # right-down
                #
                # Extract observaiton to differentiate
                obs = observation[p_ind]
                # Evaluate the derivative effect
                if fx != 0 and fy != 0:  # Bilinear interpolation:
                    result_state_np[lu_ind] += (1-fx) * (1-fy) * obs
                    result_state_np[ld_ind] += fx * (1-fy) * obs
                    result_state_np[ru_ind] += (1-fx) * fy * obs
                    result_state_np[rd_ind] += fx * fy * obs
                    #
                elif fx == 0 and fy == 0:
                    result_state_np[lu_ind] += obs
                    #
                elif fx == 0:
                    result_state_np[lu_ind] += (1-fy) * obs
                    result_state_np[ru_ind] += fy * obs
                    #
                elif fy == 0:
                    result_state_np[lu_ind] += (1-fx) * obs
                    result_state_np[ld_ind] += fx * obs
                    #
                else:
                    raise ValueError("It doesn't make sense to reach this!")

            # now wrap the the result_state in a state_vector
            result_state = self.state_vector()
            result_state[:] = result_state_np
            #
        #
        elif re.match(r'\Awind(-|_)*magnitude\Z', operator_type, re.IGNORECASE):
            #
            if self.verbose:
                elapsed_time = time.time()
            # ---------------------------------
            #      Optimization Attempt
            # ---------------------------------
            # All we need is to loop over nonzeros of the observaiton operator and apply them to the passed vector
            state_size = self._state_size
            result_state_np = np.zeros(state_size, dtype=np.float64)  # this will be the ref_vector later

            npv = self._num_prognostic_vars
            # mult = 1
            nv = (state_size - npv)  # / mult

            nx = np.int(np.floor(np.sqrt(state_size)))
            dx = 1.0 / (nx - 1)
            ny = np.int(np.floor(state_size / nx))
            dy = 1.0 / (ny - 1)
            try:
                pos = self._observations_positions
            except AttributeError:
                pos = self.get_observations_positions(attach_to_model=True)

            p = self.observation_vector_size()
            #
            wind_magnitude = np.empty(p)

            # Hu and Hv, are not consrtucted in full; just their effect is required
            for p_ind in xrange(p):

                # set indexes
                pp = pos[p_ind, :]
                pp_0 = pos[p_ind, 0]
                pp_1 = pos[p_ind, 1]
                px = np.floor(pp_0)
                py = np.floor(pp_1)
                fx = pp_0 - px
                fy = pp_1 - py
                # py -= 1
                # set indexes
                lu_ind = px + py * nx  # left-up
                ld_ind = px + 1 + py * nx  # left-down
                ru_ind = px + (py + 1) * nx  # right-up
                rd_ind = px + 1 + (py + 1) * nx  # right-down
                cd_ind = px - 1 + py * nx  # center-down
                cu_ind = px + (py - 1) * nx  # center-up

                # Extract observaiton to differentiate
                obs = observation[p_ind]
                #
                if fx != 0 and fy != 0:
                    # >>> do/dx += 1/obs * (hu * dhu/dx + hv* dhv/dx)
                    # read state entries
                    xlu = in_state[lu_ind]
                    xld = in_state[ld_ind]
                    xru = in_state[ru_ind]
                    # (d PSI)/dx --> -v component
                    # (d PSI)/dy --> u component
                    hv = (xlu-xld) / dx
                    hu = (xru-xlu) / dx
                    wind_mag = np.sqrt(hu**2 + hv**2)
                    #
                    result_state_np[lu_ind] += (hv-hu) * obs / (dx*wind_mag)
                    result_state_np[ld_ind] -= hv * obs / (dx*wind_mag)
                    result_state_np[ru_ind] += hu * obs / (dx*wind_mag)
                    #
                elif fx == 0 and fy == 0:
                    #
                    # read state entries
                    xcd = in_state[cd_ind]
                    xld = in_state[ld_ind]
                    xcu = in_state[cu_ind]
                    xru = in_state[ru_ind]
                    # (d PSI)/dx --> -v component
                    # (d PSI)/dy --> u component
                    hv = 0.5 * (xcd-xld) / dx
                    hu = 0.5 * (xru-xcu) / dx
                    wind_mag = np.sqrt(hu**2 + hv**2)
                    #
                    result_state_np[cd_ind] += 0.5 * hv * obs / (dx*wind_mag)
                    result_state_np[ld_ind] -= 0.5 * hv * obs / (dx*wind_mag)
                    result_state_np[cu_ind] -= 0.5 * hu * obs / (dx*wind_mag)
                    result_state_np[ru_ind] += 0.5 * hu * obs / (dx*wind_mag)
                    #
                elif fx == 0:
                    #
                    # read state entries
                    xcd = in_state[cd_ind]
                    xld = in_state[ld_ind]
                    xlu = in_state[lu_ind]
                    xru = in_state[ru_ind]
                    # (d PSI)/dx --> -v component
                    # (d PSI)/dy --> u component
                    hv = 0.5 * (xcd-xld) / dx
                    hu = (xru-xlu) / dx
                    wind_mag = np.sqrt(hu**2 + hv**2)
                    #
                    result_state_np[cd_ind] += 0.5 * hv * obs / (dx*wind_mag)
                    result_state_np[ld_ind] -= 0.5 * hv * obs / (dx*wind_mag)
                    result_state_np[lu_ind] -= hu * obs / (dx*wind_mag)
                    result_state_np[ru_ind] += hu * obs / (dx*wind_mag)
                    #
                elif fy == 0:
                    #
                    # read state entries
                    xld = in_state[ld_ind]
                    xlu = in_state[lu_ind]
                    xcu = in_state[cu_ind]
                    xru = in_state[ru_ind]
                    # (d PSI)/dx --> -v component
                    # (d PSI)/dy --> u component
                    hv = (xlu-xld) / dx
                    hu = 0.5 * (xru-xcu) / dx
                    wind_mag = np.sqrt(hu**2 + hv**2)
                    #
                    result_state_np[ld_ind] -= hv * obs / (dx*wind_mag)
                    result_state_np[lu_ind] += hv * obs / (dx*wind_mag)
                    result_state_np[cu_ind] -= 0.5 * hu * obs / (dx*wind_mag)
                    result_state_np[ru_ind] += 0.5 * hu * obs / (dx*wind_mag)
                    #
                else:
                    raise ValueError("It doesn't make sense to reach this!")
            #
            # wrap the result in a state_vector for return
            result_state = self.state_vector()
            result_state[:] = result_state_np
            #
        #
        if self.verbose:
            elapsed_time = time.time() - elapsed_time
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            print("Jac-transpose-vec-prod (H^T y) evaluated in [h:m:s] %ds:%d:%6.4f" %(h, m, s))
        return result_state
        #

    def observation_operator_Jacobian_prod_vec(self, in_state, state, time_point=None):
        """
        Multiply the Jacobian of the observation operator evaluated at in_state
            by a given state
            i.e. evaluate $\mathbf{H} \times \text{state}$, where $\mathbf{H}$ is evaluated at
                in_state . Should generally be time dependent...
            The result is an observation vector of course

        Args:
            in_state: StatVector at which the Jacobian of the observation operator is evaluated
            time_point: the time at which the Jacobaian of the observation operator should be evaluated if it
                is time-dependent. In this implementation time_point is ignored.
            state: StateVector to be multiplied by observation operator Jacobian transposed.

        Returns:
            result_observation: StatVector containing the result of state multiplied by the
                observation operator Jacobian.

        """
        if self.verbose:
            elapsed_time = time.time()

        # Just apply the observation operator on the state vector and return an observation vector.
        operator_type = self._observation_operator_type
        if re.match(r'\Alinear\Z', operator_type, re.IGNORECASE):
            observation_vector_np = np.empty(self.observation_vector_size())  # this will be the ref_vector later
            #
            n = self._state_size
            npv = self._num_prognostic_vars
            mult = 1
            nv = (n - npv) / mult

            try:
                nx = self.model_configs['nx']
                dx = self.model_configs['dx']
                ny = self.model_configs['ny']
                dy = self.model_configs['dy']
            except:
                nx = np.int(np.floor(np.sqrt(self.state_size())))
                dx = 1.0 / (nx - 1)
                ny = self.state_size() / nx
                dy = 1.0 / (ny - 1)

            try:
                pos = self._observations_positions
            except AttributeError:
                pos = self.get_observations_positions(attach_to_model=True)

            p = np.size(pos, 0)

            # H is not constructed in full, it is just used implicitly to update the observation vector
            for p_ind in xrange(p):
                # set indexes
                pp = pos[p_ind, :]
                pp_0 = pos[p_ind, 0]
                pp_1 = pos[p_ind, 1]
                px = np.floor(pp_0)
                py = np.floor(pp_1)
                fx = pp_0 - px
                fy = pp_1 - py
                # py -= 1
                # set indexes
                lu_ind = px + py * nx  # left-up
                ld_ind = px + 1 + py * nx  # left-down
                ru_ind = px + (py + 1) * nx  # right-up
                rd_ind = px + 1 + (py + 1) * nx  # right-down
                #
                # Evaluate observation vector
                if fx != 0 and fy != 0:  # Bilinear interpolation:
                    observation_vector_np[p_ind] = (((1-fx)*(1-fy)) * state[lu_ind]) + \
                                                ((fx*(1-fy)) *state[ld_ind]) + \
                                                (((1-fx)*fy) * state[ru_ind]) + \
                                                ((fx*fy) * state[rd_ind])
                    #
                elif fx == 0 and fy == 0:
                    observation_vector_np[p_ind] = state[lu_ind]
                    #
                elif fx == 0:
                    observation_vector_np[p_ind] = (1-fy) * state[lu_ind] + fy * state[ru_ind]
                    #
                elif fy == 0:
                    observation_vector_np[p_ind] = (1-fx) * state[lu_ind] + fx * state[ld_ind]
                    #
                else:
                    print("It doesn't make sense to reach this!")
                    raise ValueError
            #
            # wrap into observation vector:
            observation_vector = self.observation_vector(observation_vector_np)
            # observation_vector[:] = observation_vector_np

        #
        elif re.match(r'\Awind(-|_)*magnitude\Z', operator_type, re.IGNORECASE):
            raise NotImplementedError("TODO...")
            #

        #
        else:
            print("Observation operator '%s' is not supported!" % operator_type)
            raise ValueError
        #
        if self.verbose:
            elapsed_time = time.time() - elapsed_time
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            print("H(x) evaluated in [h:m:s] %ds:%d:%6.4f" %(h, m, s))
        #
        return observation_vector
        #

    def _construct_covariance_localization_operator(self,
                                                    localization_radius,
                                                    localization_function=None,
                                                    use_true_spacings=False,
                                                    coeff_threshold=1e-5,
                                                    write_to_file=False,
                                                    target_file_name=None,
                                                    attach_to_model=False
                                                    ):
        """
        Construct the localization/decorrelation operator (Decorr) in full. This should be avoided in practice.
        We need it's effect on a square covariance matrix. This should be sparse or evaluated off-line and saved to file.
        Here I construct only the upper triangular part. the diagonal is ones, and the matrix is symmetric of course.

        Args:
            localization_radius: covariance radius of influence (decorrelation radius)
            localization_function: 'Gaspari-Cohn', 'Gauss', 'Cosine', etc.
            use_true_spacings (default False): consider dx, dy, etc if True, otherwise get distances based on grid indices only
            coeff_threshold (default 1e-5): set localization coefficients less than this threshold to zero regardless the distance
            write_to_file (default False): save the decorrelation matrix to a file for later access without reconstruction
            target_file_name: name of the file to save the decorrelation matrix to (if write_to_file is True).
                The file will be written to the model path (i.e. in 'self.__model_path')
            attach_to_model (default False): If True, calculate once and attach to the model object.

        Returns:
            None

        """
        #
        if self.verbose:
            print("Warning: The covariance LOCALIZATION MATRIX SHOULD REALLY NOT be built in full!\n  \
                            Consider localization in observation space instead!")

        #
        if not (write_to_file or attach_to_model):
            print("Wasted construction of the decorrelation matrix.\n \
                   You should either attach it to the model, or write it to a file!")
            raise AssertionError

        elif write_to_file:
            if target_file_name is None:
                # Find a proper file name for the decorrelation array
                mrefin = self.model_configs['MREFIN']
                if mrefin == 5:
                    model_size = 't'  # tiny model
                elif mrefin == 6:
                    model_size = 's'  # small model
                elif mrefin == 7:
                    model_size = 'f'  # full model
                else:
                    print("This shouldn't happen! The config validation shouldn't have allowed this!")
                    raise ValueError
                    #
                target_file_name = "QG%s_Decorr.p" % model_size
                #
            model_path = self.__model_path
            target_file_path = os.path.join(model_path, target_file_name)

        #
        if self.verbose:
            elapsed_time = time.time()

        # WARNING: This should not be called online for big models. Consider calculating offline, and writing to file!
        state_size = self.state_size()
        nx = np.int(np.floor(np.sqrt(state_size)))
        ny = state_size / nx
        dx, dy = self._grid_spacings
        loc_matrix = sparse.lil_matrix((state_size, state_size), dtype=np.float)
        for i in xrange(state_size):  # loop over state_vector entries (one dimension of the covariance matrix
            ref_x_coord = i % nx
            ref_y_coord = (i - ref_x_coord) / nx
            for j in xrange(i, state_size):
                # get x and y coordinates on the grid
                x_coord = j % nx
                y_coord = (j - x_coord) / nx
                if use_true_spacings:
                    distance = np.linalg.norm(np.array([(ref_x_coord-x_coord)*dx, (ref_y_coord-y_coord)*dy]))
                    # distance = np.sqrt((np.abs(ref_x_coord-x_coord)*dy)**2 + (np.abs(ref_y_coord-y_coord)*dx)**2 )
                else:
                    distance = np.linalg.norm(np.array([ref_x_coord-x_coord, ref_y_coord-y_coord]))
                loc_coeff = utility.calculate_localization_coefficients(radius=localization_radius,
                                                                        distances=distance,
                                                                        method=localization_function)
                # print('i=%d, j=%d,\t x_coord=%d, y_coord=%d,\t ref_x_coord=%d, ref_y_coord=%d \t distance = %f, \t loc_coeff=%f'
                #       % (i, j, x_coord, y_coord, ref_x_coord, ref_y_coord, distance, loc_coeff))
                if loc_coeff > coeff_threshold:
                    if i == j:
                        loc_matrix[i, j] = loc_coeff
                    else:
                        loc_matrix[i,j] = loc_coeff
                        loc_matrix[j,i] = loc_coeff
        #
        loc_matrix = loc_matrix.tocsr()

        #
        if self.verbose:
            elapsed_time = time.time() - elapsed_time
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            print("Covariance Localization Matix Created; TIME: [h:m:s] %ds:%d:%6.4f" %(h, m, s))

        if write_to_file:
            pickle.dump(loc_matrix, open(target_file_path, "wb"))
        if attach_to_model:
            # self._covariance_localization_operator = loc_matrix.tocsr()
            self._covariance_localization_operator = loc_matrix
        # print('localization_operator created...', self._covariance_localization_operator)
        #

    def apply_state_covariance_localization(self, covariance_array,
                                            in_place=True,
                                            localization_function=None,
                                            localization_radius=None,
                                            read_loc_from_file=True,
                                            loc_mat_file_name=None
                                            ):
        """
        Apply localization/decorrelation to a given square array.
        NOTE: Assertion should be considered in all functions in general...
        This generally a point-wise multiplication of the decorrelation array/function and the passed covariance_array.

        Args:
            covariance_array: a StateMatrixNumpy or StateMatrixSpScipy containing covariances to be localized.
            in_place (default True): apply localization to covariance_array (in-place) without creating a new
                object. If False, a localized copy of covariance_array is returned.

            localization_function: 'Gaspari-Cohn', 'Gauss', 'Cosine', etc.
            localization_radius: covariance radius of influence (decorrelation radius)
            read_loc_from_file (default True): read the covariance decorrelation operator/matrix from a file in loc_mat_file_name
            loc_mat_file_name: name of the file containing the decorrelation array/operator. The file has to be in
                the model path (i.e. in 'self.__model_path')

        Returns:
            localized_covariances: a decorrelated version of (covariance_array), and of the same type.

        """
        if self.verbose:
            print("Warning: The covariance localization SHOULD NOT REALLY be applied in full space!\n  \
                            Consider localization in observation space instead!")
        #
        if self.verbose:
            elapsed_time = time.time()
        #
        # -------------------------------------------
        #       Optimization Attempt (MAY-18-2016)
        # -------------------------------------------
        # The function 'construct_localization_operator' should not be called at all.
        #
        assert isinstance(covariance_array, (StateMatrix, SparseStateMatrix))
        #
        if read_loc_from_file:
            if loc_mat_file_name is None:
                # Find a proper file name for the decorrelation array
                mrefin = self.model_configs['MREFIN']
                if mrefin == 5:
                    model_size = 't'  # tiny model
                elif mrefin == 6:
                    model_size = 's'  # small model
                elif mrefin == 7:
                    model_size = 'f'  # full model
                else:
                    raise ValueError("This shouldn't happen! The config validation shouldn't have allowed this!")
                target_file_name = "QG%s_Decorr.p" % model_size
            model_path = self.__model_path
            target_file_path = os.path.join(model_path, target_file_name)

            if not os.path.isfile(target_file_path):
                create_decorr_ans = utility.query_yes_no("Couldn't find decorrelation array in the model directory. \nShould I create it? ")
                if create_decorr_ans:
                    save_decorr_ans = utility.query_yes_no("Cool. \nShould I save it once it is created? ")
                    if save_decorr_ans:
                        if save_decorr_ans:
                            self._construct_covariance_localization_operator(localization_radius=localization_radius,
                                                                             localization_function=localization_function,
                                                                             write_to_file=True,
                                                                             attach_to_model=True
                                                                             )
                        else:
                            self._construct_covariance_localization_operator(localization_radius=localization_radius,
                                                                             localization_function=localization_function,
                                                                             write_to_file=False,
                                                                             attach_to_model=True
                                                                             )
                    # Now get the localized version...
                    localized_covariance = self.apply_state_covariance_localization(covariance_array,
                                                                                    in_place=in_place,
                                                                                    localization_function=localization_function,
                                                                                    localization_radius=localization_radius,
                                                                                    read_loc_from_file=False
                                                                                    )
                    return localized_covariance
                    #
                else:
                    print("Target file not found! The file %s has to be put in the model directory '%s' \
                            \n Consider creating the covariance localization array offline and write it to file for later access!" \
                                    % (target_file_path, model_path))
                    raise IOError
            else:
                localized_covariance = pickle.load(open(target_file_path, "rb"))  # the localization matrix is copied here
        else:
            # Construct the localization matrix
            #
            method = self.model_configs['background_errors_covariance_localization_function']
            radius = self.model_configs['background_errors_covariance_localization_radius']
            try:
                localized_covariance = self._covariance_localization_operator.copy()
            except (NameError, ValueError, AttributeError):
                self._construct_covariance_localization_operator(localization_radius=radius,
                                                                 localization_function=method,
                                                                 attach_to_model=False
                                                                 )
                localized_covariance = self._covariance_localization_operator.copy()

        #
        # Start Hadamard Product.
        # avoid using get_numpy_array() to avoid copying the array

        if sparse.issparse(localized_covariance):
            localized_covariance = sparse.csr_matrix(localized_covariance.multiply(covariance_array.get_raw_matrix_ref()))
        else:
            print("The localized covariance matrix is designed to be sparse; it is not though!")
            raise TypeError

        #
        if self.verbose:
            elapsed_time = time.time() - elapsed_time
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            print("Covariance Localization TIME: [h:m:s] %ds:%d:%6.4f" %(h, m, s))

        # wrap the loclized matrix and return:
        if in_place:
            covariance_array = SparseStateMatrix(localized_covariance)
            return covariance_array
        else:
            localized_covariance = SparseStateMatrix(localized_covariance)
            return localized_covariance
            #

    def create_observation_error_model(self, configs=None):
        """
        Create an observation error model.

        Args:
            configs: dict,
                A configurations dictionary for the observation error model
                Supported configurations:
                * errors_distribution (default 'gaussian'): probability distribution of observational
                    errors (e.g. defining the likelihood fuction in the data assimialtion settings).
                * observation_noise_level: used for creating observation error variances from observations
                    equivalent to reference trajectory if the observation_error_covariance_method is 'empirical'.
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
            configs = dict(observation_errors_covariance_method=self.model_configs['observation_errors_covariance_method'],
                           observation_errors_distribution=self.model_configs['observation_errors_distribution'],
                           observation_errors_variances=self.model_configs['observation_errors_variances']
                           )
        self.observation_error_model = QGObservationErrorModel(model=self, configs=configs)
        self.observation_error_model.construct_error_covariances(construct_inverse=True, construct_sqrtm=True)
        #

    def create_background_error_model(self, configs=None):
        """
        Create a background error model.

        Args:
            configs: dict,
                A configurations dictionary for the background error model
                Supported configurations:
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
                localize_covariances = configs['create_background_errors_correlations'] and \
                    configs['background_errors_covariance_localization_function'] is not None and \
                        configs['background_errors_covariance_localization_radius'] is not None
                configs.update({'localize_errors_covariances':localize_covariances})
            except KeyError:
                configs.update({'localize_errors_covariances':False})
        else:
            pass

        self.background_error_model = QGBackgroundErrorModel(self, configs=configs)
        self.background_error_model.construct_error_covariances(construct_inverse=True,
                                                                construct_sqrtm=True,
                                                                sqrtm_method='cholesky'
                                                                )
                                                                #
        #
        if False:
            # zero the indexes corresponding to boundaries:
            zero_indexes = self.model_configs['boundary_indexes']
            self.background_error_model.B[zero_indexes, zero_indexes] = 0.0
            self.background_error_model.sqrtB[zero_indexes, zero_indexes] = 0.0
            self.background_error_model.invB[zero_indexes, zero_indexes] = 0.0  # shall we make it huge!
        #

    def create_model_error_model(self, configs=None):
        """
        Create a model error model.

        Args:
            configs: dict,
                A configurations dictionary for the model error model
                Supported configurations:
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
            configs = dict(model_errors_distribution=self.model_configs['model_errors_distribution'],
                           model_errors_variances=self.model_configs['model_errors_variances'],
                           create_model_errors_correlations=self.model_configs['create_model_errors_correlations'],
                           model_errors_covariance_localization_function=self.model_configs['model_errors_covariance_localization_function'],
                           model_errors_covariance_localization_radius=self.model_configs['model_errors_covariance_localization_radius'],
                           model_errors_covariance_method=self.model_configs['model_errors_covariance_method']
                           )
        if not configs.has_key('localize_errors_covariances'):
            try:
                localize_covariances = configs['create_model_errors_correlations'] and \
                    configs['model_errors_covariance_localization_function'] is not None and \
                        configs['model_errors_covariance_localization_radius'] is not None
                configs.update({'localize_errors_covariances':localize_covariances})
            except KeyError:
                configs.update({'localize_errors_covariances':False})
        else:
            pass

        self.model_error_model = QGModelErrorModel(self, configs=configs)
        self.model_error_model.construct_error_covariances(construct_inverse=True, construct_sqrtm=True)
        zero_indexes = self.model_configs['boundary_indexes']
        self.background_error_model.Q[zero_indexes, zero_indexes] = 0.0
        self.background_error_model.sqrtQ[zero_indexes, zero_indexes] = 0.0
        self.background_error_model.invQ[zero_indexes, zero_indexes] = 0.0  # shall we make it huge!
        #

    def _get_QG_tiny_repo(self):
        """
        Read the repository containing ensemble members of the special QG-t (of size 33 x 33) model.
        This ensemble is downloaded automatically from enkf website if not found locally.
        """
        state_size = self.state_size()
        if state_size != 33*33:
            print("Attempting to retrieve the tiny repository for a model of different size!!")
            raise ValueError

        # Tiny QG model (QGt)
        # Attempt to read the true state from the minimal repo
        __url_enkf_samples_path = QG1p5.__url_enkf_samples_path
        #
        saved_mats_path = self._saved_mats_path
        if not os.path.isdir(saved_mats_path):
            # Create the samples directory and flag for download based on user input!
            mkdir_cmd = 'mkdir %s -p' % saved_mats_path
            os.system(mkdir_cmd)
            fresh_samples_dir = True
        else:
            fresh_samples_dir = False

        state_repo_mat_filename = 'ens.mat'
        state_repo_mat_file_path = os.path.join(saved_mats_path, state_repo_mat_filename)
        #
        if fresh_samples_dir or not os.path.isfile(state_repo_mat_file_path):
            # download the mat file. Later give the option to run the model and create the samples locally!
            self.download_QG_samples(state_repo_mat_filename, state_repo_mat_file_path , __url_enkf_samples_path)

        # load the pool of ensemble states from the mat file:
        # print('reading ensemble from: %s' % state_repo_mat_file_path)
        repo_contents = sio.loadmat(state_repo_mat_file_path)
        ensemble_repo = repo_contents['E']
        if np.size(ensemble_repo, 0) != state_size:
            raise ValueError("Dimension mismatch!")

        x_true = repo_contents['x_true']  # reference initial condition
        return x_true, ensemble_repo
        #

    def _get_ensemble_repo(self):
        """
        Read the repository containing ensemble members of the QG model.
        This ensemble is downloaded automatically from enkf website if not found locally.

        Args:
            None

        Returns:
            ensemble_repo: a mat file containing ensemble of model states obtained from the original QG-1.5 code.

        """
        __url_enkf_path = QG1p5.__url_enkf_path
        __url_enkf_samples_path = QG1p5.__url_enkf_samples_path
        #
        saved_mats_path = self._saved_mats_path
        if not os.path.isdir(saved_mats_path):
            # Create the samples directory and flag for download based on user input!
            mkdir_cmd = 'mkdir %s -p' % saved_mats_path
            os.system(mkdir_cmd)
            fresh_samples_dir = True
        else:
            fresh_samples_dir = False

        state_size = self.state_size()
        if state_size == 33*33:
            # Tiny QG model (QGt)
            state_repo_mat_filename = 'QGt_samples.mat'
            state_repo_mat_file_path = os.path.join(saved_mats_path, state_repo_mat_filename)
            #
            if fresh_samples_dir or not os.path.isfile(state_repo_mat_file_path):
                # download the mat file. Later give the option to run the model and create the samples locally!
                self.download_QG_samples(state_repo_mat_filename, state_repo_mat_file_path , __url_enkf_samples_path)

            # load the pool of ensemble states from the mat file:
            # print('reading ensemble from: %s' % state_repo_mat_file_path)
            repo_contents = sio.loadmat(state_repo_mat_file_path)
            ensemble_repo = repo_contents['S']
            if np.size(ensemble_repo, 0) != state_size:
                raise ValueError("Dimension mismatch!")

        elif state_size == 65*65:
            # Small QG model (QGs)
            state_repo_mat_filename = 'QGs_samples.mat'
            state_repo_mat_file_path = os.path.join(self.__def_saved_mats_path, state_repo_mat_filename)
            #
            if fresh_samples_dir or not os.path.isfile(state_repo_mat_file_path):
                # download the mat file. Later give the option to run the model and create the samples locally!
                self.download_QG_samples(state_repo_mat_filename, state_repo_mat_file_path , __url_enkf_samples_path)

            # load the pool of ensemble states from the mat file:
            repo_contents = sio.loadmat(state_repo_mat_file_path)
            ensemble_repo = repo_contents['S']
            if np.size(ensemble_repo, 0) != state_size:
                raise ValueError("Dimension mismatch!")

        elif state_size == 129*129:
            # Full QG model (QG)
            state_repo_mat_filename = 'QG_samples-11.mat'
            state_repo_mat_file_path = os.path.join(self.__def_saved_mats_path, state_repo_mat_filename)
            #
            if fresh_samples_dir or not os.path.isfile(state_repo_mat_file_path):
                # download the mat file. Later give the option to run the model and create the samples locally!
                self.download_QG_samples(state_repo_mat_filename, state_repo_mat_file_path , __url_enkf_samples_path)

            # load the pool of ensemble states from the mat file:
            repo_contents = sio.loadmat(state_repo_mat_file_path)
            ensemble_repo = repo_contents['S']
            if np.size(ensemble_repo, 0) != state_size:
                raise ValueError("Dimension mismatch!")
        else:
            # New settings!
            raise ValueError("Until I make sure the system is fully functional with the default settings, I won't add "
                             "more settings!")
        return ensemble_repo
        #

    def create_initial_condition(self, random_state=False, selective_index=100):
        """
        create initial condition state for QG model. This initial condition is loaded from a mat file copied from
        enkf-matlab Fortan code.

        Args:
            random_state: if true, the initial condition will be selected at random from the pool of states in the save
                repository of states, otherwise the entry in selective_index
            selective_index: the index (starting indexing at 1 rather than 0) of the state in the ensemble repository of
                the original code. This takes place if random_state is set to False.

        Returns:
            initial_condition: StateVector containing a valid initial state for QG-1.5 model

        """
        state_size = self.state_size()
        if state_size == 33*33:
            # Tiny QG model (QGt)
            try:
                initial_state, _ = self._get_QG_tiny_repo()
                initial_condition = self.state_vector()
                initial_condition[:] = np.squeeze(initial_state)
                return initial_condition
            except:
                pass
        else:
            pass

        ensemble_repo = self._get_ensemble_repo()

        if random_state:
            initial_state = ensemble_repo[:, 1+np.random.randint(np.size(ensemble_repo, 1)-1)]
        else:
            initial_state = ensemble_repo[:, selective_index]

        initial_condition = self.state_vector()
        initial_condition[:] = np.squeeze(initial_state)
        return initial_condition
        #

    def create_initial_ensemble(self, ensemble_size, ensemble_mean=None, random_ensemble=True, ensemble_from_repo=False):
        """
        create initial ensemble for QG model. This initial ensemble is loaded from a mat file copied from
        enkf-matlab code provided by Pavel Sakov.

        Args:
            ensemble_size: sample size
            ensemble_mean: StateVector; if provided it is used as the mean of the generated ensemble to which background
                noise vectors are added to create the ensemble.
                If it is None, ensemble_from_repo has to be True.
                If it is provided, ensemble_from_repo is discarded
            random_ensemble: if true, the initial ensemble will be selected at random from the pool of states in the saved
                repository of states, otherwise the first entries will be chosen.
            ensemble_from_repo: If True, the ensemble will be read from the the pool of states in the saved repository of
                states.
                - If True, and random_ensemble if True, ensemble is read randomly from repository
                - If True, and random_ensemble if False, ensemble is read from repository from the first ensemble_size of
                    entries.

        Returns:
            ensemble: list of StateVector objects serving as an initial background ensemble for Lorenze96 model

        """
        state_size = self.state_size()
        if state_size==33*33 and ensemble_size==20:
            # Tiny QG model (QGt) with 20 ensemble members...
            # print('reading tiny ensemble ....................')
            try:
                _, initial_ensemble = self._get_QG_tiny_repo()
                ensemble = []
                for ens_ind in xrange(ensemble_size):
                    member = self.state_vector()
                    member[:] = np.squeeze(initial_ensemble[:, ens_ind])
                    ensemble.append(member)
                return ensemble
            except:
                raise
        else:
            pass

        if ensemble_mean is None and not ensemble_from_repo:
            ensemble_mean = self._reference_initial_condition.copy()
            ensemble_mean = ensemble_mean.add(self.background_error_model.generate_noise_vec())

        ensemble = []
        #
        if ensemble_mean is None:
            # Read ensemble from ensemble repository
            ensemble_repo = self._get_ensemble_repo()
            repo_size = np.size(ensemble_repo, 1)
            for ens_ind in xrange(ensemble_size):
                if random_ensemble:
                    member_ind = np.random.randint(repo_size)
                else:
                    member_ind = ens_ind
                state = np.squeeze(ensemble_repo[:, member_ind])
                member = self.state_vector()
                member[:] = state
                ensemble.append(member)
        else:
            # Create an ensemble by purturbing the ensemble mean with errors from the background error model
            assert isinstance(ensemble_mean, StateVector), "the passed ensemble mean has to be an instance of StateVector!"
            forecast_state = ensemble_mean
            #
            try:
                self.background_error_model
            except(ValueError, AttributeError, NameError):
                print("Initial ensemble cannot be created before creating background error model!")
                raise ValueError
            finally:
                if self.background_error_model is None:
                    print("Initial ensemble cannot be created before creating background error model!")
                    raise ValueError

            # zeros = np.where(forecast_state == 0)
            forecast_state = forecast_state.add(self.background_error_model.generate_noise_vec())
            # # Reset boundary conditions; this can be removed for statistical consistency
            # forecast_state[forecast_state == 0] = 0.0
            for ens_ind in xrange(ensemble_size):
                state = forecast_state.copy()
                state = state.add(self.background_error_model.generate_noise_vec())
                ensemble.append(state)
                #
        return ensemble
        #

    def ensemble_covariance_matrix(self, ensemble, localize=False):
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
            perturbations[:, ens_ind] = member[:].copy()
        covariance_mat = (perturbations.dot(perturbations.T)) / (ensemble_size-1)
        covariance_mat[np.diag_indices_from(covariance_mat)] += 0.01  # avoid singularity
        # wrap it
        covariance_mat = self.state_matrix(covariance_mat)
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

        # retrieve model-grid settings
        dx = self.model_configs['dx']
        dy = self.model_configs['dy']
        nx = self.model_configs['nx']
        ny = self.model_configs['ny']

        # initilize indexes list
        indexes_list = []

        #
        if source_index.lower().startswith('state'):
            #
            # get model-grid coordinates
            model_grid_coords = self.get_model_grid()

            # num_model_grid_points = np.size(model_grid_coords, 0)
            num_model_grid_points = self._total_grid_size  # should be same as above!

            if index >= num_model_grid_points:
                print("index %d is out of range! the model grid size is %d" %(index, num_model_grid_points))
                raise IndexError

            # source index to look around is in the state space
            ref_coord = model_grid_coords[index, :]

            if target_index == source_index:
                #
                #
                for ind in xrange(num_model_grid_points):
                    coord = model_grid_coords[ind, :]
                    distance = np.linalg.norm(coord - ref_coord)
                    if distance <= radius:
                        indexes_list.append(ind)
                #
            elif target_index.lower().startswith('obs'):
                #
                # get observational-grid coordinates
                obs_grid_coords = self.get_observational_grid()
                num_obs_grid_points = np.size(obs_grid_coords, 0)

                for ind in xrange(num_obs_grid_points):
                    coord = obs_grid_coords[ind, :]
                    distance = np.linalg.norm(coord - ref_coord)
                    if distance <= radius:
                        indexes_list.append(ind)
                #
            else:
                print("Unknown target_index: '%s' !" % repr(target_index))
                raise ValueError

            #
        elif source_index.lower().startswith('obs'):
            # source index to look around is in the observation space

            # get observational-grid coordinates
            obs_grid_coords = self.get_observational_grid()
            ref_coord = obs_grid_coords[index, :]

            # get observational-grid coordinates
            num_obs_grid_points = np.size(obs_grid_coords, 0)

            if index >= num_obs_grid_points:
                print("index %d is out of range! the observational-grid is of size: %d" %(index, num_obs_grid_points))
                raise IndexError

            if target_index == source_index:
                #
                for ind in xrange(num_obs_grid_points):
                    coord = obs_grid_coords[ind, :]
                    distance = np.linalg.norm(coord - ref_coord)
                    if distance <= radius:
                        indexes_list.append(ind)

                 #
            elif target_index.lower().startswith('state'):
                # get model-grid coordinates
                model_grid_coords = self.get_model_grid()

                # num_model_grid_points = np.size(model_grid_coords, 0)
                num_model_grid_points = self._total_grid_size  # should be same as above!
                #
                for ind in xrange(num_model_grid_points):
                    coord = model_grid_coords[ind, :]
                    distance = np.linalg.norm(coord - ref_coord)
                    if distance <= radius:
                        indexes_list.append(ind)
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
        For mat-file format, I am following the same file format and contents and Pavel Sakov for ease of portability later.
        In this case each column is a state vector.

        Args:
            directory: location where observation vector will be saved
            file_name: name of the target file
            append: If set to True and the file_name exists, the observation will be appended, otherwise files will be overwritten.
            file_format: the type/format of the target file. This should be controlled by the model.

        Returns:
            None

        """
        # TODO: Update, and optimize...
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
                    except (KeyError, NameError, AttributeError):
                        raise KeyError("The file contents do not match the default format!")
                    #
                    if not (np.size(S, 0)==n and np.size(S, 1)==n_sample):
                        raise ValueError("Dimension mismatch. S shape is %s" % str(np.shape(S)))
                    else:
                        # either resize S, or create a larger version.
                        n_sample += 1
                        tmp_S = np.empty((n, n_sample))
                        tmp_S[:, :-1] = S
                        S = tmp_S
                        # Now add the new state to the last column
                        S[:, -1] = state.get_numpy_array().squeeze()
                        # Write the new values. This can be optimized by opening the file with r+ permission (will do later)
                        save_dict = dict(n=n, n_sample=n_sample, S=S)
                        sio.savemat(file_path, save_dict, oned_as='column')

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
                          file_format='mat-file', save_observation_operator=False, save_observation_grid=True):
        """
        Save an observation vector to a file. Optionally save the observation operator and the observational grid to the file as well.
        The default format is a MATLAB mat-file. the names of the variables are as follows:
            - Obs: observation(s) vector(s).
            - m:   observation vector size.
            - n_obs: number of observation vectors. Each column of Obs is an observation vector.
            - H: observation operator(s). a three dimensional array (m x n x n_obs)
            - Obs_grid: observational grid. ( m x 2 x n_obs); the observational grid at each time is a 2D array with indices.

        Args:
            directory: location where observation vector will be saved
            file_name: name of the target file
            append: If set to True and the file_name exists, the observation will be appended, otherwise files will be overwritten.
            file_format: the type/format of the target file. This should be controlled by the model.
            save_observation_operator: whether to write the observation operator along with the observation vector or not.
            save_observation_grid (default True): flag for saving the observational grid.

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
                        if save_observation_grid:
                            Obs_grid = old_dict['Obs_grid']
                    except (KeyError, NameError, AttributeError):
                        raise KeyError("The file contents do not match the default format or some variables are missing!")
                    #
                    if not (np.size(Obs, 0)==m and np.size(Obs,1)==n_obs):
                        raise ValueError("Dimension mismatch. S shape is %s" % str(np.shape(S)))
                    else:
                        # either resize S, or create a larger version.
                        n_obs += 1
                        tmp_Obs = np.empty((m, n_obs))
                        tmp_Obs[:, :-1] = Obs; Obs = tmp_Obs
                        # Now add the new observation to the last column
                        Obs[:, -1] = observation.get_numpy_array().squeeze()
                        #
                        # Now add the new observational grid if requested
                        if save_observation_grid:
                            # get observations positions
                            try:
                                observations_positions = self._observations_positions
                            except AttributeError:
                                observations_positions = self.get_observations_positions(attach_to_model=True)
                            # append the observational grid to the existing file
                            tmp_Obs_grid = np.empty((m, 2, n_obs))
                            tmp_Obs_grid[:, :, :-1] = Obs_grid; Obs_grid = tmp_Obs_grid
                            # Now add the new observational grid to the last index
                            Obs_grid[:, :, -1] = observations_positions
                        #
                        # Now add the new observation operator if requested. It will be converted to a dense version of course.
                        if save_observation_operator:
                            operator_type = self._observation_operator_type
                            if re.match(r'\Alinear\Z', operator_type, re.IGNORECASE):
                                # get the observation operator and convert it to numpy.ndarray
                                try:
                                    observation_operator = self.observation_operator
                                except (ValueError, NameError, AttributeError):
                                    self.construct_observation_operator()
                                    observation_operator = self.observation_operator
                                observation_operator = observation_operator.get_numpy_array()  # convert to numpy (full) array to be saved.
                                # append the observational grid to the existing file
                                n = self.state_size()
                                tmp_H = np.empty((m, n, n_obs))
                                tmp_H[:, :, :-1] = H[:, :]; H = tmp_H
                                # Now add the new observation operator to the last index
                                H[:, :, -1] = observation_operator
                            #
                            elif re.match(r'\Awind(_|-)*magnitude\Z', operator_type, re.IGNORECASE):
                                pass
                            else:
                                raise ValueError("Unrecognized Observation Operator")
                        # Write the new values. This can be optimized by opening the file with r+ permission (will do later)
                        if save_observation_grid and save_observation_operator:
                            save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H, Obs_grid=Obs_grid)
                        elif save_observation_grid:
                            save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, Obs_grid=Obs_grid)
                        elif save_observation_operator:
                            save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H)
                        else:
                            save_dict = dict(m=m, n_obs=n_obs, Obs=Obs)
                        sio.savemat(file_path, save_dict, oned_as='column')
                else:
                    # no file with same name in the same directory exists, it is save to write the observation a new.
                    m = self.observation_vector_size()
                    n_obs = 1
                    Obs = np.empty((m, 1))
                    Obs[:, 0] = observation.get_numpy_array().squeeze()
                    if save_observation_grid:
                        # get observations positions
                        try:
                            Obs_grid = self._observations_positions
                        except AttributeError:
                            Obs_grid = self.get_observations_positions(attach_to_model=True)

                    if save_observation_operator:
                        # get the observation operator and convert it to numpy.ndarray
                        try:
                            observation_operator = self.observation_operator
                        except (ValueError, NameError, AttributeError):
                            self.construct_observation_operator()
                            observation_operator = self.observation_operator
                        H = observation_operator.get_numpy_array()  # convert to numpy (full) array to be saved.
                    #
                    # Write the new values. This can be optimized by opening the file with r+ permission (will do later)
                    if save_observation_grid and save_observation_operator:
                        save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H, Obs_grid=Obs_grid)
                    elif save_observation_grid:
                        save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, Obs_grid=Obs_grid)
                    elif save_observation_operator:
                        save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H)
                    else:
                        save_dict = dict(m=m, n_obs=n_obs, Obs=Obs)
                    sio.savemat(file_path, save_dict, oned_as='column')
            else:
                # Go ahead and write the file, overwrite it if exists
                m = self.observation_vector_size()
                n_obs = 1
                # Obs = np.empty((m, 1))
                Obs = observation.get_numpy_array().squeeze()
                if save_observation_grid:
                    # get observations positions
                    try:
                        Obs_grid = self._observations_positions
                    except AttributeError:
                        Obs_grid = self.get_observations_positions(attach_to_model=True)
                if save_observation_operator:
                    operator_type = self._observation_operator_type
                    if re.match(r'\Alinear\Z', operator_type, re.IGNORECASE):
                        # get the observation operator and convert it to numpy.ndarray
                        try:
                            observation_operator = self.observation_operator
                        except (ValueError, NameError, AttributeError):
                            self.construct_observation_operator()
                            observation_operator = self.observation_operator
                        observation_operator = observation_operator.get_numpy_array()  # convert to numpy (full) array to be saved.
                        # append the observational grid to the existing file
                        n = self.state_size()
                        tmp_H = np.empty((m, n, n_obs))
                        tmp_H[:, :, :-1] = H[:, :]; H = tmp_H
                        # Now add the new observation operator to the last index
                        H[:, :, -1] = observation_operator
                    #
                    elif re.match(r'\Awind(-|_)*magnitude\Z', operator_type, re.IGNORECASE):
                        pass
                    else:
                        raise ValueError("Unrecognized Observation Operator")
                # Write the new values. This can be optimized by opening the file with r+ permission (will do later)
                if save_observation_grid and save_observation_operator:
                    save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H, Obs_grid=Obs_grid)
                elif save_observation_grid:
                    save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, Obs_grid=Obs_grid)
                elif save_observation_operator:
                    save_dict = dict(m=m, n_obs=n_obs, Obs=Obs, H=H)
                else:
                    save_dict = dict(m=m, n_obs=n_obs, Obs=Obs)
                sio.savemat(file_path, save_dict, oned_as='column')
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

    #
    def get_default_QG1p5_parameters(self, write_to_file=True, file_name='QG.prm', verbose=1):
        """
        Return a dictionary containing model parameters needed at any stage of the model run...
        The default parameter settings are copied from Sakov's code for the 3 model settings he recommends.
        If you wish to change the settings, you have to be certain of what you are doing!
        From Fortan parameters module:
            real(8), public :: dt            ! time step
            real(8), public :: rkb           ! bottom friction
            real(8), public :: rkh           ! horizontal friction
            real(8), public :: rkh2          ! biharmonic friction
            real(8), public :: F             ! Froud number, (L / Rd)^2
            real(8), public :: r             ! factor in front of nonlinear advection term

        More:
            tend: the final time to propagate the model to. If tend < t--> tend = t + tend. where t is the initial
                                                            time of model propagation step passed e.g. into qgstep.f90.
            outfname: the file name to write model outputs to
            dtout:  the size of step after which output from the model is saved to "outfname"
        TODO: update qgstep_wrap.f90 such that tstop
        """
        assert isinstance(verbose, int)

        state_size = self._state_size
        if state_size == 33*33:
            # Tiny QG model (QGt)
            model_parameters = dict(tend=0, dtout=5, dt=5.0, RKB=0.0e-6, RKH=2.0e-7, RKH2=4.0e-11, F=1600,
                                    r=4e-5, verbose=verbose, scheme='rk4', rstart=0, restartfname='',
                                    outfname='qgt_samples-r=4e-5,dt=5,RKH=2e-7,RKH2=4e-11.nc')
        elif state_size == 65*65:
            # Small QG model (QGs)
            model_parameters = dict(tend=0, dtout=2.5, dt=2.5, RKB=0, RKH=0, RKH2=2.0e-11, F=1600, R=1.0e-5,
                                    verbose=verbose, scheme='rk4', rstart=0, restartfname='', outfname='')
        elif state_size == 129*129:
            # Full QG model (QG)
            model_parameters = dict(tend=0, dtout=1.25, dt=1.25, RKB=0, RKH=0, RKH2=2.0e-12, F=1600, R=1.0e-5,
                                    verbose=verbose, scheme='rk4', rstart=0, restartfname='', outfname='')
        else:
            # New settings!
            raise ValueError("Until I make sure the system is fully functional with the default settings, I won't add "
                             "more settings!")

        if write_to_file:
            parameters_file_path = self.write_parameters(model_parameters, file_name)

        if write_to_file:
            return model_parameters, parameters_file_path
        else:
            return model_parameters
            #

    def write_parameters(self, parameters_dict, file_name='QG.prm', file_directory=None, file_format='F'):
        """
        write QG1p5 parameters contained in the passed dictionary to a file, and return full path to the file
        """
        if file_directory is None:
            file_directory = os.path.join(self.__model_path, 'prm/')
        if not os.path.isdir(file_directory):
            os.mkdir(file_directory)
        file_path = os.path.join(file_directory, file_name)

        # If the model requires (hopefully not) ordered parameters, consider using ordered dictionary instead!
        if file_format.lower() in ['f', 'fortran']:
            parameters_str = '&parameters\n'
            for key in parameters_dict:
                if isinstance(parameters_dict[key], str):
                    parameters_str += u"  {0:s} = '{1:s}'\n".format(key, str(parameters_dict[key]))
                else:
                    parameters_str += u"  {0:s} = {1:s}\n".format(key, str(parameters_dict[key]))
            parameters_str += "/\n"

            with open(file_path, 'w') as file_handler:
                # print('file path: %s' %file_path)
                # print('written: %s\n' %parameters_str)
                file_handler.write(parameters_str)
        else:
            raise ValueError("Parameter file format [%s] unrecognized!" % file_format)
        #
        return file_path
        #

    @staticmethod
    def download_QG_samples(state_repo_mat_filename, local_file_path , url_enkf_samples_path=None):
        """
        Download the QG samples necessary for the QG runs from enkf-MATLAB website.
        The full path of the target file (to be saved on disk) is in local_file_path.
        """
        if url_enkf_samples_path is None:
            url_enkf_samples_path = "http://enkf.nersc.no/Code/EnKF-Matlab/QG_samples"  # this should be updated if the url changes!

        url_model_samples_path = url_enkf_samples_path + state_repo_mat_filename
        print("... Downloading QG samples from %s:" % url_enkf_samples_path)

        # download the mat file. Later give the option to run the model and create the samples locally!
        file_downloader = utility.URLDownload()
        file_downloader.download(url_model_samples_path, local_file_path)
        # print("Done!\n")


#
#
if __name__ == '__main__':
    """
    This is a test procedure
    """
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation

    model = QG1p5()
    checkpoints = np.arange(0, 1250.0001, 12.5)

    # -------------------------------------------------------
    # test time integration scheme, and plot the trajectory
    # -------------------------------------------------------
    state_size = model.state_size()
    nx = model.model_configs['nx']
    ny = model.model_configs['ny']
    reference_trajectory = model.integrate_state(initial_state=model._reference_initial_condition.copy(),
                                                 checkpoints=checkpoints)
    ref_traj_reshaped = np.empty((len(reference_trajectory), nx, ny))
    for time_ind in xrange(len(reference_trajectory)):
        ref_traj_reshaped[time_ind, :, :] = np.reshape(reference_trajectory[time_ind].get_numpy_array(), (nx, nx), order='F')

    fig = plt.figure(facecolor='white')
    fig.suptitle("Reference Trajectory")
    ref_ims = []


    for i in xrange(len(reference_trajectory)):
        imgplot = plt.imshow(np.squeeze(ref_traj_reshaped[i, :, :]), animated=True)
        if i == 0:
            plt.colorbar()
        else:
            plt.autoscale()
        ref_ims.append([imgplot])

    ref_ani = animation.ArtistAnimation(fig, ref_ims, interval=50, blit=True, repeat_delay=1000)

    plt.show()
