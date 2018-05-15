
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


import cart_swe
from state_matrix_cart_swe import StateMatrixCartSWE as StateMatrix
from state_vector_cart_swe import StateVectorCartSWE as StateVector

from models_base import ModelsBase


class CartesianSWE(ModelsBase):
    """
    Cartesian shallow-water equations model. Model is written in C, with a Swig wrapper.
    """

    # Default model, assimilation, and input/output configurations.
    _def_model_configs = {'model_name': 'cartesian_swe',
                          'model_grid_type': 'cartesian',
                          'model_errors_covariance_method': 'diagonal',
                          'model_errors_distribution': 'gaussian',
                          'model_noise_level': 0.00,
                          'model_errors_steps_per_model_steps': 0
                          }
    _def_assimilation_configs = {'ignore_observations_term': False,
                                 'observation_operator_type': 'linear',
                                 'observed_variables_jump': 1,
                                 'observation_spacing_type': 'fixed',
                                 'observation_steps_per_assimilation_steps': 1,
                                 'observation_errors_distribution': 'gaussian',
                                 'observation_noise_level': 0.05,
                                 'apply_localization': False,
                                 'background_errors_covariance_method': 'diagonal',
                                 'background_noise_level': 0.08,
                                 'background_noise_type': 'gaussian'
                                 }
    _def_inout_configs = {'screen_output_iter': 1,
                          'file_output_iter': 1,
                          'file_output_moment_only': True,
                          'file_output_moment_name': 'average'
                          }

    def __init__(self, mesh_size=256, thread_count=4):
        """
        Model object constructor.
        """
        self._reference_initial_condition = StateVector()
        err = cart_swe.model_init(mesh_size, thread_count, self._reference_initial_condition._raw_vector)
        self._model_initialized = True
        # Time integration setup.
        self._default_step_size = 0.001

    def __del__(self):
        """
        Model object destructor.
        """
        err = cart_swe.model_del()

    def create_model_state_vector(self):
        """
        Create an empty state vector object and return its reference
        """
        return StateVector()

    def create_model_state_matrix(self):
        """
        Create an empty state matrix (list of model vectors) object and return its reference
        The result should be a list of model states of length equal to "num_states"
        """
        return StateMatrix()

    def construct_model_grids(self):
        """
        Either construct the spatial grids of the model (if the model is fully implemented in Python),
        or obtain model grids from the given implementation (probably by a simple run and read output files or so.)
        """
        # TODO: Think more about the latter case after interfacing with several models..
        raise NotImplementedError

    def integrate_state(self, state, time_span):
        """
        March the model state forward in time (backward for negative step or decreasing list).
        This should call
        time_span should be a float scalar (taken as one step) or a list including beginning,end or a full timespan.

        Input:
            state
        Output:
            :
        """
        raise NotImplementedError

    def integrate_state_perturbation(self, state_perturbation, time_span):
        """
        March the model state perturbation forward in time (backward for negative step or decreasing list)
        This uses the TL of the model.
        state_perturbation should be np.ndarray vector (1D array) containing difference between two model states.
        time_span should be a float scalar (taken as one step) or a list including beginning,end or a full timespan.
        """
        raise NotImplementedError

    def step_forward_function(self, time_point, in_state, out_state=StateVector()):
        """
        In the simplest case, this implements the right-hand side of the model and evaluates it at the given model state.
        """
        err = cart_swe.model_rhs(time_point, in_state._raw_vector, out_state._raw_vector)
        return out_state
    
    def step_forward_function_Jacobian(self, time_point, in_state):
        """
        This implements the Jacobian of the right-hand side of the model and evaluate it at the given model state.
        """
        return StateMatrix(time_point, in_state)

    def construct_observation_operator(self):
        """
        Construct the observation operator (H) in full. This should be avoided in practice.
        We need it's (or it's TLM) effect on a state vector always.
        """
        raise NotImplementedError

    def evaluate_theoretical_observation(self, in_state):
        """
        Evaluate the theoretical observation corresponding to a model state vector,
        i.e. evaluate H(state), where H is the observation operator.
        """
        raise NotImplementedError
    # ------------------------------------------------------------
    #

    def construct_observation_operator_Jacobian(self):
        """
        This creates the Jacobian of the observation operator (forward operator).
        """
        raise NotImplementedError
    # ------------------------------------------------------------
    #

    def evaluate_observation_operator_Jacobian(self, in_state):
        """
        Evaluate the theoretical observation corresponding to a model state vector,
        i.e. evaluate H(state), where H is the observation operator.
        """
        raise NotImplementedError
    # ------------------------------------------------------------
    #

    def construct_localization_operator(self):
        """
        Construct the localization/decorrelation operator (Decorr) in full. This should be avoided in practice.
        We need it's effect on a square covariance matrix. This should be sparse or evaluated off-line and saved to file..
        """
        raise NotImplementedError
    # ------------------------------------------------------------
    #

    def apply_localization_operator(self, covariance_array):
        """
        Apply localization/decorrelation to a given square array.
        NOTE: Assertion should be considered in all functions in general...
        This generally a point-wise multiplication of the decorrelation array/function and the passed covariance_arrya
        """
        raise NotImplementedError
    # ------------------------------------------------------------
    #

    def construct_model_error_covariances(self):
        """
        Construct the model errors covariance matrix. Shouldn't be called for large models.
        We usually need the effect of it's inverse on a vector.
        """
        raise NotImplementedError
    # ------------------------------------------------------------
    #

    def model_error_covariances_inv_prod_vec(self, in_state):
        """
        Evaluate the effect of inverse error covariance matrix on a model state vector.
        """
        raise NotImplementedError
    # ------------------------------------------------------------
    #

    def generate_model_errors(self):
        """
        return a vector of model errors.
        """
        raise NotImplementedError
    # ------------------------------------------------------------
    #
    #
