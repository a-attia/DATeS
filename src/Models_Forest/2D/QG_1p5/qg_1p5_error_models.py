
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
    A very basic (Numpy-based, and Scipy.Sparse-based) implementation of the following error models:
        + Background errors,
        + Observation errors,
        + Model errors

    This implementation is tailored for the QG-1p5 model, as an example for tweeking the basic Numpy/SpScipy-based implementations.

    QGBackgroundErrorModel:
    -----------------------
    A class implementing the Background/Prior errors (specifications) and associated operations, and functionalities.

    QGObservationErrorModel:
    ------------------------
    A class implementing the observational errors (specifications) and associated operations, and functionalities.

    QGModelErrorModel:
    ------------------
    A class implementing the dynamical model errors (specifications) and associated operations, and functionalities.

    Remarks:
    --------
    - All models implemented in this module are written making use of the Numpy and SciPy.sparse functionalities.
    - These Should be used with models implemented based on linear algebra modules built on top of Numpy functionalities,
      i.e. this module deals with states wrapped in StateVectorNumpy, and matrices wrapped in StateMatrixNumpy, and StateMatrixSpScipy.
                                                                                          
"""


import numpy as np
import scipy.sparse as sparse

import dates_utility as utility
from state_vector_base import StateVectorBase as StateVector
from observation_vector_base import ObservationVectorBase as ObservationVector
from state_matrix_numpy import StateMatrixNumpy
from state_matrix_sp_scipy import StateMatrixSpSciPy
from observation_matrix_sp_scipy import ObservationMatrixSpSciPy

from error_models_numpy import BackgroundErrorModelNumpy as BackgroundErrorModel
from error_models_numpy import ObservationErrorModelNumpy as ObservationErrorModel
from error_models_numpy import ModelErrorModelNumpy as ModelErrorModel


class QGBackgroundErrorModel(BackgroundErrorModel):
    """
    A class implementing the background errors of the QG model.
    
    Construct observation error model.
    
    Args:
        model: reference to the model object to which the error model will be attached.
        configs: a configurations dictionary for observation errors (used for constructing statistics...)
                 This should override whatever preset in the model if is not None.
    
    """
    _def_background_error_configs = {'background_errors_covariance_method': 'diagonal',
                                     'background_errors_distribution': 'gaussian',
                                     'background_errors_variances': 5.0
                                     }

    def __init__(self, model, configs=None):
        
        self._model = model
        self._state_size = model.state_size()  # this should be passed from the model...
        self._configs = utility.aggregate_configurations(configs, QGBackgroundErrorModel._def_background_error_configs)
        self.B = None  # background error covariance matrix
        self.invB = None  # inverse of background error covariance matrix
        self.sqrtB = None  # square root matrix of background error covariance matrix (for random vector generations)
        self.detB = None

        self._B_structure = self._configs['background_errors_covariance_method']
        #

    def construct_error_covariances(self, construct_inverse=False, construct_sqrtm=False, sqrtm_method='cholesky'):
        """
        Construct error covariance matrix.
        Input:
            construct_inverse: construct the full inverse of the error covariance matrix.
            construct_sqrtm: construct the square root of the matrix
        """
        error_variances = self._configs['background_errors_variances']
        state_size = self._model.state_size()
        # construct background error covariance matrix
        if self._B_structure == 'diagonal':
            #
            variances = error_variances*np.ones(state_size)
            indexes = np.empty((2, state_size))
            indexes[0, :] = np.arange(state_size)
            indexes[1, :] = np.arange(state_size)
            B = sparse.csr_matrix((variances, indexes), shape=(state_size, state_size))
            self.B = StateMatrixSpSciPy(B)
            #
            try:
                self.detB = error_variances ** state_size
            except OverflowError:
                self.detB = np.infty

        elif self._B_structure == 'full':
            # TODO: consider using the initial ensemble to create initial B instead!
            self.B = self._model.state_matrix(np.diag(error_variances*np.ones(state_size)))
            try:
                self.detB = error_variances ** state_size
            except OverflowError:
                self.detB = np.infty
        else:
            raise ValueError("Type of B should be either '%s', or '%s'! " % ('diagonal', 'full'))

        # construct inverse of the observation error covariance matrix if requested.
        if construct_inverse:
            if self._B_structure == 'diagonal':
                variances = self.B.diag()
                state_size = self._model.state_size()
                indexes = np.empty((2, state_size))
                indexes[0, :] = np.arange(state_size)
                indexes[1, :] = np.arange(state_size)
                invB = sparse.csr_matrix((1.0/variances, indexes), shape=(state_size, state_size))
                self.invB = StateMatrixSpSciPy(invB)

            elif self._B_structure == 'full':
                self.invB = self._model.state_matrix(np.diag((1.0/error_variances)*np.ones(state_size)))

            else:
                raise ValueError("Type of B should be either '%s', or '%s'! " % ('diagonal', 'full'))

        # construct the square root of the observation error covariance matrix if requested
        if construct_sqrtm:
            if self._B_structure == 'diagonal':
                variances = self.B.diag()
                state_size = self._model.state_size()
                indexes = np.empty((2, state_size))
                indexes[0, :] = np.arange(state_size)
                indexes[1, :] = np.arange(state_size)
                sqrtB = sparse.csr_matrix((np.sqrt(variances), indexes), shape=(state_size, state_size))
                self.sqrtB = StateMatrixSpSciPy(sqrtB)

            elif self._B_structure == 'full':
                self.sqrtB = self._model.state_matrix(np.diag(np.sqrt(error_variances)*np.ones(state_size)))
            else:
                raise ValueError("Type of B should be either '%s', or '%s'! " % ('diagonal', 'full'))

    def error_covariances_inv_prod_vec(self, in_state, in_place=True):
        """
        return the product of inverse error covariance matrix by a vector
        """
        if self.invB is None:
            raise ValueError("invB is not available!")

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._B_structure == 'diagonal':
            # scaled_state = scaled_state.multiply(self.invB, in_place=True)
            scaled_state = self.invB.vector_product(scaled_state)
        elif self._B_structure == 'full':
            scaled_state = self.invB.vector_product(scaled_state)
        else:
            raise ValueError("Type of B should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_state

    def error_covariances_sqrt_prod_vec(self, in_state, in_place=True):
        """
        return the product of square root error covariance matrix by a vector
        """
        if self.sqrtB is None:
            raise ValueError("sqrtB is not available!")

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._B_structure == 'diagonal':
            # scaled_state = scaled_state.multiply(self.sqrtB)
            scaled_state = self.sqrtB.vector_product(scaled_state)
        elif self._B_structure:
            scaled_state = self.sqrtB.vector_product(scaled_state)
        else:
            raise ValueError("Type of B should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_state

    def error_covariances_prod_vec(self, in_state, in_place=True):
        """
        return the product of error covariance matrix by a vector
        """
        assert isinstance(in_state, StateVector)

        if self.B is None:
            raise ValueError("B is not available!")

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._B_structure == 'diagonal':
            scaled_state = self.B.vector_product(scaled_state)
        elif self._B_structure == 'full':
            scaled_state = self.B.vector_product(scaled_state)
        else:
            raise ValueError("Type of B should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_state


    def generate_noise_vec(self, noise_type='gaussian'):
        """
        generate a random vector sampled from a PDF following noise_type. e.g. 'guassian'
        """
        background_noise_type = self._configs['background_errors_distribution']
        if background_noise_type.lower() == 'gaussian':
            randn_vec = self._model.state_vector(utility.mvn_rand_vec(self._state_size))
            self.error_covariances_sqrt_prod_vec(randn_vec, in_place=True)
        else:
            raise ValueError("noise type '%s' is not supported!" % background_noise_type)

        return randn_vec


#
#
#
class QGObservationErrorModel(ObservationErrorModel):
    """
    A class implementing the observation errors of the QG model.
    
    Construct observation error model.

    Args:
        model: reference to the model object to which the error model will be attached.
        configs: a configurations dictionary for observation errors (used for constructing statistics...)
                 This should override whatever preset in the model if is not None.
    
    """
    _def_observation_error_configs = {'observation_errors_covariance_method': 'diagonal',
                                      'observation_errors_distribution': 'gaussian',
                                      'observation_errors_variances': 4.0
                                      }

    def __init__(self, model, configs=None):
        
        self._model = model
        self._observation_vector_size = model.observation_vector_size()  #  this should be passed from the model...
        self._configs = utility.aggregate_configurations(configs, QGObservationErrorModel._def_observation_error_configs)
        self.R = None  # observation error covariance matrix
        self.invR = None  # inverse of observation error covariance matrix
        self.sqrtR = None  # square root matrix of observation error covariance matrix (for random vector generations)
        self.detR = None

        self._R_structure = self._configs['observation_errors_covariance_method']
        #

    def construct_error_covariances(self,
                                    construct_inverse=False,
                                    construct_sqrtm=False,
                                    sqrtm_method='cholesky'
                                    ):
        """
        Construct error covariance matrix.
        Input:
            construct_inverse: construct the full inverse of the error covariance matrix.
            construct_sqrtm: construct the square root of the matrix
        """
        error_variances = self._configs['observation_errors_variances']
        observation_vector_dim = self._observation_vector_size
        # construct observation error covariance matrix
        if self._R_structure == 'diagonal':
            #
            variances = error_variances*np.ones(observation_vector_dim)
            observation_size = self._model.observation_vector_size()
            indexes = np.empty((2, observation_size))
            indexes[0, :] = np.arange(observation_size)
            indexes[1, :] = np.arange(observation_size)
            R = sparse.csr_matrix((variances, indexes), shape=(observation_size, observation_size))
            self.R = ObservationMatrixSpSciPy(R)

            try:
                self.detR = error_variances ** observation_vector_dim
            except OverflowError:
                self.detR = np.infty

        elif self._R_structure == 'full':
            self.R = self._model.observation_matrix(np.diag(error_variances*np.ones(observation_vector_dim)))
            try:
                self.detR = error_variances ** observation_vector_dim
            except OverflowError:
                self.detR = np.infty

        else:
            raise ValueError("Type of R should be either '%s', or '%s'! " % ('diagonal', 'full'))

        # construct inverse of the observation error covariance matrix if requested.
        if construct_inverse:
            if self._R_structure == 'diagonal':
                variances = self.R.diag()
                observation_size = self._model.observation_vector_size()
                indexes = np.empty((2, observation_size))
                indexes[0, :] = np.arange(observation_size)
                indexes[1, :] = np.arange(observation_size)
                invR = sparse.csr_matrix((1.0/variances, indexes), shape=(observation_size, observation_size))
                self.invR = ObservationMatrixSpSciPy(invR)

            elif self._R_structure == 'full':
                self.invR = self._model.observation_matrix(np.diag((1.0/error_variances)*np.ones(observation_vector_dim)))
            else:
                raise ValueError("Type of R should be either '%s', or '%s'! " % ('diagonal', 'full'))

        # construct the square root of the observation error covariance matrix if requested
        if construct_sqrtm:
            if self._R_structure == 'diagonal':
                variances = self.R.diag()
                observation_size = self._model.observation_vector_size()
                indexes = np.empty((2, observation_size))
                indexes[0, :] = np.arange(observation_size)
                indexes[1, :] = np.arange(observation_size)
                sqrtR = sparse.csr_matrix((np.sqrt(variances), indexes), shape=(observation_size, observation_size))
                self.sqrtR = ObservationMatrixSpSciPy(sqrtR)

            elif self._R_structure == 'full':
                self.sqrtR = self._model.observation_matrix(np.diag(np.sqrt(error_variances)*np.ones(observation_vector_dim)))
            else:
                raise ValueError("Type of R should be either '%s', or '%s'! " % ('diagonal', 'full'))

    def error_covariances_inv_prod_vec(self, in_obs, in_place=True):
        """
        return the product of inverse error covariance matrix by a vector
        """
        assert isinstance(in_obs, ObservationVector)

        if self.invR is None:
            raise ValueError("invR is not available!")

        if not in_place:
            scaled_obs = in_obs.copy()
        else:
            scaled_obs = in_obs

        if self._R_structure == 'diagonal':
            # scaled_obs = scaled_obs.multiply(self.invR, in_place=True)
            scaled_obs = self.invR.vector_product(scaled_obs)
        elif self._R_structure == 'full':
            scaled_obs = self.invR.vector_product(scaled_obs)
        else:
            raise ValueError("Type of R should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_obs

    def error_covariances_sqrt_prod_vec(self, in_obs, in_place=True):
        """
        return the product of square root error covariance matrix by a vector
        """
        assert isinstance(in_obs, ObservationVector)

        if self.sqrtR is None:
            raise ValueError("sqrtR is not available!")

        if not in_place:
            scaled_obs = in_obs.copy()
        else:
            scaled_obs = in_obs

        if self._R_structure == 'diagonal':
            # scaled_obs = scaled_obs.multiply(self.sqrtR)
            scaled_obs = self.sqrtR.vector_product(scaled_obs)
        elif self._R_structure == 'full':
            scaled_obs = self.sqrtR.vector_product(scaled_obs)
        else:
            raise ValueError("Type of R should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_obs

    def error_covariances_prod_vec(self, in_obs, in_place=True):
        """
        return the product of error covariance matrix by a vector
        """
        assert isinstance(in_obs, ObservationVector)

        if self.R is None:
            raise ValueError("R is not available!")

        if not in_place:
            scaled_obs = in_obs.copy()
        else:
            scaled_obs = in_obs

        if self._R_structure == 'diagonal':
            scaled_obs = self.R.vector_product(scaled_obs)
        elif self._R_structure == 'full':
            scaled_obs = self.R.vector_product(scaled_obs)
        else:
            raise ValueError("Type of R should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_obs

    def generate_noise_vec(self, noise_type='gaussian'):
        """
        generate a random vector sampled from a PDF following noise_type. e.g. 'guassian'
        """
        observation_noise_type = self._configs['observation_errors_distribution']
        if observation_noise_type == 'gaussian':
            randn_vec = self._model.observation_vector(utility.mvn_rand_vec(self._observation_vector_size))
            self.error_covariances_sqrt_prod_vec(randn_vec, in_place=True)
        else:
            raise ValueError("noise type '%s' is not supported!" % observation_noise_type)

        return randn_vec


class QGModelErrorModel(ModelErrorModel):
    """
    A class implementing the model errors of the QG model.
    
    Construct observation error model.

    Args:
        model: reference to the model object to which the error model will be attached.
        configs: a configurations dictionary for observation errors (used for constructing statistics...)
                 This should override whatever preset in the model if is not None.
    
    """
    _def_model_error_configs = {'model_errors_covariance_method': 'diagonal',
                                'model_errors_distribution': 'gaussian',
                                'model_errors_variances': 5.0
                                }

    def __init__(self, model, configs=None):
        
        self._model = model
        self._state_size = model.state_size()  # this should be passed from the model...
        self._configs = utility.aggregate_configurations(configs, QGModelErrorModel._def_model_error_configs)
        self.Q = None  # model error covariance matrix
        self.invQ = None  # inverse of model error covariance matrix
        self.sqrtQ = None  # square root matrix of model error covariance matrix (for random vector generations)
        self.detQ = None

        self._Q_structure = self._configs['model_errors_covariance_method']
        #

    def construct_error_covariances(self, construct_inverse=False, construct_sqrtm=False, sqrtm_method='cholesky'):
        """
        Construct error covariance matrix.
        Input:
            construct_inverse: construct the full inverse of the error covariance matrix.
            construct_sqrtm: construct the square root of the matrix
        """
        error_variances = self._configs['model_errors_variances']
        state_size = self._model.state_size()
        # construct model error covariance matrix
        if self._Q_structure == 'diagonal':
            #
            variances = error_variances*np.ones(state_size)
            indexes = np.empty((2, state_size))
            indexes[0, :] = np.arange(state_size)
            indexes[1, :] = np.arange(state_size)
            Q = sparse.csr_matrix((variances, indexes), shape=(state_size, state_size))
            self.Q = StateMatrixSpSciPy(Q)
            #
            try:
                self.detQ = error_variances ** state_size
            except OverflowError:
                self.detQ = np.infty

        elif self._Q_structure == 'full':
            # TODO: consider using the initial ensemble to create initial Q instead!
            self.Q = self._model.state_matrix(np.diag(error_variances*np.ones(state_size)))
            try:
                self.detQ = error_variances ** state_size
            except OverflowError:
                self.detQ = np.infty
        else:
            raise ValueError("Type of Q should be either '%s', or '%s'! " % ('diagonal', 'full'))

        # construct inverse of the observation error covariance matrix if requested.
        if construct_inverse:
            if self._Q_structure == 'diagonal':
                variances = self.Q.diag()
                state_size = self._model.state_size()
                indexes = np.empty((2, state_size))
                indexes[0, :] = np.arange(state_size)
                indexes[1, :] = np.arange(state_size)
                invQ = sparse.csr_matrix((1.0/variances, indexes), shape=(state_size, state_size))
                self.invQ = StateMatrixSpSciPy(invQ)

            elif self._Q_structure == 'full':
                self.invQ = self._model.state_matrix(np.diag((1.0/error_variances)*np.ones(state_size)))

            else:
                raise ValueError("Type of Q should be either '%s', or '%s'! " % ('diagonal', 'full'))

        # construct the square root of the observation error covariance matrix if requested
        if construct_sqrtm:
            if self._Q_structure == 'diagonal':
                variances = self.Q.diag()
                state_size = self._model.state_size()
                indexes = np.empty((2, state_size))
                indexes[0, :] = np.arange(state_size)
                indexes[1, :] = np.arange(state_size)
                sqrtQ = sparse.csr_matrix((np.sqrt(variances), indexes), shape=(state_size, state_size))
                self.sqrtQ = StateMatrixSpSciPy(sqrtQ)

            elif self._Q_structure == 'full':
                self.sqrtQ = self._model.state_matrix(np.diag(np.sqrt(error_variances)*np.ones(state_size)))
            else:
                raise ValueError("Type of Q should be either '%s', or '%s'! " % ('diagonal', 'full'))

    def error_covariances_inv_prod_vec(self, in_state, in_place=True):
        """
        return the product of inverse error covariance matrix by a vector
        """
        if self.invQ is None:
            raise ValueError("invQ is not available!")

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._Q_structure == 'diagonal':
            # scaled_state = scaled_state.multiply(self.invQ, in_place=True)
            scaled_state = self.invQ.vector_product(scaled_state)
        elif self._Q_structure == 'full':
            scaled_state = self.invQ.vector_product(scaled_state)
        else:
            raise ValueError("Type of Q should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_state

    def error_covariances_sqrt_prod_vec(self, in_state, in_place=True):
        """
        return the product of square root error covariance matrix by a vector
        """
        if self.sqrtQ is None:
            raise ValueError("sqrtQ is not available!")

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._Q_structure == 'diagonal':
            # scaled_state = scaled_state.multiply(self.sqrtQ)
            scaled_state = self.sqrtQ.vector_product(scaled_state)
        elif self._Q_structure:
            scaled_state = self.sqrtQ.vector_product(scaled_state)
        else:
            raise ValueError("Type of Q should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_state

    def error_covariances_prod_vec(self, in_state, in_place=True):
        """
        return the product of error covariance matrix by a vector
        """
        assert isinstance(in_state, StateVector)

        if self.Q is None:
            raise ValueError("Q is not available!")

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._Q_structure == 'diagonal':
            scaled_state = self.Q.vector_product(scaled_state)
        elif self._Q_structure == 'full':
            scaled_state = self.Q.vector_product(scaled_state)
        else:
            raise ValueError("Type of Q should be either '%s', or '%s'! " % ('diagonal', 'full'))

        return scaled_state


    def generate_noise_vec(self, noise_type='gaussian'):
        """
        generate a random vector sampled from a PDF following noise_type. e.g. 'guassian'
        """
        model_noise_type = self._configs['model_errors_distribution']
        if model_noise_type.lower() == 'gaussian':
            randn_vec = self._model.state_vector(utility.mvn_rand_vec(self._state_size))
            self.error_covariances_sqrt_prod_vec(randn_vec, in_place=True)
        else:
            raise ValueError("noise type '%s' is not supported!" % model_noise_type)

        return randn_vec

