
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
       1- Background errors,
       2- Observation errors,
       3- Model errors


    BackgroundErrorModelNumpy:
    --------------------------
    A class implementing the Background/Prior errors (specifications) and associated operations, and functionalities.

    ObservationErrorModelNumpy:
    ---------------------------
    A class implementing the observational errors (specifications) and associated operations, and functionalities.

    ModelErrorModelNumpy:
    ---------------------
    A class implementing the dynamical model errors (specifications) and associated operations, and functionalities.

    Note:
    -----
    - All models implemented in this module are written making use of the Numpy and SciPy.sparse functionalities. 
      These Should be used with models implemented based on linear algebra modules built on top of Numpy functionalities, 
      i.e. this module deals with states wrapped in StateVectorNumpy, and matrices wrapped in StateMatrixNumpy, and StateMatrixSpScipy.

    - The implementations provided in this module are rather very simple and can be used with very simple model such as Lorenz-96.                                        
                                                                                          
"""


import numpy as np
import scipy.sparse as sparse

import dates_utility as utility
from state_vector_base import StateVectorBase as StateVector
from observation_vector_base import ObservationVectorBase as ObservationVector
from state_matrix_numpy import StateMatrixNumpy
from observation_matrix_numpy import ObservationMatrixNumpy
from state_matrix_sp_scipy import StateMatrixSpSciPy
from observation_matrix_sp_scipy import ObservationMatrixSpSciPy
from error_models_base import ErrorModelsBase


class BackgroundErrorModelNumpy(ErrorModelsBase):
    """
    A class implementing the background errors (specifications) and associated operations/functionalities.    
    
    Construct the background error model.
    
    Args:
        model: reference to the model object to which the error model will be attached.
        configs: a configurations dictionary for background errors (used for constructing statistics...).
            This should override whatever preset in the model if is not None.
            Supported Configurations:
            -------------------------
            * errors_covariance_method: The shape of the background error covariance matrix
                  i-  'empirical': This is a dirty way of creating background error from the reference initial
                      condition of the model and/or long runs.
                  ii- ?
            * errors_distribution: The probability distribution of the background errors.
                  Note that this is a basic implementation for the modeled version of B, while
                  ensemble-based versions can assume different assumptions such as GMM.
            * background_noise_level: This is used to create variances of the background errors, such that:
                  error variance = noise_level * signal magnitude (of initial condition).
            * create_errors_correlations (default False): Whether to create correlations between different
                  components of the state vector or not. If False, diagonal covariance matrix is construct
                  otherwise it is dense (and probably localized becoming sparse if 
                  localize_errors_covariances is set to True),
            * localize_errors_covariances (default False): Use the model to localize the error-covariance 
                  matrix with model's default settings for localization.
            * variance_adjusting_factor: a scalar factor (in [0, 1]) used to adjust variances so that the 
                covariance matrices are not deficient. 
                Diagonal of the covariance matrix is updated using the following linear rule:
                    matrix diagonal = variance_adjusting_factor + (1.0-variance_adjusting_factor) * matrix diagonal
    
    Returns:
        None
    
    """
    
    _def_background_error_configs = {'errors_covariance_method': 'empirical',
                                     'errors_distribution': 'gaussian',
                                     'background_noise_level': 0.08,
                                     'create_errors_correlations':False,
                                     'localize_errors_covariances':True,
                                     'variance_adjusting_factor':0.1
                                     }

    def __init__(self, model, configs=None):
        
        # Aggregate passed configurations with default settings
        self._configs = utility.aggregate_configurations(configs, BackgroundErrorModelNumpy._def_background_error_configs)
        
        # Check the probability distribution of background errors
        errors_distribution = self._configs['errors_distribution'].lower()
        if errors_distribution == 'gaussian':
            pass
        else:
            print("The probability distribution ['%s'] chosen for background errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError()
        
        # Check the strategy for background errors creation
        errors_covariance_method = self._configs['errors_covariance_method'].lower()
        if errors_covariance_method == 'empirical':
            pass
        else:
            print("The mothod ['%s'] chosen for construction of the background errors is not yet supported." \
                  % errors_covariance_method)
            raise NotImplementedError()
        
        # Check for the structure of the background error covariance matrix:
        create_errors_correlations = self._configs['create_errors_correlations']
        
        if not create_errors_correlations:
            self._B_structure = 'diagonal'
        else:
            self._B_structure = 'full'
        
        # Attach basic variables ( To be updated on demand )
        self._model = model
        self._state_size = model.state_size()
        self.B = None  # background error covariance matrix
        self.invB = None  # inverse of background error covariance matrix
        self.sqrtB = None  # square root matrix of background error covariance matrix (for random vector generations)
        #
        self._initialized = True
        #

    def construct_error_covariances(self,
                                    construct_inverse=False,
                                    construct_sqrtm=False, 
                                    sqrtm_method='cholesky'
                                    ):
        """
        Construct error covariance matrix.
        
        Args:
            construct_inverse (default False): construct the full inverse of the error covariance matrix.
            construct_sqrtm (default False): construct the square root of the matrix:
            sqrtm_method (default 'cholesky'): method to use for finding square root of the covariance matrix:
                1- 'cholesky': (it must be PSD) this is effective only if the covariance matrix is full.
                              
        Returns:
            None 
        
        """
        #
        errors_distribution = self._configs['errors_distribution'].lower()
        #
        if errors_distribution == 'gaussian':
            #
            errors_covariance_method = self._configs['errors_covariance_method'].lower()
            if errors_covariance_method == 'empirical':
                #
                state_size = self._state_size
                #
                # Statistics of Standard Gaussian Background Errors are constructed:
                sigma_x0 = self._configs['background_noise_level']
                #
                # add factor*I to the covariance matrix to avoid numerical issues upon inversion
                factor = self._configs['variance_adjusting_factor']
                #
                # construct background error covariance matrix
                if self._B_structure == 'diagonal':  # no correlations are created
                    # obtain background (variances) from reference initial condition
                    state_mag = self._model._reference_initial_condition.copy()
                    state_mag = state_mag.scale(sigma_x0)
                    state_mag = state_mag.get_numpy_array()
                    # variances = np.ones(self._state_size)*factor + state_mag
                    variances = state_mag
                    variances = factor + (1.0-factor) * np.square(variances)
                    try:
                        B = sparse.lil_matrix((state_size, state_size))
                        B.setdiag(variances, k=0)
                        B = B.tocsr()
                    except (TypeError):
                        indexes = np.empty((2, state_size))
                        indexes[0, :] = np.arange(state_size)
                        indexes[1, :] = np.arange(state_size)
                        B = sparse.csr_matrix((variances, indexes), shape=(state_size, state_size))
                    self.B = StateMatrixSpSciPy(B)
                    # Note that the way we initialize self.B here cannot be used with models with any kind of 
                    # ghost cells. It can be used with very simple models only, and should be derived and 
                    # updated for big models...
                #
                elif self._B_structure == 'full':
                    # obtain background (variances) from reference initial condition
                    state_mag = self._model._reference_initial_condition.copy()
                    # state_mag = state_mag.abs()
                    state_mag.scale(sigma_x0)
                    state_mag = state_mag.get_numpy_array()
                    variances = state_mag
                    #
                    variances = np.squeeze(variances)
                    #
                    localize_errors_covariances = self._configs['localize_errors_covariances']
                    if localize_errors_covariances:
                        self.B = self._model.apply_state_covariance_localization(StateMatrixNumpy(np.outer(variances, variances)))
                    else:
                        self.B = StateMatrixNumpy(np.outer(variances, variances))
                        #
                    diagonal = self.B.diagonal()
                    self.B.set_diagonal(factor + (1.0-factor)*diagonal)
                #
                else:
                    print("The structure of B: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
                    raise ValueError
                    #

                #
                # construct inverse of the background error covariance matrix if requested.
                if construct_inverse:
                    if self._B_structure == 'diagonal':
                        variances = self.B.diag()
                        try:
                            invB = sparse.lil_matrix((state_size, state_size))
                            invB.setdiag(1.0/variances, k=0)
                            invB = invB.tocsr()
                        except (TypeError):
                            indexes = np.empty((2, state_size))
                            indexes[0, :] = np.arange(state_size)
                            indexes[1, :] = np.arange(state_size)
                            invB = sparse.csr_matrix((1.0/variances, indexes), shape=(state_size, state_size))
                        self.invB = StateMatrixSpSciPy(invB)
                        # Note that the way we initialize self.B here cannot be used with models with any kind of 
                        # ghost cells. It can be used with very simple models only, and should be derived and 
                        # updated for big models...
                        
                    #
                    elif self._B_structure == 'full':
                        try:
                            self.invB = self.B.inverse(in_place=False)
                        except (NotImplementedError):
                            self.invB = StateMatrixNumpy(np.linalg.inv(self.B.get_numpy_array()))
                    #
                    else:
                        print("The structure of B: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
                        raise ValueError

                #
                # construct the square root of the background error covariance matrix if requested
                if construct_sqrtm:
                    if self._B_structure == 'diagonal':
                        variances = self.B.diag()
                        try:
                            sqrtB = sparse.lil_matrix((state_size, state_size))
                            sqrtB.setdiag(np.sqrt(variances), k=0)
                            sqrtB = sqrtB.tocsr()
                        except (TypeError):
                            state_size = self._model.state_size()
                            indexes = np.empty((2, state_size))
                            indexes[0, :] = np.arange(state_size)
                            indexes[1, :] = np.arange(state_size)
                            sqrtB = sparse.csr_matrix((np.sqrt(variances), indexes), shape=(state_size, state_size))
                        self.sqrtB = StateMatrixSpSciPy(sqrtB)
                        # Note that the way we initialize self.B here cannot be used with models with any kind of 
                        # ghost cells. It can be used with very simple models only, and should be derived and 
                        # updated for big models...
                    #
                    elif self._B_structure == 'full':
                        self.sqrtB = self.B.cholesky(in_place=False)
                    #
                    else:
                        print("The structure of B: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
                        raise ValueError
                        #
            else:
                print("The mothod ['%s'] chosen for construction of the background errors is not yet supported." \
                      % errors_covariance_method)
                raise NotImplementedError()
        #
        else:
            print("The probability distribution ['%s'] chosen for background errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError
            #
    
    def error_covariances_inv_prod_vec(self, in_state, in_place=True):
        """
        Evaluate and return the product of inverse of the error covariance matrix by a vector (in_state)
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the inverse of the covariance matrix of associated errors by the 
            passed vector in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
            
        Returns:
            scaled_state: model.state_vector; the product of inverse of the error covariance matrix by in_state
        
        """
        assert isinstance(in_state, StateVector)
        #
        if self.invB is None:
            print("Failed to retrieve the inverse of the background error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._B_structure in ['diagonal', 'full']:
            scaled_state = self.invB.vector_product(scaled_state)
        else:
            print("The structure of B: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_state
        #

    def error_covariances_sqrt_prod_vec(self, in_state, in_place=True):
        """
        Evaluate and return the product of the square root of error covariance matrix by a vector (in_state)
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the square root of the covariance matrix of associated errors by 
            the passed vector (in_state) in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
            
        Returns:
            scaled_state: model.state_vector; the product of square root of the error covariance matrix by in_state
        
        """
        assert isinstance(in_state, StateVector)
        #
        if self.sqrtB is None:
            print("Failed to retrieve the square root of the background error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._B_structure in ['diagonal', 'full']:
            scaled_state = self.sqrtB.vector_product(scaled_state)
        else:
            print("The structure of B: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_state
        #

    def error_covariances_prod_vec(self, in_state, in_place=True):
        """
        Evaluate and return the product of the error covariance matrix by a vector (in_state)
        This method might be actually needed!
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the covariance matrix of associated errors by 
            the passed vector (in_state) in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
            
        Returns:
            scaled_state: model.state_vector; the product of the error covariance matrix by in_state
        
        """
        assert isinstance(in_state, StateVector)
        #
        if self.B is None:
            print("Failed to retrieve the background error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._B_structure in ['diagonal', 'full']:
            scaled_state = self.B.vector_product(scaled_state)
        else:
            print("The structure of B: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_state
        #
    
    def generate_noise_vec(self):
        """
        Generate a random vector sampled from the Probability Distribution Fuction describing the errors
        implemented in this model. E.g. Gaussian observation noise vector with zero mean, and specified 
        background error covariance matrix
        
        Args:
        
        Returns:
            randn_vec: model.state_vector; a random noise vector sampled from the PDF of this errors model
        
        """
        errors_distribution = self._configs['errors_distribution'].lower()
        if errors_distribution == 'gaussian':
            # Generate Gaussian Random Noise Vector
            randn_numpy = utility.mvn_rand_vec(self._state_size)
            randn_vec = self._model.state_vector()
            randn_vec[:] = randn_numpy[:]
            randn_vec = self.error_covariances_sqrt_prod_vec(randn_vec, in_place=True)
            #
            return randn_vec
            #
        else:
            print("The probability distribution ['%s'] chosen for background errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError
            #


class ModelErrorModelNumpy(ErrorModelsBase):
    """
    A class implementing the model errors (specifications) and associated operations/functionalities.    
    
    Construct the forecast model error model.
    
    Args:
        model: reference to the model object to which the error model will be attached.
        configs: a configurations dictionary for model errors (used for constructing statistics...).
            This should override whatever preset in the model if is not None.
            Supported Configurations:
            * errors_covariance_method: The shape of the model error covariance matrix
                  i-  'empirical': This is a dirty way of creating model error from the reference initial
                      condition of the model and/or long runs.
                  ii- ?
            * errors_distribution: The probability distribution of the model errors.
                  Note that this is a basic implementation for the modeled version of B, while
                  ensemble-based versions can assume different assumptions such as GMM.
            * model_noise_level: This is used to create variances of the model errors, such that:
                  error variance = noise_level * signal magnitude (of initial condition).
            * create_errors_correlations (default False): Whether to create correlations between different
                  components of the state vector or not. If False, diagonal covariance matrix is construct
                  otherwise it is dense (and probably localized becoming sparse if 
                  localize_errors_covariances is set to True),
            * localize_errors_covariances (default False): Use the model to localize the error-covariance 
                  matrix with model's default settings for localization.
            * variance_adjusting_factor: a scalar factor (in [0, 1]) used to adjust variances so that the 
                covariance matrices are not deficient. 
                Diagonal of the covariance matrix is updated using the following linear rule:
                    matrix diagonal = variance_adjusting_factor + (1.0-variance_adjusting_factor) * matrix diagonal
    
    Returns:
        None
    
    """
    
    _def_model_error_configs = {'errors_covariance_method': 'empirical',
                                'errors_distribution': 'gaussian',
                                'model_noise_level': 0.08,
                                'create_errors_correlations':True,
                                'localize_errors_covariances':True,
                                'variance_adjusting_factor':0.01
                                 }

    def __init__(self, model, configs=None):
        
        # Aggregate passed configurations with default settings
        self._configs = utility.aggregate_configurations(configs, ModelErrorModelNumpy._def_model_error_configs)
        
        # Check the probability distribution of model errors
        errors_distribution = self._configs['errors_distribution'].lower()
        if errors_distribution == 'gaussian':
            pass
        else:
            print("The probability distribution ['%s'] chosen for model errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError()
        
        # Check the strategy for model errors creation
        errors_covariance_method = self._configs['errors_covariance_method'].lower()
        if errors_covariance_method == 'empirical':
            pass
        else:
            print("The mothod ['%s'] chosen for construction of the model errors is not yet supported." \
                  % errors_covariance_method)
            raise NotImplementedError()
        
        # Check for the structure of the model error covariance matrix:
        create_errors_correlations = self._configs['create_errors_correlations']
        if not create_errors_correlations:
            self._Q_structure = 'diagonal'
        else:
            self._Q_structure = 'full'
        
        # Attach basic variables ( To be updated on demand )
        self._model = model
        self._state_size = model.state_size()
        self.Q = None  # model error covariance matrix
        self.invQ = None  # inverse of model error covariance matrix
        self.sqrtQ = None  # square root matrix of model error covariance matrix (for random vector generations)
        #
        self._initialized = True
        #

    def construct_error_covariances(self,
                                    construct_inverse=False,
                                    construct_sqrtm=False, 
                                    sqrtm_method='cholesky'
                                    ):
        """
        Construct error covariance matrix.
        
        Args:
            construct_inverse (default False): construct the full inverse of the error covariance matrix.
            construct_sqrtm (default False): construct the square root of the matrix:
            sqrtm_method (default 'cholesky'): method to use for finding square root of the covariance matrix:
                    1- 'cholesky': (it must be PSD) this is effective only if the covariance matrix is full.
                              
        Returns:
            None 
        
        """
        errors_distribution = self._configs['errors_distribution'].lower()
        #
        if errors_distribution == 'gaussian':
            #
            errors_covariance_method = self._configs['errors_covariance_method'].lower()
            if errors_covariance_method == 'empirical':
                #
                # Statistics of Standard Gaussian Model Errors are constructed:
                sigma_x0 = self._configs['model_noise_level']
                #
                # add factor*I to the covariance matrix to avoid numerical issues upon inversion
                factor = self._configs['variance_adjusting_factor']
                #
                # construct model error covariance matrix
                if self._Q_structure == 'diagonal':  # no correlations are created
                    # obtain model (variances) from reference initial condition
                    state_mag = self._model._reference_initial_condition.copy()
                    state_mag = state_mag.scale(sigma_x0)
                    state_mag = state_mag.get_numpy_array()
                    # variances = np.ones(self._state_size)*factor + state_mag
                    variances = state_mag
                    variances = factor + (1.0-factor) * np.square(variances)
                    try:
                        Q = sparse.lil_matrix((state_size, state_size))
                        Q.setdiag(variances, k=0)
                        Q = Q.tocsr()
                    except (TypeError):
                        indexes = np.empty((2, state_size))
                        indexes[0, :] = np.arange(state_size)
                        indexes[1, :] = np.arange(state_size)
                        Q = sparse.csr_matrix((variances, indexes), shape=(state_size, state_size))
                    self.Q = StateMatrixSpSciPy(Q)
                    # Note that the way we initialize self.Q here cannot be used with models with any kind of 
                    # ghost cells. It can be used with very simple models only, and should be derived and 
                    # updated for big models...
                #
                elif self._Q_structure == 'full':
                    # obtain model (variances) from reference initial condition
                    state_mag = self._model._reference_initial_condition.copy()
                    # state_mag = state_mag.abs()
                    state_mag.scale(sigma_x0)
                    state_mag = state_mag.get_numpy_array()
                    #
                    variances = np.squeeze(variances)
                    #
                    localize_errors_covariances = self._configs['localize_errors_covariances']
                    print ("localize_errors_covariances", localize_errors_covariances)
                    m = input("push")
                    if localize_errors_covariances:
                        self.Q = self._model.apply_state_covariance_localization(StateMatrixNumpy(np.outer(variances, variances)))
                    else:
                        self.Q = StateMatrixNumpy(np.outer(variances, variances))
                        #
                    diagonal = self.Q.diagonal()
                    self.Q.set_diagonal(factor + (1.0-factor)*diagonal)
                #
                else:
                    print("The structure of Q: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
                    raise ValueError
                    #

                #
                # construct inverse of the model error covariance matrix if requested.
                if construct_inverse:
                    if self._Q_structure == 'diagonal':
                        variances = self.Q.diag()
                        try:
                            invQ = sparse.lil_matrix((state_size, state_size))
                            invQ.setdiag(1.0/variances, k=0)
                            invQ = invQ.tocsr()
                        except (TypeError):
                            indexes = np.empty((2, state_size))
                            indexes[0, :] = np.arange(state_size)
                            indexes[1, :] = np.arange(state_size)
                            invQ = sparse.csr_matrix((1.0/variances, indexes), shape=(state_size, state_size))
                        self.invQ = StateMatrixSpSciPy(invQ)
                        # Note that the way we initialize self.Q here cannot be used with models with any kind of 
                        # ghost cells. It can be used with very simple models only, and should be derived and 
                        # updated for big models...
                        
                    #
                    elif self._Q_structure == 'full':
                        try:
                            self.invQ = self.Q.inverse(in_place=False)
                        except (NotImplementedError):
                            self.invQ = StateMatrixNumpy(np.linalg.inv(self.Q.get_numpy_array()))
                    #
                    else:
                        print("The structure of Q: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
                        raise ValueError

                #
                # construct the square root of the model error covariance matrix if requested
                if construct_sqrtm:
                    if self._Q_structure == 'diagonal':
                        variances = self.Q.diag()
                        try:
                            sqrtQ = sparse.lil_matrix((state_size, state_size))
                            sqrtQ.setdiag(np.sqrt(variances), k=0)
                            sqrtQ = sqrtQ.tocsr()
                        except (TypeError):
                            state_size = self._model.state_size()
                            indexes = np.empty((2, state_size))
                            indexes[0, :] = np.arange(state_size)
                            indexes[1, :] = np.arange(state_size)
                            sqrtQ = sparse.csr_matrix((np.sqrt(variances), indexes), shape=(state_size, state_size))
                        self.sqrtQ = StateMatrixSpSciPy(sqrtQ)
                        # Note that the way we initialize self.Q here cannot be used with models with any kind of 
                        # ghost cells. It can be used with very simple models only, and should be derived and 
                        # updated for big models...
                    #
                    elif self._Q_structure == 'full':
                        self.sqrtQ = self.Q.cholesky(in_place=False)
                    #
                    else:
                        print("The structure of Q: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
                        raise ValueError
                        #
            else:
                print("The mothod ['%s'] chosen for construction of the model errors is not yet supported." \
                      % errors_covariance_method)
                raise NotImplementedError()
        #
        else:
            print("The probability distribution ['%s'] chosen for model errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError
            #    
    
    def error_covariances_inv_prod_vec(self, in_state, in_place=True):
        """
        Evaluate and return the product of inverse of the error covariance matrix by a vector (in_state)
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the inverse of the covariance matrix of associated errors by the 
            passed vector in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
        
        Returns:
            scaled_state: model.state_vector; the product of inverse of the error covariance matrix by in_state
        
        """
        assert isinstance(in_state, StateVector)
        #
        if self.invQ is None:
            print("Failed to retrieve the inverse of the model error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._Q_structure in ['diagonal', 'full']:
            scaled_state = self.invQ.vector_product(scaled_state)
        else:
            print("The structure of Q: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_state
        #

    def error_covariances_sqrt_prod_vec(self, in_state, in_place=True):
        """
        Evaluate and return the product of the square root of error covariance matrix by a vector (in_state)
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the square root of the covariance matrix of associated errors by 
            the passed vector (in_state) in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
        
        Returns:
            scaled_state: model.state_vector; the product of square root of the error covariance matrix by in_state
        
        """
        assert isinstance(in_state, StateVector)
        #
        if self.sqrtQ is None:
            print("Failed to retrieve the square root of the model error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._Q_structure in ['diagonal', 'full']:
            scaled_state = self.sqrtQ.vector_product(scaled_state)
        else:
            print("The structure of Q: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_state
        #

    def error_covariances_prod_vec(self, in_state, in_place=True):
        """
        Evaluate and return the product of the error covariance matrix by a vector (in_state)
        This method might be actually needed!
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the covariance matrix of associated errors by 
            the passed vector (in_state) in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
            
        Returns:
            scaled_state: model.state_vector; the product of the error covariance matrix by in_state
        
        """
        assert isinstance(in_state, StateVector)
        #
        if self.Q is None:
            print("Failed to retrieve the model error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_state = in_state.copy()
        else:
            scaled_state = in_state

        if self._Q_structure in ['diagonal', 'full']:
            scaled_state = self.Q.vector_product(scaled_state)
        else:
            print("The structure of Q: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_state
        #
    
    def generate_noise_vec(self):
        """
        Generate a random vector sampled from the Probability Distribution Fuction describing the errors
        implemented in this model. E.g. Gaussian observation noise vector with zero mean, and specified 
        observational error covariance matrix
        
        Args:
        
        Returns:
            randn_vec: model.state_vector; a random noise vector sampled from the PDF of this errors model
        
        """
        errors_distribution = self._configs['errors_distribution'].lower()
        if errors_distribution == 'gaussian':
            # Generate Gaussian Random Noise Vector
            randn_numpy = utility.mvn_rand_vec(self._state_size)
            randn_vec = self._model.state_vector()
            randn_vec[:] = randn_numpy[:]
            randn_vec = self.error_covariances_sqrt_prod_vec(randn_vec, in_place=True)
            #
            return randn_vec
            #
        else:
            print("The probability distribution ['%s'] chosen for model errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError
            #


class ObservationErrorModelNumpy(ErrorModelsBase):
    """
    A class implementing the observation errors (specifications) and associated operations/functionalities.    
    
    Construct the observation error model.
        
        Args:
            model: reference to the model object to which the error model will be attached.
            configs: a configurations dictionary for observation errors (used for constructing statistics...).
                This should override whatever preset in the model if is not None.
                Supported Configurations:
                * errors_covariance_method: The shape of the observation error covariance matrix
                      i-  'empirical': This is a dirty way of creating observation error from the reference initial
                          condition of the model and/or long runs.
                      ii- ?
                * errors_distribution: The probability distribution of the observation errors.
                      Note that this is a basic implementation for the modeled version of B, while
                      ensemble-based versions can assume different assumptions such as GMM.
                * observation_noise_level: This is used to create variances of the observation errors, such that:
                      error variance = noise_level * signal magnitude (of initial condition).
                * create_errors_correlations (default False): Whether to create correlations between different
                      components of the observation vector or not. If False, diagonal covariance matrix is construct
                      otherwise it is dense (and probably localized becoming sparse if 
                      localize_errors_covariances is set to True),
                * localize_errors_covariances (default False): Use the model to localize the error-covariance 
                      matrix with model's default settings for localization.
                      Here this is observation-error localization
                * variance_adjusting_factor: a scalar factor (in [0, 1]) used to adjust variances so that the 
                    covariance matrices are not deficient. 
                    Diagonal of the covariance matrix is updated using the following linear rule:
                        matrix diagonal = variance_adjusting_factor + (1.0-variance_adjusting_factor) * matrix diagonal
        
        Returns:
            None
        
    """
    
    _def_observation_error_configs = {'errors_covariance_method': 'empirical',
                                      'errors_distribution': 'gaussian',
                                      'observation_noise_level': 0.05,
                                      'create_errors_correlations':False,
                                      'localize_errors_covariances':False,
                                      'variance_adjusting_factor':0.0
                                      }

    def __init__(self, model, configs=None):
        
        # Aggregate passed configurations with default settings
        self._configs = utility.aggregate_configurations(configs, ObservationErrorModelNumpy._def_observation_error_configs)
        
        # Check the probability distribution of observation errors
        errors_distribution = self._configs['errors_distribution'].lower()
        if errors_distribution == 'gaussian':
            pass
        else:
            print("The probability distribution ['%s'] chosen for observation errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError()
        
        # Check the strategy for observation errors creation
        errors_covariance_method = self._configs['errors_covariance_method'].lower()
        if errors_covariance_method == 'empirical':
            pass
        else:
            print("The mothod ['%s'] chosen for construction of the observation errors is not yet supported." \
                  % errors_covariance_method)
            raise NotImplementedError()
        
        # Check for the structure of the observation error covariance matrix:
        create_errors_correlations = self._configs['create_errors_correlations']
        if not create_errors_correlations:
            self._R_structure = 'diagonal'
        else:
            self._R_structure = 'full'
            
        # Attach basic variables ( To be updated on demand )
        self._model = model
        self._observation_vector_size = model.observation_vector_size()
        self.R = None  # observation error covariance matrix
        self.invR = None  # inverse of observation error covariance matrix
        self.sqrtR = None  # square root matrix of observation error covariance matrix (for random vector generations)
        self.detR = None  # determinant of the observation error covariance matrix R.
        self.log_detR = None  # logarithm of the determinant of the observation error covariance matrix R.
        #
        self._initialized = True
        #

    def construct_error_covariances(self,
                                    construct_inverse=False,
                                    construct_sqrtm=False, 
                                    sqrtm_method='cholesky',
                                    observation_checkpoints=None,
                                    def_num_obs_points=30
                                    ):
        """
        Construct error covariance matrix.
        
        Args:
            construct_inverse (default False): construct the full inverse of the error covariance matrix.
            construct_sqrtm (default False): construct the square root of the matrix:
            sqrtm_method (default 'cholesky'): method to use for finding square root of the covariance matrix:
                1- 'cholesky': (it must be PSD) this is effective only if the covariance matrix is full.
            observation_checkpoints: iterable containing timespan used to construct observation trajectory to 
                calculate time-dependent correlations.
                              
        Returns:
            None 
        
        """
        errors_distribution = self._configs['errors_distribution'].lower()
        #
        if errors_distribution == 'gaussian':
            #
            errors_covariance_method = self._configs['errors_covariance_method'].lower()
            if errors_covariance_method == 'empirical':
                #
                observation_size = self._observation_vector_size
                #
                # Statistics of Standard Gaussian Observation Errors are constructed:
                sigma_obs = self._configs['observation_noise_level']
                #
                # add factor*I to the covariance matrix to avoid numerical issues upon inversion
                factor = self._configs['variance_adjusting_factor']
                #
                # construct observation error covariance matrix
                if self._R_structure == 'diagonal':
                    #
                    if observation_checkpoints is None:
                        raise ValueError("You must pass observation timespan in the option \
                                          'observation_checkpoints' of the configurations dictionary!"
                                          )
                    else:
                        num_obs_points = len(observation_checkpoints)-2
                    
                    # get an initial observation vector:
                    model_traject = self._model.integrate_state(self._model._reference_initial_condition.copy(),
                                                                checkpoints=observation_checkpoints[0:2]
                                                                )
                    if isinstance(model_traject, list):
                        cum_observation_mag = self._model.evaluate_theoretical_observation(model_traject[-1])
                    elif isinstance(model_traject, StateVector):
                        cum_observation_mag = self._model.evaluate_theoretical_observation(model_traject)
                    else:
                        print("The model output is not a trajectory or a StateVector!")
                        raise TypeError
                    
                    #
                    # Accumulate observation magnitudes...
                    cum_observation_mag = cum_observation_mag.abs()
                    for obs_ind in xrange(1, num_obs_points):
                        model_traject = self._model.integrate_state(model_traject[-1],
                                                                    checkpoints=observation_checkpoints[obs_ind:obs_ind+2]
                                                                    )
                        if isinstance(model_traject, list):
                            observation = self._model.evaluate_theoretical_observation(model_traject[-1])
                        elif isinstance(model_traject, StateVector):
                            observation = self._model.evaluate_theoretical_observation(model_traject)
                        else:
                            raise TypeError("The model output is not a trajectory or a StateVector!")
                        observation = observation.abs()
                        cum_observation_mag = cum_observation_mag.add(observation, in_place=True)
                        #
                    
                    # obtain observation (variances) from reference trajectory
                    cum_observation_mag = cum_observation_mag.scale(sigma_obs/float(num_obs_points))
                    variances = cum_observation_mag.get_numpy_array()  # Numpy array now
                    variances = factor + (1.0-factor) * np.square(variances)
                    
                    indexes = np.empty((2, observation_size))
                    indexes[0, :] = np.arange(observation_size)
                    indexes[1, :] = np.arange(observation_size)
                    R = sparse.csr_matrix((variances, indexes), shape=(observation_size, observation_size))
                    self.R = ObservationMatrixSpSciPy(R)
                    #
                    # Try evaluating the determinant of the observation error covariance matrix
                    try:
                        self.detR = variances.prod()
                    except OverflowError:
                        self.detR = np.infty
                    
                    try:
                        self.log_detR = np.sum(np.log(variances))
                    except OverflowError:
                        self.log_detR = np.infty
                        
                #
                elif self._R_structure == 'full':
                    #
                    if observation_checkpoints is None:
                        raise ValueError("You must pass observation timespan in the option \
                                          'observation_checkpoints' of the configurations dictionary!"
                                          )
                    else:
                        num_obs_points = len(observation_checkpoints)-2
                    
                    # get an initial observation vector:
                    model_traject = self._model.integrate_state(self._model._reference_initial_condition.copy(),
                                                                checkpoints=observation_checkpoints[0:2]
                                                                )
                    if isinstance(model_traject, list):
                        cum_observation_mag = self._model.evaluate_theoretical_observation(model_traject[-1])
                    elif isinstance(model_traject, StateVector):
                        cum_observation_mag = self._model.evaluate_theoretical_observation(model_traject)
                    else:
                        print("The model output is not a trajectory or a StateVector!")
                        raise TypeError
                    
                    #
                    # Accumulate observation magnitudes...
                    cum_observation_mag = cum_observation_mag.abs()
                    for obs_ind in xrange(1, num_obs_points):
                        model_traject = self._model.integrate_state(model_traject[-1],
                                                                    checkpoints=observation_checkpoints[obs_ind:obs_ind+2]
                                                                    )
                        if isinstance(model_traject, list):
                            observation = self._model.evaluate_theoretical_observation(model_traject[-1])
                        elif isinstance(model_traject, StateVector):
                            observation = self._model.evaluate_theoretical_observation(model_traject)
                        else:
                            raise TypeError("The model output is not a trajectory or a StateVector!")
                        observation = observation.abs()
                        cum_observation_mag = cum_observation_mag.add(observation, in_place=True)
                        #
                    
                    # obtain observation (variances) from reference trajectory
                    cum_observation_mag = cum_observation_mag.scale(sigma_obs/float(num_obs_points))
                    cum_observation_mag = cum_observation_mag.get_numpy_array()
                    variances = cum_observation_mag  # Numpy array now
                    variances = np.squeeze(variances)
                    #
                    localize_errors_covariances = self._configs['localize_errors_covariances']
                    if localize_errors_covariances:
                        self.R = self._model.apply_observation_covariance_localization(ObservationMatrixNumpy(np.outer(variances, variances)))
                    else:
                        self.R = ObservationMatrixNumpy(np.outer(variances, variances))
                        #            
                    diagonal = self.R.diagonal()
                    self.R.set_diagonal(factor + (1.0-factor)*diagonal)
                    #   
                    # Try evaluating the determinant of the observation error covariance matrix
                    try:
                        self.detR = self.R.det()
                    except (OverflowError, AttributeError):
                        self.detR = np.infty
                    
                    if np.isinf(self.detR):
                        self.log_detR = np.infty
                    else:
                        self.log_detR = np.log(self.log_detR)
                        #

                #
                # construct inverse of the observation error covariance matrix if requested.
                if construct_inverse:
                    if self._R_structure == 'diagonal':
                        variances = self.R.diag()
                        try:
                            invR = sparse.lil_matrix((observation_size, observation_size))
                            invR.setdiag(1.0/variances, k=0)
                            invR = invR.tocsr()
                        except (TypeError):
                            indexes = np.empty((2, observation_size))
                            indexes[0, :] = np.arange(observation_size)
                            indexes[1, :] = np.arange(observation_size)
                            invR = sparse.csr_matrix((1.0/variances, indexes), shape=(observation_size, observation_size))
                        self.invR = ObservationMatrixSpSciPy(invR)
                        # Note that the way we initialize self.R here cannot be used with models with any kind of 
                        # ghost cells. It can be used with very simple models only, and should be derived and 
                        # updated for big models...
                        
                    #
                    elif self._R_structure == 'full':
                        try:
                            self.invR = self.R.inverse(in_place=False)
                        except (NotImplementedError):
                            self.invR = ObservationMatrixNumpy(np.linalg.inv(self.R.get_numpy_array()))
                    #
                    else:
                        print("The structure of R: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
                        raise ValueError

                #
                # construct the square root of the observation error covariance matrix if requested
                if construct_sqrtm:
                    if self._R_structure == 'diagonal':
                        variances = self.R.diag()
                        try:
                            sqrtR = sparse.lil_matrix((observation_size, observation_size))
                            sqrtR.setdiag(np.sqrt(variances), k=0)
                            sqrtR = sqrtR.tocsr()
                        except (TypeError):
                            indexes = np.empty((2, observation_size))
                            indexes[0, :] = np.arange(observation_size)
                            indexes[1, :] = np.arange(observation_size)
                            sqrtR = sparse.csr_matrix((np.sqrt(variances), indexes), shape=(observation_size, observation_size))
                        self.sqrtR = ObservationMatrixSpSciPy(sqrtR)
                        # Note that the way we initialize self.R here cannot be used with models with any kind of 
                        # ghost cells. It can be used with very simple models only, and should be derived and 
                        # updated for big models...
                    #
                    elif self._R_structure == 'full':
                        self.sqrtR = self.R.cholesky(in_place=False)
                    #
                    else:
                        print("The structure of R: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
                        raise ValueError
                        #
            else:
                print("The mothod ['%s'] chosen for construction of the observation errors is not yet supported." \
                      % errors_covariance_method)
                raise NotImplementedError()               
        #
        else:
            print("The probability distribution ['%s'] chosen for observation errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError
            #
    
    def error_covariances_inv_prod_vec(self, in_observation, in_place=True):
        """
        Evaluate and return the product of inverse of the error covariance matrix by a vector (in_observation)
        
        Args:
            in_observation: model.observation_vector
            in_place (default True): multiply the inverse of the covariance matrix of associated errors by the 
            passed vector in place. This overwrites the passed observation. 
                If False, a new model.observation_vector object is created and returned.
            
        Returns:
            scaled_observation: model.observation_vector; the product of inverse of the error covariance matrix 
                by in_observation
        
        """
        assert isinstance(in_observation, ObservationVector)
        #
        if self.invR is None:
            print("Failed to retrieve the inverse of the observation error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_observation = in_observation.copy()
        else:
            scaled_observation = in_observation

        if self._R_structure in ['diagonal', 'full']:
            scaled_observation = self.invR.vector_product(scaled_observation)
        else:
            print("The structure of R: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_observation
        #
    
    def error_covariances_sqrt_prod_vec(self, in_observation, in_place=True):
        """
        Evaluate and return the product of square root of the error covariance matrix by a vector (in_observation)
        
        Args:
            in_observation: model.observation_vector
            in_place (default True): multiply the square root of the covariance matrix of associated errors by the 
            passed vector in place. This overwrites the passed observation. 
                If False, a new model.observation_vector object is created and returned.
            
        Returns:
            scaled_observation: model.observation_vector; the product of square root of the error covariance matrix 
                by in_observation
        
        """
        assert isinstance(in_observation, ObservationVector)
        #
        if self.sqrtR is None:
            print("Failed to retrieve the square root of the observation error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_observation = in_observation.copy()
        else:
            scaled_observation = in_observation

        if self._R_structure in ['diagonal', 'full']:
            scaled_observation = self.sqrtR.vector_product(scaled_observation)
        else:
            print("The structure of R: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_observation
        #
    
    def error_covariances_prod_vec(self, in_observation, in_place=True):
        """
        Evaluate and return the product of the error covariance matrix by a vector (in_observation)
        
        Args:
            in_observation: model.observation_vector
            in_place (default True): multiply the covariance matrix of associated errors by the 
            passed vector in place. This overwrites the passed observation. 
                If False, a new model.observation_vector object is created and returned.
        
        Returns:
            scaled_observation: model.observation_vector; the product of the error covariance matrix 
                by in_observation
        
        """
        assert isinstance(in_observation, ObservationVector)
        #
        if self.R is None:
            print("Failed to retrieve the observation error coariance matrix!")
            raise ValueError

        if not in_place:
            scaled_observation = in_observation.copy()
        else:
            scaled_observation = in_observation

        if self._R_structure in ['diagonal', 'full']:
            scaled_observation = self.R.vector_product(scaled_observation)
        else:
            print("The structure of R: ['%s'] is not recognized! " % repr(['diagonal', 'full']))
            raise ValueError()
            #
        return scaled_observation
        #
    
    def generate_noise_vec(self):
        """
        Generate a random vector sampled from the Probability Distribution Fuction describing the errors
        implemented in this model. E.g. Gaussian observation noise vector with zero mean, and specified 
        observational error covariance matrix
        
        Args:
        
        Returns:
            randn_vec: model.observation_vector; a random noise vector sampled from the PDF of this errors model
        
        """
        errors_distribution = self._configs['errors_distribution'].lower()
        if errors_distribution == 'gaussian':
            # Generate Gaussian Random Noise Vector
            randn_numpy = utility.mvn_rand_vec(self._observation_vector_size)
            randn_vec = self._model.observation_vector()
            randn_vec[:] = randn_numpy[:]
            randn_vec = self.error_covariances_sqrt_prod_vec(randn_vec, in_place=True)
            #
            return randn_vec
            #
        else:
            print("The probability distribution ['%s'] chosen for observation errors is not yet supported." \
                  % errors_distribution)
            raise NotImplementedError
            #
            
            
