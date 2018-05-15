
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


import sys
import numpy as np

from state_vector_numpy import StateVectorNumpy as StateVector
from state_matrix_numpy import StateMatrixNumpy as StateMatrix
from time_integration_base import TimeIntegratorBase as TimeIntegrator
import dates_utility as utility


class ErrorModelsBase(object):
    """
    Base class for models describing uncertatinties associated with different sources of information in the Data Assimilation framework.                                     
    This class should support all (at least basic) tasks/functionalities related to uncertainty and errors associated with information sources such as Model, Prior, and observations.
    
    Base class for models describing uncertatinties associated with different sources of information 
    in the Data Assimilation framework.
    A base class for error models such as:
        1. Observation errors model,
        2. Background errors model,
        3. Model errors model.
    
    Args:
        model: reference to the model object to which the error model will be attached.
        configs: a configurations dictionary for the error model to be created.
        This should override whatever present in the model configurations if is not None.
    
    """
    
    def __init__(self, model, configs=None, *argv, **kwargs):
        
        # self._model = model
        raise NotImplementedError()

    def construct_error_covariances(self, *argv, **kwargs):
        """
        Construct error covariance matrix/model of the errors attached to this specific source of information.
        For example construct the observation error covariance matrix of Gaussian observational errors.
        
        Args:
            *argv:
            **kwargs:
            
        Returns:
        
        """
        raise NotImplementedError()

    def error_covariances_inv_prod_vec(self, in_state, in_place=True, *argv, **kwargs):
        """
        Evaluate and return the product of inverse of the error covariance matrix by a vector
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the inverse of the covariance matrix of associated errors by the 
            passed vector in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
            *argv:
            **kwargs:
            
        
        Returns:
            out_state: model.state_vecotr; the product of inverse of the error covariance matrix by in_state
        
        """
        raise NotImplementedError()

    def error_covariances_sqrt_prod_vec(self, in_state, in_place=True, *argv, **kwargs):
        """
        Evaluate and return the product of square root of the error covariance matrix by a vector
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the square root (e.g. Cholesky decomposition) of the covariance 
            matrix of associated errors by the passed vector in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
            *argv:
            **kwargs:
            
            
        Returns:
            out_state: model.state_vecotr; the product of square root of the error covariance matrix by in_state.
        
        """
        raise NotImplementedError()

    def error_covariances_prod_vec(self, in_state, in_place=True, *argv, **kwargs):
        """
        Evaluate and return the product of the error covariance matrix by a vector.
        This method might be not needed!
        
        Args:
            in_state: model.state_vector
            in_place (default True): multiply the covariance matrix of associated errors by the passed vector 
                in place. This overwrites the passed state. 
                If False, a new model.state_vector object is created and returned.
            *argv:
            **kwargs:
            
        
        Returns:
            out_state: model.state_vecotr; the product of the error covariance matrix by in_state.
        
        """
        raise NotImplementedError()

    def generate_noise_vec(self, *argv, **kwargs):
        """
        Generate a random vector sampled from the Probability Distribution Fuction describing the errors
        implemented in this model. E.g. Gaussian observation noise vector with zero mean, and specified 
        observational error covariance matrix
        
        Args:
            *argv:
            **kwargs:
            
        Returns:
            out_state: model.state_vecotr; a random noise vector sampled from the PDF of this errors model
        
        """
        raise NotImplementedError()
        
        
