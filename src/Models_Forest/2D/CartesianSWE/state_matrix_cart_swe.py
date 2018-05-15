
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


from state_matrix_base import StateMatrixBase
from state_vector_cart_swe import StateVectorCartSWE
import cart_swe

class StateMatrixCartSWE(StateMatrixBase):
    """
    Cartesian shallow water equations Jacobian.

    NOTE: Has only Jacobian vector products (no actual matrix).
    
    Matrix constructor.

    Input:
        model_matrix_ref: a reference to the model's matrix object to be wrapped.
        
    """

    def __init__(self, time_point, in_state):
        
        self._time_point = time_point
        self._in_state = in_state

    def vector_product(self, vector):
        """
        Return the vector resulting from the right multiplication of the matrix and vector.

        Input:
            vector:

        Output:
            :
        """
        out_vector = StateVectorCartSWE()
        err = cart_swe.model_jac_vec(self._time_point, self._in_state, vector._raw_vector, out_vector._raw_vector)
        return out_vector

